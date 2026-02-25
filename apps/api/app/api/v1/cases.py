import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from app.config import settings
from app.core.validators import validate_cnj, validate_client_document
from app.dependencies import get_current_user, get_tenant_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cases", tags=["cases"])

TABLE = "cases"


class CaseCreate(BaseModel):
    title: str
    cnj_number: Optional[str] = None
    description: Optional[str] = None
    area: Optional[str] = None
    client_name: Optional[str] = None
    client_document: Optional[str] = None
    opposing_party: Optional[str] = None
    court: Optional[str] = None
    judge_name: Optional[str] = None
    estimated_value: Optional[float] = None

    _validate_cnj = field_validator("cnj_number", mode="before")(validate_cnj)
    _validate_doc = field_validator("client_document", mode="before")(validate_client_document)


class CaseUpdate(BaseModel):
    title: Optional[str] = None
    cnj_number: Optional[str] = None
    description: Optional[str] = None
    area: Optional[str] = None
    status: Optional[str] = None
    client_name: Optional[str] = None
    opposing_party: Optional[str] = None
    court: Optional[str] = None
    judge_name: Optional[str] = None

    _validate_cnj = field_validator("cnj_number", mode="before")(validate_cnj)


class CaseResponse(BaseModel):
    id: str
    title: str
    cnj_number: Optional[str] = None
    description: Optional[str] = None
    area: Optional[str] = None
    status: str = "active"
    client_name: Optional[str] = None
    client_document: Optional[str] = None
    opposing_party: Optional[str] = None
    court: Optional[str] = None
    judge_name: Optional[str] = None
    estimated_value: Optional[float] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    model_config = {"from_attributes": True}


class CaseListResponse(BaseModel):
    cases: list[CaseResponse]
    total: int


def _headers():
    return {
        "apikey": settings.SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _base_url():
    return f"{settings.SUPABASE_URL}/rest/v1"


def _row_to_response(row: dict) -> CaseResponse:
    return CaseResponse(
        id=str(row.get("id", "")),
        title=row.get("title", ""),
        cnj_number=row.get("cnj_number"),
        description=row.get("description"),
        area=row.get("area"),
        status=row.get("status", "active"),
        client_name=row.get("client_name"),
        client_document=row.get("client_document"),
        opposing_party=row.get("opposing_party"),
        court=row.get("court"),
        judge_name=row.get("judge_name"),
        estimated_value=row.get("estimated_value"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


@router.get(
    "",
    response_model=CaseListResponse,
    summary="Listar processos",
    description="Retorna processos paginados do tenant autenticado com contagem total exata.",
    responses={401: {"description": "Token JWT inválido ou ausente"}},
)
async def list_cases(
    area: Optional[str] = None,
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    from app.api.v1.helpers import supabase_list

    params: dict[str, str] = {
        "select": "*",
        "tenant_id": f"eq.{tenant_id}",
        "order": "created_at.desc",
    }
    if area:
        params["area"] = f"eq.{area}"
    if status:
        params["status"] = f"eq.{status}"

    try:
        rows, total = await supabase_list(TABLE, params=params, skip=skip, limit=limit)
    except Exception as e:
        logger.error("Failed to list cases: %s", e)
        rows, total = [], 0

    cases = [_row_to_response(r) for r in rows]
    return CaseListResponse(cases=cases, total=total)


@router.post(
    "",
    response_model=CaseResponse,
    status_code=201,
    summary="Criar processo",
    description="Cria um novo processo. Valida CNJ e CPF/CNPJ automaticamente.",
    responses={422: {"description": "Dados inválidos (CNJ, CPF/CNPJ)"}},
)
async def create_case(
    case_data: CaseCreate,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    case_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "id": case_id,
        "tenant_id": tenant_id,
        "created_at": now,
        "updated_at": now,
        "status": "active",
        **case_data.model_dump(exclude_none=True),
    }
    user_id = user.get("id", "")
    if user_id:
        payload["created_by"] = user_id

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{_base_url()}/{TABLE}", headers=_headers(), json=payload
            )
            if resp.status_code == 409 and "created_by" in payload:
                payload.pop("created_by", None)
                resp = await client.post(
                    f"{_base_url()}/{TABLE}", headers=_headers(), json=payload
                )
            resp.raise_for_status()
            rows = resp.json()
            row = rows[0] if isinstance(rows, list) else rows
    except Exception as e:
        logger.error("Failed to create case: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    return _row_to_response(row)


@router.get("/{case_id}", response_model=CaseResponse)
async def get_case(
    case_id: str,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{_base_url()}/{TABLE}",
                headers={**_headers(), "Accept": "application/vnd.pgrst.object+json"},
                params={"id": f"eq.{case_id}", "tenant_id": f"eq.{tenant_id}"},
            )
            if resp.status_code == 406:
                raise HTTPException(status_code=404, detail="Case not found")
            resp.raise_for_status()
            row = resp.json()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get case: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    return _row_to_response(row)


@router.patch("/{case_id}", response_model=CaseResponse)
async def update_case(
    case_id: str,
    case_data: CaseUpdate,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    update_data = case_data.model_dump(exclude_none=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No data to update")

    update_data["updated_at"] = datetime.now(timezone.utc).isoformat()

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.patch(
                f"{_base_url()}/{TABLE}",
                headers=_headers(),
                params={"id": f"eq.{case_id}", "tenant_id": f"eq.{tenant_id}"},
                json=update_data,
            )
            resp.raise_for_status()
            rows = resp.json()
            if not rows:
                raise HTTPException(status_code=404, detail="Case not found")
            row = rows[0] if isinstance(rows, list) else rows
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update case: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    return _row_to_response(row)


@router.post(
    "/{case_id}/analyze",
    summary="Raio-X do Processo",
    description=(
        "Executa análise completa do processo: brechas, estratégias, "
        "perfil do juiz, predição e jurisprudência similar via RAG."
    ),
    responses={
        404: {"description": "Processo não encontrado"},
        500: {"description": "Falha na análise (serviço externo indisponível)"},
    },
)
async def analyze_case(
    case_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    from app.services.case_analyzer import CaseAnalyzer

    analyzer = CaseAnalyzer()
    try:
        result = await analyzer.analyze(case_id, tenant_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Case analysis failed for %s: %s", case_id, e)
        raise HTTPException(status_code=500, detail=f"Análise falhou: {e}")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.patch(
                f"{_base_url()}/{TABLE}",
                headers=_headers(),
                params={"id": f"eq.{case_id}", "tenant_id": f"eq.{tenant_id}"},
                json={
                    "last_analysis": result.model_dump(),
                    "updated_at": result.generated_at,
                },
            )
    except Exception as e:
        logger.warning("Failed to cache analysis: %s", e)

    return result.model_dump()


@router.get(
    "/{case_id}/analysis",
    summary="Obter última análise",
    description="Retorna a última análise Raio-X em cache para o processo.",
    responses={404: {"description": "Processo ou análise não encontrada"}},
)
async def get_case_analysis(
    case_id: str,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    """Get the last cached analysis for a case."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{_base_url()}/{TABLE}",
                headers={**_headers(), "Accept": "application/vnd.pgrst.object+json"},
                params={
                    "id": f"eq.{case_id}",
                    "tenant_id": f"eq.{tenant_id}",
                    "select": "id,last_analysis",
                },
            )
            if resp.status_code == 406:
                raise HTTPException(status_code=404, detail="Case not found")
            resp.raise_for_status()
            row = resp.json()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get analysis: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    analysis = row.get("last_analysis")
    if not analysis:
        raise HTTPException(
            status_code=404,
            detail="Nenhuma análise encontrada. Execute POST /analyze primeiro.",
        )
    return analysis


@router.get("/{case_id}/related-updates")
async def get_related_updates(
    case_id: str,
    days: int = 30,
    limit: int = 20,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    """Get content_updates related to a case's area, from the last N days."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{_base_url()}/{TABLE}",
                headers={**_headers(), "Accept": "application/vnd.pgrst.object+json"},
                params={"id": f"eq.{case_id}", "tenant_id": f"eq.{tenant_id}", "select": "area,title"},
            )
            if resp.status_code == 406:
                raise HTTPException(status_code=404, detail="Case not found")
            resp.raise_for_status()
            case_data = resp.json()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get case for related updates: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    case_area = (case_data.get("area") or "").lower()
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

    params = {
        "select": "id,title,category,subcategory,summary,court_or_organ,publication_date,source_url,relevance_score,areas,captured_at",
        "captured_at": f"gte.{since}T00:00:00Z",
        "order": "relevance_score.desc",
        "limit": str(limit),
    }
    if case_area:
        params["areas"] = f"cs.{{{case_area}}}"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{_base_url()}/content_updates",
                headers=_headers(),
                params=params,
            )
            resp.raise_for_status()
            updates = resp.json()
    except Exception as e:
        logger.error("Failed to fetch related updates: %s", e)
        updates = []

    return {"case_id": case_id, "updates": updates, "total": len(updates)}


@router.get(
    "/{case_id}/analyze/stream",
    summary="Raio-X com streaming SSE",
    description="Executa o Raio-X e transmite progresso em tempo real via Server-Sent Events.",
)
async def analyze_case_stream(
    case_id: str,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    """Stream analysis progress via SSE."""
    from app.services.case_analyzer import CaseAnalyzer, CaseAnalysis

    async def event_stream() -> AsyncGenerator[str, None]:
        analyzer = CaseAnalyzer()

        def sse(step: str, progress: int, data: Optional[dict] = None) -> str:
            payload = {"step": step, "progress": progress}
            if data:
                payload["data"] = data
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        try:
            yield sse("fetching_case", 5)
            case_data = await analyzer._fetch_case(case_id, tenant_id)
            if not case_data:
                yield sse("error", 0, {"message": f"Caso {case_id} não encontrado"})
                return

            yield sse("fetching_documents", 15)
            documents = await analyzer._fetch_case_documents(case_id, tenant_id)
            doc_content = await analyzer._fetch_indexed_content(documents)

            yield sse("searching_jurisprudence", 30)
            query_text = analyzer._build_search_query(case_data, doc_content)
            similar_cases = await analyzer._retrieve_similar(query_text, tenant_id)

            judge_profile = None
            judge_name = case_data.get("judge_name")
            if judge_name:
                yield sse("judge_profile", 45)
                judge_profile = await analyzer._get_judge_profile(
                    judge_name, case_data.get("court")
                )

            yield sse("outcome_prediction", 55)
            prediction = await analyzer._predict_outcome(case_data, doc_content)

            yield sse("llm_analysis", 70)
            llm_result = await analyzer._run_llm_analysis(
                case_data=case_data,
                doc_content=doc_content,
                similar_cases=similar_cases,
                judge_profile=judge_profile,
                prediction=prediction,
            )

            yield sse("finalizing", 90)
            from app.services.case_analyzer import TimelineEvent, Vulnerability, Strategy
            similar_cases_summary = [
                {
                    "title": c.get("title", c.get("document_title", "")),
                    "court": c.get("court", ""),
                    "date": c.get("date", ""),
                    "doc_type": c.get("doc_type", ""),
                    "snippet": c.get("content", "")[:300],
                }
                for c in similar_cases
            ]

            analysis = CaseAnalysis(
                case_id=case_id,
                summary=llm_result.get("summary", ""),
                timeline=[TimelineEvent(**e) for e in llm_result.get("timeline", [])],
                legal_framework=llm_result.get("legal_framework", ""),
                vulnerabilities=[Vulnerability(**v) for v in llm_result.get("vulnerabilities", [])],
                strategies=[Strategy(**s) for s in llm_result.get("strategies", [])],
                judge_profile=judge_profile,
                prediction=prediction,
                similar_cases=similar_cases_summary,
                risk_level=llm_result.get("risk_level", "médio"),
                risk_assessment=llm_result.get("risk_assessment", ""),
                model_used=llm_result.get("_model_used", ""),
                generated_at=datetime.now(timezone.utc).isoformat(),
            )

            from app.services.cache import CacheService
            _cache = CacheService()
            await _cache.set(
                f"case_analysis:{tenant_id}:{case_id}",
                analysis.model_dump(),
                ttl=CacheService.TTL_ANALYSIS,
            )

            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.patch(
                        f"{_base_url()}/{TABLE}",
                        headers=_headers(),
                        params={"id": f"eq.{case_id}", "tenant_id": f"eq.{tenant_id}"},
                        json={
                            "last_analysis": analysis.model_dump(),
                            "updated_at": analysis.generated_at,
                        },
                    )
            except Exception:
                pass

            yield sse("complete", 100, analysis.model_dump())

        except Exception as e:
            logger.error("SSE analysis failed for %s: %s", case_id, e)
            yield sse("error", 0, {"message": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/{case_id}", status_code=204)
async def delete_case(
    case_id: str,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.delete(
                f"{_base_url()}/{TABLE}",
                headers=_headers(),
                params={"id": f"eq.{case_id}", "tenant_id": f"eq.{tenant_id}"},
            )
            resp.raise_for_status()
    except Exception as e:
        logger.error("Failed to delete case: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
