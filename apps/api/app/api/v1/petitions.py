import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from app.config import settings
from app.dependencies import get_current_user, get_tenant_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/petitions", tags=["petitions"])

AI_LABEL = "Conteúdo gerado com auxílio de IA — CNJ Res. 615/2025"
TABLE = "petitions"


class PetitionCreate(BaseModel):
    title: str
    case_id: Optional[str] = None
    petition_type: Optional[str] = None
    content: Optional[str] = None


class PetitionUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    status: Optional[str] = None
    petition_type: Optional[str] = None


class PetitionGenerateRequest(BaseModel):
    petition_type: str = "peticao_inicial"
    case_id: Optional[str] = None
    title: str = ""
    case_context: dict[str, Any] = {}
    variables: dict[str, str] = {}


class PetitionResponse(BaseModel):
    id: str
    title: str
    petition_type: Optional[str] = None
    content: Optional[str] = None
    status: str = "draft"
    case_id: Optional[str] = None
    ai_generated: bool = False
    ai_label: str = AI_LABEL
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class PetitionListResponse(BaseModel):
    petitions: list[PetitionResponse]
    total: int


def _headers():
    return {
        "apikey": settings.SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _base_url():
    return f"{settings.SUPABASE_URL}/rest/v1/{TABLE}"


@router.get("", response_model=PetitionListResponse)
async def list_petitions(
    case_id: Optional[str] = None,
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    params: dict[str, str] = {
        "tenant_id": f"eq.{tenant_id}",
        "order": "created_at.desc",
        "offset": str(skip),
        "limit": str(limit),
        "select": "*",
    }
    if case_id:
        params["case_id"] = f"eq.{case_id}"
    if status:
        params["status"] = f"eq.{status}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(_base_url(), headers=_headers(), params=params)
            if resp.status_code == 200:
                rows = resp.json()
                return PetitionListResponse(
                    petitions=[_row_to_response(r) for r in rows],
                    total=len(rows),
                )
    except Exception as e:
        logger.error("List petitions failed: %s", e)

    return PetitionListResponse(petitions=[], total=0)


@router.post("", response_model=PetitionResponse, status_code=201)
async def create_petition(
    petition: PetitionCreate,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    now = datetime.now(timezone.utc).isoformat()
    data = {
        "id": str(uuid.uuid4()),
        "tenant_id": tenant_id,
        "title": petition.title,
        "petition_type": petition.petition_type,
        "content": petition.content or "",
        "status": "draft",
        "case_id": petition.case_id,
        "ai_generated": False,
        "created_by": user.get("id", ""),
        "created_at": now,
        "updated_at": now,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(_base_url(), headers=_headers(), json=data)
            if resp.status_code in (200, 201):
                rows = resp.json()
                return _row_to_response(rows[0] if isinstance(rows, list) else rows)
    except Exception as e:
        logger.error("Create petition failed: %s", e)

    return _row_to_response(data)


@router.post("/generate", response_model=PetitionResponse, status_code=201)
async def generate_petition(
    request: PetitionGenerateRequest,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    """Generate a petition using AI."""
    from app.services.petition.generator import PetitionGenerator

    generator = PetitionGenerator()

    content = await generator.generate(
        petition_type=request.petition_type,
        case_context=request.case_context,
        template_id=request.petition_type,
        tenant_id=tenant_id,
        variables=request.variables,
    )

    now = datetime.now(timezone.utc).isoformat()
    data = {
        "id": str(uuid.uuid4()),
        "tenant_id": tenant_id,
        "title": request.title or f"Petição - {request.petition_type}",
        "petition_type": request.petition_type,
        "content": content,
        "status": "draft",
        "case_id": request.case_id,
        "ai_generated": True,
        "created_by": user.get("id", ""),
        "created_at": now,
        "updated_at": now,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(_base_url(), headers=_headers(), json=data)
            if resp.status_code in (200, 201):
                rows = resp.json()
                return _row_to_response(rows[0] if isinstance(rows, list) else rows)
    except Exception as e:
        logger.error("Save generated petition failed: %s", e)

    return _row_to_response(data)


@router.get("/templates")
async def get_templates(
    category: Optional[str] = None,
    user: dict = Depends(get_current_user),
):
    from app.services.petition.templates import list_templates
    return {"templates": list_templates(category=category)}


@router.get("/{petition_id}", response_model=PetitionResponse)
async def get_petition(
    petition_id: str,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                _base_url(),
                headers=_headers(),
                params={"id": f"eq.{petition_id}", "tenant_id": f"eq.{tenant_id}"},
            )
            if resp.status_code == 200:
                rows = resp.json()
                if rows:
                    return _row_to_response(rows[0])
    except Exception as e:
        logger.error("Get petition failed: %s", e)

    raise HTTPException(status_code=404, detail="Petition not found")


@router.patch("/{petition_id}", response_model=PetitionResponse)
async def update_petition(
    petition_id: str,
    update: PetitionUpdate,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    update_data = {k: v for k, v in update.model_dump().items() if v is not None}
    update_data["updated_at"] = datetime.now(timezone.utc).isoformat()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.patch(
                _base_url(),
                headers=_headers(),
                params={"id": f"eq.{petition_id}", "tenant_id": f"eq.{tenant_id}"},
                json=update_data,
            )
            if resp.status_code == 200:
                rows = resp.json()
                if rows:
                    return _row_to_response(rows[0])
    except Exception as e:
        logger.error("Update petition failed: %s", e)

    raise HTTPException(status_code=404, detail="Petition not found")


@router.delete("/{petition_id}", status_code=204)
async def delete_petition(
    petition_id: str,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.delete(
                _base_url(),
                headers={**_headers(), "Prefer": "return=minimal"},
                params={"id": f"eq.{petition_id}", "tenant_id": f"eq.{tenant_id}"},
            )
            if resp.status_code in (200, 204):
                return
    except Exception as e:
        logger.error("Delete petition failed: %s", e)

    raise HTTPException(status_code=404, detail="Petition not found")


class FormatABNTRequest(BaseModel):
    content: str
    include_oab: bool = False
    ai_generated: bool = False


@router.post("/format-abnt")
async def format_abnt(
    request: FormatABNTRequest,
    user: dict = Depends(get_current_user),
):
    """Apply ABNT formatting to petition content."""
    from app.services.petition.formatter import PetitionFormatter

    formatter = PetitionFormatter()
    if request.include_oab:
        formatted = formatter.format_oab(request.content)
    else:
        formatted = formatter.format_abnt(request.content)

    if request.ai_generated:
        formatted = formatter.add_ai_label(formatted)

    return {"content": formatted}


@router.post("/{petition_id}/verify-citations")
async def verify_citations(
    petition_id: str,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    """Verify all citations in a petition."""
    # Fetch petition content
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                _base_url(),
                headers=_headers(),
                params={"id": f"eq.{petition_id}", "tenant_id": f"eq.{tenant_id}", "select": "content"},
            )
            if resp.status_code != 200 or not resp.json():
                raise HTTPException(status_code=404, detail="Petition not found")
            content = resp.json()[0].get("content", "")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    from app.services.petition.citation_verifier import CitationVerifier

    verifier = CitationVerifier()
    citations = await verifier.verify_all(content)

    return {
        "petition_id": petition_id,
        "total_citations": len(citations),
        "verified": sum(1 for c in citations if c.status == "verified"),
        "not_found": sum(1 for c in citations if c.status == "not_found"),
        "revoked": sum(1 for c in citations if c.status == "revoked"),
        "citations": [c.model_dump() for c in citations],
    }


async def _fetch_petition_row(petition_id: str, tenant_id: str) -> dict:
    """Fetch a single petition row from Supabase."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            _base_url(),
            headers=_headers(),
            params={"id": f"eq.{petition_id}", "tenant_id": f"eq.{tenant_id}"},
        )
        if resp.status_code != 200 or not resp.json():
            raise HTTPException(status_code=404, detail="Petition not found")
        return resp.json()[0]


@router.get("/{petition_id}/export/pdf")
async def export_petition_pdf(
    petition_id: str,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    """Export petition as PDF with ABNT formatting."""
    row = await _fetch_petition_row(petition_id, tenant_id)

    from app.services.petition.exporter import export_pdf

    try:
        pdf_bytes = export_pdf(
            content=row.get("content", ""),
            title=row.get("title", "Peticao"),
            ai_generated=row.get("ai_generated", False),
        )
    except Exception as e:
        logger.error("PDF export failed: %s", e)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

    filename = f"{row.get('title', 'peticao')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{petition_id}/export/docx")
async def export_petition_docx(
    petition_id: str,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    """Export petition as DOCX with ABNT formatting."""
    row = await _fetch_petition_row(petition_id, tenant_id)

    from app.services.petition.exporter import export_docx

    try:
        docx_bytes = export_docx(
            content=row.get("content", ""),
            title=row.get("title", "Peticao"),
            ai_generated=row.get("ai_generated", False),
        )
    except Exception as e:
        logger.error("DOCX export failed: %s", e)
        raise HTTPException(status_code=500, detail=f"DOCX generation failed: {e}")

    filename = f"{row.get('title', 'peticao')}.docx"
    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _row_to_response(row: dict) -> PetitionResponse:
    return PetitionResponse(
        id=row.get("id", ""),
        title=row.get("title", ""),
        petition_type=row.get("petition_type"),
        content=row.get("content"),
        status=row.get("status", "draft"),
        case_id=row.get("case_id"),
        ai_generated=row.get("ai_generated", False),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )
