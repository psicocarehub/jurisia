import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.config import settings
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


@router.get("", response_model=CaseListResponse)
async def list_cases(
    area: Optional[str] = None,
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    params: dict = {
        "select": "*",
        "tenant_id": f"eq.{tenant_id}",
        "order": "created_at.desc",
        "offset": str(skip),
        "limit": str(limit),
    }
    if area:
        params["area"] = f"eq.{area}"
    if status:
        params["status"] = f"eq.{status}"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{_base_url()}/{TABLE}", headers=_headers(), params=params
            )
            resp.raise_for_status()
            rows = resp.json()
    except Exception as e:
        logger.error("Failed to list cases: %s", e)
        rows = []

    cases = [_row_to_response(r) for r in rows]
    return CaseListResponse(cases=cases, total=len(cases))


@router.post("", response_model=CaseResponse, status_code=201)
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
