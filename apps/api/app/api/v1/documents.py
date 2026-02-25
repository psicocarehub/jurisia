import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel

from app.config import settings
from app.dependencies import get_current_user, get_tenant_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

TABLE = "documents"


class DocumentResponse(BaseModel):
    id: str
    title: str
    doc_type: Optional[str] = None
    source: Optional[str] = None
    ocr_status: str = "pending"
    classification_label: Optional[str] = None
    storage_key: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    created_at: Optional[str] = None

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
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


def _row_to_response(row: dict) -> DocumentResponse:
    return DocumentResponse(
        id=str(row.get("id", "")),
        title=row.get("title", ""),
        doc_type=row.get("doc_type"),
        source=row.get("source"),
        ocr_status=row.get("ocr_status", "pending"),
        classification_label=row.get("classification_label"),
        storage_key=row.get("file_path"),
        file_size=row.get("file_size"),
        mime_type=row.get("mime_type"),
        created_at=row.get("created_at"),
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    case_id: Optional[str] = None,
    doc_type: Optional[str] = None,
    ocr_status: Optional[str] = None,
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
    if case_id:
        params["case_id"] = f"eq.{case_id}"
    if doc_type:
        params["doc_type"] = f"eq.{doc_type}"
    if ocr_status:
        params["ocr_status"] = f"eq.{ocr_status}"

    try:
        rows, total = await supabase_list(TABLE, params=params, skip=skip, limit=limit)
    except Exception as e:
        logger.error("Failed to list documents: %s", e)
        rows, total = [], 0

    return DocumentListResponse(
        documents=[_row_to_response(r) for r in rows],
        total=total,
    )


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    case_id: Optional[str] = None,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    from app.services.storage.service import StorageService

    file_data = await file.read()
    storage = StorageService()

    storage_key = await storage.upload(
        data=file_data,
        filename=file.filename or "document",
        tenant_id=tenant_id,
        content_type=file.content_type or "application/octet-stream",
    )

    doc_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "id": doc_id,
        "tenant_id": tenant_id,
        "case_id": case_id,
        "title": file.filename or "Untitled",
        "source": "upload",
        "mime_type": file.content_type,
        "file_size": len(file_data),
        "file_path": storage_key,
        "ocr_status": "pending",
        "uploaded_by": user.get("id", ""),
        "created_at": now,
        "updated_at": now,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{_base_url()}/{TABLE}", headers=_headers(), json=payload
            )
            resp.raise_for_status()
            rows = resp.json()
            row = rows[0] if isinstance(rows, list) else rows
    except Exception as e:
        logger.error("Failed to create document: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    from app.services.document_pipeline import process_document

    background_tasks.add_task(
        process_document,
        document_id=doc_id,
        storage_key=storage_key,
        tenant_id=tenant_id,
        filename=file.filename or "document",
        mime_type=file.content_type or "",
    )

    return _row_to_response(row)


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{_base_url()}/{TABLE}",
                headers={**_headers(), "Accept": "application/vnd.pgrst.object+json"},
                params={"id": f"eq.{document_id}", "tenant_id": f"eq.{tenant_id}"},
            )
            if resp.status_code == 406:
                raise HTTPException(status_code=404, detail="Document not found")
            resp.raise_for_status()
            row = resp.json()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get document: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    return _row_to_response(row)


@router.get("/{document_id}/download")
async def download_document(
    document_id: str,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{_base_url()}/{TABLE}",
                headers={**_headers(), "Accept": "application/vnd.pgrst.object+json"},
                params={"id": f"eq.{document_id}", "tenant_id": f"eq.{tenant_id}"},
            )
            if resp.status_code == 406:
                raise HTTPException(status_code=404, detail="Document not found")
            resp.raise_for_status()
            row = resp.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    storage_key = row.get("file_path")
    if not storage_key:
        raise HTTPException(status_code=404, detail="File not in storage")

    from app.services.storage.service import StorageService

    storage = StorageService()
    file_data = await storage.download(storage_key)
    if not file_data:
        raise HTTPException(status_code=404, detail="File not found in storage")

    return Response(
        content=file_data,
        media_type=row.get("mime_type") or "application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{row.get("title", "doc")}"'},
    )


@router.get("/{document_id}/status")
async def document_status(
    document_id: str,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{_base_url()}/{TABLE}",
                headers={**_headers(), "Accept": "application/vnd.pgrst.object+json"},
                params={
                    "select": "id,ocr_status,classification_label,title",
                    "id": f"eq.{document_id}",
                    "tenant_id": f"eq.{tenant_id}",
                },
            )
            if resp.status_code == 406:
                raise HTTPException(status_code=404, detail="Document not found")
            resp.raise_for_status()
            row = resp.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "id": row.get("id"),
        "ocr_status": row.get("ocr_status"),
        "classification_label": row.get("classification_label"),
        "title": row.get("title"),
    }


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{_base_url()}/{TABLE}",
                headers={**_headers(), "Accept": "application/vnd.pgrst.object+json"},
                params={"id": f"eq.{document_id}", "tenant_id": f"eq.{tenant_id}"},
            )
            if resp.status_code == 406:
                raise HTTPException(status_code=404, detail="Document not found")
            resp.raise_for_status()
            row = resp.json()

            storage_key = row.get("file_path")
            if storage_key:
                from app.services.storage.service import StorageService
                storage = StorageService()
                await storage.delete(storage_key)

            await client.delete(
                f"{_base_url()}/{TABLE}",
                headers=_headers(),
                params={"id": f"eq.{document_id}", "tenant_id": f"eq.{tenant_id}"},
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete document: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
