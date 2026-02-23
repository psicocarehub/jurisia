import logging
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_user, get_tenant_id, get_db
from app.db.models import Document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentResponse(BaseModel):
    id: str
    title: str
    doc_type: Optional[str] = None
    source: Optional[str] = None
    ocr_status: str = "pending"
    classification_label: Optional[str] = None
    storage_key: Optional[str] = None
    file_size: Optional[int] = None
    created_at: Optional[str] = None

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int


def _doc_to_response(d: Document) -> DocumentResponse:
    return DocumentResponse(
        id=str(d.id),
        title=d.title,
        doc_type=d.doc_type,
        source=d.source,
        ocr_status=d.ocr_status,
        classification_label=d.classification_label,
        storage_key=getattr(d, "storage_key", None),
        file_size=getattr(d, "file_size", None),
        created_at=str(d.created_at) if d.created_at else None,
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
    db: AsyncSession = Depends(get_db),
):
    query = select(Document).where(Document.tenant_id == uuid.UUID(tenant_id))
    if case_id:
        query = query.where(Document.case_id == uuid.UUID(case_id))
    if doc_type:
        query = query.where(Document.doc_type == doc_type)
    if ocr_status:
        query = query.where(Document.ocr_status == ocr_status)
    query = query.offset(skip).limit(limit)

    result = await db.execute(query)
    documents = result.scalars().all()

    return DocumentListResponse(
        documents=[_doc_to_response(d) for d in documents],
        total=len(documents),
    )


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    case_id: Optional[str] = None,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
    db: AsyncSession = Depends(get_db),
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

    doc = Document(
        tenant_id=uuid.UUID(tenant_id),
        case_id=uuid.UUID(case_id) if case_id else None,
        title=file.filename or "Untitled",
        source="upload",
        mime_type=file.content_type,
        file_size=len(file_data),
        ocr_status="pending",
        uploaded_by=uuid.UUID(user["id"]),
        storage_key=storage_key,
    )
    db.add(doc)
    await db.flush()

    doc_id = str(doc.id)

    from app.services.document_pipeline import process_document

    background_tasks.add_task(
        process_document,
        document_id=doc_id,
        storage_key=storage_key,
        tenant_id=tenant_id,
        filename=file.filename or "document",
        mime_type=file.content_type or "",
    )

    return _doc_to_response(doc)


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Document).where(
            Document.id == uuid.UUID(document_id),
            Document.tenant_id == uuid.UUID(tenant_id),
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return _doc_to_response(doc)


@router.get("/{document_id}/download")
async def download_document(
    document_id: str,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Document).where(
            Document.id == uuid.UUID(document_id),
            Document.tenant_id == uuid.UUID(tenant_id),
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    storage_key = getattr(doc, "storage_key", None)
    if not storage_key:
        raise HTTPException(status_code=404, detail="File not in storage")

    from app.services.storage.service import StorageService

    storage = StorageService()
    file_data = await storage.download(storage_key)
    if not file_data:
        raise HTTPException(status_code=404, detail="File not found in storage")

    return Response(
        content=file_data,
        media_type=doc.mime_type or "application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{doc.title}"',
        },
    )


@router.get("/{document_id}/status")
async def document_status(
    document_id: str,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Document).where(
            Document.id == uuid.UUID(document_id),
            Document.tenant_id == uuid.UUID(tenant_id),
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "id": str(doc.id),
        "ocr_status": doc.ocr_status,
        "classification_label": doc.classification_label,
        "title": doc.title,
    }


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Document).where(
            Document.id == uuid.UUID(document_id),
            Document.tenant_id == uuid.UUID(tenant_id),
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    storage_key = getattr(doc, "storage_key", None)
    if storage_key:
        from app.services.storage.service import StorageService

        storage = StorageService()
        await storage.delete(storage_key)

    await db.delete(doc)
