import uuid

from fastapi import APIRouter, Depends, UploadFile, File
from pydantic import BaseModel
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_user, get_tenant_id, get_db
from app.db.models import Document

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentResponse(BaseModel):
    id: str
    title: str
    doc_type: Optional[str] = None
    source: Optional[str] = None
    ocr_status: str = "pending"
    classification_label: Optional[str] = None
    created_at: Optional[str] = None

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    case_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    query = select(Document).where(Document.tenant_id == uuid.UUID(tenant_id))
    if case_id:
        query = query.where(Document.case_id == uuid.UUID(case_id))
    query = query.offset(skip).limit(limit)

    result = await db.execute(query)
    documents = result.scalars().all()

    return DocumentListResponse(
        documents=[
            DocumentResponse(
                id=str(d.id),
                title=d.title,
                doc_type=d.doc_type,
                source=d.source,
                ocr_status=d.ocr_status,
                classification_label=d.classification_label,
                created_at=str(d.created_at) if d.created_at else None,
            )
            for d in documents
        ],
        total=len(documents),
    )


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    case_id: Optional[str] = None,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    doc = Document(
        tenant_id=uuid.UUID(tenant_id),
        case_id=uuid.UUID(case_id) if case_id else None,
        title=file.filename or "Untitled",
        source="upload",
        mime_type=file.content_type,
        file_size=file.size,
        ocr_status="pending",
        uploaded_by=uuid.UUID(user["id"]),
    )
    db.add(doc)
    await db.flush()

    # TODO: Save file to S3/MinIO and trigger OCR pipeline

    return DocumentResponse(
        id=str(doc.id),
        title=doc.title,
        doc_type=doc.doc_type,
        source=doc.source,
        ocr_status=doc.ocr_status,
    )
