"""
Document processing pipeline: OCR -> Classification -> Chunking -> RAG Indexing.

Designed to run as a background task (via BackgroundTasks or Celery).
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


async def process_document(
    document_id: str,
    storage_key: str,
    tenant_id: str,
    filename: str,
    mime_type: str = "",
) -> None:
    """
    Full document processing pipeline:
    1. Download from storage
    2. OCR (if PDF/image)
    3. Classify document type
    4. Chunk text
    5. Index into Elasticsearch + Qdrant
    6. Update document record with results
    """
    from app.services.storage.service import StorageService
    from app.services.nlp.classifier import DocumentClassifier
    from app.services.rag.chunker import LegalChunker
    from app.services.rag.indexer import IncrementalIndexer

    storage = StorageService()
    classifier = DocumentClassifier()

    try:
        await _update_doc_status(document_id, "processing")

        file_data = await storage.download(storage_key)
        if not file_data:
            await _update_doc_status(document_id, "error", error="File not found in storage")
            return

        text = ""
        is_pdf = mime_type == "application/pdf" or filename.lower().endswith(".pdf")
        is_image = mime_type.startswith("image/") if mime_type else False

        if is_pdf or is_image:
            text = await _run_ocr(file_data, filename, mime_type)
        else:
            try:
                text = file_data.decode("utf-8", errors="replace")
            except Exception:
                text = file_data.decode("latin-1", errors="replace")

        if not text.strip():
            await _update_doc_status(document_id, "error", error="No text extracted")
            return

        label, confidence = classifier.classify(text)
        logger.info("Document %s classified as %s (%.2f)", document_id, label, confidence)

        chunker = LegalChunker()
        chunks = chunker.chunk(text)

        indexer = IncrementalIndexer()
        indexed = 0
        for i, chunk_text in enumerate(chunks):
            doc_data = {
                "content": chunk_text,
                "document_id": document_id,
                "document_title": filename,
                "doc_type": label,
                "tenant_id": tenant_id,
                "court": "",
                "date": "",
                "metadata": {
                    "chunk_index": i,
                    "source": "upload",
                    "storage_key": storage_key,
                },
            }
            try:
                await indexer.index_document(doc_data, tenant_id=tenant_id)
                indexed += 1
            except Exception as e:
                logger.warning("Failed to index chunk %d: %s", i, e)

        await _update_doc_status(
            document_id,
            "completed",
            classification_label=label,
            classification_confidence=confidence,
            ocr_text_length=len(text),
            chunks_indexed=indexed,
        )
        logger.info(
            "Document %s processed: %s, %d chunks indexed",
            document_id, label, indexed,
        )

    except Exception as e:
        logger.error("Document pipeline failed for %s: %s", document_id, e)
        await _update_doc_status(document_id, "error", error=str(e))


async def _run_ocr(file_data: bytes, filename: str, mime_type: str) -> str:
    """Run OCR on PDF/image file. Try PaddleOCR first, fallback to text extraction."""
    if settings.PADDLE_OCR_ENDPOINT:
        try:
            from app.services.ocr.paddle import PaddleOCRService

            with tempfile.NamedTemporaryFile(
                suffix=Path(filename).suffix, delete=False
            ) as tmp:
                tmp.write(file_data)
                tmp_path = tmp.name

            paddle = PaddleOCRService()
            result = await paddle.process_pdf(tmp_path)
            Path(tmp_path).unlink(missing_ok=True)

            if result.text:
                return result.text
        except Exception as e:
            logger.warning("PaddleOCR failed: %s", e)

    # Fallback: try PyPDF2/pdfplumber for text-based PDFs
    if filename.lower().endswith(".pdf"):
        try:
            import io

            try:
                from pdfplumber import open as pdf_open

                with pdf_open(io.BytesIO(file_data)) as pdf:
                    pages = [p.extract_text() or "" for p in pdf.pages]
                    return "\n\n".join(pages)
            except ImportError:
                pass

            from PyPDF2 import PdfReader

            reader = PdfReader(io.BytesIO(file_data))
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n\n".join(pages)
        except Exception as e:
            logger.warning("PDF text extraction failed: %s", e)

    return ""


async def _update_doc_status(
    document_id: str,
    status: str,
    error: str = "",
    classification_label: str = "",
    classification_confidence: float = 0.0,
    ocr_text_length: int = 0,
    chunks_indexed: int = 0,
) -> None:
    """Update document record in Supabase."""
    import httpx

    if not settings.SUPABASE_URL:
        return

    update_data: dict = {"ocr_status": status}
    if error:
        update_data["ocr_error"] = error
    if classification_label:
        update_data["classification_label"] = classification_label
    if ocr_text_length:
        update_data["ocr_text_length"] = ocr_text_length

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.patch(
                f"{settings.SUPABASE_URL}/rest/v1/documents",
                headers={
                    "apikey": settings.SUPABASE_ANON_KEY,
                    "Authorization": f"Bearer {settings.SUPABASE_ANON_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
                params={"id": f"eq.{document_id}"},
                json=update_data,
            )
    except Exception as e:
        logger.warning("Failed to update document status: %s", e)
