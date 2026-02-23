import logging
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.dependencies import get_current_user, get_tenant_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str
    area: Optional[str] = None
    court: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    top_k: int = 10
    use_reranker: bool = True


class SearchResult(BaseModel):
    id: str
    content: str
    score: float
    document_title: str
    doc_type: str = ""
    court: str = ""
    date: str = ""
    document_id: str = ""


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total: int
    query: str


@router.post("", response_model=SearchResponse)
async def semantic_search(
    request: SearchRequest,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    from app.services.rag.retriever import HybridRetriever

    retriever = HybridRetriever()

    filters: dict = {}
    if request.area:
        filters["area"] = request.area
    if request.court:
        filters["court"] = request.court
    if request.date_from:
        filters["date_from"] = request.date_from
    if request.date_to:
        filters["date_to"] = request.date_to

    try:
        chunks = await retriever.retrieve(
            query=request.query,
            tenant_id=tenant_id,
            top_k=request.top_k,
            filters=filters if filters else None,
            use_reranker=request.use_reranker,
        )
    except Exception as e:
        logger.error("Search failed: %s", e)
        return SearchResponse(results=[], total=0, query=request.query)

    results = [
        SearchResult(
            id=chunk.id,
            content=chunk.content[:500],
            score=chunk.score,
            document_title=chunk.document_title,
            doc_type=chunk.doc_type,
            court=chunk.court,
            date=chunk.date,
            document_id=chunk.document_id,
        )
        for chunk in chunks
    ]

    return SearchResponse(
        results=results,
        total=len(results),
        query=request.query,
    )
