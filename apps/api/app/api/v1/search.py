from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.dependencies import get_current_user, get_tenant_id

router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str
    area: Optional[str] = None
    court: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    top_k: int = 10


class SearchResult(BaseModel):
    id: str
    content: str
    score: float
    document_title: str
    doc_type: str = ""
    court: str = ""
    date: str = ""


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
    # TODO: Implement HybridRetriever integration
    return SearchResponse(results=[], total=0, query=request.query)
