from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.dependencies import get_current_user, get_tenant_id

router = APIRouter(prefix="/memory", tags=["memory"])


class MemoryQuery(BaseModel):
    query: str
    case_id: Optional[str] = None


@router.post("/search")
async def search_memory(
    request: MemoryQuery,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    # TODO: Implement MemoryManager search across 4 tiers
    return {"facts": [], "knowledge": [], "query": request.query}
