import logging
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.dependencies import get_current_user, get_tenant_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory", tags=["memory"])


class MemoryQuery(BaseModel):
    query: str
    case_id: Optional[str] = None


class MemoryStore(BaseModel):
    fact: str
    case_id: str
    source: str = "manual"


def _get_memory_manager():
    from app.services.memory.manager import MemoryManager
    from app.services.memory.graphiti import GraphitiClient
    from app.services.memory.mem0 import Mem0Client

    return MemoryManager(
        graphiti_client=GraphitiClient(),
        mem0_client=Mem0Client(),
    )


@router.post("/search")
async def search_memory(
    request: MemoryQuery,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    manager = _get_memory_manager()
    try:
        results = await manager.search_facts(
            tenant_id=tenant_id,
            query=request.query,
            case_id=request.case_id,
        )
        return results
    except Exception as e:
        logger.error("Memory search failed: %s", e)
        return {"facts": [], "knowledge": [], "query": request.query}


@router.post("/store")
async def store_memory(
    request: MemoryStore,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    manager = _get_memory_manager()
    try:
        await manager.store_fact(
            tenant_id=tenant_id,
            case_id=request.case_id,
            fact=request.fact,
            source=request.source,
        )
        return {"status": "stored"}
    except Exception as e:
        logger.error("Memory store failed: %s", e)
        return {"status": "error", "detail": str(e)}
