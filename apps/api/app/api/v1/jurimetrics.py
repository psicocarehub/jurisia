from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.dependencies import get_current_user, get_tenant_id

router = APIRouter(prefix="/jurimetrics", tags=["jurimetrics"])


class JudgeProfileResponse(BaseModel):
    name: str
    court: Optional[str] = None
    total_decisions: int = 0
    avg_decision_time_days: Optional[float] = None
    favorability: dict = {}


@router.get("/judges/{judge_name}")
async def get_judge_profile(
    judge_name: str,
    court: Optional[str] = None,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    # TODO: Implement judge profiling service
    return JudgeProfileResponse(name=judge_name, court=court)
