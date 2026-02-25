import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.dependencies import get_current_user, get_tenant_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jurimetrics", tags=["jurimetrics"])


# --- Judge Profile ---

@router.get("/judges/{judge_name}")
async def get_judge_profile(
    judge_name: str,
    court: Optional[str] = None,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    from app.services.jurimetrics.judge_profile import JurimetricsService

    service = JurimetricsService()
    profile = await service.get_judge_profile(judge_name, court=court)
    if profile:
        return profile.model_dump()
    return {"name": judge_name, "court": court, "total_decisions": 0}


# --- Court Stats ---

@router.get("/courts/{tribunal}/stats")
async def get_court_stats(
    tribunal: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    area: Optional[str] = None,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    from app.services.jurimetrics.court_stats import CourtStatsService

    service = CourtStatsService()
    stats = await service.get_court_statistics(
        tribunal=tribunal, date_from=date_from, date_to=date_to, area=area,
    )
    return stats.model_dump()


@router.get("/areas/{area}/stats")
async def get_area_stats(
    area: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    tribunal: Optional[str] = None,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    from app.services.jurimetrics.court_stats import CourtStatsService

    service = CourtStatsService()
    stats = await service.get_area_statistics(
        area=area, date_from=date_from, date_to=date_to, tribunal=tribunal,
    )
    return stats.model_dump()


# --- Prediction ---

class PredictionRequest(BaseModel):
    area: str
    tribunal: Optional[str] = None
    judge_name: Optional[str] = None
    estimated_value: Optional[float] = None
    tipo_acao: Optional[str] = None
    num_partes: int = 2


@router.post("/predict")
async def predict_outcome(
    request: PredictionRequest,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    from app.services.jurimetrics.predictor import OutcomePredictor

    predictor = OutcomePredictor()

    case_data: dict[str, Any] = {
        "area": request.area,
        "court": request.tribunal or "",
        "tribunal": request.tribunal or "",
        "estimated_value": request.estimated_value or 0,
        "tipo_acao": request.tipo_acao or "",
        "num_partes": request.num_partes,
    }

    # Enrich with judge favorability if available
    if request.judge_name:
        try:
            from app.services.jurimetrics.judge_profile import JurimetricsService

            service = JurimetricsService()
            profile = await service.get_judge_profile(
                request.judge_name, court=request.tribunal
            )
            if profile and profile.favorability.get("geral"):
                case_data["judge_favorability"] = profile.favorability["geral"].get("autor", 50.0)
        except Exception as e:
            logger.warning("Failed to enrich with judge favorability for %s: %s", request.judge_name, e)

    result = predictor.predict(case_data, area=request.area)
    return result.model_dump()


# --- Trends ---

@router.get("/trends")
async def get_trends(
    area: Optional[str] = None,
    tribunal: Optional[str] = None,
    months: int = 12,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    from app.services.jurimetrics.court_stats import CourtStatsService

    service = CourtStatsService()
    trends = await service.get_trends(area=area, tribunal=tribunal, months=months)
    return {"trends": [t.model_dump() for t in trends]}
