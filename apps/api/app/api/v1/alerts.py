"""
Alerts API endpoints.
Lets users retrieve, read, and subscribe to legislative change alerts.
"""

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.monitoring.alert_service import AlertService

router = APIRouter(prefix="/alerts", tags=["alerts"])
_service = AlertService()


class SubscribeRequest(BaseModel):
    user_id: str
    tenant_id: str
    areas: list[str]
    change_types: Optional[list[str]] = None
    min_severity: str = "medium"


@router.get("")
async def get_alerts(
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    area: Optional[str] = None,
    unread_only: bool = True,
    limit: int = 50,
):
    areas = [area] if area else None
    alerts = await _service.get_alerts(
        user_id=user_id,
        tenant_id=tenant_id,
        areas=areas,
        unread_only=unread_only,
        limit=limit,
    )
    return {"alerts": alerts, "count": len(alerts)}


@router.patch("/{alert_id}/read")
async def mark_alert_read(alert_id: str):
    await _service.mark_read(alert_id)
    return {"status": "ok"}


@router.patch("/read-all")
async def mark_all_read(user_id: str, tenant_id: Optional[str] = None):
    await _service.mark_all_read(user_id, tenant_id)
    return {"status": "ok"}


@router.post("/subscribe", status_code=201)
async def subscribe(req: SubscribeRequest):
    sub_id = await _service.subscribe(
        user_id=req.user_id,
        tenant_id=req.tenant_id,
        areas=req.areas,
        change_types=req.change_types,
        min_severity=req.min_severity,
    )
    return {"subscription_id": sub_id, "status": "active"}
