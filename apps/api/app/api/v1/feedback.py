"""
Feedback API endpoints.
Captures user reactions (thumbs up/down, edits, flags) on AI responses.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.feedback.feedback_service import (
    FeedbackEntry,
    FeedbackService,
    FeedbackType,
)

router = APIRouter(prefix="/feedback", tags=["feedback"])
_service = FeedbackService()


class FeedbackRequest(BaseModel):
    feedback_type: FeedbackType
    message_id: str
    conversation_id: str
    user_id: str
    tenant_id: str
    original_response: Optional[str] = None
    edited_response: Optional[str] = None
    source_ids: list[str] = []
    query: Optional[str] = None
    area: Optional[str] = None
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: str
    status: str


@router.post("", response_model=FeedbackResponse, status_code=201)
async def submit_feedback(req: FeedbackRequest):
    entry = FeedbackEntry(**req.model_dump())
    feedback_id = await _service.submit_feedback(entry)
    if not feedback_id:
        raise HTTPException(status_code=500, detail="Failed to store feedback")
    return FeedbackResponse(id=feedback_id, status="received")


@router.get("/source-score/{source_id}")
async def get_source_score(source_id: str):
    score = await _service.get_source_quality_score(source_id)
    return {"source_id": source_id, "quality_score": score}
