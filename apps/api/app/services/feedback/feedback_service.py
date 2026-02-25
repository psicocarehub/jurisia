"""
Feedback Service: captures user feedback on AI responses.

Types of feedback:
- Rating: thumbs up/down on a response
- Edit: user edits the AI-generated text (correction)
- Citation usage: user copies a response into a petition

This data feeds two downstream systems:
1. RAG Reranker (immediate): demote sources that led to bad responses
2. Data Accumulator (batch): edited responses become training data
"""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel

from app.db.supabase_client import supabase_db

logger = logging.getLogger("jurisai.feedback")

FEEDBACK_TABLE = "feedback_log"
SOURCE_SCORES_TABLE = "source_quality_scores"


class FeedbackType(str, Enum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    EDIT = "edit"
    CITATION_USED = "citation_used"
    FLAG_INCORRECT = "flag_incorrect"
    FLAG_OUTDATED = "flag_outdated"


class FeedbackEntry(BaseModel):
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


class FeedbackService:
    """Processes user feedback for continuous improvement."""

    async def ensure_tables(self) -> None:
        """Create feedback tables."""
        try:
            await supabase_db.rpc("exec_sql", {"query": """
                CREATE TABLE IF NOT EXISTS feedback_log (
                    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                    feedback_type VARCHAR(30) NOT NULL,
                    message_id VARCHAR(100),
                    conversation_id VARCHAR(100),
                    user_id UUID NOT NULL,
                    tenant_id UUID NOT NULL,
                    original_response TEXT,
                    edited_response TEXT,
                    source_ids TEXT[] DEFAULT '{}',
                    query TEXT,
                    area VARCHAR(100),
                    comment TEXT,
                    processed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_fb_user ON feedback_log(user_id);
                CREATE INDEX IF NOT EXISTS idx_fb_type ON feedback_log(feedback_type);
                CREATE INDEX IF NOT EXISTS idx_fb_processed ON feedback_log(processed);
                CREATE INDEX IF NOT EXISTS idx_fb_tenant ON feedback_log(tenant_id);

                CREATE TABLE IF NOT EXISTS source_quality_scores (
                    source_id VARCHAR(200) PRIMARY KEY,
                    positive_count INTEGER DEFAULT 0,
                    negative_count INTEGER DEFAULT 0,
                    quality_score FLOAT DEFAULT 0.5,
                    last_updated TIMESTAMPTZ DEFAULT NOW()
                );
            """})
        except Exception as e:
            logger.warning("Could not create feedback tables: %s", e)

    async def submit_feedback(self, entry: FeedbackEntry) -> str:
        """
        Store feedback and trigger immediate actions.

        Returns the feedback ID.
        """
        try:
            result = await supabase_db.insert(FEEDBACK_TABLE, {
                "feedback_type": entry.feedback_type.value,
                "message_id": entry.message_id,
                "conversation_id": entry.conversation_id,
                "user_id": entry.user_id,
                "tenant_id": entry.tenant_id,
                "original_response": entry.original_response,
                "edited_response": entry.edited_response,
                "source_ids": entry.source_ids,
                "query": entry.query,
                "area": entry.area,
                "comment": entry.comment,
            })
            feedback_id = result.get("id", "")
        except Exception as e:
            logger.error("Failed to store feedback: %s", e)
            return ""

        if entry.feedback_type in (
            FeedbackType.THUMBS_DOWN,
            FeedbackType.FLAG_INCORRECT,
            FeedbackType.FLAG_OUTDATED,
        ):
            await self._demote_sources(entry.source_ids)
        elif entry.feedback_type in (FeedbackType.THUMBS_UP, FeedbackType.CITATION_USED):
            await self._promote_sources(entry.source_ids)

        return feedback_id

    async def get_training_candidates(
        self,
        limit: int = 1000,
        min_quality: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Get unprocessed feedback entries suitable for training data.

        Includes:
        - Edits (user corrections = gold training pairs)
        - Thumbs up on responses (positive examples)
        - Citation usage (highest quality signal)
        """
        try:
            results = await supabase_db.select(
                FEEDBACK_TABLE,
                filters={"processed": False},
            )
            if not isinstance(results, list):
                results = [results] if results else []
        except Exception as e:
            logger.warning("Failed to fetch training candidates: %s", e)
            return []

        candidates = []
        for r in results:
            ft = r.get("feedback_type", "")
            if ft == FeedbackType.EDIT.value and r.get("edited_response"):
                candidates.append({
                    "type": "correction",
                    "query": r.get("query", ""),
                    "original": r.get("original_response", ""),
                    "corrected": r.get("edited_response", ""),
                    "area": r.get("area", "geral"),
                    "feedback_id": r.get("id"),
                })
            elif ft in (FeedbackType.THUMBS_UP.value, FeedbackType.CITATION_USED.value):
                if r.get("original_response"):
                    candidates.append({
                        "type": "positive_example",
                        "query": r.get("query", ""),
                        "response": r.get("original_response", ""),
                        "area": r.get("area", "geral"),
                        "feedback_id": r.get("id"),
                    })

        return candidates[:limit]

    async def mark_processed(self, feedback_ids: list[str]) -> None:
        """Mark feedback entries as processed by the training pipeline."""
        for fid in feedback_ids:
            try:
                await supabase_db.update(
                    FEEDBACK_TABLE,
                    {"processed": True},
                    filters={"id": fid},
                )
            except Exception as e:
                logger.warning("Failed to mark feedback %s as processed: %s", fid, e)
                continue

    async def get_source_quality_score(self, source_id: str) -> float:
        """Get quality score for a source (used by reranker)."""
        try:
            result = await supabase_db.select(
                SOURCE_SCORES_TABLE,
                filters={"source_id": source_id},
                single=True,
            )
            if result and isinstance(result, dict):
                return result.get("quality_score", 0.5)
        except Exception as e:
            logger.debug("Failed to get source quality score for %s: %s", source_id, e)
        return 0.5

    async def get_source_adjustments(
        self, source_ids: list[str],
    ) -> dict[str, float]:
        """
        Get quality score adjustments for multiple sources.
        Used by the reranker to boost/demote sources.
        """
        adjustments: dict[str, float] = {}
        for sid in source_ids:
            score = await self.get_source_quality_score(sid)
            adjustments[sid] = score
        return adjustments

    async def _demote_sources(self, source_ids: list[str]) -> None:
        """Decrease quality score for sources that led to bad responses."""
        for sid in source_ids:
            if not sid:
                continue
            try:
                existing = await supabase_db.select(
                    SOURCE_SCORES_TABLE,
                    filters={"source_id": sid},
                )
                if existing and isinstance(existing, list) and len(existing) > 0:
                    record = existing[0]
                    neg = record.get("negative_count", 0) + 1
                    pos = record.get("positive_count", 0)
                    score = pos / max(pos + neg, 1)
                    await supabase_db.update(
                        SOURCE_SCORES_TABLE,
                        {"negative_count": neg, "quality_score": round(score, 4)},
                        filters={"source_id": sid},
                    )
                else:
                    await supabase_db.insert(SOURCE_SCORES_TABLE, {
                        "source_id": sid,
                        "negative_count": 1,
                        "quality_score": 0.0,
                    })
            except Exception as e:
                logger.warning("Failed to demote source %s: %s", sid, e)

    async def _promote_sources(self, source_ids: list[str]) -> None:
        """Increase quality score for sources that led to good responses."""
        for sid in source_ids:
            if not sid:
                continue
            try:
                existing = await supabase_db.select(
                    SOURCE_SCORES_TABLE,
                    filters={"source_id": sid},
                )
                if existing and isinstance(existing, list) and len(existing) > 0:
                    record = existing[0]
                    pos = record.get("positive_count", 0) + 1
                    neg = record.get("negative_count", 0)
                    score = pos / max(pos + neg, 1)
                    await supabase_db.update(
                        SOURCE_SCORES_TABLE,
                        {"positive_count": pos, "quality_score": round(score, 4)},
                        filters={"source_id": sid},
                    )
                else:
                    await supabase_db.insert(SOURCE_SCORES_TABLE, {
                        "source_id": sid,
                        "positive_count": 1,
                        "quality_score": 1.0,
                    })
            except Exception as e:
                logger.warning("Failed to promote source %s: %s", sid, e)
