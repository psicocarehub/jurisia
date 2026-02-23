"""
Jurimetrics: judge profiles via Elasticsearch aggregation.
CNJ Res. 615/2025: criminal prediction discouraged.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from elasticsearch import AsyncElasticsearch
from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger(__name__)


class JudgeProfile(BaseModel):
    name: str
    court: str = ""
    jurisdiction: str = ""
    total_decisions: int = 0
    avg_decision_time_days: float = 0.0
    favorability: Dict[str, Dict[str, float]] = {}
    top_citations: list = []
    decision_patterns: Dict[str, Any] = {}
    conciliation_rate: float = 0.0
    reform_rate: float = 0.0
    areas: Dict[str, int] = {}
    recent_decisions: list[dict] = []


class JurimetricsService:
    """Judge profiling via Elasticsearch aggregation on indexed decisions."""

    def __init__(self) -> None:
        self.es = AsyncElasticsearch(settings.ELASTICSEARCH_URL)
        self.index = f"{settings.ES_INDEX_PREFIX}_chunks"

    async def get_judge_profile(
        self, judge_name: str, court: Optional[str] = None
    ) -> Optional[JudgeProfile]:
        """Build judge profile from Elasticsearch aggregation."""
        must_clauses: list[dict] = [
            {"match": {"metadata.judge_name": judge_name}},
        ]
        if court:
            must_clauses.append({"term": {"court": court}})

        body = {
            "size": 0,
            "query": {"bool": {"must": must_clauses}},
            "aggs": {
                "total": {"value_count": {"field": "_id"}},
                "by_area": {"terms": {"field": "metadata.area.keyword", "size": 20}},
                "by_court": {"terms": {"field": "court.keyword", "size": 10}},
                "by_outcome": {"terms": {"field": "metadata.outcome.keyword", "size": 10}},
                "by_doc_type": {"terms": {"field": "doc_type.keyword", "size": 10}},
                "top_laws": {"terms": {"field": "metadata.cited_laws.keyword", "size": 20}},
                "recent": {
                    "top_hits": {
                        "size": 5,
                        "sort": [{"date": {"order": "desc"}}],
                        "_source": ["document_title", "court", "date", "doc_type", "content"],
                    }
                },
            },
        }

        try:
            resp = await self.es.search(index=self.index, body=body)
            aggs = resp.get("aggregations", {})

            total = aggs.get("total", {}).get("value", 0)
            if total == 0:
                # Fallback: try broader text match
                body["query"] = {"bool": {"must": [
                    {"multi_match": {"query": judge_name, "fields": ["content", "metadata.judge_name"]}},
                ]}}
                resp = await self.es.search(index=self.index, body=body)
                aggs = resp.get("aggregations", {})
                total = aggs.get("total", {}).get("value", 0)

            areas = {}
            for bucket in aggs.get("by_area", {}).get("buckets", []):
                areas[bucket["key"]] = bucket["doc_count"]

            courts = [b["key"] for b in aggs.get("by_court", {}).get("buckets", [])]
            primary_court = court or (courts[0] if courts else "")

            outcomes = {}
            for bucket in aggs.get("by_outcome", {}).get("buckets", []):
                outcomes[bucket["key"]] = bucket["doc_count"]

            favorability = {}
            if total > 0:
                procedente = outcomes.get("procedente", 0) + outcomes.get("parcialmente_procedente", 0)
                improcedente = outcomes.get("improcedente", 0)
                if procedente + improcedente > 0:
                    favorability["geral"] = {
                        "autor": round(procedente / (procedente + improcedente) * 100, 1),
                        "reu": round(improcedente / (procedente + improcedente) * 100, 1),
                    }

            top_citations = [
                {"law": b["key"], "count": b["doc_count"]}
                for b in aggs.get("top_laws", {}).get("buckets", [])
            ]

            recent_decisions = []
            for hit in aggs.get("recent", {}).get("hits", {}).get("hits", []):
                src = hit.get("_source", {})
                recent_decisions.append({
                    "title": src.get("document_title", ""),
                    "court": src.get("court", ""),
                    "date": src.get("date", ""),
                    "doc_type": src.get("doc_type", ""),
                    "snippet": src.get("content", "")[:200],
                })

            return JudgeProfile(
                name=judge_name,
                court=primary_court,
                jurisdiction=", ".join(courts),
                total_decisions=total,
                favorability=favorability,
                top_citations=top_citations,
                decision_patterns=outcomes,
                areas=areas,
                recent_decisions=recent_decisions,
            )

        except Exception as e:
            logger.error("Judge profile aggregation failed: %s", e)
            return JudgeProfile(name=judge_name, court=court or "")
