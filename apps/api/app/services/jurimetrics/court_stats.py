"""
Court statistics service — Elasticsearch aggregation for tribunal and area stats.
"""

import logging
from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from app.config import settings
from app.services.clients import create_es_client

logger = logging.getLogger(__name__)


class CourtStats(BaseModel):
    tribunal: str
    total_decisions: int = 0
    by_area: dict[str, int] = {}
    by_class: dict[str, int] = {}
    by_month: dict[str, int] = {}
    top_subjects: list[dict] = []
    period_start: Optional[str] = None
    period_end: Optional[str] = None


class AreaStats(BaseModel):
    area: str
    total_decisions: int = 0
    by_tribunal: dict[str, int] = {}
    avg_duration_days: Optional[float] = None
    by_month: dict[str, int] = {}
    period_start: Optional[str] = None
    period_end: Optional[str] = None


class TrendPoint(BaseModel):
    period: str
    count: int
    area: str = ""
    tribunal: str = ""


class CourtStatsService:
    def __init__(self) -> None:
        self.es = create_es_client()
        self.index = f"{settings.ES_INDEX_PREFIX}_chunks"

    async def get_court_statistics(
        self,
        tribunal: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        area: Optional[str] = None,
    ) -> CourtStats:
        must: list[dict] = [{"term": {"court.keyword": tribunal}}]
        if area:
            must.append({"term": {"metadata.area.keyword": area}})
        if date_from or date_to:
            range_q: dict = {}
            if date_from:
                range_q["gte"] = date_from
            if date_to:
                range_q["lte"] = date_to
            must.append({"range": {"date": range_q}})

        body = {
            "size": 0,
            "query": {"bool": {"must": must}},
            "aggs": {
                "total": {"value_count": {"field": "_id"}},
                "by_area": {"terms": {"field": "metadata.area.keyword", "size": 20}},
                "by_class": {"terms": {"field": "doc_type.keyword", "size": 20}},
                "by_month": {"date_histogram": {"field": "date", "calendar_interval": "month", "format": "yyyy-MM"}},
                "top_subjects": {"terms": {"field": "metadata.subject.keyword", "size": 15}},
            },
        }

        try:
            resp = await self.es.search(index=self.index, body=body)
            aggs = resp.get("aggregations", {})

            by_area = {b["key"]: b["doc_count"] for b in aggs.get("by_area", {}).get("buckets", [])}
            by_class = {b["key"]: b["doc_count"] for b in aggs.get("by_class", {}).get("buckets", [])}
            by_month = {b["key_as_string"]: b["doc_count"] for b in aggs.get("by_month", {}).get("buckets", [])}
            top_subjects = [
                {"subject": b["key"], "count": b["doc_count"]}
                for b in aggs.get("top_subjects", {}).get("buckets", [])
            ]

            return CourtStats(
                tribunal=tribunal,
                total_decisions=aggs.get("total", {}).get("value", 0),
                by_area=by_area,
                by_class=by_class,
                by_month=by_month,
                top_subjects=top_subjects,
                period_start=date_from,
                period_end=date_to,
            )
        except Exception as e:
            logger.error("Court stats aggregation failed: %s", e)
            return CourtStats(tribunal=tribunal)

    async def get_area_statistics(
        self,
        area: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        tribunal: Optional[str] = None,
    ) -> AreaStats:
        must: list[dict] = [{"term": {"metadata.area.keyword": area}}]
        if tribunal:
            must.append({"term": {"court.keyword": tribunal}})
        if date_from or date_to:
            range_q: dict = {}
            if date_from:
                range_q["gte"] = date_from
            if date_to:
                range_q["lte"] = date_to
            must.append({"range": {"date": range_q}})

        body = {
            "size": 0,
            "query": {"bool": {"must": must}},
            "aggs": {
                "total": {"value_count": {"field": "_id"}},
                "by_tribunal": {"terms": {"field": "court.keyword", "size": 30}},
                "by_month": {"date_histogram": {"field": "date", "calendar_interval": "month", "format": "yyyy-MM"}},
            },
        }

        try:
            resp = await self.es.search(index=self.index, body=body)
            aggs = resp.get("aggregations", {})

            by_tribunal = {b["key"]: b["doc_count"] for b in aggs.get("by_tribunal", {}).get("buckets", [])}
            by_month = {b["key_as_string"]: b["doc_count"] for b in aggs.get("by_month", {}).get("buckets", [])}

            return AreaStats(
                area=area,
                total_decisions=aggs.get("total", {}).get("value", 0),
                by_tribunal=by_tribunal,
                by_month=by_month,
                period_start=date_from,
                period_end=date_to,
            )
        except Exception as e:
            logger.error("Area stats aggregation failed: %s", e)
            return AreaStats(area=area)

    async def get_trends(
        self,
        area: Optional[str] = None,
        tribunal: Optional[str] = None,
        months: int = 12,
    ) -> list[TrendPoint]:
        """Get monthly trend data."""
        must: list[dict] = []
        if area:
            must.append({"term": {"metadata.area.keyword": area}})
        if tribunal:
            must.append({"term": {"court.keyword": tribunal}})

        body = {
            "size": 0,
            "query": {"bool": {"must": must}} if must else {"match_all": {}},
            "aggs": {
                "monthly": {
                    "date_histogram": {
                        "field": "date",
                        "calendar_interval": "month",
                        "format": "yyyy-MM",
                    },
                    "aggs": {
                        "by_area": {"terms": {"field": "metadata.area.keyword", "size": 5}},
                    },
                }
            },
        }

        try:
            resp = await self.es.search(index=self.index, body=body)
            buckets = resp.get("aggregations", {}).get("monthly", {}).get("buckets", [])
            recent = buckets[-months:] if len(buckets) > months else buckets

            points = []
            for b in recent:
                points.append(TrendPoint(
                    period=b["key_as_string"],
                    count=b["doc_count"],
                    area=area or "",
                    tribunal=tribunal or "",
                ))
            return points
        except Exception as e:
            logger.error("Trends aggregation failed: %s", e)
            return []
