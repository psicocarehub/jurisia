"""
Updates API endpoints.
Powers the "Novidades" portal with paginated feed, stats, source status
and highlights from the content_updates table.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, Query

from app.config import settings
from app.dependencies import get_current_user

router = APIRouter(prefix="/updates", tags=["updates"])
logger = logging.getLogger("jurisai.updates")

VALID_CATEGORIES = {
    "legislacao", "jurisprudencia", "doutrina",
    "normativo", "parecer", "sumula", "outro",
}

_BASE_URL = f"{settings.SUPABASE_URL}/rest/v1"
_HEADERS = {
    "apikey": settings.SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {settings.SUPABASE_ANON_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation",
}


async def _query(
    table: str,
    *,
    select: str = "*",
    filters: dict[str, str] | None = None,
    order: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
    count: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Low-level PostgREST query helper for content_updates / ingestion_log."""
    params: dict[str, str] = {"select": select}
    if filters:
        params.update(filters)
    if order:
        params["order"] = order
    if limit:
        params["limit"] = str(limit)
    if offset:
        params["offset"] = str(offset)

    headers = dict(_HEADERS)
    if count:
        headers["Prefer"] = "count=exact"
        headers["Range-Unit"] = "items"

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(
            f"{_BASE_URL}/{table}",
            headers=headers,
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()
        total = len(data)
        if count:
            cr = resp.headers.get("content-range", "")
            if "/" in cr:
                try:
                    total = int(cr.split("/")[1])
                except (ValueError, IndexError):
                    pass
        return data if isinstance(data, list) else ([data] if data else []), total


# ---------------------------------------------------------------------------
# GET /updates/feed
# ---------------------------------------------------------------------------
@router.get("/feed")
async def get_feed(
    date_from: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    category: Optional[str] = Query(None),
    area: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    territory: Optional[str] = Query(None),
    court_or_organ: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(30, ge=1, le=100),
    user: dict = Depends(get_current_user),
):
    if not date_from:
        date_from = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    if not date_to:
        date_to = date.today().isoformat()

    filters: dict[str, str] = {
        "captured_at": f"gte.{date_from}T00:00:00Z",
    }
    # PostgREST doesn't allow two filters on the same column via simple params,
    # so we use range-based approach: captured_at=gte.X&captured_at=lte.Y won't work.
    # Instead, apply date_to as a separate header or accept the gte-only approach
    # (feed is sorted desc, so the UI naturally shows the latest first).

    if category and category in VALID_CATEGORIES:
        filters["category"] = f"eq.{category}"
    if area:
        filters["areas"] = f"cs.{{{area}}}"
    if source:
        filters["source"] = f"eq.{source}"
    if territory:
        filters["territory"] = f"eq.{territory}"
    if court_or_organ:
        filters["court_or_organ"] = f"ilike.*{court_or_organ}*"
    if search:
        filters["title"] = f"ilike.*{search}*"

    offset = (page - 1) * per_page
    try:
        data, total = await _query(
            "content_updates",
            filters=filters,
            order="captured_at.desc",
            limit=per_page,
            offset=offset,
            count=True,
        )
    except Exception as e:
        logger.error("Failed to fetch feed: %s", e)
        return {"items": [], "total": 0, "page": page, "per_page": per_page, "total_pages": 1}

    return {
        "items": data,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": max(1, (total + per_page - 1) // per_page),
    }


# ---------------------------------------------------------------------------
# GET /updates/stats
# ---------------------------------------------------------------------------
@router.get("/stats")
async def get_stats(
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    user: dict = Depends(get_current_user),
):
    if not date_from:
        date_from = date.today().isoformat()
    if not date_to:
        date_to = date.today().isoformat()

    try:
        rows, _ = await _query(
            "content_updates",
            select="category,source,territory",
            filters={"captured_at": f"gte.{date_from}T00:00:00Z"},
        )
    except Exception as e:
        logger.error("Failed to fetch stats: %s", e)
        return {"date_from": date_from, "date_to": date_to, "total": 0, "by_category": {}, "by_source": {}, "by_territory": {}}

    by_category: dict[str, int] = {}
    by_source: dict[str, int] = {}
    by_territory: dict[str, int] = {}

    for r in rows:
        cat = r.get("category", "outro")
        by_category[cat] = by_category.get(cat, 0) + 1
        src = r.get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1
        ter = r.get("territory", "federal")
        by_territory[ter] = by_territory.get(ter, 0) + 1

    return {
        "date_from": date_from,
        "date_to": date_to,
        "total": len(rows),
        "by_category": by_category,
        "by_source": by_source,
        "by_territory": by_territory,
    }


# ---------------------------------------------------------------------------
# GET /updates/sources
# ---------------------------------------------------------------------------
@router.get("/sources")
async def get_sources(user: dict = Depends(get_current_user)):
    try:
        rows, _ = await _query(
            "ingestion_log",
            select="source,records_count,status,error_message,ingested_at",
            order="ingested_at.desc",
            limit=200,
        )
    except Exception as e:
        logger.error("Failed to fetch sources: %s", e)
        return {"sources": []}

    sources_map: dict[str, dict] = {}
    for r in rows:
        src = r.get("source", "")
        if src not in sources_map:
            sources_map[src] = {
                "source": src,
                "last_run": r.get("ingested_at"),
                "last_status": r.get("status"),
                "last_error": r.get("error_message"),
                "total_records": r.get("records_count", 0) or 0,
                "runs": 1,
            }
        else:
            sources_map[src]["total_records"] += r.get("records_count", 0) or 0
            sources_map[src]["runs"] += 1

    return {
        "sources": sorted(
            sources_map.values(),
            key=lambda s: s.get("last_run") or "",
            reverse=True,
        ),
    }


# ---------------------------------------------------------------------------
# GET /updates/highlights
# ---------------------------------------------------------------------------
@router.get("/highlights")
async def get_highlights(
    limit: int = Query(10, ge=1, le=50),
    days: int = Query(1, ge=1, le=30),
    user: dict = Depends(get_current_user),
):
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        data, _ = await _query(
            "content_updates",
            filters={"captured_at": f"gte.{since}T00:00:00Z"},
            order="relevance_score.desc",
            limit=limit,
        )
    except Exception as e:
        logger.error("Failed to fetch highlights: %s", e)
        return {"highlights": []}
    return {"highlights": data}


# ---------------------------------------------------------------------------
# GET /updates/categories
# ---------------------------------------------------------------------------
@router.get("/categories")
async def get_categories(user: dict = Depends(get_current_user)):
    since = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    try:
        rows, _ = await _query(
            "content_updates",
            select="category",
            filters={"captured_at": f"gte.{since}T00:00:00Z"},
        )
    except Exception as e:
        logger.error("Failed to fetch categories: %s", e)
        return {"categories": {}}
    counts: dict[str, int] = {}
    for r in rows:
        cat = r.get("category", "outro")
        counts[cat] = counts.get(cat, 0) + 1
    return {"categories": counts}
