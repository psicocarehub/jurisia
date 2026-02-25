"""
Updates API endpoints.
Powers the "Novidades" portal with paginated feed, stats, source status
and highlights from the content_updates table.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query

from app.config import settings
from app.dependencies import get_current_user, get_tenant_id

router = APIRouter(prefix="/updates", tags=["updates"])
logger = logging.getLogger("jurisai.updates")

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_date(value: str, name: str) -> str:
    if not _DATE_RE.match(value):
        raise HTTPException(status_code=400, detail=f"Invalid {name} format. Use YYYY-MM-DD")
    return value

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
    and_filter: str | None = None,
    order: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
    count: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Low-level PostgREST query helper for content_updates / ingestion_log."""
    params: dict[str, str] = {"select": select}
    if filters:
        params.update(filters)
    if and_filter:
        params["and"] = and_filter
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
    else:
        _validate_date(date_from, "date_from")
    if not date_to:
        date_to = date.today().isoformat()
    else:
        _validate_date(date_to, "date_to")

    filters: dict[str, str] = {}
    and_filter = f"(captured_at.gte.{date_from}T00:00:00Z,captured_at.lte.{date_to}T23:59:59Z)"

    if category:
        if category not in VALID_CATEGORIES:
            raise HTTPException(status_code=400, detail=f"Invalid category. Valid: {', '.join(sorted(VALID_CATEGORIES))}")
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
            and_filter=and_filter,
            order="captured_at.desc",
            limit=per_page,
            offset=offset,
            count=True,
        )
    except httpx.HTTPStatusError as e:
        logger.error("Supabase error fetching feed: %s", e)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.error("Failed to fetch feed: %s", e)
        raise HTTPException(status_code=500, detail="Internal error")

    return {
        "items": data,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": max(1, (total + per_page - 1) // per_page),
    }


# ---------------------------------------------------------------------------
# GET /updates/feed/personalized
# ---------------------------------------------------------------------------
@router.get("/feed/personalized")
async def get_personalized_feed(
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(30, ge=1, le=100),
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    """Feed personalized by user's subscribed areas, boosting relevant items."""
    if not date_from:
        date_from = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    else:
        _validate_date(date_from, "date_from")
    if not date_to:
        date_to = date.today().isoformat()
    else:
        _validate_date(date_to, "date_to")

    user_id = user.get("id", "")
    user_areas: list[str] = []

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{_BASE_URL}/alert_subscriptions",
                headers=_HEADERS,
                params={
                    "select": "areas",
                    "user_id": f"eq.{user_id}",
                    "is_active": "eq.true",
                    "limit": "1",
                },
            )
            if resp.status_code == 200:
                subs = resp.json()
                if subs and isinstance(subs, list):
                    user_areas = subs[0].get("areas", [])
    except Exception as e:
        logger.warning("Failed to fetch user areas: %s", e)

    and_filter = f"(captured_at.gte.{date_from}T00:00:00Z,captured_at.lte.{date_to}T23:59:59Z)"
    offset = (page - 1) * per_page

    try:
        data, total = await _query(
            "content_updates",
            and_filter=and_filter,
            order="relevance_score.desc,captured_at.desc",
            limit=per_page * 2,
            offset=0,
            count=True,
        )
    except Exception as e:
        logger.error("Failed to fetch personalized feed: %s", e)
        raise HTTPException(status_code=500, detail="Internal error")

    if user_areas:
        area_set = set(a.lower() for a in user_areas)
        for item in data:
            item_areas = set(a.lower() for a in (item.get("areas") or []))
            if area_set.intersection(item_areas):
                item["relevance_score"] = min(1.0, (item.get("relevance_score") or 0.5) + 0.15)

        data.sort(key=lambda x: (-x.get("relevance_score", 0), x.get("captured_at", "")))

    paginated = data[offset : offset + per_page]

    return {
        "items": paginated,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": max(1, (total + per_page - 1) // per_page),
        "personalized": bool(user_areas),
        "user_areas": user_areas,
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
    else:
        _validate_date(date_from, "date_from")
    if not date_to:
        date_to = date.today().isoformat()
    else:
        _validate_date(date_to, "date_to")

    and_filter = f"(captured_at.gte.{date_from}T00:00:00Z,captured_at.lte.{date_to}T23:59:59Z)"

    try:
        rows, _ = await _query(
            "content_updates",
            select="category,source,territory",
            and_filter=and_filter,
            limit=10000,
        )
    except httpx.HTTPStatusError as e:
        logger.error("Supabase error fetching stats: %s", e)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.error("Failed to fetch stats: %s", e)
        raise HTTPException(status_code=500, detail="Internal error")

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
    except httpx.HTTPStatusError as e:
        logger.error("Supabase error fetching sources: %s", e)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.error("Failed to fetch sources: %s", e)
        raise HTTPException(status_code=500, detail="Internal error")

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
    except httpx.HTTPStatusError as e:
        logger.error("Supabase error fetching highlights: %s", e)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.error("Failed to fetch highlights: %s", e)
        raise HTTPException(status_code=500, detail="Internal error")
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
    except httpx.HTTPStatusError as e:
        logger.error("Supabase error fetching categories: %s", e)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.error("Failed to fetch categories: %s", e)
        raise HTTPException(status_code=500, detail="Internal error")
    counts: dict[str, int] = {}
    for r in rows:
        cat = r.get("category", "outro")
        counts[cat] = counts.get(cat, 0) + 1
    return {"categories": counts}


# ---------------------------------------------------------------------------
# GET /updates/bookmarks
# ---------------------------------------------------------------------------
@router.get("/bookmarks")
async def list_bookmarks(
    page: int = Query(1, ge=1),
    per_page: int = Query(30, ge=1, le=100),
    user: dict = Depends(get_current_user),
):
    """List bookmarked content updates for the current user."""
    user_id = user.get("id", "")
    offset = (page - 1) * per_page
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{_BASE_URL}/user_bookmarks",
                headers={**_HEADERS, "Prefer": "count=exact"},
                params={
                    "select": "id,created_at,notes,content_update_id,content_updates(*)",
                    "user_id": f"eq.{user_id}",
                    "order": "created_at.desc",
                    "limit": str(per_page),
                    "offset": str(offset),
                },
            )
            resp.raise_for_status()
            data = resp.json()
            total = len(data)
            cr = resp.headers.get("content-range", "")
            if "/" in cr:
                try:
                    total = int(cr.split("/")[1])
                except (ValueError, IndexError):
                    pass
    except Exception as e:
        logger.error("Failed to list bookmarks: %s", e)
        raise HTTPException(status_code=500, detail="Internal error")

    return {
        "bookmarks": data if isinstance(data, list) else [],
        "total": total,
        "page": page,
        "per_page": per_page,
    }


# ---------------------------------------------------------------------------
# POST /updates/{update_id}/bookmark
# ---------------------------------------------------------------------------
@router.post("/{update_id}/bookmark")
async def add_bookmark(
    update_id: str,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    """Bookmark a content update."""
    user_id = user.get("id", "")
    payload = {
        "user_id": user_id,
        "tenant_id": tenant_id,
        "content_update_id": update_id,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{_BASE_URL}/user_bookmarks",
                headers={**_HEADERS, "Prefer": "return=representation,resolution=ignore-duplicates"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return {"bookmark": data[0] if isinstance(data, list) and data else data}
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 409:
            return {"bookmark": None, "message": "Already bookmarked"}
        logger.error("Failed to bookmark: %s", e)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.error("Failed to bookmark: %s", e)
        raise HTTPException(status_code=500, detail="Internal error")


# ---------------------------------------------------------------------------
# DELETE /updates/{update_id}/bookmark
# ---------------------------------------------------------------------------
@router.delete("/{update_id}/bookmark")
async def remove_bookmark(
    update_id: str,
    user: dict = Depends(get_current_user),
):
    """Remove a bookmark."""
    user_id = user.get("id", "")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.delete(
                f"{_BASE_URL}/user_bookmarks",
                headers=_HEADERS,
                params={"user_id": f"eq.{user_id}", "content_update_id": f"eq.{update_id}"},
            )
            resp.raise_for_status()
    except Exception as e:
        logger.error("Failed to remove bookmark: %s", e)
        raise HTTPException(status_code=500, detail="Internal error")
    return {"removed": True}


# ---------------------------------------------------------------------------
# GET /updates/export
# ---------------------------------------------------------------------------
@router.get("/export")
async def export_updates(
    format: str = Query("csv", description="Export format: csv or json"),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    user: dict = Depends(get_current_user),
):
    """Export content updates as CSV or JSON."""
    if not date_from:
        date_from = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    else:
        _validate_date(date_from, "date_from")
    if not date_to:
        date_to = date.today().isoformat()
    else:
        _validate_date(date_to, "date_to")

    filters: dict[str, str] = {}
    and_filter_str = f"(captured_at.gte.{date_from}T00:00:00Z,captured_at.lte.{date_to}T23:59:59Z)"
    if category and category in VALID_CATEGORIES:
        filters["category"] = f"eq.{category}"

    try:
        data, _ = await _query(
            "content_updates",
            select="title,category,subcategory,court_or_organ,territory,publication_date,source_url,relevance_score,areas,captured_at",
            filters=filters,
            and_filter=and_filter_str,
            order="captured_at.desc",
            limit=5000,
        )
    except Exception as e:
        logger.error("Failed to export: %s", e)
        raise HTTPException(status_code=500, detail="Export failed")

    if format == "json":
        from fastapi.responses import JSONResponse
        return JSONResponse(
            content={"items": data, "total": len(data)},
            headers={"Content-Disposition": f"attachment; filename=novidades_{date_from}_{date_to}.json"},
        )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Título", "Categoria", "Subcategoria", "Órgão", "Território", "Data Publicação", "URL", "Relevância", "Áreas", "Capturado em"])
    for item in data:
        writer.writerow([
            item.get("title", ""),
            item.get("category", ""),
            item.get("subcategory", ""),
            item.get("court_or_organ", ""),
            item.get("territory", ""),
            item.get("publication_date", ""),
            item.get("source_url", ""),
            item.get("relevance_score", ""),
            ", ".join(item.get("areas", [])),
            item.get("captured_at", ""),
        ])

    from fastapi.responses import StreamingResponse as FastStreamingResponse
    csv_content = output.getvalue()
    output.close()
    return FastStreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=novidades_{date_from}_{date_to}.csv"},
    )
