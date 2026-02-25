"""
Shared PostgREST query helpers for pagination with exact counts.
"""

from typing import Any, Optional

import httpx

from app.config import settings


def _supabase_headers(*, count: bool = False) -> dict[str, str]:
    headers = {
        "apikey": settings.SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    if count:
        headers["Prefer"] = "return=representation, count=exact"
        headers["Range-Unit"] = "items"
    return headers


def _base_url() -> str:
    return f"{settings.SUPABASE_URL}/rest/v1"


def _parse_content_range(response: httpx.Response) -> Optional[int]:
    """Extract total count from PostgREST Content-Range header (e.g. '0-9/42')."""
    cr = response.headers.get("content-range", "")
    if "/" in cr:
        try:
            total_str = cr.split("/")[1]
            if total_str != "*":
                return int(total_str)
        except (ValueError, IndexError):
            pass
    return None


async def supabase_list(
    table: str,
    *,
    params: dict[str, str],
    skip: int = 0,
    limit: int = 50,
    timeout: float = 15.0,
) -> tuple[list[dict[str, Any]], int]:
    """Query a Supabase table with pagination and exact total count."""
    params = {**params, "offset": str(skip), "limit": str(limit)}
    headers = _supabase_headers(count=True)

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(
            f"{_base_url()}/{table}",
            headers=headers,
            params=params,
        )
        resp.raise_for_status()
        rows = resp.json()
        if not isinstance(rows, list):
            rows = [rows] if rows else []

        total = _parse_content_range(resp)
        if total is None:
            total = len(rows)

        return rows, total
