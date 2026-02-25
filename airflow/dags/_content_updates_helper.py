"""
Shared helper for inserting rows into the content_updates table from any DAG.
"""

from __future__ import annotations

from typing import Optional

from airflow.providers.postgres.hooks.postgres import PostgresHook


def insert_content_update(
    hook: PostgresHook,
    *,
    source: str,
    category: str,
    title: str,
    subcategory: str | None = None,
    summary: str | None = None,
    content_preview: str | None = None,
    areas: list[str] | None = None,
    court_or_organ: str | None = None,
    territory: str | None = None,
    publication_date: str | None = None,
    source_url: str | None = None,
    relevance_score: float = 0.5,
    metadata: dict | None = None,
) -> None:
    """Insert a single content update, silently ignoring duplicates on (source, title, publication_date)."""
    import json

    areas_literal = _pg_array(areas) if areas else "{}"
    meta_json = json.dumps(metadata or {})

    hook.run(
        """
        INSERT INTO content_updates
            (source, category, subcategory, title, summary, content_preview,
             areas, court_or_organ, territory, publication_date,
             source_url, relevance_score, metadata)
        VALUES (%s, %s, %s, %s, %s, %s,
                %s::text[], %s, %s, %s::date,
                %s, %s, %s::jsonb)
        ON CONFLICT DO NOTHING
        """,
        parameters=(
            source,
            category,
            subcategory,
            title[:1000],
            (summary or "")[:2000] or None,
            (content_preview or "")[:500] or None,
            areas_literal,
            court_or_organ,
            territory or "federal",
            publication_date,
            source_url,
            relevance_score,
            meta_json,
        ),
    )


def _pg_array(items: list[str]) -> str:
    """Convert a Python list to a PostgreSQL text array literal."""
    escaped = [i.replace("\\", "\\\\").replace('"', '\\"') for i in items]
    return "{" + ",".join(f'"{e}"' for e in escaped) + "}"
