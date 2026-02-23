"""
Graphiti Client — knowledge graph backed by Supabase/PostgreSQL jsonb.

Simplified KG: nodes stored in a Supabase table with jsonb metadata.
Edges stored as references within node metadata.
Keyword search with ts_vector for retrieval.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger(__name__)

TABLE_NODES = "kg_nodes"
TABLE_EDGES = "kg_edges"


class GraphitiSearchResult(BaseModel):
    node_id: str
    content: str
    metadata: dict[str, Any] = {}
    score: float = 0.0
    fact: str = ""
    source: str = ""


class GraphitiClient:
    """
    Knowledge graph backed by Supabase REST API (PostgREST).
    Uses two tables: kg_nodes (id, namespace, content, metadata, ts)
    and kg_edges (id, source_id, target_id, relation, namespace).
    """

    def __init__(self, namespace: str = "default") -> None:
        self.namespace = namespace
        self._base = f"{settings.SUPABASE_URL}/rest/v1"
        self._headers = {
            "apikey": settings.SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_ANON_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

    async def search(
        self,
        query: str,
        namespace: Optional[str] = None,
        top_k: int = 10,
        limit: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[GraphitiSearchResult]:
        """Full-text search on kg_nodes within namespace."""
        ns = namespace or self.namespace
        query_words = " & ".join(query.split()[:8])

        params: dict[str, str] = {
            "namespace": f"eq.{ns}",
            "limit": str(top_k or limit),
            "order": "created_at.desc",
        }

        if query_words:
            params["content"] = f"fts.{query_words}"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self._base}/{TABLE_NODES}",
                    headers=self._headers,
                    params=params,
                )
                if resp.status_code == 200:
                    rows = resp.json()
                    return [
                        GraphitiSearchResult(
                            node_id=r["id"],
                            content=r.get("content", ""),
                            metadata=r.get("metadata", {}),
                            score=1.0,
                            fact=r.get("content", ""),
                            source=r.get("metadata", {}).get("source", ""),
                        )
                        for r in rows
                    ]

                # Fallback: ilike search
                params.pop("content", None)
                params["content"] = f"ilike.*{query.split()[0]}*" if query.split() else "ilike.*"
                resp2 = await client.get(
                    f"{self._base}/{TABLE_NODES}",
                    headers=self._headers,
                    params=params,
                )
                if resp2.status_code == 200:
                    rows = resp2.json()
                    return [
                        GraphitiSearchResult(
                            node_id=r["id"],
                            content=r.get("content", ""),
                            metadata=r.get("metadata", {}),
                            score=0.5,
                            fact=r.get("content", ""),
                            source=r.get("metadata", {}).get("source", ""),
                        )
                        for r in rows
                    ]
        except Exception as e:
            logger.warning("GraphitiClient search failed: %s", e)

        return []

    async def add_episode(
        self,
        name: str = "",
        episode_body: str = "",
        content: str = "",
        source_description: str = "",
        namespace: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        relationships: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        """Add a fact node and optional relationship edges."""
        ns = namespace or self.namespace
        node_id = str(uuid.uuid4())
        body = content or episode_body

        node_data = {
            "id": node_id,
            "namespace": ns,
            "content": body,
            "metadata": {
                **(metadata or {}),
                "name": name,
                "source": source_description,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self._base}/{TABLE_NODES}",
                    headers=self._headers,
                    json=node_data,
                )
                if resp.status_code not in (200, 201):
                    logger.warning("Failed to insert kg_node: %s", resp.text)
                    return ""

                if relationships:
                    edges = [
                        {
                            "id": str(uuid.uuid4()),
                            "source_id": node_id,
                            "target_id": target_id,
                            "relation": rel_type,
                            "namespace": ns,
                        }
                        for target_id, rel_type in relationships
                    ]
                    await client.post(
                        f"{self._base}/{TABLE_EDGES}",
                        headers=self._headers,
                        json=edges,
                    )

        except Exception as e:
            logger.warning("GraphitiClient add_episode failed: %s", e)
            return ""

        return node_id

    async def get_related(
        self,
        node_id: str,
        relation: Optional[str] = None,
    ) -> list[GraphitiSearchResult]:
        """Get nodes related to a given node via edges."""
        params: dict[str, str] = {
            "source_id": f"eq.{node_id}",
            "limit": "20",
        }
        if relation:
            params["relation"] = f"eq.{relation}"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self._base}/{TABLE_EDGES}",
                    headers=self._headers,
                    params=params,
                )
                if resp.status_code != 200:
                    return []

                edges = resp.json()
                target_ids = [e["target_id"] for e in edges]
                if not target_ids:
                    return []

                ids_filter = ",".join(f'"{tid}"' for tid in target_ids)
                resp2 = await client.get(
                    f"{self._base}/{TABLE_NODES}",
                    headers=self._headers,
                    params={"id": f"in.({','.join(target_ids)})"},
                )
                if resp2.status_code == 200:
                    rows = resp2.json()
                    return [
                        GraphitiSearchResult(
                            node_id=r["id"],
                            content=r.get("content", ""),
                            metadata=r.get("metadata", {}),
                            score=1.0,
                            fact=r.get("content", ""),
                            source=r.get("metadata", {}).get("source", ""),
                        )
                        for r in rows
                    ]
        except Exception as e:
            logger.warning("GraphitiClient get_related failed: %s", e)

        return []
