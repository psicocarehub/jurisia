"""
Querido Diário API Client — diários oficiais (queridodiario.ok.org.br).

Integração com a API do Querido Diário (Open Knowledge Brasil)
para busca em diários oficiais municipais, estaduais e federais.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import httpx
from pydantic import BaseModel

logger = logging.getLogger("jurisai.querido_diario")


class GazetteItem(BaseModel):
    """Item de diário oficial."""

    id: str
    source: str
    date: date
    title: str
    excerpt: str
    url: Optional[str] = None
    territory_id: Optional[str] = None


class QueridoDiarioClient:
    """
    Cliente para a API Querido Diário (queridodiario.ok.org.br).

    Permite busca em diários oficiais de prefeituras e governos.
    """

    API_URL = "https://queridodiario.ok.org.br/api/gazettes"

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.api_url = base_url or self.API_URL

    async def search_gazettes(
        self,
        query: str,
        territory_id: Optional[str] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[GazetteItem]:
        """
        Busca em diários oficiais via API Querido Diário.
        Docs: https://queridodiario.ok.org.br/api/docs
        """
        params: dict[str, str | int] = {
            "querystring": query,
            "size": limit,
            "offset": offset,
        }
        if territory_id:
            params["territory_ids"] = territory_id
        if date_from:
            params["published_since"] = date_from.isoformat()
        if date_to:
            params["published_until"] = date_to.isoformat()

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(self.api_url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.error("Querido Diário search failed: %s", e)
            return []

        items: list[GazetteItem] = []
        for g in data.get("gazettes", []):
            try:
                items.append(
                    GazetteItem(
                        id=g.get("territory_id", "") + "-" + g.get("date", ""),
                        source="querido_diario",
                        date=g.get("date", "2000-01-01"),
                        title=g.get("territory_name", "Diário Oficial"),
                        excerpt=(g.get("excerpts", [""])[0] if g.get("excerpts") else ""),
                        url=g.get("url"),
                        territory_id=g.get("territory_id"),
                    )
                )
            except Exception as e:
                logger.warning("Failed to parse gazette item: %s", e)
                continue

        return items
