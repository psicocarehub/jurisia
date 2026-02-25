"""
JUIT API Client — 70M decisoes judiciais brasileiras com texto integral.

Projetado para RAG com LLMs. Acessa jurisprudencia de todos os tribunais
brasileiros via REST API (https://juit.com.br).
"""

import logging
from datetime import datetime
from typing import Any, Optional

import httpx
from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger("jurisai.juit")


class JUITDecision(BaseModel):
    """Decisao retornada pela JUIT API."""

    id: str
    tribunal: str
    processo: str
    classe: str = ""
    assunto: str = ""
    relator: str = ""
    data_julgamento: Optional[datetime] = None
    data_publicacao: Optional[datetime] = None
    ementa: str = ""
    inteiro_teor: str = ""
    area: str = ""
    metadata: dict[str, Any] = {}


class JUITClient:
    """
    Client REST para JUIT API (jurisprudencia brasileira).

    Busca decisoes com texto integral para indexacao no RAG pipeline.
    Requer JUIT_API_KEY configurada.
    """

    TRIBUNAIS_PRIORITARIOS = ["STJ", "STF", "TJSP", "TJRJ", "TJMG", "TJRS", "TJPR"]

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key or settings.JUIT_API_KEY
        self.base_url = (base_url or settings.JUIT_API_URL).rstrip("/")
        self._rate_limit_remaining: int = 100

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def search(
        self,
        query: str = "",
        tribunal: str | None = None,
        area: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        """
        Search decisions via JUIT API.

        Args:
            query: Full-text search query
            tribunal: Filter by tribunal (e.g. "STJ", "TJSP")
            area: Legal area filter
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            page: Page number (1-indexed)
            page_size: Results per page (max 50)
        """
        if not self.api_key:
            logger.warning("JUIT_API_KEY nao configurada")
            return {"results": [], "total": 0}

        params: dict[str, Any] = {
            "page": page,
            "pageSize": min(page_size, 50),
        }
        if query:
            params["q"] = query
        if tribunal:
            params["tribunal"] = tribunal
        if area:
            params["area"] = area
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{self.base_url}/jurisprudence",
                    headers=self._headers(),
                    params=params,
                )
                self._update_rate_limit(resp)
                resp.raise_for_status()
                data = resp.json()

            results = [self._parse_decision(d) for d in data.get("results", [])]
            return {
                "results": results,
                "total": data.get("total", 0),
                "page": data.get("page", page),
                "pages": data.get("pages", 1),
            }
        except httpx.HTTPStatusError as e:
            logger.error("JUIT API error %s: %s", e.response.status_code, e.response.text[:500])
            return {"results": [], "total": 0, "error": str(e)}
        except Exception as e:
            logger.error("JUIT request failed: %s", e)
            return {"results": [], "total": 0, "error": str(e)}

    async def get_decision(self, decision_id: str) -> JUITDecision | None:
        """Fetch a single decision by ID (full text)."""
        if not self.api_key:
            return None

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{self.base_url}/jurisprudence/{decision_id}",
                    headers=self._headers(),
                )
                self._update_rate_limit(resp)
                resp.raise_for_status()
                return self._parse_decision(resp.json())
        except Exception as e:
            logger.error("JUIT get_decision(%s) failed: %s", decision_id, e)
            return None

    async def fetch_recent(
        self,
        tribunal: str,
        since_date: str,
        max_pages: int = 10,
    ) -> list[JUITDecision]:
        """Fetch all recent decisions for a tribunal since a given date."""
        all_results: list[JUITDecision] = []

        for page in range(1, max_pages + 1):
            data = await self.search(
                tribunal=tribunal,
                date_from=since_date,
                page=page,
                page_size=50,
            )
            results = data.get("results", [])
            if not results:
                break
            all_results.extend(results)

            if page >= data.get("pages", 1):
                break

            if self._rate_limit_remaining < 5:
                import asyncio
                logger.info("Rate limit baixo (%d), aguardando...", self._rate_limit_remaining)
                await asyncio.sleep(2.0)

        logger.info(
            "JUIT fetch_recent(%s, since=%s): %d decisoes",
            tribunal, since_date, len(all_results),
        )
        return all_results

    def _parse_decision(self, raw: dict[str, Any]) -> JUITDecision:
        return JUITDecision(
            id=str(raw.get("id", "")),
            tribunal=raw.get("tribunal", ""),
            processo=raw.get("processo", raw.get("numero", "")),
            classe=raw.get("classe", ""),
            assunto=raw.get("assunto", ""),
            relator=raw.get("relator", ""),
            data_julgamento=self._parse_date(raw.get("dataJulgamento")),
            data_publicacao=self._parse_date(raw.get("dataPublicacao")),
            ementa=raw.get("ementa", ""),
            inteiro_teor=raw.get("inteiroTeor", raw.get("texto", "")),
            area=raw.get("area", ""),
            metadata={
                k: v for k, v in raw.items()
                if k not in {
                    "id", "tribunal", "processo", "numero", "classe",
                    "assunto", "relator", "dataJulgamento", "dataPublicacao",
                    "ementa", "inteiroTeor", "texto", "area",
                }
            },
        )

    @staticmethod
    def _parse_date(val: Any) -> datetime | None:
        if not val:
            return None
        if isinstance(val, datetime):
            return val
        try:
            return datetime.fromisoformat(str(val).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    def _update_rate_limit(self, resp: httpx.Response) -> None:
        remaining = resp.headers.get("X-RateLimit-Remaining")
        if remaining is not None:
            try:
                self._rate_limit_remaining = int(remaining)
            except ValueError:
                logger.debug("Non-integer X-RateLimit-Remaining: %s", remaining)
