"""
Tribunal Scraper generico — extrai decisoes de tribunais estaduais.

Suporta os 5 maiores tribunais por volume:
- TJSP (via e-SAJ)
- TJRJ (via portal jurisprudencia)
- TJMG (via portal jurisprudencia)
- TJRS (via portal jurisprudencia)
- TJPR (via portal jurisprudencia)

Baseado nos patterns do projeto courtsbr (ABJ).
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Optional

import httpx
from pydantic import BaseModel

logger = logging.getLogger("jurisai.tribunal_scraper")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


class TribunalDecision(BaseModel):
    """Decisao extraida de tribunal estadual."""

    processo: str
    tribunal: str
    classe: str = ""
    assunto: str = ""
    relator: str = ""
    data_julgamento: Optional[datetime] = None
    data_publicacao: Optional[datetime] = None
    ementa: str = ""
    decisao: str = ""
    orgao_julgador: str = ""
    comarca: str = ""
    grau: str = ""
    metadata: dict[str, Any] = {}


TRIBUNAL_CONFIGS: dict[str, dict[str, Any]] = {
    "TJRJ": {
        "base_url": "http://www4.tjrj.jus.br",
        "search_path": "/ejuris/ConsultarJurisprudencia.aspx",
        "search_method": "POST",
        "params_template": {
            "txtPesquisaLivre": "{query}",
            "txtDataIni": "{date_from}",
            "txtDataFim": "{date_to}",
        },
    },
    "TJMG": {
        "base_url": "https://www5.tjmg.jus.br",
        "search_path": "/jurisprudencia/pesquisaPalavrasEspelhoAcordao.do",
        "search_method": "GET",
        "params_template": {
            "palavras": "{query}",
            "dataPublicacaoInicial": "{date_from}",
            "dataPublicacaoFinal": "{date_to}",
            "paginaNumero": "{page}",
        },
    },
    "TJRS": {
        "base_url": "https://www.tjrs.jus.br",
        "search_path": "/buscas/jurisprudencia/",
        "search_method": "GET",
        "params_template": {
            "q": "{query}",
            "data_ini": "{date_from}",
            "data_fim": "{date_to}",
            "pagina": "{page}",
        },
    },
    "TJPR": {
        "base_url": "https://portal.tjpr.jus.br",
        "search_path": "/jurisprudencia/publico/pesquisa.do",
        "search_method": "GET",
        "params_template": {
            "actionType": "pesquisar",
            "criterio.pesquisa": "{query}",
            "criterio.dtJulgamentoInicio": "{date_from}",
            "criterio.dtJulgamentoFim": "{date_to}",
            "paginacao.pageNumber": "{page}",
        },
    },
}


class TribunalScraper:
    """
    Scraper generico para tribunais estaduais.

    Usa patterns do courtsbr (ABJ toolkit) adaptados para scraping HTTP.
    Rate limiting agressivo (min 3s entre requests) para nao sobrecarregar.
    """

    MIN_DELAY = 3.0

    def __init__(self, tribunal: str) -> None:
        self.tribunal = tribunal.upper()
        if self.tribunal == "TJSP":
            from app.services.ingestion.esaj import ESAJClient
            self._esaj = ESAJClient("TJSP")
        else:
            self._esaj = None

        self.config = TRIBUNAL_CONFIGS.get(self.tribunal, {})
        self._last_request = 0.0

    async def _rate_limit(self) -> None:
        import time
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < self.MIN_DELAY:
            await asyncio.sleep(self.MIN_DELAY - elapsed)
        self._last_request = time.time()

    async def search(
        self,
        query: str = "",
        date_from: str | None = None,
        date_to: str | None = None,
        page: int = 1,
        limit: int = 20,
    ) -> list[TribunalDecision]:
        """
        Busca decisoes no tribunal configurado.

        Args:
            query: Pesquisa livre
            date_from: Data inicio (DD/MM/YYYY)
            date_to: Data fim (DD/MM/YYYY)
            page: Pagina
            limit: Maximo de resultados
        """
        if self._esaj:
            esaj_results = await self._esaj.search_decisions(
                query=query, date_from=date_from, date_to=date_to, limit=limit,
            )
            return [
                TribunalDecision(
                    processo=r.processo,
                    tribunal=self.tribunal,
                    classe=r.classe,
                    assunto=r.assunto,
                    relator=r.relator,
                    data_julgamento=r.data_decisao,
                    ementa=r.ementa or r.texto,
                    orgao_julgador=r.vara,
                    comarca=r.comarca,
                    grau=r.metadata.get("grau", ""),
                    metadata=r.metadata,
                )
                for r in esaj_results
            ]

        if not self.config:
            logger.warning("Tribunal %s nao configurado", self.tribunal)
            return []

        return await self._generic_search(query, date_from, date_to, page)

    async def fetch_recent(
        self,
        since_date: str,
        max_pages: int = 10,
    ) -> list[TribunalDecision]:
        """Fetch recent decisions since a given date."""
        all_results: list[TribunalDecision] = []
        today = datetime.utcnow().strftime("%d/%m/%Y")

        for page in range(1, max_pages + 1):
            results = await self.search(
                date_from=since_date,
                date_to=today,
                page=page,
            )
            if not results:
                break
            all_results.extend(results)

        logger.info(
            "%s fetch_recent(since=%s): %d decisoes",
            self.tribunal, since_date, len(all_results),
        )
        return all_results

    async def _generic_search(
        self,
        query: str,
        date_from: str | None,
        date_to: str | None,
        page: int,
    ) -> list[TribunalDecision]:
        """Generic HTTP-based search for tribunals with configured patterns."""
        config = self.config
        base_url = config["base_url"]
        search_path = config["search_path"]
        method = config.get("search_method", "GET")

        params: dict[str, str] = {}
        for key, template in config.get("params_template", {}).items():
            val = (
                template
                .replace("{query}", query or "")
                .replace("{date_from}", date_from or "")
                .replace("{date_to}", date_to or "")
                .replace("{page}", str(page))
            )
            if val.strip():
                params[key] = val

        await self._rate_limit()

        try:
            async with httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
            ) as client:
                if method == "POST":
                    resp = await client.post(f"{base_url}{search_path}", data=params)
                else:
                    resp = await client.get(f"{base_url}{search_path}", params=params)
                resp.raise_for_status()

            return self._parse_generic_results(resp.text)
        except Exception as e:
            logger.error("%s search failed: %s", self.tribunal, e)
            return []

    def _parse_generic_results(self, html: str) -> list[TribunalDecision]:
        """
        Generic HTML parser for tribunal results.

        Attempts to extract decisions using common patterns found in
        Brazilian tribunal websites.
        """
        decisions: list[TribunalDecision] = []

        cnj_pattern = r'(\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4})'
        processos = re.findall(cnj_pattern, html)

        blocks = re.split(cnj_pattern, html)
        for i in range(1, len(blocks), 2):
            processo = blocks[i]
            content = blocks[i + 1] if i + 1 < len(blocks) else ""

            relator = self._extract(content, r'[Rr]elator[a]?[:\s]+([^<\n]{3,80})')
            classe = self._extract(content, r'[Cc]lasse[:\s]+([^<\n]{3,80})')
            assunto = self._extract(content, r'[Aa]ssunto[:\s]+([^<\n]{3,80})')
            data_str = self._extract(content, r'(\d{2}/\d{2}/\d{4})')
            orgao = self._extract(content, r'[ÓOo]rg[ãa]o\s+[Jj]ulgador[:\s]+([^<\n]{3,100})')
            comarca = self._extract(content, r'[Cc]omarca[:\s]+([^<\n]{3,80})')

            ementa_match = re.search(
                r'[Ee]menta[:\s]*(.*?)(?=<div|<br\s*/?\s*>\s*<br|$)',
                content, re.DOTALL,
            )
            ementa = self._clean_html(ementa_match.group(1)) if ementa_match else ""

            decisions.append(TribunalDecision(
                processo=processo,
                tribunal=self.tribunal,
                classe=(classe or "").strip(),
                assunto=(assunto or "").strip(),
                relator=(relator or "").strip(),
                data_julgamento=self._parse_date(data_str),
                ementa=ementa[:5000],
                orgao_julgador=(orgao or "").strip(),
                comarca=(comarca or "").strip(),
                metadata={"source": "scraper"},
            ))

        return decisions

    @staticmethod
    def _extract(text: str, pattern: str) -> str | None:
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    @staticmethod
    def _clean_html(html: str) -> str:
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def _parse_date(val: Any) -> datetime | None:
        if not val:
            return None
        for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(str(val).strip(), fmt)
            except (ValueError, TypeError):
                continue
        return None


def get_scraper(tribunal: str) -> TribunalScraper:
    """Factory function to get scraper for a specific tribunal."""
    return TribunalScraper(tribunal)


SUPPORTED_TRIBUNAIS = ["TJSP", "TJRJ", "TJMG", "TJRS", "TJPR"]
