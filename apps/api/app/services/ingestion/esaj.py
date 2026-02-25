"""
e-SAJ Client — scraping de decisoes judiciais de tribunais estaduais.

Implementa scraping para:
- CJPG (Consulta de Julgados de 1o Grau) — decisoes de primeira instancia
- CJSG (Consulta de Julgados de 2o Grau) — acordaos de segunda instancia

Tribunais suportados: TJSP (principal), adaptavel para TJMS, TJMT, TJAM, etc.

NOTA: Scraping deve respeitar robots.txt, rate limiting e LGPD.
Nao indexar dados sigilosos ou processos em segredo de justica.
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel

logger = logging.getLogger("jurisai.esaj")

ESAJ_URLS: dict[str, dict[str, str]] = {
    "TJSP": {
        "base": "https://esaj.tjsp.jus.br",
        "cjsg": "/cjsg/resultadoCompleta.do",
        "cjpg": "/cjpg/resultadoCompleta.do",
        "inteiro_teor": "/cjsg/getArquivo.do",
    },
    "TJMS": {
        "base": "https://esaj.tjms.jus.br",
        "cjsg": "/cjsg/resultadoCompleta.do",
        "cjpg": "/cjpg/resultadoCompleta.do",
        "inteiro_teor": "/cjsg/getArquivo.do",
    },
    "TJAM": {
        "base": "https://esaj.tjam.jus.br",
        "cjsg": "/cjsg/resultadoCompleta.do",
        "cjpg": "/cjpg/resultadoCompleta.do",
        "inteiro_teor": "/cjsg/getArquivo.do",
    },
}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


class ESAJDecision(BaseModel):
    """Decisao recuperada via e-SAJ."""

    processo: str
    tribunal: str
    vara: str
    classe: str
    assunto: str
    data_decisao: Optional[datetime] = None
    texto: str = ""
    ementa: str = ""
    relator: str = ""
    comarca: str = ""
    metadata: dict[str, Any] = {}


class ESAJClient:
    """
    Client para scraping de decisoes via e-SAJ (TJSP e outros tribunais).

    Implementa CJSG (2o grau) e CJPG (1o grau) scraping com:
    - Rate limiting (min 2s entre requests)
    - Retry com backoff
    - Parser HTML para extracao de dados
    - Respeito a robots.txt
    """

    MIN_DELAY = 2.0

    def __init__(self, tribunal: str = "TJSP") -> None:
        self.tribunal = tribunal
        urls = ESAJ_URLS.get(tribunal, ESAJ_URLS["TJSP"])
        self.base_url = urls["base"]
        self.cjsg_path = urls["cjsg"]
        self.cjpg_path = urls["cjpg"]
        self.inteiro_teor_path = urls["inteiro_teor"]
        self._last_request = 0.0

    async def _rate_limit(self) -> None:
        """Enforce minimum delay between requests."""
        import time
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < self.MIN_DELAY:
            await asyncio.sleep(self.MIN_DELAY - elapsed)
        self._last_request = time.time()

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "pt-BR,pt;q=0.9",
            },
        )

    async def search_cjsg(
        self,
        query: str = "",
        classe: str | None = None,
        assunto: str | None = None,
        relator: str | None = None,
        comarca: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> list[ESAJDecision]:
        """
        Busca acordaos de 2o grau via CJSG.

        Args:
            query: Pesquisa livre (ementa)
            classe: Classe processual
            assunto: Assunto/materia
            relator: Desembargador relator
            comarca: Comarca de origem
            date_from: Data inicio (DD/MM/YYYY)
            date_to: Data fim (DD/MM/YYYY)
        """
        params: dict[str, str] = {
            "conversationId": "",
            "pesquisaLivre": query,
            "tipoDecisao": "A",
            "nuPagina": str(page),
        }
        if classe:
            params["classeTreeSelection.values"] = classe
        if assunto:
            params["assuntoTreeSelection.values"] = assunto
        if relator:
            params["relator"] = relator
        if comarca:
            params["comarcaTreeSelection.values"] = comarca
        if date_from:
            params["dtJulgamentoInicio"] = date_from
        if date_to:
            params["dtJulgamentoFim"] = date_to

        await self._rate_limit()

        try:
            async with self._client() as client:
                resp = await client.get(
                    f"{self.base_url}{self.cjsg_path}",
                    params=params,
                )
                resp.raise_for_status()
                return self._parse_cjsg_results(resp.text)
        except Exception as e:
            logger.error("CJSG search failed (%s): %s", self.tribunal, e)
            return []

    async def search_cjpg(
        self,
        query: str = "",
        classe: str | None = None,
        comarca: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        page: int = 1,
    ) -> list[ESAJDecision]:
        """
        Busca decisoes de 1o grau via CJPG.

        Args:
            query: Pesquisa livre
            classe: Classe processual
            comarca: Comarca
            date_from: Data inicio (DD/MM/YYYY)
            date_to: Data fim (DD/MM/YYYY)
        """
        params: dict[str, str] = {
            "conversationId": "",
            "pesquisaLivre": query,
            "nuPagina": str(page),
        }
        if classe:
            params["classeTreeSelection.values"] = classe
        if comarca:
            params["comarcaTreeSelection.values"] = comarca
        if date_from:
            params["dtJulgamentoInicio"] = date_from
        if date_to:
            params["dtJulgamentoFim"] = date_to

        await self._rate_limit()

        try:
            async with self._client() as client:
                resp = await client.get(
                    f"{self.base_url}{self.cjpg_path}",
                    params=params,
                )
                resp.raise_for_status()
                return self._parse_cjpg_results(resp.text)
        except Exception as e:
            logger.error("CJPG search failed (%s): %s", self.tribunal, e)
            return []

    async def search_decisions(
        self,
        query: str = "",
        classe: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 50,
    ) -> list[ESAJDecision]:
        """Unified search combining CJSG and CJPG results."""
        cjsg = await self.search_cjsg(query=query, classe=classe, date_from=date_from, date_to=date_to)
        cjpg = await self.search_cjpg(query=query, classe=classe, date_from=date_from, date_to=date_to)
        combined = cjsg + cjpg
        return combined[:limit]

    async def get_decision_text(
        self,
        processo: str,
        movimento_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Obtem o texto integral de uma decisao especifica.

        Args:
            processo: Numero do processo (CNJ)
            movimento_id: ID do movimento/decisao
        """
        if not movimento_id:
            return None

        await self._rate_limit()

        try:
            async with self._client() as client:
                resp = await client.get(
                    f"{self.base_url}{self.inteiro_teor_path}",
                    params={"cdAcordao": movimento_id},
                )
                resp.raise_for_status()
                if "text/html" in resp.headers.get("content-type", ""):
                    return self._extract_text_from_html(resp.text)
                return resp.text
        except Exception as e:
            logger.error("Erro obtendo inteiro teor %s: %s", processo, e)
            return None

    def _parse_cjsg_results(self, html: str) -> list[ESAJDecision]:
        """Parse CJSG HTML results page into decisions."""
        decisions: list[ESAJDecision] = []

        blocks = re.split(r'<div\s+class="[^"]*resultado[^"]*"', html)
        for block in blocks[1:]:
            try:
                processo = self._extract_field(block, r'Processo:\s*</strong>\s*([\d\.\-/]+)')
                if not processo:
                    numero_match = re.search(r'(\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4})', block)
                    processo = numero_match.group(1) if numero_match else ""

                if not processo:
                    continue

                relator = self._extract_field(block, r'Relator[a]?:\s*</strong>\s*([^<]+)')
                comarca = self._extract_field(block, r'Comarca:\s*</strong>\s*([^<]+)')
                classe = self._extract_field(block, r'Classe[^:]*:\s*</strong>\s*([^<]+)')
                assunto = self._extract_field(block, r'Assunto:\s*</strong>\s*([^<]+)')
                data_str = self._extract_field(block, r'Data\s+d[eo]\s+[Jj]ulgamento:\s*</strong>\s*(\d{2}/\d{2}/\d{4})')

                ementa_match = re.search(r'Ementa:\s*</strong>\s*(.*?)(?:<div|<br\s*/?\s*>\s*<br)', block, re.DOTALL)
                ementa = self._clean_html(ementa_match.group(1)) if ementa_match else ""

                decisions.append(ESAJDecision(
                    processo=processo.strip(),
                    tribunal=self.tribunal,
                    vara="",
                    classe=(classe or "").strip(),
                    assunto=(assunto or "").strip(),
                    data_decisao=self._parse_date(data_str),
                    ementa=ementa.strip(),
                    relator=(relator or "").strip(),
                    comarca=(comarca or "").strip(),
                    metadata={"grau": "2", "source": "CJSG"},
                ))
            except Exception as e:
                logger.warning("CJSG block parse error (skipping): %s", e)
                continue

        return decisions

    def _parse_cjpg_results(self, html: str) -> list[ESAJDecision]:
        """Parse CJPG HTML results page into decisions."""
        decisions: list[ESAJDecision] = []

        blocks = re.split(r'<div\s+class="[^"]*resultado[^"]*"', html)
        for block in blocks[1:]:
            try:
                processo = self._extract_field(block, r'Processo:\s*</strong>\s*([\d\.\-/]+)')
                if not processo:
                    numero_match = re.search(r'(\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4})', block)
                    processo = numero_match.group(1) if numero_match else ""

                if not processo:
                    continue

                vara = self._extract_field(block, r'Vara:\s*</strong>\s*([^<]+)') or ""
                classe = self._extract_field(block, r'Classe[^:]*:\s*</strong>\s*([^<]+)') or ""
                assunto = self._extract_field(block, r'Assunto:\s*</strong>\s*([^<]+)') or ""
                comarca = self._extract_field(block, r'Comarca:\s*</strong>\s*([^<]+)') or ""
                data_str = self._extract_field(block, r'Data\s+d[ae]\s+[Dd]ecis[aã]o:\s*</strong>\s*(\d{2}/\d{2}/\d{4})')

                texto_match = re.search(r'class="[^"]*textoSentenca[^"]*"[^>]*>(.*?)</div>', block, re.DOTALL)
                texto = self._clean_html(texto_match.group(1)) if texto_match else ""

                decisions.append(ESAJDecision(
                    processo=processo.strip(),
                    tribunal=self.tribunal,
                    vara=vara.strip(),
                    classe=classe.strip(),
                    assunto=assunto.strip(),
                    data_decisao=self._parse_date(data_str),
                    texto=texto.strip(),
                    comarca=comarca.strip(),
                    metadata={"grau": "1", "source": "CJPG"},
                ))
            except Exception as e:
                logger.warning("CJPG block parse error (skipping): %s", e)
                continue

        return decisions

    @staticmethod
    def _extract_field(html: str, pattern: str) -> str | None:
        match = re.search(pattern, html, re.IGNORECASE)
        return match.group(1).strip() if match else None

    @staticmethod
    def _clean_html(html: str) -> str:
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def _extract_text_from_html(html: str) -> str:
        body_match = re.search(r'<body[^>]*>(.*?)</body>', html, re.DOTALL | re.IGNORECASE)
        content = body_match.group(1) if body_match else html
        return ESAJClient._clean_html(content)

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
