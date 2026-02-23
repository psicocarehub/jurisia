"""
LexML API Client -- legislacao brasileira (lexml.gov.br).

Uses the LexML OAI-PMH and SRU/SRW interfaces for structured
access to Brazilian legislation with persistent URN LEX identifiers.
"""

import re
import xml.etree.ElementTree as ET
from typing import Any, Optional

import httpx
from pydantic import BaseModel

from app.config import settings


class LexMLNorma(BaseModel):
    """Norma/legislacao recuperada via LexML."""

    urn: str
    tipo: str
    titulo: str
    ementa: Optional[str] = None
    texto_completo: Optional[str] = None
    data_publicacao: Optional[str] = None
    metadata: dict[str, Any] = {}


_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "mods": "http://www.loc.gov/mods/v3",
    "srw": "http://www.loc.gov/zing/srw/",
    "dc": "http://purl.org/dc/elements/1.1/",
}

TIPO_MAP = {
    "lei": "lei",
    "lei.complementar": "lei_complementar",
    "decreto": "decreto",
    "decreto-lei": "decreto_lei",
    "medida.provisoria": "medida_provisoria",
    "resolucao": "resolucao",
    "emenda.constitucional": "emenda_constitucional",
    "portaria": "portaria",
    "sumula": "sumula",
}


class LexMLClient:
    """
    Client for LexML (lexml.gov.br).

    Supports two interfaces:
    - SRU/SRW: structured search with CQL queries
    - OAI-PMH: incremental harvesting for bulk ingestion
    """

    SRU_URL = "https://www.lexml.gov.br/busca/SRU"
    OAI_URL = "https://www.lexml.gov.br/oai/oai.php"
    RESOLVER_URL = "https://www.lexml.gov.br/urn"

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.sru_url = base_url or self.SRU_URL

    async def search_legislation(
        self,
        query: str,
        tipo: Optional[str] = None,
        esfera: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[LexMLNorma]:
        """
        Search legislation via SRU/SRW with CQL query syntax.

        Args:
            query: Free-text search or CQL expression
            tipo: Filter by type (lei, decreto, resolucao, etc.)
            esfera: Filter by sphere (federal, estadual, municipal)
            limit: Max results (up to 100 per request)
            offset: Start position for pagination
        """
        cql_parts = []

        if query:
            escaped = query.replace('"', '\\"')
            cql_parts.append(f'dc.description all "{escaped}"')

        if tipo:
            cql_parts.append(f'dc.type = "{tipo}"')

        if esfera:
            cql_parts.append(f'dc.coverage = "{esfera}"')

        cql = " AND ".join(cql_parts) if cql_parts else "dc.type any lei"

        params = {
            "operation": "searchRetrieve",
            "version": "1.1",
            "query": cql,
            "maximumRecords": str(min(limit, 100)),
            "startRecord": str(offset + 1),
            "recordSchema": "mods",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(self.sru_url, params=params)
            response.raise_for_status()

        return self._parse_sru_response(response.text)

    async def get_by_urn(self, urn: str) -> Optional[LexMLNorma]:
        """
        Retrieve a specific norm by its URN LEX identifier.

        URN LEX format: urn:lex:br:federal:lei:2002-01-10;10406
        """
        url = f"{self.RESOLVER_URL}/{urn}"

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url)
            if response.status_code != 200:
                return None

        tipo = _extract_tipo_from_urn(urn)
        title = _extract_title_from_urn(urn)

        text = _strip_html(response.text)

        return LexMLNorma(
            urn=urn,
            tipo=tipo,
            titulo=title,
            ementa=_extract_ementa(text),
            texto_completo=text[:100000],
            data_publicacao=_extract_date_from_urn(urn),
            metadata={"source": "lexml_resolver", "url": str(response.url)},
        )

    async def harvest_recent(
        self,
        from_date: Optional[str] = None,
        set_spec: Optional[str] = None,
        limit: int = 200,
    ) -> list[LexMLNorma]:
        """
        Harvest recent records via OAI-PMH ListRecords.

        Args:
            from_date: ISO date (YYYY-MM-DD) to fetch records modified after
            set_spec: OAI set filter (e.g., "tipo:lei" or "esfera:federal")
            limit: Max records to fetch
        """
        params: dict[str, str] = {
            "verb": "ListRecords",
            "metadataPrefix": "mods",
        }
        if from_date:
            params["from"] = from_date
        if set_spec:
            params["set"] = set_spec

        results: list[LexMLNorma] = []

        async with httpx.AsyncClient(timeout=60.0) as client:
            while len(results) < limit:
                response = await client.get(self.OAI_URL, params=params)
                if response.status_code != 200:
                    break

                batch, resumption = self._parse_oai_response(response.text)
                results.extend(batch)

                if not resumption:
                    break
                params = {
                    "verb": "ListRecords",
                    "resumptionToken": resumption,
                }

        return results[:limit]

    async def search_by_article(
        self,
        law_type: str,
        law_number: str,
    ) -> Optional[LexMLNorma]:
        """
        Find a specific law by type and number.

        Args:
            law_type: Type of law (lei, decreto, etc.)
            law_number: Number with optional year (e.g., "10406/2002")
        """
        cql = f'dc.identifier = "{law_type} {law_number}"'
        params = {
            "operation": "searchRetrieve",
            "version": "1.1",
            "query": cql,
            "maximumRecords": "1",
            "recordSchema": "mods",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(self.sru_url, params=params)
            if response.status_code != 200:
                return None

        results = self._parse_sru_response(response.text)
        return results[0] if results else None

    def _parse_sru_response(self, xml_text: str) -> list[LexMLNorma]:
        """Parse SRU/SRW XML response into LexMLNorma objects."""
        results: list[LexMLNorma] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return results

        records = root.findall(".//srw:record", _NS)
        for record in records:
            mods = record.find(".//mods:mods", _NS)
            if mods is None:
                continue

            urn = ""
            identifier = mods.find("mods:identifier[@type='uri']", _NS)
            if identifier is not None and identifier.text:
                urn = identifier.text

            title_el = mods.find("mods:titleInfo/mods:title", _NS)
            titulo = title_el.text.strip() if title_el is not None and title_el.text else ""

            abstract_el = mods.find("mods:abstract", _NS)
            ementa = abstract_el.text.strip() if abstract_el is not None and abstract_el.text else ""

            date_el = mods.find("mods:originInfo/mods:dateIssued", _NS)
            data_pub = date_el.text.strip() if date_el is not None and date_el.text else ""

            genre_el = mods.find("mods:genre", _NS)
            tipo_raw = genre_el.text.strip().lower() if genre_el is not None and genre_el.text else ""
            tipo = TIPO_MAP.get(tipo_raw, tipo_raw)

            results.append(LexMLNorma(
                urn=urn,
                tipo=tipo,
                titulo=titulo,
                ementa=ementa,
                data_publicacao=data_pub,
                metadata={"source": "lexml_sru"},
            ))

        return results

    def _parse_oai_response(self, xml_text: str) -> tuple[list[LexMLNorma], Optional[str]]:
        """Parse OAI-PMH ListRecords response."""
        results: list[LexMLNorma] = []
        resumption_token: Optional[str] = None

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return results, None

        records = root.findall(".//oai:record", _NS)
        for record in records:
            header = record.find("oai:header", _NS)
            if header is not None:
                status = header.get("status", "")
                if status == "deleted":
                    continue

            mods = record.find(".//mods:mods", _NS)
            if mods is None:
                dc = record.find(".//oai:metadata", _NS)
                if dc is not None:
                    title_el = dc.find("dc:title", _NS)
                    ident_el = dc.find("dc:identifier", _NS)
                    results.append(LexMLNorma(
                        urn=ident_el.text if ident_el is not None and ident_el.text else "",
                        tipo="unknown",
                        titulo=title_el.text if title_el is not None and title_el.text else "",
                        metadata={"source": "lexml_oai"},
                    ))
                continue

            urn = ""
            identifier = mods.find("mods:identifier[@type='uri']", _NS)
            if identifier is not None and identifier.text:
                urn = identifier.text

            title_el = mods.find("mods:titleInfo/mods:title", _NS)
            titulo = title_el.text.strip() if title_el is not None and title_el.text else ""

            abstract_el = mods.find("mods:abstract", _NS)
            ementa = abstract_el.text.strip() if abstract_el is not None and abstract_el.text else ""

            genre_el = mods.find("mods:genre", _NS)
            tipo_raw = genre_el.text.strip().lower() if genre_el is not None and genre_el.text else ""
            tipo = TIPO_MAP.get(tipo_raw, tipo_raw)

            results.append(LexMLNorma(
                urn=urn,
                tipo=tipo,
                titulo=titulo,
                ementa=ementa,
                metadata={"source": "lexml_oai"},
            ))

        token_el = root.find(".//oai:resumptionToken", _NS)
        if token_el is not None and token_el.text:
            resumption_token = token_el.text

        return results, resumption_token


def _extract_tipo_from_urn(urn: str) -> str:
    parts = urn.split(":")
    if len(parts) >= 5:
        raw = parts[4].replace(".", "_")
        return TIPO_MAP.get(raw, raw)
    return "unknown"


def _extract_title_from_urn(urn: str) -> str:
    parts = urn.split(":")
    if len(parts) >= 6:
        info = parts[5]
        date_part, _, number = info.partition(";")
        tipo = parts[4].replace(".", " ").title() if len(parts) > 4 else ""
        return f"{tipo} nº {number} de {date_part}" if number else f"{tipo} {date_part}"
    return urn


def _extract_date_from_urn(urn: str) -> str:
    parts = urn.split(":")
    if len(parts) >= 6:
        info = parts[5]
        date_part = info.split(";")[0]
        return date_part
    return ""


def _strip_html(html: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "\n", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()


def _extract_ementa(text: str) -> str:
    match = re.search(
        r"((?:Dispõe|Altera|Institui|Regulamenta|Dá nova redação|Estabelece).*?\.)",
        text[:3000],
        re.DOTALL,
    )
    return match.group(1).strip()[:2000] if match else ""
