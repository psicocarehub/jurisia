"""
CARF Client — Conselho Administrativo de Recursos Fiscais.

Download e parser de decisoes tributarias do CARF (Ministerio da Fazenda).
Materias: IRPF, IRPJ, PIS/COFINS, CSLL, IPI, aduaneiro, etc.

Fonte: https://carf.fazenda.gov.br/
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx
from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger("jurisai.carf")

CARF_BASE = "https://carf.fazenda.gov.br"
CARF_SEARCH = f"{CARF_BASE}/sincon/public/pages/ConsultarJurisprudencia"


class CARFDecision(BaseModel):
    """Decisao do CARF."""

    id: str
    numero_processo: str
    numero_acordao: str = ""
    turma: str = ""
    secao: str = ""
    relator: str = ""
    data_sessao: Optional[datetime] = None
    data_publicacao: Optional[datetime] = None
    ementa: str = ""
    decisao: str = ""
    materia: str = ""
    assunto: str = ""
    recurso_tipo: str = ""
    resultado: str = ""
    metadata: dict[str, Any] = {}


MATERIA_KEYWORDS: dict[str, list[str]] = {
    "IRPF": ["imposto de renda pessoa física", "IRPF", "declaração de ajuste"],
    "IRPJ": ["imposto de renda pessoa jurídica", "IRPJ", "lucro real", "lucro presumido"],
    "PIS_COFINS": ["PIS", "COFINS", "contribuição social", "não cumulatividade"],
    "CSLL": ["CSLL", "contribuição social sobre o lucro"],
    "IPI": ["IPI", "imposto sobre produtos industrializados"],
    "IOF": ["IOF", "imposto sobre operações financeiras"],
    "CONTRIBUICOES": ["contribuição previdenciária", "INSS patronal"],
    "ADUANEIRO": ["aduana", "importação", "exportação", "classificação fiscal"],
    "SIMPLES": ["Simples Nacional", "MEI", "microempresa"],
    "ITCMD": ["ITCMD", "transmissão causa mortis", "doação"],
}


class CARFClient:
    """
    Client para download e busca de decisoes do CARF.

    Acessa o sistema SINCON (Sistema de Consulta) do CARF para
    pesquisa de jurisprudencia tributaria.
    """

    def __init__(self) -> None:
        self.base_url = CARF_BASE

    async def search_decisions(
        self,
        query: str = "",
        materia: str | None = None,
        turma: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> list[CARFDecision]:
        """
        Search CARF decisions via the SINCON public search.

        Args:
            query: Full-text search (ementa, decisao)
            materia: Tax subject filter (IRPF, IRPJ, PIS_COFINS, etc.)
            turma: CARF chamber/turma filter
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            page: Page number
            page_size: Results per page
        """
        params: dict[str, Any] = {"pagina": page, "tamanhoPagina": page_size}
        if query:
            params["ementa"] = query
        if materia:
            params["materia"] = materia
        if turma:
            params["turma"] = turma
        if date_from:
            params["dataSessaoInicio"] = date_from
        if date_to:
            params["dataSessaoFim"] = date_to

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{CARF_SEARCH}/listarJurisprudencia",
                    params=params,
                )
                resp.raise_for_status()
                data = resp.json()

            results = data if isinstance(data, list) else data.get("registros", data.get("data", []))
            return [self._parse_decision(r) for r in results]

        except httpx.HTTPStatusError as e:
            logger.error("CARF search error %s: %s", e.response.status_code, e.response.text[:500])
            return []
        except Exception as e:
            logger.error("CARF search failed: %s", e)
            return []

    async def get_decision(self, numero_acordao: str) -> CARFDecision | None:
        """Fetch a specific CARF decision by acordao number."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{CARF_SEARCH}/detalharJurisprudencia",
                    params={"numeroAcordao": numero_acordao},
                )
                resp.raise_for_status()
                return self._parse_decision(resp.json())
        except Exception as e:
            logger.error("CARF get_decision(%s) failed: %s", numero_acordao, e)
            return None

    async def fetch_recent(
        self,
        since_date: str,
        materia: str | None = None,
        max_pages: int = 20,
    ) -> list[CARFDecision]:
        """Fetch all recent decisions since a given date."""
        all_results: list[CARFDecision] = []

        for page in range(1, max_pages + 1):
            results = await self.search_decisions(
                materia=materia,
                date_from=since_date,
                page=page,
                page_size=50,
            )
            if not results:
                break
            all_results.extend(results)

        logger.info("CARF fetch_recent(since=%s): %d decisoes", since_date, len(all_results))
        return all_results

    def classify_materia(self, text: str) -> str:
        """Classify the tax subject of a decision based on keywords."""
        text_lower = text.lower()
        scores: dict[str, int] = {}
        for materia, keywords in MATERIA_KEYWORDS.items():
            scores[materia] = sum(1 for kw in keywords if kw.lower() in text_lower)

        if not scores or max(scores.values()) == 0:
            return "outros"
        return max(scores, key=lambda k: scores[k])

    def _parse_decision(self, raw: dict[str, Any]) -> CARFDecision:
        ementa = raw.get("ementa", raw.get("textoEmenta", ""))
        decisao = raw.get("decisao", raw.get("textoDecisao", ""))
        materia = raw.get("materia", "")
        if not materia:
            materia = self.classify_materia(ementa + " " + decisao)

        return CARFDecision(
            id=str(raw.get("id", raw.get("numeroAcordao", ""))),
            numero_processo=raw.get("numeroProcesso", ""),
            numero_acordao=raw.get("numeroAcordao", ""),
            turma=raw.get("turma", raw.get("nomeTurma", "")),
            secao=raw.get("secao", raw.get("nomeSecao", "")),
            relator=raw.get("relator", raw.get("nomeRelator", "")),
            data_sessao=self._parse_date(raw.get("dataSessao")),
            data_publicacao=self._parse_date(raw.get("dataPublicacao")),
            ementa=ementa,
            decisao=decisao,
            materia=materia,
            assunto=raw.get("assunto", ""),
            recurso_tipo=raw.get("tipoRecurso", ""),
            resultado=raw.get("resultado", raw.get("tipoResultado", "")),
            metadata={
                k: v for k, v in raw.items()
                if k not in {
                    "id", "numeroProcesso", "numeroAcordao", "turma", "nomeTurma",
                    "secao", "nomeSecao", "relator", "nomeRelator", "dataSessao",
                    "dataPublicacao", "ementa", "textoEmenta", "decisao", "textoDecisao",
                    "materia", "assunto", "tipoRecurso", "resultado", "tipoResultado",
                }
            },
        )

    @staticmethod
    def _parse_date(val: Any) -> datetime | None:
        if not val:
            return None
        if isinstance(val, datetime):
            return val
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(str(val), fmt)
            except (ValueError, TypeError):
                continue
        return None
