"""
STF Client — decisoes do Supremo Tribunal Federal.

Integra:
1. API de jurisprudencia do portal STF (busca textual)
2. Corte Aberta (portal.stf.jus.br/hotsites/corteaberta/) — CSVs de votos,
   decisoes e ~1.400 temas de repercussao geral

Dados Abertos STF: https://transparencia.stf.jus.br/single/?appid=615fc495-804d-4b55-b740-1a5e3d1e4e34
"""

import csv
import io
import logging
import re
from datetime import datetime
from typing import Any, Optional

import httpx
from pydantic import BaseModel

logger = logging.getLogger("jurisai.stf")

CORTE_ABERTA_BASE = "https://transparencia.stf.jus.br"
STF_API_BASE = "https://portal.stf.jus.br"

CORTE_ABERTA_CSVS = {
    "decisoes": "/extensions/app/corteaberta/pautas.csv",
    "votos": "/extensions/app/corteaberta/votacoes.csv",
    "repercussao": "/extensions/app/corteaberta/repercussao_geral.csv",
}


class STFDecision(BaseModel):
    """Decisao do STF."""

    id: str
    processo: str
    relator: Optional[str] = None
    classe: str
    data_julgamento: Optional[datetime] = None
    ementa: str = ""
    texto_inteiro: str = ""
    metadata: dict[str, Any] = {}


class STFVoto(BaseModel):
    """Voto individual de ministro em julgamento do STF."""

    processo: str
    ministro: str
    voto: str
    data_sessao: Optional[datetime] = None
    tipo_julgamento: str = ""
    metadata: dict[str, Any] = {}


class RepercussaoGeral(BaseModel):
    """Tema de Repercussao Geral do STF."""

    tema_numero: int
    titulo: str
    descricao: str = ""
    relator: str = ""
    leading_case: str = ""
    situacao: str = ""
    tese: str = ""
    data_reconhecimento: Optional[datetime] = None
    areas: list[str] = []
    metadata: dict[str, Any] = {}


class STFClient:
    """
    Cliente para API de decisoes do STF + Corte Aberta.

    Combina busca textual de jurisprudencia com dados estruturados
    de votacoes e temas de repercussao geral.
    """

    BASE_URL = "https://portal.stf.jus.br/jurisprudencia"

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = base_url or self.BASE_URL

    async def search_decisions(
        self,
        query: str = "",
        classe: Optional[str] = None,
        relator: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[STFDecision]:
        """
        Busca decisoes na API do STF.

        Args:
            query: Termo de busca (ementa, processo, etc.)
            classe: Classe processual (ADI, ADPF, RE, etc.)
            relator: Nome do ministro relator
            date_from: Data inicial (YYYY-MM-DD)
            date_to: Data final (YYYY-MM-DD)
            limit: Maximo de resultados
            offset: Offset para paginacao
        """
        params: dict[str, Any] = {"pageSize": limit, "offset": offset}
        if query:
            params["pesquisa"] = query
        if classe:
            params["classe"] = classe
        if relator:
            params["relator"] = relator
        if date_from:
            params["dataInicio"] = date_from
        if date_to:
            params["dataFim"] = date_to

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{self.base_url}/pesquisa",
                    params=params,
                )
                resp.raise_for_status()
                data = resp.json()

            results = data if isinstance(data, list) else data.get("result", data.get("items", []))
            return [self._parse_decision(r) for r in results]
        except Exception as e:
            logger.error("STF search failed: %s", e)
            return []

    async def get_decision(self, processo: str) -> Optional[STFDecision]:
        """Recupera decisao especifica pelo numero do processo."""
        results = await self.search_decisions(query=processo, limit=1)
        return results[0] if results else None

    async def fetch_corte_aberta_votos(self) -> list[STFVoto]:
        """
        Baixa CSV de votacoes da Corte Aberta e parseia.

        Retorna lista de votos individuais de ministros.
        """
        csv_url = f"{CORTE_ABERTA_BASE}{CORTE_ABERTA_CSVS['votos']}"
        return await self._fetch_csv(csv_url, self._parse_voto_row)

    async def fetch_corte_aberta_repercussao(self) -> list[RepercussaoGeral]:
        """
        Baixa CSV de temas de repercussao geral da Corte Aberta.

        ~1.400 temas com teses firmadas.
        """
        csv_url = f"{CORTE_ABERTA_BASE}{CORTE_ABERTA_CSVS['repercussao']}"
        return await self._fetch_csv(csv_url, self._parse_repercussao_row)

    async def fetch_corte_aberta_decisoes(self) -> list[STFDecision]:
        """Baixa CSV de pautas/decisoes da Corte Aberta."""
        csv_url = f"{CORTE_ABERTA_BASE}{CORTE_ABERTA_CSVS['decisoes']}"
        return await self._fetch_csv(csv_url, self._parse_corte_aberta_decisao_row)

    async def get_voting_patterns(self, ministro: str) -> dict[str, Any]:
        """
        Analisa padroes de votacao de um ministro.

        Usa dados da Corte Aberta para mapear tendencias.
        """
        votos = await self.fetch_corte_aberta_votos()
        ministro_votos = [v for v in votos if ministro.lower() in v.ministro.lower()]

        if not ministro_votos:
            return {"ministro": ministro, "total_votos": 0}

        voto_types: dict[str, int] = {}
        for v in ministro_votos:
            vt = v.voto.lower().strip()
            voto_types[vt] = voto_types.get(vt, 0) + 1

        return {
            "ministro": ministro,
            "total_votos": len(ministro_votos),
            "distribuicao_votos": voto_types,
            "periodo": {
                "inicio": min(
                    (v.data_sessao for v in ministro_votos if v.data_sessao),
                    default=None,
                ),
                "fim": max(
                    (v.data_sessao for v in ministro_votos if v.data_sessao),
                    default=None,
                ),
            },
        }

    async def _fetch_csv(self, url: str, parser: Any) -> list:
        """Generic CSV downloader and parser."""
        try:
            async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()

            text = resp.text
            reader = csv.DictReader(io.StringIO(text), delimiter=";")
            results = []
            for row in reader:
                try:
                    parsed = parser(row)
                    if parsed:
                        results.append(parsed)
                except Exception:
                    continue

            logger.info("Corte Aberta CSV %s: %d registros", url.split("/")[-1], len(results))
            return results

        except Exception as e:
            logger.error("Erro baixando CSV %s: %s", url, e)
            return []

    def _parse_decision(self, raw: dict[str, Any]) -> STFDecision:
        return STFDecision(
            id=str(raw.get("id", raw.get("incidente", ""))),
            processo=raw.get("processo", raw.get("numero", "")),
            relator=raw.get("relator", raw.get("ministroRelator", "")),
            classe=raw.get("classe", raw.get("classeProcesso", "")),
            data_julgamento=self._parse_date(raw.get("dataJulgamento", raw.get("dataSessao"))),
            ementa=raw.get("ementa", ""),
            texto_inteiro=raw.get("inteiroTeor", raw.get("textoInteiro", "")),
            metadata={k: v for k, v in raw.items() if k not in {
                "id", "incidente", "processo", "numero", "relator",
                "ministroRelator", "classe", "classeProcesso",
                "dataJulgamento", "dataSessao", "ementa", "inteiroTeor", "textoInteiro",
            }},
        )

    @staticmethod
    def _parse_voto_row(row: dict[str, str]) -> STFVoto | None:
        ministro = row.get("Ministro", row.get("ministro", ""))
        if not ministro:
            return None
        return STFVoto(
            processo=row.get("Processo", row.get("processo", "")),
            ministro=ministro,
            voto=row.get("Voto", row.get("voto", "")),
            data_sessao=STFClient._parse_date(row.get("Data da Sessão", row.get("dataSessao"))),
            tipo_julgamento=row.get("Tipo de Julgamento", row.get("tipoJulgamento", "")),
            metadata=row,
        )

    @staticmethod
    def _parse_repercussao_row(row: dict[str, str]) -> RepercussaoGeral | None:
        tema_str = row.get("Tema", row.get("tema", row.get("numero", "")))
        try:
            tema_num = int(re.sub(r"\D", "", str(tema_str))[:6] or "0")
        except (ValueError, TypeError):
            return None
        if tema_num == 0:
            return None

        return RepercussaoGeral(
            tema_numero=tema_num,
            titulo=row.get("Título", row.get("titulo", "")),
            descricao=row.get("Descrição", row.get("descricao", "")),
            relator=row.get("Relator", row.get("relator", "")),
            leading_case=row.get("Leading Case", row.get("leadingCase", "")),
            situacao=row.get("Situação", row.get("situacao", "")),
            tese=row.get("Tese", row.get("tese", "")),
            data_reconhecimento=STFClient._parse_date(
                row.get("Data de Reconhecimento", row.get("dataReconhecimento"))
            ),
            metadata=row,
        )

    @staticmethod
    def _parse_corte_aberta_decisao_row(row: dict[str, str]) -> STFDecision | None:
        processo = row.get("Processo", row.get("processo", ""))
        if not processo:
            return None
        return STFDecision(
            id=row.get("Incidente", row.get("incidente", processo)),
            processo=processo,
            relator=row.get("Relator", row.get("relator")),
            classe=row.get("Classe", row.get("classe", "")),
            data_julgamento=STFClient._parse_date(
                row.get("Data da Sessão", row.get("dataSessao"))
            ),
            ementa=row.get("Ementa", row.get("ementa", "")),
            metadata=row,
        )

    @staticmethod
    def _parse_date(val: Any) -> datetime | None:
        if not val:
            return None
        if isinstance(val, datetime):
            return val
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(str(val).strip(), fmt)
            except (ValueError, TypeError):
                continue
        return None
