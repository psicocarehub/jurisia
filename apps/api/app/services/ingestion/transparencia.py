"""
Portal da Transparencia Client — CEIS, CNEP e CEAF.

Acessa a API publica do Portal da Transparencia do Governo Federal:
- CEIS: Cadastro de Empresas Inidoneas e Suspensas
- CNEP: Cadastro Nacional de Empresas Punidas
- CEAF: Cadastro de Expulsoes da Administracao Federal

Licenca: CC0 (dominio publico). API key gratuita.
Docs: https://api.portaldatransparencia.gov.br/swagger-ui.html
"""

import logging
from datetime import datetime
from typing import Any, Optional

import httpx
from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger("jurisai.transparencia")

API_BASE = "https://api.portaldatransparencia.gov.br/api-de-dados"


class Sancao(BaseModel):
    """Sancao encontrada no Portal da Transparencia."""

    tipo: str  # CEIS, CNEP ou CEAF
    cpf_cnpj: str
    nome: str
    orgao_sancionador: str = ""
    data_inicio: Optional[str] = None
    data_fim: Optional[str] = None
    fundamentacao_legal: str = ""
    descricao: str = ""
    uf: str = ""
    fonte: str = ""
    metadata: dict[str, Any] = {}


class TransparenciaClient:
    """
    Client para API do Portal da Transparencia.

    Verifica sancoes de empresas (CEIS/CNEP) e servidores (CEAF).
    Util para due diligence e compliance em processos judiciais.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or settings.TRANSPARENCIA_API_KEY
        if not self.api_key:
            logger.warning("TRANSPARENCIA_API_KEY nao configurada")

    def _headers(self) -> dict[str, str]:
        return {
            "chave-api-dados": self.api_key,
            "Accept": "application/json",
        }

    async def check_ceis(self, cnpj: str) -> list[Sancao]:
        """
        Verifica se empresa consta no CEIS (Empresas Inidoneas e Suspensas).

        Args:
            cnpj: CNPJ da empresa (com ou sem formatacao)
        """
        cnpj_clean = cnpj.replace(".", "").replace("/", "").replace("-", "")
        return await self._search_endpoint("/ceis", {"cnpjSancionado": cnpj_clean}, "CEIS")

    async def check_cnep(self, cnpj: str) -> list[Sancao]:
        """
        Verifica se empresa consta no CNEP (Empresas Punidas — Lei Anticorrupcao).

        Args:
            cnpj: CNPJ da empresa
        """
        cnpj_clean = cnpj.replace(".", "").replace("/", "").replace("-", "")
        return await self._search_endpoint("/cnep", {"cnpjSancionado": cnpj_clean}, "CNEP")

    async def check_ceaf(self, cpf: str) -> list[Sancao]:
        """
        Verifica se servidor consta no CEAF (Expulsoes da Administracao Federal).

        Args:
            cpf: CPF do servidor
        """
        cpf_clean = cpf.replace(".", "").replace("-", "")
        return await self._search_endpoint("/ceaf", {"cpfSancionado": cpf_clean}, "CEAF")

    async def check_all(self, identifier: str) -> dict[str, list[Sancao]]:
        """
        Verifica sancoes em todos os cadastros (CEIS, CNEP, CEAF).

        Detecta automaticamente se e CNPJ ou CPF pelo tamanho.
        """
        clean = identifier.replace(".", "").replace("/", "").replace("-", "")
        result: dict[str, list[Sancao]] = {}

        if len(clean) == 14:
            result["ceis"] = await self.check_ceis(clean)
            result["cnep"] = await self.check_cnep(clean)
        elif len(clean) == 11:
            result["ceaf"] = await self.check_ceaf(clean)
        else:
            logger.warning("Identificador invalido: %s (esperado CPF ou CNPJ)", identifier)

        return result

    async def list_ceis(
        self,
        pagina: int = 1,
        uf: str | None = None,
    ) -> list[Sancao]:
        """Lista sancoes CEIS com paginacao."""
        params: dict[str, Any] = {"pagina": pagina}
        if uf:
            params["ufSancionado"] = uf
        return await self._search_endpoint("/ceis", params, "CEIS")

    async def _search_endpoint(
        self,
        endpoint: str,
        params: dict[str, Any],
        tipo: str,
    ) -> list[Sancao]:
        if not self.api_key:
            logger.warning("Transparencia API key nao configurada, retornando vazio")
            return []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{API_BASE}{endpoint}",
                    headers=self._headers(),
                    params=params,
                )
                resp.raise_for_status()
                data = resp.json()

            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = data.get("data", data.get("registros", [data]))
            else:
                records = []

            return [self._parse_sancao(r, tipo) for r in records]

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return []
            logger.error("Transparencia %s error %s: %s", endpoint, e.response.status_code, e.response.text[:300])
            return []
        except Exception as e:
            logger.error("Transparencia %s request failed: %s", endpoint, e)
            return []

    @staticmethod
    def _parse_sancao(raw: dict[str, Any], tipo: str) -> Sancao:
        cpf_cnpj = (
            raw.get("cnpjSancionado", "")
            or raw.get("cpfSancionado", "")
            or raw.get("numeroCPFCNPJ", "")
        )
        return Sancao(
            tipo=tipo,
            cpf_cnpj=cpf_cnpj,
            nome=raw.get("nomeSancionado", raw.get("nomeServidor", "")),
            orgao_sancionador=raw.get("orgaoSancionador", raw.get("orgaoLotacao", "")),
            data_inicio=raw.get("dataInicioSancao", raw.get("dataPublicacao", "")),
            data_fim=raw.get("dataFimSancao", ""),
            fundamentacao_legal=raw.get("fundamentacaoLegal", ""),
            descricao=raw.get("textoInformacaoAdicional", raw.get("tipoSancao", "")),
            uf=raw.get("ufSancionado", ""),
            fonte=f"Portal da Transparencia - {tipo}",
            metadata={k: v for k, v in raw.items() if k not in {
                "cnpjSancionado", "cpfSancionado", "nomeSancionado",
                "orgaoSancionador", "dataInicioSancao", "dataFimSancao",
                "fundamentacaoLegal", "textoInformacaoAdicional",
            }},
        )
