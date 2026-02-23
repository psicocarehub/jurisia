"""
STF API Client — decisões do Supremo Tribunal Federal.

Cliente para consulta à API de jurisprudência do STF.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class STFDecision(BaseModel):
    """Decisão do STF."""

    id: str
    processo: str
    relator: Optional[str] = None
    classe: str
    data_julgamento: Optional[datetime] = None
    ementa: str = ""
    texto_inteiro: str = ""
    metadata: dict[str, Any] = {}


class STFClient:
    """
    Cliente para API de decisões do STF.

    Consulta jurisprudência do Supremo Tribunal Federal.
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
        Busca decisões na API do STF.

        Args:
            query: Termo de busca (ementa, processo, etc.)
            classe: Classe processual
            relator: Nome do ministro relator
            date_from: Data inicial (YYYY-MM-DD)
            date_to: Data final (YYYY-MM-DD)
            limit: Máximo de resultados
            offset: Offset para paginação

        Returns:
            Lista de decisões (stub: retorno vazio)
        """
        # Stub: implementação real via API do portal STF
        _ = query, classe, relator, date_from, date_to, limit, offset
        return []

    async def get_decision(self, processo: str) -> Optional[STFDecision]:
        """
        Recupera decisão específica pelo número do processo.

        Args:
            processo: Número do processo (formato STF)

        Returns:
            Decisão ou None se não encontrada (stub)
        """
        # Stub: implementação real via API
        _ = processo
        return None
