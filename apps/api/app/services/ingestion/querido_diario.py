"""
Querido Diário API Client — diários oficiais (queridodiario.ok.org.br).

Integração com a API do Querido Diário (Open Knowledge Brasil)
para busca em diários oficiais municipais, estaduais e federais.
"""

from datetime import date
from typing import Optional

from pydantic import BaseModel


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

    BASE_URL = "https://queridodiario.ok.org.br"

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = base_url or self.BASE_URL

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

        Args:
            query: Termo de busca
            territory_id: ID do território (município/estado)
            date_from: Data inicial
            date_to: Data final
            limit: Máximo de resultados
            offset: Offset para paginação

        Returns:
            Lista de itens de diário (stub: retorno vazio)
        """
        # Stub: implementação real via API REST
        # GET /api/gazettes?q=...&territory_id=...
        _ = query, territory_id, date_from, date_to, limit, offset
        return []
