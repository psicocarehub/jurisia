"""
e-SAJ Client — scraping de decisões (TJSP e outros tribunais).

Cliente para extração de dados de decisões judiciais via e-SAJ
(Tribunais de Justiça). Atenção à LGPD e termos de uso.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class ESAJDecision(BaseModel):
    """Decisão recuperada via e-SAJ."""

    processo: str
    tribunal: str
    vara: str
    classe: str
    assunto: str
    data_decisao: Optional[datetime] = None
    texto: str = ""
    metadata: dict[str, Any] = {}


class ESAJClient:
    """
    Cliente para scraping de decisões via e-SAJ (TJSP etc).

    NOTA: Implementação stub. Scraping deve respeitar:
    - robots.txt e termos de uso do tribunal
    - LGPD: não indexar dados sigilosos
    - Rate limiting adequado
    """

    def __init__(self, tribunal: str = "TJSP") -> None:
        """
        Args:
            tribunal: Sigla do tribunal (TJSP, TJMG, etc.)
        """
        self.tribunal = tribunal

    async def search_decisions(
        self,
        query: str = "",
        classe: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 50,
    ) -> list[ESAJDecision]:
        """
        Busca decisões no e-SAJ do tribunal configurado.

        Args:
            query: Termo de busca (partes, assunto, etc.)
            classe: Código/classe processual
            date_from: Data inicial (YYYY-MM-DD)
            date_to: Data final (YYYY-MM-DD)
            limit: Máximo de resultados

        Returns:
            Lista de decisões (stub: retorno vazio)
        """
        # Stub: implementação real requer scraping com Playwright/Selenium
        _ = query, classe, date_from, date_to, limit
        return []

    async def get_decision_text(
        self,
        processo: str,
        movimento_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Obtém o texto integral de uma decisão específica.

        Args:
            processo: Número do processo (CNJ)
            movimento_id: ID do movimento/decisão, se houver

        Returns:
            Texto da decisão ou None (stub)
        """
        # Stub: implementação real via scraping da página de inteiro teor
        _ = processo, movimento_id
        return None
