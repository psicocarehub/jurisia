"""
Mem0 Client — extração e armazenamento de fatos.

Cliente para Mem0 (fact extraction) com escopo por usuário.
"""

from typing import Any, Optional

from pydantic import BaseModel


class Mem0Fact(BaseModel):
    """Fato extraído/armazenado."""

    id: str
    content: str
    metadata: dict[str, Any] = {}
    user_id: Optional[str] = None


class Mem0Client:
    """
    Cliente para Mem0 — fact extraction e memória.

    Fatos são escopados por user_id para isolamento.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        default_user_id: Optional[str] = None,
    ) -> None:
        """
        Args:
            url: URL do serviço Mem0
            default_user_id: User ID padrão para operações
        """
        self.url = url
        self.default_user_id = default_user_id

    async def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        top_k: int = 10,
    ) -> list[Mem0Fact]:
        """
        Busca fatos por similaridade semântica.

        Args:
            query: Texto de busca
            user_id: Escopo do usuário (usa default se não informado)
            top_k: Máximo de fatos a retornar

        Returns:
            Lista de fatos relevantes (stub)
        """
        # Stub: implementação real via API Mem0
        _ = query, user_id or self.default_user_id, top_k
        return []

    async def add(
        self,
        content: str,
        user_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Adiciona fato à memória do usuário.

        Args:
            content: Conteúdo do fato
            user_id: Escopo do usuário
            metadata: Metadados opcionais

        Returns:
            ID do fato criado (stub)
        """
        # Stub: implementação real via API Mem0
        _ = content, user_id or self.default_user_id, metadata
        return ""
