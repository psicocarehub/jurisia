"""
LangGraph Checkpointer — PostgresSaver.

Configuração do checkpointer PostgreSQL para persistência
de estado nos grafos LangGraph.
"""

from typing import Optional

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config import settings


_checkpointer: Optional[AsyncPostgresSaver] = None


async def get_checkpointer() -> AsyncPostgresSaver:
    """
    Factory para obter o checkpointer PostgreSQL do LangGraph.

    Usa DATABASE_URL. O checkpointer deve ser inicializado
    (setup) antes do primeiro uso.

    Returns:
        Instância configurada de AsyncPostgresSaver
    """
    global _checkpointer

    if _checkpointer is None:
        # Remover asyncpg driver para conexão síncrona do PostgresSaver
        conn_string = settings.DATABASE_URL.replace(
            "postgresql+asyncpg://", "postgresql://"
        )
        _checkpointer = AsyncPostgresSaver.from_conn_string(conn_string)
        await _checkpointer.setup()

    return _checkpointer
