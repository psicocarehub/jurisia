"""
Reranker service (stub). Production: Jina-ColBERT-v2 or bge-reranker-v2-m3.
"""

from typing import List

from app.services.rag.retriever import RetrievedChunk


class RerankerService:
    """Reranks retrieved chunks. Stub implementation returns top_k by score."""

    async def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = 10,
    ) -> List[RetrievedChunk]:
        """Rerank chunks by relevance to query. Stub: return top_k by score."""
        # Stub: sort by score and return top_k
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
        return sorted_chunks[:top_k]
