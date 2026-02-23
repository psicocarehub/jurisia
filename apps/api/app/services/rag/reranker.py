"""
Reranker service using Voyage AI voyage-rerank-2.
"""

from typing import List

import httpx

from app.config import settings
from app.services.rag.retriever import RetrievedChunk


class RerankerService:
    """Reranks retrieved chunks using Voyage AI reranking API."""

    BASE_URL = "https://api.voyageai.com/v1"

    def __init__(self) -> None:
        self.model = settings.RERANK_MODEL
        self.api_key = settings.VOYAGE_API_KEY

    async def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = 10,
    ) -> List[RetrievedChunk]:
        """Rerank chunks by relevance to query using Voyage rerank API."""
        if not chunks:
            return []

        if not self.api_key:
            sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
            return sorted_chunks[:top_k]

        documents = [c.content for c in chunks]

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.BASE_URL}/rerank",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "query": query,
                        "documents": documents,
                        "top_k": top_k,
                    },
                )
                response.raise_for_status()
                data = response.json()

            reranked = []
            for result in data.get("data", []):
                idx = result["index"]
                chunk = chunks[idx]
                chunk.score = result["relevance_score"]
                reranked.append(chunk)

            return reranked

        except (httpx.HTTPError, KeyError):
            sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
            return sorted_chunks[:top_k]
