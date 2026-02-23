"""
Embedding service using Voyage AI for Brazilian legal documents.
Supports voyage-3-large (general) and voyage-law-2 (legal-specific).
"""

from typing import List

import httpx

from app.config import settings


class EmbeddingService:
    """Voyage AI embeddings for legal text (voyage-3-large / voyage-law-2)."""

    BASE_URL = "https://api.voyageai.com/v1"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.EMBEDDING_MODEL
        self.api_key = settings.VOYAGE_API_KEY

    async def embed_query(self, text: str) -> List[float]:
        """Embed a query string. Uses input_type 'query' for better retrieval."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.BASE_URL}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "input": [text],
                    "input_type": "query",
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents. Uses input_type 'document'."""
        if not texts:
            return []

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.BASE_URL}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "input": texts,
                    "input_type": "document",
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            return [d["embedding"] for d in data["data"]]
