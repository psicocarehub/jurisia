"""
Memory Client using SuperMemory API for fact extraction and storage.
"""

from typing import Any, Optional

import httpx
from pydantic import BaseModel

from app.config import settings


class Mem0Fact(BaseModel):
    """Fato extraido/armazenado."""

    id: str
    content: str
    metadata: dict[str, Any] = {}
    user_id: Optional[str] = None


class Mem0Client:
    """
    Memory client backed by SuperMemory API.
    Facts are scoped by user_id for tenant isolation.
    """

    BASE_URL = "https://api.supermemory.ai/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_user_id: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or settings.SUPERMEMORY_API_KEY
        self.default_user_id = default_user_id

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        top_k: int = 10,
    ) -> list[Mem0Fact]:
        """Search facts by semantic similarity."""
        if not self.api_key:
            return []

        uid = user_id or self.default_user_id or "default"

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{self.BASE_URL}/search",
                    headers=self._headers(),
                    json={
                        "query": query,
                        "limit": top_k,
                        "filters": {"user_id": uid},
                    },
                )
                response.raise_for_status()
                data = response.json()

            results = []
            for item in data.get("results", []):
                results.append(Mem0Fact(
                    id=item.get("id", ""),
                    content=item.get("content", ""),
                    metadata=item.get("metadata", {}),
                    user_id=uid,
                ))
            return results
        except (httpx.HTTPError, KeyError):
            return []

    async def add(
        self,
        content: str,
        user_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add a fact to user's memory."""
        if not self.api_key:
            return ""

        uid = user_id or self.default_user_id or "default"

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{self.BASE_URL}/memories",
                    headers=self._headers(),
                    json={
                        "content": content,
                        "metadata": {
                            **(metadata or {}),
                            "user_id": uid,
                            "source": "jurisai",
                        },
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data.get("id", "")
        except (httpx.HTTPError, KeyError):
            return ""
