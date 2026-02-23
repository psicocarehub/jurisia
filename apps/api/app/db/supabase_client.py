"""
Supabase REST client for database operations.
Uses PostgREST API with service_role_key — no DB password needed.
"""

from typing import Any, Optional

import httpx

from app.config import settings


class SupabaseDB:
    """Supabase PostgREST client for CRUD operations."""

    def __init__(self) -> None:
        self.base_url = f"{settings.SUPABASE_URL}/rest/v1"
        self.headers = {
            "apikey": settings.SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_ANON_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

    async def select(
        self,
        table: str,
        columns: str = "*",
        filters: Optional[dict[str, Any]] = None,
        single: bool = False,
    ) -> list[dict] | dict | None:
        params = {"select": columns}
        if filters:
            for key, value in filters.items():
                params[key] = f"eq.{value}"

        if single:
            self.headers["Accept"] = "application/vnd.pgrst.object+json"

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{self.base_url}/{table}",
                headers=self.headers,
                params=params,
            )
            resp.raise_for_status()
            return resp.json()

    async def insert(self, table: str, data: dict[str, Any]) -> dict:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{self.base_url}/{table}",
                headers=self.headers,
                json=data,
            )
            resp.raise_for_status()
            result = resp.json()
            return result[0] if isinstance(result, list) else result

    async def update(
        self, table: str, data: dict[str, Any], filters: dict[str, Any]
    ) -> dict:
        params = {}
        for key, value in filters.items():
            params[key] = f"eq.{value}"

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.patch(
                f"{self.base_url}/{table}",
                headers=self.headers,
                params=params,
                json=data,
            )
            resp.raise_for_status()
            result = resp.json()
            return result[0] if isinstance(result, list) else result

    async def delete(self, table: str, filters: dict[str, Any]) -> None:
        params = {}
        for key, value in filters.items():
            params[key] = f"eq.{value}"

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.delete(
                f"{self.base_url}/{table}",
                headers=self.headers,
                params=params,
            )
            resp.raise_for_status()

    async def rpc(self, function_name: str, params: dict[str, Any]) -> Any:
        """Call a Postgres function via RPC."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.base_url}/rpc/{function_name}",
                headers=self.headers,
                json=params,
            )
            resp.raise_for_status()
            return resp.json()


supabase_db = SupabaseDB()
