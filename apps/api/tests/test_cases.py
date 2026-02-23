"""Tests for case management endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.mark.asyncio
async def test_list_cases_requires_auth():
    """GET /api/v1/cases should require authentication."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/cases")
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_create_case_requires_auth():
    """POST /api/v1/cases should require authentication."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/cases",
            json={"title": "Test case", "description": "desc"},
        )
    assert resp.status_code in (401, 403)
