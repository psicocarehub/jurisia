"""Tests for petition endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.mark.asyncio
async def test_list_petitions_requires_auth():
    """GET /api/v1/petitions should require authentication."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/petitions")
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_create_petition_requires_auth():
    """POST /api/v1/petitions should require authentication."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/petitions",
            json={"title": "Test petition"},
        )
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_get_petition_requires_auth():
    """GET /api/v1/petitions/{id} should require authentication."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/petitions/fake-id")
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_export_pdf_requires_auth():
    """GET /api/v1/petitions/{id}/export/pdf should require authentication."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/petitions/fake-id/export/pdf")
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_export_docx_requires_auth():
    """GET /api/v1/petitions/{id}/export/docx should require authentication."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/petitions/fake-id/export/docx")
    assert resp.status_code in (401, 403)
