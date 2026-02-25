"""Tests for health check and general API setup."""

import pytest


@pytest.mark.asyncio
async def test_health_check(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_openapi_docs(client):
    resp = await client.get("/docs")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_openapi_json(client):
    resp = await client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert schema["info"]["title"] == "Juris.AI API"
    assert "paths" in schema


@pytest.mark.asyncio
async def test_security_headers(client):
    resp = await client.get("/health")
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("X-Frame-Options") == "DENY"
    assert "Strict-Transport-Security" in resp.headers


@pytest.mark.asyncio
async def test_cors_preflight(client):
    resp = await client.options(
        "/api/v1/cases",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code == 200
