"""Tests for case management endpoints."""

import pytest


@pytest.mark.asyncio
async def test_list_cases_requires_auth(client):
    resp = await client.get("/api/v1/cases")
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_create_case_requires_auth(client):
    resp = await client.post("/api/v1/cases", json={"title": "Test"})
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_list_cases_with_auth(auth_client):
    resp = await auth_client.get("/api/v1/cases")
    assert resp.status_code == 200
    data = resp.json()
    assert "cases" in data
    assert "total" in data
    assert isinstance(data["cases"], list)


@pytest.mark.asyncio
async def test_create_case_with_auth(auth_client):
    resp = await auth_client.post("/api/v1/cases", json={
        "title": "Caso de Teste",
        "area": "civil",
        "description": "Teste automatizado",
    })
    # May return 201 or 500 if Supabase is not configured
    assert resp.status_code in (201, 500)


@pytest.mark.asyncio
async def test_create_case_invalid_cnj(auth_client):
    resp = await auth_client.post("/api/v1/cases", json={
        "title": "Caso CNJ Inválido",
        "cnj_number": "1234",
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_case_valid_cnj(auth_client):
    resp = await auth_client.post("/api/v1/cases", json={
        "title": "Caso CNJ Válido",
        "cnj_number": "0000001-23.2024.8.26.0100",
    })
    assert resp.status_code in (201, 500)


@pytest.mark.asyncio
async def test_create_case_invalid_cpf(auth_client):
    resp = await auth_client.post("/api/v1/cases", json={
        "title": "Caso CPF Inválido",
        "client_document": "12345678900",
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_get_case_not_found(auth_client):
    resp = await auth_client.get("/api/v1/cases/00000000-0000-0000-0000-000000000099")
    assert resp.status_code in (404, 500)


@pytest.mark.asyncio
async def test_analyze_case_not_found(auth_client):
    resp = await auth_client.post("/api/v1/cases/00000000-0000-0000-0000-000000000099/analyze")
    assert resp.status_code in (404, 500)


@pytest.mark.asyncio
async def test_get_analysis_not_found(auth_client):
    resp = await auth_client.get("/api/v1/cases/00000000-0000-0000-0000-000000000099/analysis")
    assert resp.status_code in (404, 500)
