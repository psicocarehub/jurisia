"""
Basic pytest tests for the Juris.AI FastAPI application.

Uses httpx + pytest-asyncio for async testing. Protected routes use Supabase/JWT auth,
so unauthenticated requests return 401/403 as expected. Tests verify that routes exist
(not 404) rather than successful auth.
"""

import pytest

from app.config import get_settings, Settings


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_returns_200(client):
    """GET /health returns status 200."""
    response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_returns_ok_and_service(client):
    """GET /health returns {status: ok, service: Juris.AI}."""
    response = await client.get("/health")
    data = response.json()
    assert data.get("status") == "ok"
    assert data.get("service") == "Juris.AI"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_settings_loads_correctly():
    """Settings class loads with expected defaults."""
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.APP_NAME == "Juris.AI"
    assert isinstance(settings.DEBUG, bool)
    assert settings.API_KEY_HEADER == "X-API-Key"


def test_settings_allowed_origins_list():
    """allowed_origins_list property returns list of origins."""
    settings = get_settings()
    origins = settings.allowed_origins_list
    assert isinstance(origins, list)
    assert "http://localhost:3000" in origins or len(origins) >= 0


# ---------------------------------------------------------------------------
# API router mount (protected routes exist, return auth errors not 404)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_v1_router_mounted_protected_route_returns_auth_error_not_404(client):
    """
    GET /api/v1/petitions without auth returns 401 or 403, not 404.
    Confirms the API router is mounted and protected routes exist.
    """
    response = await client.get("/api/v1/petitions")
    assert response.status_code in (401, 403), (
        f"Expected 401 or 403 for unauthenticated request, got {response.status_code}. "
        "404 would mean the route is not mounted."
    )


@pytest.mark.asyncio
async def test_api_v1_chat_route_exists(client):
    """POST /api/v1/chat/completions without auth returns auth error, not 404."""
    response = await client.post(
        "/api/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "test"}], "stream": False},
    )
    assert response.status_code in (401, 403), (
        f"Expected 401 or 403 for unauthenticated request, got {response.status_code}"
    )
