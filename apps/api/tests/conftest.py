"""Pytest fixtures for API tests."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.auth import create_access_token
from app.main import app

TEST_TENANT_ID = "00000000-0000-0000-0000-000000000001"
TEST_USER_ID = "00000000-0000-0000-0000-000000000002"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    """Unauthenticated async HTTP client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def auth_token() -> str:
    """Generate a valid JWT token for testing."""
    return create_access_token({
        "sub": TEST_USER_ID,
        "tenant_id": TEST_TENANT_ID,
        "role": "admin",
        "email": "test@jurisai.com.br",
    })


@pytest.fixture
def auth_headers(auth_token: str) -> dict[str, str]:
    """Authorization headers with valid JWT."""
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture
async def auth_client(auth_token: str):
    """Authenticated async HTTP client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Authorization": f"Bearer {auth_token}"},
    ) as ac:
        yield ac
