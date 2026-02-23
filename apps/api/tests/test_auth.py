"""
Testes de autenticação.
"""

import pytest
from jose import jwt

from app.core.auth import (
    ALGORITHM,
    create_access_token,
    get_current_user,
)
from app.config import settings


def test_create_access_token():
    """Testa criação de token JWT."""
    data = {"sub": "user-123", "tenant_id": "tenant-456", "email": "test@example.com"}
    token = create_access_token(data)
    assert isinstance(token, str)
    assert len(token) > 0


def test_create_access_token_decodable():
    """Testa que o token pode ser decodificado corretamente."""
    data = {"sub": "user-123", "tenant_id": "tenant-456"}
    token = create_access_token(data)
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
    assert payload["sub"] == "user-123"
    assert payload["tenant_id"] == "tenant-456"
    assert "exp" in payload


def test_invalid_token_raises():
    """Testa que token inválido resulta em exceção ao decodificar."""
    from jose import JWTError

    with pytest.raises(JWTError):
        jwt.decode("invalid-token", settings.SECRET_KEY, algorithms=[ALGORITHM])


def test_token_with_wrong_secret_raises():
    """Testa que token com secret errado falha na decodificação."""
    from jose import JWTError

    data = {"sub": "user-123", "tenant_id": "tenant-456"}
    token = create_access_token(data)
    with pytest.raises(JWTError):
        jwt.decode(token, "wrong-secret-key", algorithms=[ALGORITHM])
