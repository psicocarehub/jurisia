"""
Per-tenant rate limiter using SlowAPI.

Extracts tenant_id from the JWT token for rate-limiting isolation.
Falls back to remote IP address if no valid token is present.
"""

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings


def _tenant_key_func(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        try:
            from jose import jwt as jose_jwt
            payload = jose_jwt.decode(
                auth[7:], settings.SECRET_KEY, algorithms=["HS256"],
                options={"verify_exp": False},
            )
            tid = payload.get("tenant_id")
            if tid:
                return f"tenant:{tid}"
        except Exception:
            pass
    return get_remote_address(request)


limiter = Limiter(key_func=_tenant_key_func, default_limits=["120/minute"])
