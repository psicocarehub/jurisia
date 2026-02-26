from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.config import settings

security = HTTPBearer()
optional_security = HTTPBearer(auto_error=False)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

GUEST_USER = {
    "id": "guest",
    "tenant_id": "__public__",
    "role": "guest",
    "email": "",
}


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    try:
        payload = jwt.decode(credentials.credentials, settings.SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        tenant_id = payload.get("tenant_id")
        if not user_id or not tenant_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return {
            "id": user_id,
            "tenant_id": tenant_id,
            "role": payload.get("role", "lawyer"),
            "email": payload.get("email", ""),
        }
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(optional_security),
) -> dict:
    """Returns authenticated user if token present, otherwise guest user."""
    if credentials is None:
        return GUEST_USER
    try:
        payload = jwt.decode(credentials.credentials, settings.SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        tenant_id = payload.get("tenant_id")
        if not user_id or not tenant_id:
            return GUEST_USER
        return {
            "id": user_id,
            "tenant_id": tenant_id,
            "role": payload.get("role", "lawyer"),
            "email": payload.get("email", ""),
        }
    except JWTError:
        return GUEST_USER


def get_tenant_id(user: dict = Depends(get_current_user)) -> str:
    return user["tenant_id"]


def get_tenant_id_optional(user: dict = Depends(get_current_user_optional)) -> str:
    return user["tenant_id"]
