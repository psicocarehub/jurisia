import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import create_access_token
from app.core.security import hash_password, verify_password
from app.db.models import Tenant, User
from app.db.session import get_db

router = APIRouter(prefix="/admin", tags=["admin"])


class TenantCreate(BaseModel):
    name: str
    slug: str
    plan: str = "starter"


class UserRegister(BaseModel):
    email: str
    name: str
    password: str
    tenant_slug: str
    role: str = "lawyer"
    oab_number: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str
    tenant_slug: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    tenant_id: str
    user_id: str


@router.post("/tenants", status_code=201)
async def create_tenant(
    data: TenantCreate,
    db: AsyncSession = Depends(get_db),
):
    existing = await db.execute(select(Tenant).where(Tenant.slug == data.slug))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Tenant slug already exists")

    tenant = Tenant(name=data.name, slug=data.slug, plan=data.plan)
    db.add(tenant)
    await db.flush()

    return {"id": str(tenant.id), "name": tenant.name, "slug": tenant.slug}


@router.post("/register", response_model=TokenResponse)
async def register_user(
    data: UserRegister,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Tenant).where(Tenant.slug == data.tenant_slug))
    tenant = result.scalar_one_or_none()
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    existing = await db.execute(
        select(User).where(User.tenant_id == tenant.id, User.email == data.email)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="User already exists")

    user = User(
        tenant_id=tenant.id,
        email=data.email,
        name=data.name,
        hashed_password=hash_password(data.password),
        role=data.role,
        oab_number=data.oab_number,
    )
    db.add(user)
    await db.flush()

    token = create_access_token(
        data={
            "sub": str(user.id),
            "tenant_id": str(tenant.id),
            "role": user.role,
            "email": user.email,
        }
    )

    return TokenResponse(
        access_token=token, tenant_id=str(tenant.id), user_id=str(user.id)
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    data: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Tenant).where(Tenant.slug == data.tenant_slug))
    tenant = result.scalar_one_or_none()
    if not tenant:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    result = await db.execute(
        select(User).where(User.tenant_id == tenant.id, User.email == data.email)
    )
    user = result.scalar_one_or_none()
    if not user or not user.hashed_password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(
        data={
            "sub": str(user.id),
            "tenant_id": str(tenant.id),
            "role": user.role,
            "email": user.email,
        }
    )

    return TokenResponse(
        access_token=token, tenant_id=str(tenant.id), user_id=str(user.id)
    )
