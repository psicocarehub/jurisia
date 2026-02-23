from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.auth import create_access_token
from app.core.security import hash_password, verify_password
from app.db.supabase_client import supabase_db

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
async def create_tenant(data: TenantCreate):
    existing = await supabase_db.select("tenants", filters={"slug": data.slug})
    if existing:
        raise HTTPException(status_code=409, detail="Tenant slug already exists")

    tenant = await supabase_db.insert("tenants", {
        "name": data.name,
        "slug": data.slug,
        "plan": data.plan,
    })
    return {"id": tenant["id"], "name": tenant["name"], "slug": tenant["slug"]}


@router.post("/register", response_model=TokenResponse)
async def register_user(data: UserRegister):
    tenants = await supabase_db.select("tenants", filters={"slug": data.tenant_slug})
    if not tenants:
        raise HTTPException(status_code=404, detail="Tenant not found")
    tenant = tenants[0]

    existing = await supabase_db.select(
        "users", filters={"tenant_id": tenant["id"], "email": data.email}
    )
    if existing:
        raise HTTPException(status_code=409, detail="User already exists")

    user = await supabase_db.insert("users", {
        "tenant_id": tenant["id"],
        "email": data.email,
        "name": data.name,
        "hashed_password": hash_password(data.password),
        "role": data.role,
        "oab_number": data.oab_number,
    })

    token = create_access_token(data={
        "sub": user["id"],
        "tenant_id": tenant["id"],
        "role": user["role"],
        "email": user["email"],
    })

    return TokenResponse(
        access_token=token, tenant_id=tenant["id"], user_id=user["id"]
    )


@router.post("/login", response_model=TokenResponse)
async def login(data: LoginRequest):
    tenants = await supabase_db.select("tenants", filters={"slug": data.tenant_slug})
    if not tenants:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    tenant = tenants[0]

    users = await supabase_db.select(
        "users", filters={"tenant_id": tenant["id"], "email": data.email}
    )
    if not users:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    user = users[0]

    if not user.get("hashed_password"):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(data={
        "sub": user["id"],
        "tenant_id": tenant["id"],
        "role": user["role"],
        "email": user["email"],
    })

    return TokenResponse(
        access_token=token, tenant_id=tenant["id"], user_id=user["id"]
    )
