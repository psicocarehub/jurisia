"""Seed the database with initial data for development."""

import asyncio
import uuid

from app.core.security import hash_password
from app.db.models import Tenant, User
from app.db.session import AsyncSessionLocal


async def seed():
    async with AsyncSessionLocal() as session:
        tenant = Tenant(
            id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
            name="Escritório Demo",
            slug="demo",
            plan="professional",
        )
        session.add(tenant)

        admin = User(
            tenant_id=tenant.id,
            email="admin@demo.juris.ai",
            name="Administrador Demo",
            role="admin",
            hashed_password=hash_password("admin123"),
            is_active=True,
            ai_consent_given=True,
        )
        session.add(admin)

        lawyer = User(
            tenant_id=tenant.id,
            email="advogado@demo.juris.ai",
            name="Dr. João Silva",
            role="lawyer",
            oab_number="SP123456",
            hashed_password=hash_password("lawyer123"),
            is_active=True,
            ai_consent_given=True,
        )
        session.add(lawyer)

        await session.commit()
        print("Seed data created successfully.")


if __name__ == "__main__":
    asyncio.run(seed())
