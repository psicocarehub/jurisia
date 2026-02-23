import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_user, get_tenant_id, get_db
from app.db.models import Case

router = APIRouter(prefix="/cases", tags=["cases"])


class CaseCreate(BaseModel):
    title: str
    cnj_number: Optional[str] = None
    description: Optional[str] = None
    area: Optional[str] = None
    client_name: Optional[str] = None
    client_document: Optional[str] = None
    opposing_party: Optional[str] = None
    court: Optional[str] = None
    judge_name: Optional[str] = None
    estimated_value: Optional[float] = None


class CaseUpdate(BaseModel):
    title: Optional[str] = None
    cnj_number: Optional[str] = None
    description: Optional[str] = None
    area: Optional[str] = None
    status: Optional[str] = None
    client_name: Optional[str] = None
    opposing_party: Optional[str] = None
    court: Optional[str] = None
    judge_name: Optional[str] = None


class CaseResponse(BaseModel):
    id: str
    title: str
    cnj_number: Optional[str] = None
    description: Optional[str] = None
    area: Optional[str] = None
    status: str = "active"
    client_name: Optional[str] = None
    opposing_party: Optional[str] = None
    court: Optional[str] = None
    judge_name: Optional[str] = None
    created_at: Optional[str] = None

    model_config = {"from_attributes": True}


class CaseListResponse(BaseModel):
    cases: list[CaseResponse]
    total: int


@router.get("", response_model=CaseListResponse)
async def list_cases(
    area: Optional[str] = None,
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    query = select(Case).where(Case.tenant_id == uuid.UUID(tenant_id))
    if area:
        query = query.where(Case.area == area)
    if status:
        query = query.where(Case.status == status)
    query = query.offset(skip).limit(limit)

    result = await db.execute(query)
    cases = result.scalars().all()

    return CaseListResponse(
        cases=[
            CaseResponse(
                id=str(c.id),
                title=c.title,
                cnj_number=c.cnj_number,
                description=c.description,
                area=c.area,
                status=c.status,
                client_name=c.client_name,
                opposing_party=c.opposing_party,
                court=c.court,
                judge_name=c.judge_name,
                created_at=str(c.created_at) if c.created_at else None,
            )
            for c in cases
        ],
        total=len(cases),
    )


@router.post("", response_model=CaseResponse, status_code=201)
async def create_case(
    case_data: CaseCreate,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    case = Case(
        tenant_id=uuid.UUID(tenant_id),
        created_by=uuid.UUID(user["id"]),
        **case_data.model_dump(exclude_none=True),
    )
    db.add(case)
    await db.flush()

    return CaseResponse(
        id=str(case.id),
        title=case.title,
        cnj_number=case.cnj_number,
        description=case.description,
        area=case.area,
        status=case.status,
        client_name=case.client_name,
        opposing_party=case.opposing_party,
        court=case.court,
        judge_name=case.judge_name,
    )


@router.get("/{case_id}", response_model=CaseResponse)
async def get_case(
    case_id: str,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Case).where(
            Case.id == uuid.UUID(case_id),
            Case.tenant_id == uuid.UUID(tenant_id),
        )
    )
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    return CaseResponse(
        id=str(case.id),
        title=case.title,
        cnj_number=case.cnj_number,
        description=case.description,
        area=case.area,
        status=case.status,
        client_name=case.client_name,
        opposing_party=case.opposing_party,
        court=case.court,
        judge_name=case.judge_name,
        created_at=str(case.created_at) if case.created_at else None,
    )


@router.patch("/{case_id}", response_model=CaseResponse)
async def update_case(
    case_id: str,
    case_data: CaseUpdate,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Case).where(
            Case.id == uuid.UUID(case_id),
            Case.tenant_id == uuid.UUID(tenant_id),
        )
    )
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    update_data = case_data.model_dump(exclude_none=True)
    for field, value in update_data.items():
        setattr(case, field, value)

    await db.flush()

    return CaseResponse(
        id=str(case.id),
        title=case.title,
        cnj_number=case.cnj_number,
        description=case.description,
        area=case.area,
        status=case.status,
        client_name=case.client_name,
        opposing_party=case.opposing_party,
        court=case.court,
        judge_name=case.judge_name,
        created_at=str(case.created_at) if case.created_at else None,
    )


@router.delete("/{case_id}", status_code=204)
async def delete_case(
    case_id: str,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Case).where(
            Case.id == uuid.UUID(case_id),
            Case.tenant_id == uuid.UUID(tenant_id),
        )
    )
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    await db.delete(case)
