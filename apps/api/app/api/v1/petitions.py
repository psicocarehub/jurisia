from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.dependencies import get_current_user, get_tenant_id

router = APIRouter(prefix="/petitions", tags=["petitions"])


class PetitionCreate(BaseModel):
    title: str
    case_id: Optional[str] = None
    petition_type: Optional[str] = None
    content: Optional[str] = None


class PetitionResponse(BaseModel):
    id: str
    title: str
    petition_type: Optional[str] = None
    status: str = "draft"
    ai_generated: bool = True
    ai_label: str = "Conteúdo gerado com auxílio de IA — CNJ Res. 615/2025"


@router.get("")
async def list_petitions(
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    return {"petitions": [], "total": 0}


@router.post("", response_model=PetitionResponse, status_code=201)
async def create_petition(
    petition: PetitionCreate,
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
):
    # TODO: Implement petition creation with AI generation
    return PetitionResponse(
        id="placeholder",
        title=petition.title,
        petition_type=petition.petition_type,
    )
