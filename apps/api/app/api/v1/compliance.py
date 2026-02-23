"""
Compliance API — verificacao de sancoes e impedimentos por CNPJ/CPF.

Consulta Portal da Transparencia (CEIS/CNEP/CEAF) e base CNPJ da Receita
Federal para due diligence e compliance em processos judiciais.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services.ingestion.cnpj import CNPJClient
from app.services.ingestion.transparencia import TransparenciaClient

router = APIRouter(prefix="/compliance", tags=["compliance"])


class SancaoResponse(BaseModel):
    tipo: str
    cpf_cnpj: str
    nome: str
    orgao_sancionador: str = ""
    data_inicio: str | None = None
    data_fim: str | None = None
    fundamentacao_legal: str = ""
    descricao: str = ""


class ComplianceCheckResult(BaseModel):
    identifier: str
    has_sanctions: bool
    total_sanctions: int
    sanctions: list[SancaoResponse]
    company_info: dict | None = None


class CNPJSearchResult(BaseModel):
    cnpj: str
    razao_social: str
    nome_fantasia: str = ""
    situacao_cadastral: str = ""
    uf: str = ""
    municipio: str = ""


@router.get("/check/{identifier}", response_model=ComplianceCheckResult)
async def check_sanctions(identifier: str):
    """
    Verifica sancoes de uma empresa (CNPJ) ou servidor (CPF).

    Consulta CEIS (empresas inidoneas), CNEP (Lei Anticorrupcao) e
    CEAF (servidores expulsos).
    """
    clean = identifier.replace(".", "").replace("/", "").replace("-", "")
    if len(clean) not in (11, 14):
        raise HTTPException(400, "Identificador invalido: esperado CPF (11 digitos) ou CNPJ (14 digitos)")

    client = TransparenciaClient()
    results = await client.check_all(clean)

    all_sanctions: list[SancaoResponse] = []
    for sanctions_list in results.values():
        for s in sanctions_list:
            all_sanctions.append(SancaoResponse(
                tipo=s.tipo,
                cpf_cnpj=s.cpf_cnpj,
                nome=s.nome,
                orgao_sancionador=s.orgao_sancionador,
                data_inicio=s.data_inicio,
                data_fim=s.data_fim,
                fundamentacao_legal=s.fundamentacao_legal,
                descricao=s.descricao,
            ))

    company_info = None
    if len(clean) == 14:
        cnpj_client = CNPJClient()
        empresa = await cnpj_client.search_by_cnpj(clean)
        if empresa:
            company_info = {
                "cnpj": empresa.cnpj,
                "razao_social": empresa.razao_social,
                "nome_fantasia": empresa.nome_fantasia,
                "situacao_cadastral": empresa.situacao_cadastral,
                "uf": empresa.uf,
            }

    return ComplianceCheckResult(
        identifier=clean,
        has_sanctions=len(all_sanctions) > 0,
        total_sanctions=len(all_sanctions),
        sanctions=all_sanctions,
        company_info=company_info,
    )


@router.get("/cnpj/{cnpj}", response_model=Optional[CNPJSearchResult])
async def search_cnpj(cnpj: str):
    """Busca dados de uma empresa pelo CNPJ na base da Receita Federal."""
    clean = cnpj.replace(".", "").replace("/", "").replace("-", "")
    if len(clean) != 14:
        raise HTTPException(400, "CNPJ invalido: esperado 14 digitos")

    client = CNPJClient()
    empresa = await client.search_by_cnpj(clean)
    if not empresa:
        raise HTTPException(404, "CNPJ nao encontrado")

    return CNPJSearchResult(
        cnpj=empresa.cnpj,
        razao_social=empresa.razao_social,
        nome_fantasia=empresa.nome_fantasia,
        situacao_cadastral=empresa.situacao_cadastral,
        uf=empresa.uf,
        municipio=empresa.municipio,
    )


@router.get("/cnpj/search", response_model=list[CNPJSearchResult])
async def search_cnpj_by_name(
    q: str = Query(..., min_length=3, description="Razao social ou nome fantasia"),
    limit: int = Query(10, ge=1, le=50),
):
    """Busca empresas por razao social na base CNPJ."""
    client = CNPJClient()
    empresas = await client.search_by_name(q, limit=limit)

    return [
        CNPJSearchResult(
            cnpj=e.cnpj,
            razao_social=e.razao_social,
            nome_fantasia=e.nome_fantasia,
            situacao_cadastral=e.situacao_cadastral,
            uf=e.uf,
            municipio=e.municipio,
        )
        for e in empresas
    ]
