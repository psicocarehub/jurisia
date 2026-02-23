"""
DataJud API - CNJ Resolution 331/2020.
api-publica.datajud.cnj.jus.br
"""

from datetime import datetime
from typing import List, Optional

import httpx
from pydantic import BaseModel

from app.config import settings


class DataJudMovement(BaseModel):
    """Process movement from DataJud."""

    code: int
    name: str
    date: datetime


class DataJudProcess(BaseModel):
    """Process metadata from DataJud API."""

    cnj_number: str
    court: str
    class_name: str
    subject: str
    filing_date: datetime
    judge: Optional[str] = None
    movements: List[DataJudMovement] = []
    parties: List[dict] = []
    metadata: dict = {}


class DataJudClient:
    """Client for CNJ DataJud API."""

    BASE_URL = "https://api-publica.datajud.cnj.jus.br"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or settings.DATAJUD_API_KEY
        self.headers = {
            "Authorization": f"APIKey {self.api_key}",
            "Content-Type": "application/json",
        }

    async def search_processes(
        self,
        tribunal: str,
        query: str = "*",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        class_code: Optional[int] = None,
        size: int = 100,
        from_offset: int = 0,
    ) -> List[DataJudProcess]:
        """Search processes in DataJud. LGPD: excludes sigiloso parties."""
        url = f"{self.BASE_URL}/api_publica_{tribunal}/_search"

        must_clauses: List[dict] = []
        if query != "*":
            must_clauses.append({"query_string": {"query": query}})
        if class_code is not None:
            must_clauses.append({"term": {"classe.codigo": class_code}})
        if date_from or date_to:
            range_clause: dict = {"range": {"dataAjuizamento": {}}}
            if date_from:
                range_clause["range"]["dataAjuizamento"]["gte"] = date_from
            if date_to:
                range_clause["range"]["dataAjuizamento"]["lte"] = date_to
            must_clauses.append(range_clause)

        body = {
            "size": size,
            "from": from_offset,
            "query": {
                "bool": {
                    "must": must_clauses if must_clauses else [{"match_all": {}}]
                }
            },
            "sort": [{"dataAjuizamento": {"order": "desc"}}],
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url, headers=self.headers, json=body
            )
            response.raise_for_status()
            data = response.json()

        processes: List[DataJudProcess] = []
        for hit in data.get("hits", {}).get("hits", []):
            src = hit.get("_source", {})

            # LGPD: exclude sigiloso parties
            parties = [
                {"name": p.get("nome", ""), "type": p.get("tipo", "")}
                for p in src.get("partes", [])
                if not p.get("sigiloso", False)
            ]

            assuntos = src.get("assuntos", [{}])
            subject = assuntos[0].get("nome", "") if assuntos else ""

            processes.append(
                DataJudProcess(
                    cnj_number=src.get("numeroProcesso", ""),
                    court=tribunal.upper(),
                    class_name=src.get("classe", {}).get("nome", ""),
                    subject=subject,
                    filing_date=src.get("dataAjuizamento", datetime.now()),
                    judge=src.get("orgaoJulgador", {}).get("nomeJuiz"),
                    movements=[
                        DataJudMovement(
                            code=m.get("codigo", 0),
                            name=m.get("nome", ""),
                            date=m.get("dataHora", datetime.now()),
                        )
                        for m in src.get("movimentos", [])
                    ],
                    parties=parties,
                    metadata=src,
                )
            )

        return processes
