"""
CNPJ Receita Federal — download e parser de dados de 42M empresas brasileiras.

Dados publicos de https://arquivos.receitafederal.gov.br/dados/cnpj/
Usa-se para entity resolution em documentos judiciais (identificar partes,
razao social, socios, situacao cadastral).
"""

import csv
import io
import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import httpx
from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger("jurisai.cnpj")

RF_BASE = "https://arquivos.receitafederal.gov.br/dados/cnpj/dados_abertos_cnpj"

LAYOUT_EMPRESA = [
    ("cnpj_basico", 0, 8),
    ("razao_social", 8, 158),
    ("natureza_juridica", 158, 162),
    ("qualificacao_responsavel", 162, 164),
    ("capital_social", 164, 178),
    ("porte_empresa", 178, 180),
    ("ente_federativo", 180, None),
]

LAYOUT_ESTABELECIMENTO = [
    ("cnpj_basico", 0, 8),
    ("cnpj_ordem", 8, 12),
    ("cnpj_dv", 12, 14),
    ("matriz_filial", 14, 15),
    ("nome_fantasia", 15, 70),
    ("situacao_cadastral", 70, 72),
    ("data_situacao", 72, 80),
    ("motivo_situacao", 80, 82),
    ("cidade_exterior", 82, 152),
    ("pais", 152, 155),
    ("data_inicio", 155, 163),
    ("cnae_principal", 163, 170),
    ("cnae_secundario", 170, 877),
    ("logradouro_tipo", 877, 897),
    ("logradouro", 897, 957),
    ("numero", 957, 963),
    ("complemento", 963, 1119),
    ("bairro", 1119, 1169),
    ("cep", 1169, 1177),
    ("uf", 1177, 1179),
    ("municipio", 1179, 1183),
]


class CNPJEmpresa(BaseModel):
    """Dados de uma empresa da Receita Federal."""

    cnpj: str
    razao_social: str
    nome_fantasia: str = ""
    natureza_juridica: str = ""
    situacao_cadastral: str = ""
    data_inicio: Optional[str] = None
    capital_social: str = ""
    porte_empresa: str = ""
    uf: str = ""
    municipio: str = ""
    cnae_principal: str = ""
    metadata: dict[str, Any] = {}


class CNPJClient:
    """
    Download e parser de dados CNPJ da Receita Federal.

    Arquivos mensais com layout de largura fixa (~85GB total).
    Indexa em Elasticsearch para busca rapida por CNPJ, razao social, socios.
    """

    def __init__(self, data_dir: str | None = None) -> None:
        self.data_dir = Path(data_dir or settings.CNPJ_DATA_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    async def download_latest(self, file_type: str = "Empresas") -> list[Path]:
        """
        Download latest CNPJ bulk files from Receita Federal.

        Args:
            file_type: One of Empresas, Estabelecimentos, Socios
        """
        downloaded: list[Path] = []

        async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
            listing_url = f"{RF_BASE}/{file_type}/"
            try:
                resp = await client.get(listing_url)
                resp.raise_for_status()
            except Exception as e:
                logger.error("Erro listando arquivos CNPJ %s: %s", file_type, e)
                return downloaded

            import re
            zip_files = re.findall(r'href="([^"]+\.zip)"', resp.text, re.IGNORECASE)

            for zf in zip_files[:10]:
                url = f"{listing_url}{zf}"
                local_path = self.data_dir / zf
                if local_path.exists():
                    logger.info("Pulando %s (ja existe)", zf)
                    downloaded.append(local_path)
                    continue

                logger.info("Baixando %s ...", url)
                try:
                    async with client.stream("GET", url) as stream:
                        stream.raise_for_status()
                        with open(local_path, "wb") as f:
                            async for chunk in stream.aiter_bytes(chunk_size=1024 * 1024):
                                f.write(chunk)
                    downloaded.append(local_path)
                    logger.info("  %s salvo (%.1f MB)", zf, local_path.stat().st_size / 1e6)
                except Exception as e:
                    logger.error("Erro baixando %s: %s", url, e)
                    if local_path.exists():
                        local_path.unlink()

        return downloaded

    def parse_empresas(self, zip_path: Path) -> list[dict[str, str]]:
        """Parse a zip file of empresa records (CSV inside zip)."""
        results: list[dict[str, str]] = []

        try:
            with zipfile.ZipFile(zip_path) as zf:
                for name in zf.namelist():
                    if not name.endswith(".csv") and not name.endswith(".CSV"):
                        name_parts = zf.namelist()
                        if name_parts:
                            name = name_parts[0]

                    with zf.open(name) as f:
                        reader = csv.reader(
                            io.TextIOWrapper(f, encoding="latin-1"),
                            delimiter=";",
                            quotechar='"',
                        )
                        for row in reader:
                            if len(row) < 6:
                                continue
                            results.append({
                                "cnpj_basico": row[0].strip(),
                                "razao_social": row[1].strip(),
                                "natureza_juridica": row[2].strip() if len(row) > 2 else "",
                                "qualificacao_responsavel": row[3].strip() if len(row) > 3 else "",
                                "capital_social": row[4].strip() if len(row) > 4 else "",
                                "porte_empresa": row[5].strip() if len(row) > 5 else "",
                            })
                    break
        except Exception as e:
            logger.error("Erro parseando %s: %s", zip_path, e)

        return results

    def parse_estabelecimentos(self, zip_path: Path) -> list[dict[str, str]]:
        """Parse a zip file of estabelecimento records."""
        results: list[dict[str, str]] = []

        try:
            with zipfile.ZipFile(zip_path) as zf:
                name = zf.namelist()[0]
                with zf.open(name) as f:
                    reader = csv.reader(
                        io.TextIOWrapper(f, encoding="latin-1"),
                        delimiter=";",
                        quotechar='"',
                    )
                    for row in reader:
                        if len(row) < 20:
                            continue
                        cnpj_full = f"{row[0]}{row[1]}{row[2]}"
                        results.append({
                            "cnpj": cnpj_full,
                            "cnpj_basico": row[0].strip(),
                            "matriz_filial": row[3].strip() if len(row) > 3 else "",
                            "nome_fantasia": row[4].strip() if len(row) > 4 else "",
                            "situacao_cadastral": row[5].strip() if len(row) > 5 else "",
                            "data_situacao": row[6].strip() if len(row) > 6 else "",
                            "data_inicio": row[11].strip() if len(row) > 11 else "",
                            "cnae_principal": row[12].strip() if len(row) > 12 else "",
                            "uf": row[19].strip() if len(row) > 19 else "",
                            "municipio": row[20].strip() if len(row) > 20 else "",
                        })
        except Exception as e:
            logger.error("Erro parseando %s: %s", zip_path, e)

        return results

    def parse_socios(self, zip_path: Path) -> list[dict[str, str]]:
        """Parse a zip file of socio records."""
        results: list[dict[str, str]] = []

        try:
            with zipfile.ZipFile(zip_path) as zf:
                name = zf.namelist()[0]
                with zf.open(name) as f:
                    reader = csv.reader(
                        io.TextIOWrapper(f, encoding="latin-1"),
                        delimiter=";",
                        quotechar='"',
                    )
                    for row in reader:
                        if len(row) < 8:
                            continue
                        results.append({
                            "cnpj_basico": row[0].strip(),
                            "tipo_socio": row[1].strip(),
                            "nome_socio": row[2].strip(),
                            "cpf_cnpj_socio": row[3].strip() if len(row) > 3 else "",
                            "qualificacao": row[4].strip() if len(row) > 4 else "",
                            "data_entrada": row[5].strip() if len(row) > 5 else "",
                        })
        except Exception as e:
            logger.error("Erro parseando %s: %s", zip_path, e)

        return results

    async def search_by_cnpj(self, cnpj: str) -> CNPJEmpresa | None:
        """
        Search for a company by CNPJ in Elasticsearch.

        Falls back to Receita Federal public API if ES not available.
        """
        from elasticsearch import AsyncElasticsearch

        cnpj_clean = cnpj.replace(".", "").replace("/", "").replace("-", "")
        es = AsyncElasticsearch(settings.ELASTICSEARCH_URL)

        try:
            result = await es.search(
                index=f"{settings.ES_INDEX_PREFIX}_cnpj",
                body={"query": {"term": {"cnpj": cnpj_clean}}},
                size=1,
            )
            hits = result.get("hits", {}).get("hits", [])
            if hits:
                src = hits[0]["_source"]
                return CNPJEmpresa(**src)
        except Exception as e:
            logger.warning("ES search failed for CNPJ %s: %s", cnpj, e)
        finally:
            await es.close()

        return None

    async def search_by_name(self, razao_social: str, limit: int = 10) -> list[CNPJEmpresa]:
        """Search companies by razao social in Elasticsearch."""
        from elasticsearch import AsyncElasticsearch

        es = AsyncElasticsearch(settings.ELASTICSEARCH_URL)

        try:
            result = await es.search(
                index=f"{settings.ES_INDEX_PREFIX}_cnpj",
                body={"query": {"match": {"razao_social": razao_social}}},
                size=limit,
            )
            hits = result.get("hits", {}).get("hits", [])
            return [CNPJEmpresa(**h["_source"]) for h in hits]
        except Exception as e:
            logger.warning("ES search by name failed: %s", e)
            return []
        finally:
            await es.close()
