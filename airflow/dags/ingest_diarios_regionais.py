"""
DAG: Ingestao diaria de diarios oficiais estaduais e municipais.

Usa a API do Querido Diario (queridodiario.ok.org.br) para buscar
publicacoes de municipios e estados relevantes.  Captura portarias,
decretos municipais, editais e outros atos locais que nao aparecem
no DOU federal.

Executa diariamente as 7h UTC (apos o DAG do DOU federal).
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from typing import Any

import httpx
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

from _content_updates_helper import insert_content_update

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "jurisai",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(hours=3),
}

QUERIDO_DIARIO_API = "https://queridodiario.ok.org.br/api/gazettes"

# Major municipalities and state capitals with active legal publishing.
# territory_id follows IBGE codes.  Expand as needed.
TERRITORIES = {
    # Capitais / grandes cidades
    "3550308": {"name": "Sao Paulo", "uf": "SP"},
    "3304557": {"name": "Rio de Janeiro", "uf": "RJ"},
    "3106200": {"name": "Belo Horizonte", "uf": "MG"},
    "4106902": {"name": "Curitiba", "uf": "PR"},
    "4314902": {"name": "Porto Alegre", "uf": "RS"},
    "2927408": {"name": "Salvador", "uf": "BA"},
    "2304400": {"name": "Fortaleza", "uf": "CE"},
    "1302603": {"name": "Manaus", "uf": "AM"},
    "2611606": {"name": "Recife", "uf": "PE"},
    "5208707": {"name": "Goiania", "uf": "GO"},
    "1501402": {"name": "Belem", "uf": "PA"},
    "5002704": {"name": "Campo Grande", "uf": "MS"},
    "1100205": {"name": "Porto Velho", "uf": "RO"},
    "2507507": {"name": "Joao Pessoa", "uf": "PB"},
    "2408102": {"name": "Natal", "uf": "RN"},
    "2704302": {"name": "Maceio", "uf": "AL"},
    "2800308": {"name": "Aracaju", "uf": "SE"},
    "2211001": {"name": "Teresina", "uf": "PI"},
    "2111300": {"name": "Sao Luis", "uf": "MA"},
    "1721000": {"name": "Palmas", "uf": "TO"},
    "1600303": {"name": "Macapa", "uf": "AP"},
    "1400100": {"name": "Boa Vista", "uf": "RR"},
    "5103403": {"name": "Cuiaba", "uf": "MT"},
    "5300108": {"name": "Brasilia", "uf": "DF"},
    # Cidades com forte atividade aduaneira / portuária
    "3548500": {"name": "Santos", "uf": "SP"},
    "4209102": {"name": "Itajai", "uf": "SC"},
    "4113700": {"name": "Paranagua", "uf": "PR"},
    "3518800": {"name": "Guarulhos", "uf": "SP"},
    "3509502": {"name": "Campinas", "uf": "SP"},
    "4205407": {"name": "Florianopolis", "uf": "SC"},
}

KEYWORDS = [
    "PORTARIA", "DECRETO", "RESOLUÇÃO", "EDITAL",
    "INSTRUÇÃO NORMATIVA", "LEI", "LEI COMPLEMENTAR",
    "ATO", "PARECER",
]


def _get_last_date(hook: PostgresHook, territory_id: str) -> str | None:
    """Retrieve the last ingested date for a territory."""
    hook.run("""
        CREATE TABLE IF NOT EXISTS ingestion_diarios_state (
            territory_id VARCHAR(20) PRIMARY KEY,
            last_date DATE,
            last_ingested_at TIMESTAMPTZ
        )
    """)
    row = hook.get_first(
        "SELECT last_date FROM ingestion_diarios_state WHERE territory_id = %s",
        parameters=(territory_id,),
    )
    return row[0].isoformat() if row and row[0] else None


def _update_state(hook: PostgresHook, territory_id: str, last_date: str, count: int) -> None:
    hook.run("""
        INSERT INTO ingestion_diarios_state (territory_id, last_date, last_ingested_at)
        VALUES (%s, %s::date, NOW())
        ON CONFLICT (territory_id) DO UPDATE SET
            last_date = GREATEST(
                COALESCE(ingestion_diarios_state.last_date, '1970-01-01'::date),
                EXCLUDED.last_date
            ),
            last_ingested_at = NOW()
    """, parameters=(territory_id, last_date))
    hook.run(
        "INSERT INTO ingestion_log (source, records_count, status) VALUES (%s, %s, 'completed')",
        parameters=(f"diario_{territory_id}", count),
    )


def _classify(excerpt: str) -> tuple[str, str]:
    """Return (category, subcategory) from the gazette excerpt."""
    upper = excerpt.upper()[:300]
    if any(k in upper for k in ("LEI Nº", "LEI COMPLEMENTAR", "LEI MUNICIPAL")):
        return "legislacao", "lei_municipal"
    if "DECRETO" in upper:
        return "legislacao", "decreto_municipal"
    if "PORTARIA" in upper:
        return "normativo", "portaria"
    if "INSTRUÇÃO NORMATIVA" in upper or "INSTRUCAO NORMATIVA" in upper:
        return "normativo", "instrucao_normativa"
    if "EDITAL" in upper:
        return "normativo", "edital"
    if "RESOLUÇÃO" in upper or "RESOLUCAO" in upper:
        return "normativo", "resolucao"
    if "PARECER" in upper:
        return "parecer", "parecer_municipal"
    return "outro", "publicacao_oficial"


def ingest_territory(territory_id: str, **kwargs: Any) -> None:
    """Fetch recent gazette entries for a single territory."""
    info = TERRITORIES.get(territory_id, {"name": territory_id, "uf": "?"})
    hook = PostgresHook(postgres_conn_id="jurisai_db")

    last = _get_last_date(hook, territory_id) or (
        date.today() - timedelta(days=3)
    ).isoformat()
    today = date.today().isoformat()

    total = 0
    max_date = last

    with httpx.Client(timeout=60.0) as client:
        for keyword in KEYWORDS:
            try:
                resp = client.get(
                    QUERIDO_DIARIO_API,
                    params={
                        "territory_id": territory_id,
                        "querystring": keyword,
                        "published_since": last,
                        "published_until": today,
                        "size": 50,
                    },
                )
                if resp.status_code != 200:
                    logger.info("HTTP %d for %s", resp.status_code, keyword)
                    continue
                data = resp.json()
            except Exception as e:
                logger.warning("[diario_%s] Error fetching %s: %s", territory_id, keyword, e)
                continue

            for item in data.get("gazettes", []):
                pub_date = item.get("date", today)
                excerpts = item.get("excerpts", [])
                excerpt = excerpts[0] if excerpts else ""
                title = _extract_title(excerpt, keyword, info["name"])
                category, subcategory = _classify(excerpt)

                insert_content_update(
                    hook,
                    source="diario_regional",
                    category=category,
                    subcategory=subcategory,
                    title=title,
                    summary=excerpt[:2000] if excerpt else None,
                    publication_date=pub_date,
                    source_url=item.get("url", ""),
                    territory=info["uf"],
                    court_or_organ=f"Diario Oficial - {info['name']}",
                    areas=_infer_areas(excerpt),
                )
                total += 1
                if pub_date > max_date:
                    max_date = pub_date

            time.sleep(1.0)

    _update_state(hook, territory_id, max_date, total)


def _extract_title(text: str, keyword: str, city: str) -> str:
    import re
    match = re.search(
        rf"({keyword}\s*[Nn]?[ºo°]?\s*[\d./]+[^.;]*)",
        text,
        re.IGNORECASE,
    )
    base = match.group(1).strip()[:400] if match else keyword
    return f"[{city}] {base}"


def _infer_areas(text: str) -> list[str]:
    """Simple keyword-based area detection."""
    areas = []
    upper = text.upper()[:1000]
    area_keywords = {
        "tributario": ["TRIBUT", "ICMS", "ISS", "IPTU", "FISCAL"],
        "aduaneiro": ["ADUAN", "ALFÂNDEG", "ALFANDEG", "IMPORTA", "EXPORTA", "PORTO", "DESPACHO ADUANEIRO"],
        "ambiental": ["AMBIENT", "LICENÇA AMBIENTAL", "LICENCA AMBIENTAL", "IBAMA"],
        "trabalhista": ["TRABALH", "CLT", "EMPREGAD"],
        "urbanistico": ["URBAN", "ZONEAMENTO", "HABITE-SE", "ALVARÁ"],
        "saude": ["SAÚDE", "SAUDE", "SANITÁR", "SANITARI", "ANVISA"],
        "educacao": ["EDUCAÇ", "EDUCAC", "ESCOLA", "UNIVERSIDADE"],
        "licitacao": ["LICITAÇ", "LICITAC", "PREGÃO", "PREGAO", "TOMADA DE PREÇO"],
    }
    for area, kws in area_keywords.items():
        if any(k in upper for k in kws):
            areas.append(area)
    return areas or ["geral"]


with DAG(
    dag_id="ingest_diarios_regionais",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 7 * * *",
    start_date=datetime(2026, 2, 1),
    catchup=False,
    max_active_tasks=5,
    tags=["ingestion", "diarios", "municipios", "estados"],
) as dag:
    for tid in TERRITORIES:
        PythonOperator(
            task_id=f"ingest_{tid}",
            python_callable=ingest_territory,
            op_kwargs={"territory_id": tid},
        )
