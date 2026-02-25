"""
DAG: Ingestao diaria de normativos da Receita Federal do Brasil.

Captura Instrucoes Normativas (IN RFB), Solucoes de Consulta COSIT/COANA,
Atos Declaratorios e outros normativos tributarios e aduaneiros.

Fontes:
  - Portal de normas da RFB (normas.receita.fazenda.gov.br)
  - DOU secao 1 (via Querido Diario filtrando por RFB)

Executa diariamente as 8h UTC.
"""

from __future__ import annotations

import logging
import re
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
}

NORMAS_RFB_URL = "https://normas.receita.fazenda.gov.br/sijut2consulta/consulta.action"
QUERIDO_DIARIO_API = "https://queridodiario.ok.org.br/api/gazettes"

RFB_KEYWORDS = [
    "INSTRUÇÃO NORMATIVA RFB",
    "SOLUÇÃO DE CONSULTA COSIT",
    "SOLUÇÃO DE CONSULTA COANA",
    "ATO DECLARATÓRIO EXECUTIVO",
    "PORTARIA RFB",
    "PORTARIA COANA",
    "SOLUÇÃO DE DIVERGÊNCIA",
]


def _get_state(hook: PostgresHook) -> str | None:
    hook.run("""
        CREATE TABLE IF NOT EXISTS ingestion_receita_state (
            source VARCHAR(100) PRIMARY KEY,
            last_date DATE,
            last_ingested_at TIMESTAMPTZ
        )
    """)
    row = hook.get_first(
        "SELECT last_date FROM ingestion_receita_state WHERE source = 'rfb_normas'",
    )
    return row[0].isoformat() if row and row[0] else None


def _update_state(hook: PostgresHook, last_date: str, count: int) -> None:
    hook.run("""
        INSERT INTO ingestion_receita_state (source, last_date, last_ingested_at)
        VALUES ('rfb_normas', %s::date, NOW())
        ON CONFLICT (source) DO UPDATE SET
            last_date = GREATEST(
                COALESCE(ingestion_receita_state.last_date, '1970-01-01'::date),
                EXCLUDED.last_date
            ),
            last_ingested_at = NOW()
    """, parameters=(last_date,))
    hook.run(
        "INSERT INTO ingestion_log (source, records_count, status) VALUES ('receita_federal', %s, 'completed')",
        parameters=(count,),
    )


def _classify_rfb(keyword: str) -> tuple[str, str, list[str]]:
    """Return (category, subcategory, areas) for a given RFB keyword."""
    kw = keyword.upper()
    if "INSTRUÇÃO NORMATIVA" in kw:
        return "normativo", "instrucao_normativa_rfb", ["tributario"]
    if "COSIT" in kw:
        return "parecer", "solucao_consulta_cosit", ["tributario"]
    if "COANA" in kw:
        return "normativo", "solucao_consulta_coana", ["aduaneiro"]
    if "ATO DECLARATÓRIO" in kw:
        return "normativo", "ato_declaratorio", ["tributario"]
    if "PORTARIA COANA" in kw:
        return "normativo", "portaria_coana", ["aduaneiro"]
    if "PORTARIA RFB" in kw:
        return "normativo", "portaria_rfb", ["tributario"]
    if "DIVERGÊNCIA" in kw or "DIVERGENCIA" in kw:
        return "parecer", "solucao_divergencia", ["tributario"]
    return "normativo", "normativo_rfb", ["tributario"]


def ingest_rfb_via_dou(**kwargs: Any) -> None:
    """Fetch RFB normatives from DOU via Querido Diario API."""
    hook = PostgresHook(postgres_conn_id="jurisai_db")
    last = _get_state(hook) or (date.today() - timedelta(days=7)).isoformat()
    today = date.today().isoformat()
    total = 0
    max_date = last

    with httpx.Client(timeout=60.0) as client:
        for keyword in RFB_KEYWORDS:
            try:
                resp = client.get(
                    QUERIDO_DIARIO_API,
                    params={
                        "territory_id": "5300108",
                        "querystring": keyword,
                        "published_since": last,
                        "published_until": today,
                        "size": 100,
                    },
                )
                if resp.status_code != 200:
                    logger.info("HTTP %d for %s", resp.status_code, keyword)
                    continue
                data = resp.json()
            except Exception as e:
                logger.warning("[receita] Error fetching %s: %s", keyword, e)
                continue

            category, subcategory, areas = _classify_rfb(keyword)

            for item in data.get("gazettes", []):
                pub_date = item.get("date", today)
                excerpts = item.get("excerpts", [])
                excerpt = excerpts[0] if excerpts else ""

                title_match = re.search(
                    rf"({keyword}\s*[Nn]?[ºo°]?\s*[\d./]+[^.;]*)",
                    excerpt,
                    re.IGNORECASE,
                )
                title = title_match.group(1).strip()[:500] if title_match else keyword

                insert_content_update(
                    hook,
                    source="receita_federal",
                    category=category,
                    subcategory=subcategory,
                    title=title,
                    summary=excerpt[:2000] if excerpt else None,
                    publication_date=pub_date,
                    source_url=item.get("url", ""),
                    territory="federal",
                    court_or_organ="Receita Federal do Brasil",
                    areas=areas,
                )
                total += 1
                if pub_date > max_date:
                    max_date = pub_date

            time.sleep(1.0)

    _update_state(hook, max_date, total)


def ingest_rfb_portal(**kwargs: Any) -> None:
    """Scrape the RFB normas portal for recent normatives."""
    hook = PostgresHook(postgres_conn_id="jurisai_db")
    total = 0

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.get(
                NORMAS_RFB_URL,
                params={
                    "tipoAto": "",
                    "dataPublicacaoInicial": (date.today() - timedelta(days=7)).strftime("%d/%m/%Y"),
                    "dataPublicacaoFinal": date.today().strftime("%d/%m/%Y"),
                },
            )
            if resp.status_code != 200:
                return

        links = re.findall(
            r'href="(link\.action\?idAto=\d+[^"]*)"[^>]*>([^<]+)</a>',
            resp.text,
        )

        for href, title in links[:100]:
            full_url = f"https://normas.receita.fazenda.gov.br/sijut2consulta/{href}"
            category, subcategory, areas = _classify_rfb(title)
            insert_content_update(
                hook,
                source="receita_federal_portal",
                category=category,
                subcategory=subcategory,
                title=title.strip()[:500],
                source_url=full_url,
                territory="federal",
                court_or_organ="Receita Federal do Brasil",
                areas=areas,
            )
            total += 1

    except Exception as e:
        hook.run(
            "INSERT INTO ingestion_log (source, records_count, status, error_message) VALUES ('receita_federal_portal', 0, 'failed', %s)",
            parameters=(str(e)[:500],),
        )
        return

    hook.run(
        "INSERT INTO ingestion_log (source, records_count, status) VALUES ('receita_federal_portal', %s, 'completed')",
        parameters=(total,),
    )


with DAG(
    dag_id="ingest_receita_federal",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 8 * * *",
    start_date=datetime(2026, 2, 1),
    catchup=False,
    tags=["ingestion", "receita_federal", "tributario", "aduaneiro"],
) as dag:
    task_dou = PythonOperator(
        task_id="ingest_rfb_via_dou",
        python_callable=ingest_rfb_via_dou,
    )
    task_portal = PythonOperator(
        task_id="ingest_rfb_portal",
        python_callable=ingest_rfb_portal,
    )
