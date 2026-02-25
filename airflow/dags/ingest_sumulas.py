"""
DAG: Ingestao semanal de sumulas e teses repetitivas.

Captura novas sumulas e teses fixadas em recursos repetitivos/repercussao geral:
  - STJ: Sumulas e temas de recursos repetitivos
  - STF: Sumulas vinculantes e temas de repercussao geral
  - TST: Sumulas e OJs (Orientacoes Jurisprudenciais)

Executa semanalmente (sexta-feira, 4h UTC).
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

QUERIDO_DIARIO_API = "https://queridodiario.ok.org.br/api/gazettes"

SUMULA_KEYWORDS = [
    "SÚMULA VINCULANTE",
    "SÚMULA STJ",
    "SÚMULA STF",
    "TEMA DE REPERCUSSÃO GERAL",
    "RECURSO REPETITIVO",
    "TESE FIXADA",
    "SÚMULA TST",
    "ORIENTAÇÃO JURISPRUDENCIAL",
]

STJ_REPETITIVOS_URL = "https://processo.stj.jus.br/repetitivos/temas_repetitivos"
STF_RG_URL = "https://portal.stf.jus.br/jurisprudenciaRepercussao"


def ingest_sumulas_dou(**kwargs: Any) -> None:
    """Search DOU for new sumulas and binding precedents."""
    hook = PostgresHook(postgres_conn_id="jurisai_db")
    since = (date.today() - timedelta(days=7)).isoformat()
    today = date.today().isoformat()
    total = 0

    with httpx.Client(timeout=60.0) as client:
        for keyword in SUMULA_KEYWORDS:
            try:
                resp = client.get(
                    QUERIDO_DIARIO_API,
                    params={
                        "territory_id": "5300108",
                        "querystring": keyword,
                        "published_since": since,
                        "published_until": today,
                        "size": 50,
                    },
                )
                if resp.status_code != 200:
                    logger.info("HTTP %d for %s", resp.status_code, keyword)
                    continue
                data = resp.json()
            except Exception as e:
                logger.warning("[sumulas] Error fetching %s: %s", keyword, e)
                continue

            for item in data.get("gazettes", []):
                pub_date = item.get("date", today)
                excerpts = item.get("excerpts", [])
                excerpt = excerpts[0] if excerpts else ""

                title_match = re.search(
                    rf"({keyword}\s*[Nn]?[ºo°]?\s*[\d./-]*[^.;]*)",
                    excerpt,
                    re.IGNORECASE,
                )
                title = title_match.group(1).strip()[:500] if title_match else keyword

                court = "STJ"
                if "STF" in keyword or "REPERCUSSÃO" in keyword or "REPERCUSSAO" in keyword:
                    court = "STF"
                elif "TST" in keyword or "ORIENTAÇÃO JURISPRUDENCIAL" in keyword:
                    court = "TST"

                insert_content_update(
                    hook,
                    source=f"sumulas_{court.lower()}",
                    category="sumula",
                    subcategory=_classify_sumula(keyword),
                    title=title,
                    summary=excerpt[:2000] if excerpt else None,
                    publication_date=pub_date,
                    source_url=item.get("url", ""),
                    territory="federal",
                    court_or_organ=court,
                    relevance_score=0.9,
                )
                total += 1

            time.sleep(1.0)

    hook.run(
        "INSERT INTO ingestion_log (source, records_count, status) VALUES ('sumulas', %s, 'completed')",
        parameters=(total,),
    )


def ingest_stj_repetitivos(**kwargs: Any) -> None:
    """Scrape STJ for recently decided repetitive themes."""
    hook = PostgresHook(postgres_conn_id="jurisai_db")
    total = 0

    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            resp = client.get(STJ_REPETITIVOS_URL)
            if resp.status_code != 200:
                return

        themes = re.findall(
            r'Tema\s+(\d+)\s*[:\-–]\s*([^<\n]+)',
            resp.text,
        )

        for num, desc in themes[:100]:
            insert_content_update(
                hook,
                source="stj_repetitivos",
                category="sumula",
                subcategory="tema_repetitivo",
                title=f"STJ - Tema {num}: {desc.strip()}"[:500],
                summary=desc.strip()[:2000],
                territory="federal",
                court_or_organ="STJ",
                relevance_score=0.9,
                source_url=f"{STJ_REPETITIVOS_URL}/{num}",
            )
            total += 1

    except Exception as e:
        hook.run(
            "INSERT INTO ingestion_log (source, records_count, status, error_message) VALUES ('stj_repetitivos', 0, 'failed', %s)",
            parameters=(str(e)[:500],),
        )
        return

    hook.run(
        "INSERT INTO ingestion_log (source, records_count, status) VALUES ('stj_repetitivos', %s, 'completed')",
        parameters=(total,),
    )


def ingest_stf_repercussao(**kwargs: Any) -> None:
    """Scrape STF for recently decided general repercussion themes."""
    hook = PostgresHook(postgres_conn_id="jurisai_db")
    total = 0

    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            resp = client.get(STF_RG_URL)
            if resp.status_code != 200:
                return

        themes = re.findall(
            r'Tema\s+(\d+)\s*[:\-–]\s*([^<\n]+)',
            resp.text,
        )

        for num, desc in themes[:100]:
            insert_content_update(
                hook,
                source="stf_repercussao",
                category="sumula",
                subcategory="tema_repercussao_geral",
                title=f"STF - Tema RG {num}: {desc.strip()}"[:500],
                summary=desc.strip()[:2000],
                territory="federal",
                court_or_organ="STF",
                relevance_score=0.9,
                source_url=f"{STF_RG_URL}/{num}",
            )
            total += 1

    except Exception as e:
        hook.run(
            "INSERT INTO ingestion_log (source, records_count, status, error_message) VALUES ('stf_repercussao', 0, 'failed', %s)",
            parameters=(str(e)[:500],),
        )
        return

    hook.run(
        "INSERT INTO ingestion_log (source, records_count, status) VALUES ('stf_repercussao', %s, 'completed')",
        parameters=(total,),
    )


def _classify_sumula(keyword: str) -> str:
    kw = keyword.upper()
    if "VINCULANTE" in kw:
        return "sumula_vinculante"
    if "REPERCUSSÃO" in kw or "REPERCUSSAO" in kw:
        return "tema_repercussao_geral"
    if "REPETITIVO" in kw:
        return "tema_repetitivo"
    if "ORIENTAÇÃO" in kw or "ORIENTACAO" in kw:
        return "orientacao_jurisprudencial"
    if "TST" in kw:
        return "sumula_tst"
    if "STJ" in kw:
        return "sumula_stj"
    if "STF" in kw:
        return "sumula_stf"
    return "sumula"


with DAG(
    dag_id="ingest_sumulas_teses",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 4 * * 5",
    start_date=datetime(2026, 2, 1),
    catchup=False,
    tags=["ingestion", "sumulas", "repetitivos", "repercussao_geral"],
) as dag:
    t1 = PythonOperator(
        task_id="ingest_sumulas_dou",
        python_callable=ingest_sumulas_dou,
    )
    t2 = PythonOperator(
        task_id="ingest_stj_repetitivos",
        python_callable=ingest_stj_repetitivos,
    )
    t3 = PythonOperator(
        task_id="ingest_stf_repercussao",
        python_callable=ingest_stf_repercussao,
    )
