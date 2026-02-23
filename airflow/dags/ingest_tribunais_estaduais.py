"""
DAG: Ingestao semanal de decisoes de tribunais estaduais.

Scraping de decisoes dos 5 maiores tribunais estaduais por volume:
TJSP > TJRJ > TJMG > TJRS > TJPR

Cada tribunal roda como task independente com rate limiting para
nao sobrecarregar os portais.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

logger = logging.getLogger("jurisai.dag.tribunais")

TRIBUNAIS = ["TJSP", "TJRJ", "TJMG", "TJRS", "TJPR"]

default_args = {
    "owner": "jurisai",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(hours=2),
    "execution_timeout": timedelta(hours=6),
}


def _scrape_tribunal(tribunal: str, **context):
    """Scrape recent decisions from a specific state tribunal."""
    from app.services.ingestion.tribunal_scraper import TribunalScraper
    from app.services.rag.indexer import IncrementalIndexer

    last_sync_key = f"tribunal_last_sync_{tribunal}"
    try:
        since = Variable.get(last_sync_key)
    except KeyError:
        since = (datetime.utcnow() - timedelta(days=14)).strftime("%d/%m/%Y")

    scraper = TribunalScraper(tribunal)
    indexer = IncrementalIndexer()

    decisions = asyncio.run(scraper.fetch_recent(since_date=since, max_pages=15))
    logger.info("[%s] %d decisoes encontradas desde %s", tribunal, len(decisions), since)

    indexed = 0
    for dec in decisions:
        text = dec.ementa or dec.decisao
        if not text or len(text.strip()) < 50:
            continue

        doc = {
            "doc_id": f"{tribunal.lower()}_{dec.processo}",
            "title": f"{tribunal} - {dec.processo} - {dec.classe}",
            "text": text,
            "source": f"tribunal_{tribunal.lower()}",
            "tribunal": tribunal,
            "area": dec.assunto[:100] if dec.assunto else "",
            "metadata": {
                "processo": dec.processo,
                "relator": dec.relator,
                "classe": dec.classe,
                "assunto": dec.assunto,
                "orgao_julgador": dec.orgao_julgador,
                "comarca": dec.comarca,
                "grau": dec.grau,
                "data_julgamento": dec.data_julgamento.isoformat() if dec.data_julgamento else None,
            },
        }
        try:
            asyncio.run(indexer.index_document(doc))
            indexed += 1
        except Exception as e:
            logger.warning("[%s] Erro indexando %s: %s", tribunal, dec.processo, e)

    Variable.set(last_sync_key, datetime.utcnow().strftime("%d/%m/%Y"))
    logger.info("[%s] %d/%d decisoes indexadas", tribunal, indexed, len(decisions))
    return indexed


with DAG(
    dag_id="ingest_tribunais_estaduais",
    default_args=default_args,
    description="Ingestao semanal de decisoes de tribunais estaduais (TJSP, TJRJ, TJMG, TJRS, TJPR)",
    schedule_interval="0 1 * * 0",
    start_date=datetime(2026, 2, 1),
    catchup=False,
    tags=["ingestion", "tribunais", "scraping"],
    max_active_runs=1,
) as dag:
    prev_task = None
    for tribunal in TRIBUNAIS:
        task = PythonOperator(
            task_id=f"scrape_{tribunal.lower()}",
            python_callable=_scrape_tribunal,
            op_kwargs={"tribunal": tribunal},
        )
        if prev_task:
            prev_task >> task
        prev_task = task
