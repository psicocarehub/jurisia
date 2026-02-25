"""
DAG: Ingestao diaria de decisoes via JUIT API.

Sincroniza novas decisoes dos tribunais prioritarios (STJ, STF, TJSP, TJRJ,
TJMG, TJRS, TJPR), chunka e indexa no pipeline RAG (Elasticsearch + Qdrant).
"""

import asyncio
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

from _content_updates_helper import insert_content_update

logger = logging.getLogger("jurisai.dag.juit")

TRIBUNAIS = ["STJ", "STF", "TJSP", "TJRJ", "TJMG", "TJRS", "TJPR"]

default_args = {
    "owner": "jurisai",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}


def _fetch_and_index(tribunal: str, **context):
    """Fetch recent decisions for a tribunal and index them."""
    from app.services.ingestion.juit import JUITClient
    from app.services.rag.indexer import IncrementalIndexer

    last_sync_key = f"juit_last_sync_{tribunal}"
    try:
        since = Variable.get(last_sync_key)
    except KeyError:
        since = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

    client = JUITClient()
    indexer = IncrementalIndexer()

    decisions = asyncio.run(client.fetch_recent(tribunal, since_date=since, max_pages=20))
    logger.info("[%s] %d decisoes encontradas desde %s", tribunal, len(decisions), since)

    indexed = 0
    for dec in decisions:
        if not dec.inteiro_teor and not dec.ementa:
            continue
        text = dec.inteiro_teor or dec.ementa
        doc = {
            "doc_id": f"juit_{dec.id}",
            "title": f"{dec.tribunal} - {dec.processo} - {dec.classe}",
            "text": text,
            "source": "juit",
            "tribunal": dec.tribunal,
            "area": dec.area,
            "metadata": {
                "processo": dec.processo,
                "relator": dec.relator,
                "classe": dec.classe,
                "assunto": dec.assunto,
                "data_julgamento": dec.data_julgamento.isoformat() if dec.data_julgamento else None,
            },
        }
        try:
            asyncio.run(indexer.index_document(doc))
            indexed += 1
        except Exception as e:
            logger.warning("Erro indexando %s: %s", dec.id, e)

    hook = PostgresHook(postgres_conn_id="jurisai_db")
    for dec in decisions:
        if not dec.inteiro_teor and not dec.ementa:
            continue
        insert_content_update(
            hook,
            source="juit",
            category="jurisprudencia",
            subcategory=dec.classe or "decisao",
            title=f"{dec.tribunal} - {dec.processo} - {dec.classe}"[:500],
            summary=(dec.ementa or "")[:2000],
            court_or_organ=dec.tribunal,
            territory="federal",
            publication_date=dec.data_julgamento.isoformat()[:10] if dec.data_julgamento else None,
            areas=[dec.area] if dec.area else [],
        )

    Variable.set(last_sync_key, datetime.utcnow().strftime("%Y-%m-%d"))
    logger.info("[%s] %d/%d decisoes indexadas", tribunal, indexed, len(decisions))
    return indexed


with DAG(
    dag_id="ingest_juit",
    default_args=default_args,
    description="Ingestao diaria de decisoes via JUIT API",
    schedule_interval="0 6 * * *",
    start_date=datetime(2026, 2, 1),
    catchup=False,
    tags=["ingestion", "juit", "rag"],
    max_active_runs=1,
) as dag:
    for tribunal in TRIBUNAIS:
        PythonOperator(
            task_id=f"fetch_{tribunal.lower()}",
            python_callable=_fetch_and_index,
            op_kwargs={"tribunal": tribunal},
        )
