"""
DAG: Ingestao mensal de decisoes do CARF.

Busca novas decisoes tributarias do CARF (Conselho Administrativo de Recursos
Fiscais), classifica por materia e indexa no pipeline RAG.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

from _content_updates_helper import insert_content_update

logger = logging.getLogger("jurisai.dag.carf")

default_args = {
    "owner": "jurisai",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(hours=1),
}


def _fetch_carf(**context):
    from app.services.ingestion.carf import CARFClient

    try:
        since = Variable.get("carf_last_sync")
    except KeyError:
        since = (datetime.utcnow() - timedelta(days=60)).strftime("%Y-%m-%d")

    client = CARFClient()
    decisions = asyncio.run(client.fetch_recent(since_date=since, max_pages=30))
    logger.info("CARF: %d decisoes encontradas desde %s", len(decisions), since)

    context["ti"].xcom_push(key="decisions_count", value=len(decisions))
    context["ti"].xcom_push(key="decisions", value=[d.model_dump(mode="json") for d in decisions])
    return len(decisions)


def _index_carf(**context):
    from app.services.rag.indexer import IncrementalIndexer

    decisions_data = context["ti"].xcom_pull(task_ids="fetch_carf", key="decisions") or []
    indexer = IncrementalIndexer()

    indexed = 0
    for dec_data in decisions_data:
        text = dec_data.get("ementa", "") + "\n\n" + dec_data.get("decisao", "")
        if len(text.strip()) < 50:
            continue

        doc = {
            "doc_id": f"carf_{dec_data.get('id', dec_data.get('numero_acordao', ''))}",
            "title": f"CARF - Acordao {dec_data.get('numero_acordao', '')} - {dec_data.get('materia', '')}",
            "text": text,
            "source": "carf",
            "tribunal": "CARF",
            "area": f"tributario_{dec_data.get('materia', '').lower()}",
            "metadata": {
                "numero_processo": dec_data.get("numero_processo"),
                "numero_acordao": dec_data.get("numero_acordao"),
                "turma": dec_data.get("turma"),
                "relator": dec_data.get("relator"),
                "materia": dec_data.get("materia"),
                "resultado": dec_data.get("resultado"),
                "data_sessao": dec_data.get("data_sessao"),
            },
        }
        try:
            asyncio.run(indexer.index_document(doc))
            indexed += 1
        except Exception as e:
            logger.warning("Erro indexando CARF %s: %s", dec_data.get("id"), e)

    hook = PostgresHook(postgres_conn_id="jurisai_db")
    for dec_data in decisions_data:
        insert_content_update(
            hook,
            source="carf",
            category="jurisprudencia",
            subcategory="acordao_tributario",
            title=f"CARF - Acordao {dec_data.get('numero_acordao', '')} - {dec_data.get('materia', '')}"[:500],
            summary=(dec_data.get("ementa") or "")[:2000],
            court_or_organ="CARF",
            territory="federal",
            publication_date=dec_data.get("data_sessao"),
            areas=["tributario"],
            metadata={
                "turma": dec_data.get("turma"),
                "relator": dec_data.get("relator"),
                "resultado": dec_data.get("resultado"),
            },
        )

    Variable.set("carf_last_sync", datetime.utcnow().strftime("%Y-%m-%d"))
    logger.info("CARF: %d/%d decisoes indexadas", indexed, len(decisions_data))
    return indexed


with DAG(
    dag_id="ingest_carf",
    default_args=default_args,
    description="Ingestao mensal de decisoes tributarias do CARF",
    schedule_interval="0 3 15 * *",
    start_date=datetime(2026, 2, 1),
    catchup=False,
    tags=["ingestion", "carf", "tributario"],
    max_active_runs=1,
) as dag:
    fetch = PythonOperator(task_id="fetch_carf", python_callable=_fetch_carf)
    index = PythonOperator(task_id="index_carf", python_callable=_index_carf)
    fetch >> index
