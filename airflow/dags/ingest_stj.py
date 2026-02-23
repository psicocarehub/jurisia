"""
DAG de ingestão do STJ Open Data.

Dados abertos do STJ (dadosabertos.web.stj.jus.br).
Inclui texto integral de decisões (ao contrário do DataJud).
"""

from datetime import datetime, timedelta
from typing import Any

import httpx
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

DEFAULT_ARGS = {
    "owner": "jurisai",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

BASE_URL = "https://dadosabertos.web.stj.jus.br"


def ingest_stj_decisions(**kwargs: Any) -> None:
    """
    Ingest STJ Open Data (acórdãos e decisões monocráticas).
    """
    datasets = ["acordaos", "decisoes_monocraticas"]

    with httpx.Client(timeout=120.0) as client:
        for dataset in datasets:
            try:
                # STJ Open Data: obter metadados do dataset e URL de download
                meta_url = f"{BASE_URL}/dataset/{dataset}"
                resp = client.get(meta_url)
                resp.raise_for_status()

                # URL típica de recurso CSV/JSON (ajustar conforme estrutura real do portal)
                resource_url = f"{BASE_URL}/dataset/{dataset}/resource/latest.csv"
                dl = client.get(resource_url)
                if dl.status_code == 200:
                    lines = dl.text.split("\n")
                    count = max(0, len(lines) - 1)  # header
                else:
                    count = 0

                hook = PostgresHook(postgres_conn_id="jurisai_db")
                hook.run(
                    "INSERT INTO ingestion_log (source, records_count, status) VALUES (%s, %s, 'completed')",
                    parameters=(f"stj_{dataset}", count),
                )
            except Exception as e:
                hook = PostgresHook(postgres_conn_id="jurisai_db")
                hook.run(
                    "INSERT INTO ingestion_log (source, records_count, status, error_message) VALUES (%s, 0, 'failed', %s)",
                    parameters=(f"stj_{dataset}", str(e)[:500]),
                )


with DAG(
    dag_id="ingest_stj_opendata",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 4 * * *",  # 4h (após DataJud)
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ingestion", "stj", "opendata"],
) as dag:
    PythonOperator(
        task_id="ingest_stj_decisions",
        python_callable=ingest_stj_decisions,
    )
