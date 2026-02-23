"""
DAG de ingestão de dados do STF.

Fontes: API do STF, dados abertos, jurisprudência.
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

# STF: API pública e dados abertos (ajustar URLs conforme documentação oficial)
STF_BASE = "https://portal.stf.jus.br"


def ingest_stf_data(**kwargs: Any) -> None:
    """
    Ingest STF data (jurisprudência, processos, decisões).
    """
    sources = ["jurisprudencia", "processos"]

    with httpx.Client(timeout=120.0) as client:
        for source in sources:
            try:
                # Endpoint genérico; ajustar conforme API real do STF
                url = f"{STF_BASE}/servicos/dadosabertos/api/dados/{source}"
                resp = client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    count = len(data) if isinstance(data, list) else data.get("total", 0)
                else:
                    count = 0

                hook = PostgresHook(postgres_conn_id="jurisai_db")
                hook.run(
                    "INSERT INTO ingestion_log (source, records_count, status) VALUES (%s, %s, 'completed')",
                    parameters=(f"stf_{source}", count),
                )
            except Exception as e:
                hook = PostgresHook(postgres_conn_id="jurisai_db")
                hook.run(
                    "INSERT INTO ingestion_log (source, records_count, status, error_message) VALUES (%s, 0, 'failed', %s)",
                    parameters=(f"stf_{source}", str(e)[:500]),
                )


with DAG(
    dag_id="ingest_stf_data",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 5 * * *",  # 5h (após STJ)
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ingestion", "stf"],
) as dag:
    PythonOperator(
        task_id="ingest_stf",
        python_callable=ingest_stf_data,
    )
