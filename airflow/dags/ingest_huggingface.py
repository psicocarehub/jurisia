"""
DAG de ingestão semanal de datasets HuggingFace jurídicos.

Fontes (celsowm collection):
  - Jurisprudência: STJ, STF, TJSP, TJRJ, TJMG, TJDF, TRF2, TRT1, TJPR
  - Súmulas: STF vinculantes, STF, STJ
  - Legislação: CF/88, CC, CPC, CDC, CLT, CP, CPP, CTN, ECA, Lei Licitações
  - Leis ordinárias federais (6.5k+)
  - Modelos de petições
  - Vademecum jurídico (14.6k+)

Executa semanalmente aos domingos às 5h UTC.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    "owner": "jurisai",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(hours=6),
}

HF_CATEGORIES = [
    "jurisprudencia",
    "sumulas",
    "legislacao",
    "leis_ordinarias",
    "peticoes",
    "vademecum",
]


def ingest_hf_category(category: str, **kwargs: Any) -> None:
    """Run ingest_huggingface.py for a specific category."""
    import asyncio
    import logging
    import sys
    from pathlib import Path

    logger = logging.getLogger(f"hf_ingest.{category}")
    root = Path(__file__).resolve().parents[2]
    scripts_dir = root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    if str(root / "apps" / "api") not in sys.path:
        sys.path.insert(0, str(root / "apps" / "api"))

    limit = int(Variable.get(f"hf_limit_{category}", default_var="5000"))

    from ingest_huggingface import main
    logger.info("Starting HF ingestion: category=%s, limit=%d", category, limit)
    asyncio.run(main(category, limit))
    logger.info("Completed HF ingestion: category=%s", category)


with DAG(
    dag_id="ingest_huggingface_weekly",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 5 * * 0",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_tasks=2,
    tags=["ingestion", "huggingface", "datasets"],
    doc_md="""
    ## HuggingFace Legal Datasets Weekly Ingestion

    Re-indexa datasets legais do HuggingFace (celsowm collection).
    Inclui jurisprudência, súmulas, legislação, petições, vademecum.
    Hash-based deduplication previne documentos duplicados.
    """,
) as dag:
    for category in HF_CATEGORIES:
        PythonOperator(
            task_id=f"ingest_{category}",
            python_callable=ingest_hf_category,
            op_kwargs={"category": category},
            pool="hf_pool",
        )
