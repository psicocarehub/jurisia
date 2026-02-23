"""
DAG de ingestão diária do DataJud (API CNJ).

Executa para cada tribunal configurado às 3h.
Rastreia última data de ingestão por tribunal.
"""

from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

# 15+ tribunais: STF, STJ, TST, TRFs, TRTs, TJs
TRIBUNAIS = [
    "stf",
    "stj",
    "tst",
    "tse",
    "stm",
    "trf1",
    "trf2",
    "trf3",
    "trf4",
    "trf5",
    "trf6",
    "trt1",
    "trt2",
    "trt3",
    "trt4",
    "trt5",
    "trt6",
    "trt7",
    "trt8",
    "trt9",
    "trt10",
    "trt11",
    "trt12",
    "trt13",
    "trt14",
    "trt15",
    "trt16",
    "trt17",
    "trt18",
    "trt19",
    "trt20",
    "trt21",
    "trt22",
    "trt23",
    "trt24",
    "tjsp",
    "tjrj",
    "tjmg",
    "tjrs",
    "tjpr",
    "tjba",
    "tjsc",
    "tjpe",
    "tjce",
    "tjgo",
    "tjpb",
    "tjpa",
    "tjrn",
    "tjal",
    "tjse",
    "tjac",
    "tjro",
    "tjmt",
    "tjms",
    "tjam",
    "tjap",
    "tjpi",
    "tjrr",
    "tjto",
    "tjdft",
]

DEFAULT_ARGS = {
    "owner": "jurisai",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}


def _ensure_state_table(hook: PostgresHook) -> None:
    """Create ingestion_state table if not exists (tracks last_filing_date per tribunal)."""
    hook.run("""
        CREATE TABLE IF NOT EXISTS ingestion_datajud_state (
            source VARCHAR(100) PRIMARY KEY,
            last_filing_date DATE,
            last_ingested_at TIMESTAMPTZ
        )
    """)


def _get_last_filing_date(hook: PostgresHook, tribunal: str) -> str | None:
    """Get last filing date for tribunal from state table."""
    source = f"datajud_{tribunal}"
    row = hook.get_first(
        "SELECT last_filing_date FROM ingestion_datajud_state WHERE source = %s",
        parameters=(source,),
    )
    return row[0].isoformat() if row and row[0] else None


def _update_state(hook: PostgresHook, tribunal: str, max_date: str | None, count: int) -> None:
    """Update ingestion state and log."""
    source = f"datajud_{tribunal}"
    if max_date:
        hook.run("""
            INSERT INTO ingestion_datajud_state (source, last_filing_date, last_ingested_at)
            VALUES (%s, %s::date, NOW())
            ON CONFLICT (source) DO UPDATE SET
                last_filing_date = GREATEST(
                    COALESCE(ingestion_datajud_state.last_filing_date, '1970-01-01'::date),
                    EXCLUDED.last_filing_date
                ),
                last_ingested_at = NOW()
        """, parameters=(source, max_date))
    hook.run(
        "INSERT INTO ingestion_log (source, records_count, status) VALUES (%s, %s, 'completed')",
        parameters=(source, count),
    )


def ingest_tribunal(tribunal: str, **kwargs: Any) -> None:
    """
    Ingest latest data from a specific tribunal.

    Uses DataJud API. Indexer is pluggable via env/import.
    """
    import httpx

    api_key = Variable.get("datajud_api_key", default_var="")
    if not api_key:
        raise ValueError("Variável datajud_api_key não configurada no Airflow")

    hook = PostgresHook(postgres_conn_id="jurisai_db")
    _ensure_state_table(hook)

    date_from = _get_last_filing_date(hook, tribunal) or "2024-01-01"
    url = f"https://api-publica.datajud.cnj.jus.br/api_publica_{tribunal}/_search"

    body = {
        "size": 1000,
        "from": 0,
        "query": {
            "bool": {
                "must": [{"range": {"dataAjuizamento": {"gte": date_from}}}]
            }
        },
        "sort": [{"dataAjuizamento": {"order": "desc"}}],
    }

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            url,
            headers={
                "Authorization": f"APIKey {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()

    hits = data.get("hits", {}).get("hits", [])
    processes = [h["_source"] for h in hits]

    max_date = None
    for p in processes:
        d = p.get("dataAjuizamento")
        if d:
            d_str = d[:10] if isinstance(d, str) else str(d)[:10]
            max_date = d_str if (max_date is None or d_str > max_date) else max_date

    # Indexação (plugável: importar Indexer de apps.api quando disponível)
    try:
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from apps.api.app.services.rag.indexer import Indexer
        indexer = Indexer()
        for p in processes:
            indexer.index_process(p)
    except ImportError:
        pass  # Indexer opcional; dados já foram buscados

    _update_state(hook, tribunal, max_date, len(processes))


with DAG(
    dag_id="ingest_datajud_daily",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 3 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ingestion", "datajud"],
) as dag:
    for tribunal in TRIBUNAIS:
        PythonOperator(
            task_id=f"ingest_{tribunal}",
            python_callable=ingest_tribunal,
            op_kwargs={"tribunal": tribunal},
        )
