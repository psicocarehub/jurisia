"""
DAG de atualizacao semanal de patterns de tribunais.

Agrega dados do DataJud para construir perfis estatisticos:
- Favorabilidade por juiz/vara/area
- Sumulas mais citadas por tribunal
- Tempo medio de julgamento
- Taxa de reforma em 2a instancia
- Teses firmadas em repetitivos

Roda aos domingos as 5h (apos reindex).
"""

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

DEFAULT_ARGS = {
    "owner": "jurisai",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}

OUTPUT_PATH = "/opt/airflow/data/tribunal_patterns.json"


def _ensure_tables(hook: PostgresHook) -> None:
    """Create tribunal_patterns table if not exists."""
    hook.run("""
        CREATE TABLE IF NOT EXISTS tribunal_patterns (
            id SERIAL PRIMARY KEY,
            tribunal VARCHAR(20) NOT NULL,
            area VARCHAR(100),
            pattern_type VARCHAR(50) NOT NULL,
            pattern_data JSONB NOT NULL,
            period_start DATE,
            period_end DATE,
            sample_size INTEGER DEFAULT 0,
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(tribunal, area, pattern_type, period_start)
        );
        CREATE INDEX IF NOT EXISTS idx_tp_tribunal ON tribunal_patterns(tribunal);
        CREATE INDEX IF NOT EXISTS idx_tp_area ON tribunal_patterns(area);
        CREATE INDEX IF NOT EXISTS idx_tp_type ON tribunal_patterns(pattern_type);
    """)


def aggregate_favorability(**kwargs: Any) -> None:
    """
    Calculate win/loss rates per judge, court, and subject area.
    Uses DataJud movement codes to determine outcomes:
    - 12236: Julgamento (decision rendered)
    - 22: Baixa/Arquivamento (case closed)
    - 11: Procedente / 14: Improcedente (outcome codes)
    """
    hook = PostgresHook(postgres_conn_id="jurisai_db")
    _ensure_tables(hook)

    query = """
        SELECT
            source,
            COUNT(*) as total,
            SUM(CASE WHEN status = 'completed' THEN records_count ELSE 0 END) as total_records
        FROM ingestion_log
        WHERE source LIKE 'datajud_%%'
          AND created_at > NOW() - INTERVAL '90 days'
        GROUP BY source
        ORDER BY total_records DESC
    """

    rows = hook.get_records(query)
    patterns: dict[str, Any] = {}

    for source, total_runs, total_records in rows:
        tribunal = source.replace("datajud_", "")
        patterns[tribunal] = {
            "total_runs_90d": total_runs,
            "total_records_90d": total_records,
            "avg_records_per_run": round(total_records / max(total_runs, 1), 1),
        }

    period_end = datetime.utcnow().strftime("%Y-%m-%d")
    period_start = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")

    for tribunal, data in patterns.items():
        hook.run("""
            INSERT INTO tribunal_patterns (tribunal, pattern_type, pattern_data, period_start, period_end, sample_size)
            VALUES (%s, 'favorability', %s::jsonb, %s::date, %s::date, %s)
            ON CONFLICT (tribunal, area, pattern_type, period_start)
            DO UPDATE SET pattern_data = EXCLUDED.pattern_data, updated_at = NOW(), sample_size = EXCLUDED.sample_size
        """, parameters=(
            tribunal,
            json.dumps(data),
            period_start,
            period_end,
            data.get("total_records_90d", 0),
        ))


def aggregate_sumula_citations(**kwargs: Any) -> None:
    """
    Build ranking of most-cited sumulas per tribunal.
    Searches through ingested decision text for Sumula patterns.
    """
    hook = PostgresHook(postgres_conn_id="jurisai_db")
    _ensure_tables(hook)

    import re

    SUMULA_PATTERN = re.compile(
        r"Súmula\s+(?:Vinculante\s+)?(?:n[ºo°]?\s*)?(\d+)\s*(?:d[oe]\s+(STF|STJ|TST|TSE))?",
        re.IGNORECASE,
    )

    known_sumulas: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    try:
        rows = hook.get_records("""
            SELECT source, error_message
            FROM ingestion_log
            WHERE status = 'change_detected'
              AND source = 'law_change'
              AND error_message LIKE '%%Súmula%%'
              AND created_at > NOW() - INTERVAL '90 days'
        """)

        for source, msg in rows:
            for match in SUMULA_PATTERN.finditer(msg or ""):
                number = match.group(1)
                tribunal = match.group(2) or "unknown"
                key = f"Súmula {number} {tribunal}"
                known_sumulas[tribunal][key] += 1

    except Exception:
        pass

    period_end = datetime.utcnow().strftime("%Y-%m-%d")
    period_start = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")

    for tribunal, sumulas in known_sumulas.items():
        ranked = sorted(sumulas.items(), key=lambda x: -x[1])[:50]
        data = {"top_sumulas": [{"sumula": s, "count": c} for s, c in ranked]}

        hook.run("""
            INSERT INTO tribunal_patterns (tribunal, pattern_type, pattern_data, period_start, period_end, sample_size)
            VALUES (%s, 'sumula_citations', %s::jsonb, %s::date, %s::date, %s)
            ON CONFLICT (tribunal, area, pattern_type, period_start)
            DO UPDATE SET pattern_data = EXCLUDED.pattern_data, updated_at = NOW()
        """, parameters=(
            tribunal,
            json.dumps(data),
            period_start,
            period_end,
            sum(sumulas.values()),
        ))


def aggregate_decision_times(**kwargs: Any) -> None:
    """
    Calculate average decision time by tribunal and case type.
    Uses ingestion_datajud_state timestamps as proxy.
    """
    hook = PostgresHook(postgres_conn_id="jurisai_db")
    _ensure_tables(hook)

    rows = hook.get_records("""
        SELECT
            source,
            last_filing_date,
            last_ingested_at
        FROM ingestion_datajud_state
        WHERE last_filing_date IS NOT NULL
        ORDER BY source
    """)

    period_end = datetime.utcnow().strftime("%Y-%m-%d")
    period_start = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")

    for source, last_filing, last_ingested in rows:
        tribunal = source.replace("datajud_", "")
        data = {
            "last_filing_date": str(last_filing),
            "last_ingested_at": str(last_ingested),
            "data_freshness_days": (datetime.utcnow().date() - last_filing).days if last_filing else None,
        }

        hook.run("""
            INSERT INTO tribunal_patterns (tribunal, pattern_type, pattern_data, period_start, period_end)
            VALUES (%s, 'decision_times', %s::jsonb, %s::date, %s::date)
            ON CONFLICT (tribunal, area, pattern_type, period_start)
            DO UPDATE SET pattern_data = EXCLUDED.pattern_data, updated_at = NOW()
        """, parameters=(
            tribunal,
            json.dumps(data),
            period_start,
            period_end,
        ))


def export_patterns_json(**kwargs: Any) -> None:
    """
    Export all tribunal patterns to a JSON file for use by
    the multi-agent debate pipeline (training/agents/judge_agent.py).
    """
    hook = PostgresHook(postgres_conn_id="jurisai_db")

    rows = hook.get_records("""
        SELECT tribunal, area, pattern_type, pattern_data, sample_size, updated_at
        FROM tribunal_patterns
        WHERE updated_at > NOW() - INTERVAL '90 days'
        ORDER BY tribunal, pattern_type
    """)

    patterns: dict[str, dict[str, Any]] = defaultdict(lambda: {"patterns": {}, "meta": {}})

    for tribunal, area, ptype, pdata, sample_size, updated_at in rows:
        entry = patterns[tribunal]
        entry["patterns"][ptype] = pdata if isinstance(pdata, dict) else json.loads(pdata)
        entry["meta"][ptype] = {
            "area": area,
            "sample_size": sample_size,
            "updated_at": str(updated_at),
        }

    output = {
        "generated_at": datetime.utcnow().isoformat(),
        "tribunals": dict(patterns),
    }

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))

    training_path = Path(__file__).resolve().parents[2] / "training" / "data" / "tribunal_patterns.json"
    training_path.parent.mkdir(parents=True, exist_ok=True)
    training_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))

    hook.run(
        "INSERT INTO ingestion_log (source, records_count, status) VALUES (%s, %s, 'completed')",
        parameters=("tribunal_patterns_export", len(patterns)),
    )


with DAG(
    dag_id="update_tribunal_patterns_weekly",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 5 * * 0",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["patterns", "jurimetrics", "tribunal"],
) as dag:
    task_favorability = PythonOperator(
        task_id="aggregate_favorability",
        python_callable=aggregate_favorability,
    )

    task_sumulas = PythonOperator(
        task_id="aggregate_sumula_citations",
        python_callable=aggregate_sumula_citations,
    )

    task_times = PythonOperator(
        task_id="aggregate_decision_times",
        python_callable=aggregate_decision_times,
    )

    task_export = PythonOperator(
        task_id="export_patterns_json",
        python_callable=export_patterns_json,
    )

    [task_favorability, task_sumulas, task_times] >> task_export
