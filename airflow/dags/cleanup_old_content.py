"""
DAG: Cleanup old content updates.

Removes content_updates older than 6 months with low relevance scores.
High-relevance items are preserved indefinitely.
Also cleans old ingestion_log entries (12+ months).

Runs monthly on the 1st at 02:00 UTC.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

logger = logging.getLogger("jurisai.dag.cleanup")

DEFAULT_ARGS = {
    "owner": "jurisai",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


def cleanup_old_content(**kwargs: Any) -> None:
    """Delete low-relevance content_updates older than 6 months."""
    hook = PostgresHook(postgres_conn_id="jurisai_db")

    count_before = hook.get_first(
        "SELECT count(*) FROM content_updates WHERE captured_at < NOW() - INTERVAL '6 months' AND relevance_score < 0.7"
    )
    total_before = count_before[0] if count_before else 0

    if total_before == 0:
        logger.info("No old low-relevance content to clean up")
        return

    logger.info("Found %d old low-relevance content updates to remove", total_before)

    hook.run("""
        DELETE FROM content_updates
        WHERE captured_at < NOW() - INTERVAL '6 months'
          AND relevance_score < 0.7
    """)

    logger.info("Removed %d old content updates", total_before)


def cleanup_old_logs(**kwargs: Any) -> None:
    """Delete ingestion_log entries older than 12 months."""
    hook = PostgresHook(postgres_conn_id="jurisai_db")

    count_before = hook.get_first(
        "SELECT count(*) FROM ingestion_log WHERE ingested_at < NOW() - INTERVAL '12 months'"
    )
    total_before = count_before[0] if count_before else 0

    if total_before == 0:
        logger.info("No old ingestion logs to clean up")
        return

    hook.run("""
        DELETE FROM ingestion_log
        WHERE ingested_at < NOW() - INTERVAL '12 months'
    """)

    logger.info("Removed %d old ingestion log entries", total_before)


def cleanup_old_bookmarks(**kwargs: Any) -> None:
    """Remove bookmarks that reference deleted content updates."""
    hook = PostgresHook(postgres_conn_id="jurisai_db")

    hook.run("""
        DELETE FROM user_bookmarks
        WHERE content_update_id NOT IN (SELECT id FROM content_updates)
    """)

    logger.info("Cleaned up orphaned bookmarks")


with DAG(
    dag_id="cleanup_old_content",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 2 1 * *",
    start_date=datetime(2026, 2, 1),
    catchup=False,
    tags=["cleanup", "maintenance"],
) as dag:
    t_content = PythonOperator(
        task_id="cleanup_old_content",
        python_callable=cleanup_old_content,
    )
    t_logs = PythonOperator(
        task_id="cleanup_old_logs",
        python_callable=cleanup_old_logs,
    )
    t_bookmarks = PythonOperator(
        task_id="cleanup_old_bookmarks",
        python_callable=cleanup_old_bookmarks,
    )
    t_content >> t_bookmarks
    t_content >> t_logs
