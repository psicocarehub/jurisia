"""
DAG de avaliacao mensal do modelo GAIA.

Roda benchmark OAB, detecta drift, e trigga retreino
automatico se a qualidade cair abaixo do threshold.

Executa no primeiro dia de cada mes as 8h.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

DEFAULT_ARGS = {
    "owner": "jurisai",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=15),
}

EVAL_RESULTS_DIR = "training/continuous/eval_results"
DRIFT_THRESHOLD = 0.05
MIN_ACCURACY = 0.60


def _ensure_tables(hook: PostgresHook) -> None:
    hook.run("""
        CREATE TABLE IF NOT EXISTS model_eval_history (
            id SERIAL PRIMARY KEY,
            model_version VARCHAR(100),
            model_path TEXT,
            eval_date DATE NOT NULL,
            accuracy FLOAT NOT NULL,
            total_questions INTEGER,
            correct_questions INTEGER,
            area_metrics JSONB DEFAULT '{}',
            drift_detected BOOLEAN DEFAULT FALSE,
            drift_areas TEXT[] DEFAULT '{}',
            notes TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_eval_date ON model_eval_history(eval_date);
        CREATE INDEX IF NOT EXISTS idx_eval_version ON model_eval_history(model_version);
    """)


def run_evaluation(**kwargs: Any) -> dict[str, Any]:
    """
    Run OAB benchmark evaluation on the current production model.
    """
    hook = PostgresHook(postgres_conn_id="jurisai_db")
    _ensure_tables(hook)

    import sys
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    versions_manifest = root / "training" / "continuous" / "versions" / "manifest.json"
    model_path = str(root / "training" / "sft" / "gaia-legal-sft")
    model_version = "base"

    if versions_manifest.exists():
        manifest = json.loads(versions_manifest.read_text())
        current = manifest.get("current_version")
        if current:
            for v in manifest.get("versions", []):
                if v["version_id"] == current:
                    model_path = v.get("model_path", model_path)
                    model_version = current
                    break

    questions_path = str(root / "training" / "data" / "oab_questions.jsonl")
    if not Path(questions_path).exists():
        result = {
            "model_version": model_version,
            "accuracy": 0.0,
            "total": 0,
            "correct": 0,
            "status": "no_questions",
        }
        kwargs["ti"].xcom_push(key="eval_result", value=result)
        return result

    try:
        from training.eval.eval_oab import run_eval

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = str(root / EVAL_RESULTS_DIR / f"monthly_{timestamp}.json")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        metrics = run_eval(
            model_path=model_path,
            questions_path=questions_path,
            output_path=output_path,
        )
    except Exception as e:
        result = {
            "model_version": model_version,
            "accuracy": 0.0,
            "status": "eval_failed",
            "error": str(e)[:500],
        }
        kwargs["ti"].xcom_push(key="eval_result", value=result)
        return result

    area_metrics = {
        k.replace("accuracy_", ""): v
        for k, v in metrics.items()
        if k.startswith("accuracy_")
    }

    result = {
        "model_version": model_version,
        "model_path": model_path,
        "accuracy": metrics.get("accuracy", 0.0),
        "total": int(metrics.get("total", 0)),
        "correct": int(metrics.get("correct", 0)),
        "area_metrics": area_metrics,
        "status": "completed",
    }

    hook.run("""
        INSERT INTO model_eval_history
            (model_version, model_path, eval_date, accuracy, total_questions, correct_questions, area_metrics)
        VALUES (%s, %s, %s::date, %s, %s, %s, %s::jsonb)
    """, parameters=(
        model_version,
        model_path,
        datetime.utcnow().strftime("%Y-%m-%d"),
        result["accuracy"],
        result["total"],
        result["correct"],
        json.dumps(area_metrics),
    ))

    kwargs["ti"].xcom_push(key="eval_result", value=result)
    return result


def detect_drift(**kwargs: Any) -> dict[str, Any]:
    """
    Compare current evaluation against historical baseline.
    Detect if model quality has degraded in any area.
    """
    hook = PostgresHook(postgres_conn_id="jurisai_db")
    ti = kwargs["ti"]
    current = ti.xcom_pull(task_ids="run_evaluation", key="eval_result")

    if not current or current.get("status") != "completed":
        drift_result = {"drift_detected": False, "reason": "eval_incomplete"}
        ti.xcom_push(key="drift_result", value=drift_result)
        return drift_result

    previous = hook.get_records("""
        SELECT accuracy, area_metrics
        FROM model_eval_history
        WHERE eval_date < CURRENT_DATE
        ORDER BY eval_date DESC
        LIMIT 3
    """)

    if not previous:
        drift_result = {"drift_detected": False, "reason": "no_baseline"}
        ti.xcom_push(key="drift_result", value=drift_result)
        return drift_result

    baseline_accuracies = [row[0] for row in previous]
    baseline_avg = sum(baseline_accuracies) / len(baseline_accuracies)
    current_acc = current["accuracy"]

    overall_drift = baseline_avg - current_acc
    drift_detected = overall_drift > DRIFT_THRESHOLD

    degraded_areas: list[str] = []
    baseline_areas: dict[str, list[float]] = {}
    for _, area_json in previous:
        if area_json:
            areas = area_json if isinstance(area_json, dict) else json.loads(area_json)
            for area, acc in areas.items():
                baseline_areas.setdefault(area, []).append(acc)

    for area, vals in baseline_areas.items():
        avg = sum(vals) / len(vals)
        curr = current.get("area_metrics", {}).get(area, 0.0)
        if avg - curr > DRIFT_THRESHOLD:
            degraded_areas.append(area)

    if degraded_areas:
        drift_detected = True

    below_minimum = current_acc < MIN_ACCURACY

    drift_result = {
        "drift_detected": drift_detected,
        "below_minimum": below_minimum,
        "overall_drift": round(overall_drift, 4),
        "baseline_avg": round(baseline_avg, 4),
        "current_accuracy": round(current_acc, 4),
        "degraded_areas": degraded_areas,
        "needs_retrain": drift_detected or below_minimum,
    }

    hook.run("""
        UPDATE model_eval_history
        SET drift_detected = %s, drift_areas = %s, notes = %s
        WHERE eval_date = CURRENT_DATE
        ORDER BY created_at DESC
        LIMIT 1
    """, parameters=(
        drift_detected,
        degraded_areas,
        json.dumps(drift_result),
    ))

    ti.xcom_push(key="drift_result", value=drift_result)
    return drift_result


def check_retrain_needed(**kwargs: Any) -> str:
    """Branch: decide if retraining is needed."""
    ti = kwargs["ti"]
    drift = ti.xcom_pull(task_ids="detect_drift", key="drift_result")

    if drift and drift.get("needs_retrain"):
        return "trigger_retrain"
    return "log_healthy"


def trigger_retrain(**kwargs: Any) -> None:
    """
    Trigger the incremental fine-tuning pipeline.
    Collects accumulated data, trains, and registers candidate.
    """
    import sys
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    hook = PostgresHook(postgres_conn_id="jurisai_db")

    try:
        from training.continuous.data_accumulator import DataAccumulator
        import asyncio

        acc = DataAccumulator()
        asyncio.run(acc.collect_all())

        if acc.total_accumulated < 100:
            hook.run(
                "INSERT INTO ingestion_log (source, records_count, status, error_message) VALUES (%s, %s, %s, %s)",
                parameters=("model_retrain", acc.total_accumulated, "skipped", "Insufficient data for retraining"),
            )
            return

        acc.export_for_training()

        from training.continuous.incremental_finetune import incremental_train, VersionManager

        result = incremental_train()
        if "error" not in result:
            vm = VersionManager()
            vm.register_version(
                version_id=result["version_id"],
                model_path=result["model_path"],
                metrics={"train_loss": result["train_loss"]},
                training_data_count=result["data_count"],
            )
            hook.run(
                "INSERT INTO ingestion_log (source, records_count, status, error_message) VALUES (%s, %s, %s, %s)",
                parameters=("model_retrain", result["data_count"], "completed", f"Version {result['version_id']} created"),
            )
        else:
            hook.run(
                "INSERT INTO ingestion_log (source, records_count, status, error_message) VALUES (%s, 0, %s, %s)",
                parameters=("model_retrain", "failed", result.get("detail", result.get("error", "unknown"))),
            )

    except Exception as e:
        hook.run(
            "INSERT INTO ingestion_log (source, records_count, status, error_message) VALUES (%s, 0, %s, %s)",
            parameters=("model_retrain", "failed", str(e)[:500]),
        )


def log_healthy(**kwargs: Any) -> None:
    """Log that the model is performing within acceptable parameters."""
    hook = PostgresHook(postgres_conn_id="jurisai_db")
    ti = kwargs["ti"]
    current = ti.xcom_pull(task_ids="run_evaluation", key="eval_result")
    accuracy = current.get("accuracy", 0.0) if current else 0.0

    hook.run(
        "INSERT INTO ingestion_log (source, records_count, status, error_message) VALUES (%s, 0, %s, %s)",
        parameters=("model_eval", "completed", f"Model healthy: accuracy={accuracy:.2%}"),
    )


with DAG(
    dag_id="eval_model_monthly",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 8 1 * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["evaluation", "model", "drift"],
) as dag:
    task_eval = PythonOperator(
        task_id="run_evaluation",
        python_callable=run_evaluation,
    )

    task_drift = PythonOperator(
        task_id="detect_drift",
        python_callable=detect_drift,
    )

    task_branch = BranchPythonOperator(
        task_id="check_retrain_needed",
        python_callable=check_retrain_needed,
    )

    task_retrain = PythonOperator(
        task_id="trigger_retrain",
        python_callable=trigger_retrain,
    )

    task_healthy = PythonOperator(
        task_id="log_healthy",
        python_callable=log_healthy,
    )

    task_eval >> task_drift >> task_branch
    task_branch >> task_retrain
    task_branch >> task_healthy
