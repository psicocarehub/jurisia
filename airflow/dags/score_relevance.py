"""
DAG: Scoring de relevancia de conteudo novo.

Processa content_updates recentes que ainda nao foram scored (relevance_score = 0.5)
e atribui um score de relevancia baseado em:
  1. Tipo de conteudo (sumulas/teses > legislacao > jurisprudencia > doutrina)
  2. Abrangencia (federal > estadual > municipal)
  3. Impacto potencial (revogacoes, novas leis > portarias)
  4. Cruzamento com casos ativos dos tenants

Adicionalmente, gera alertas quando conteudo novo impacta casos ativos.

Executa diariamente as 10h UTC (apos todos os DAGs de ingestao).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

logger = logging.getLogger("jurisai.dag.relevance")

DEFAULT_ARGS = {
    "owner": "jurisai",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Base relevance weights by category
CATEGORY_WEIGHTS = {
    "sumula": 0.95,
    "legislacao": 0.85,
    "normativo": 0.75,
    "parecer": 0.70,
    "jurisprudencia": 0.60,
    "doutrina": 0.50,
    "outro": 0.40,
}

# Subcategory boosters
SUBCATEGORY_BOOST = {
    "sumula_vinculante": 0.10,
    "tema_repercussao_geral": 0.10,
    "tema_repetitivo": 0.10,
    "lei": 0.05,
    "lei_complementar": 0.08,
    "emenda_constitucional": 0.10,
    "medida_provisoria": 0.08,
    "instrucao_normativa_rfb": 0.05,
    "solucao_consulta_cosit": 0.05,
    "solucao_consulta_coana": 0.05,
}

TERRITORY_WEIGHTS = {
    "federal": 0.05,
}


def score_new_content(**kwargs: Any) -> None:
    """Score recently ingested content that still has the default relevance_score."""
    hook = PostgresHook(postgres_conn_id="jurisai_db")

    unscored = hook.get_records("""
        SELECT id, category, subcategory, territory, title, areas
        FROM content_updates
        WHERE relevance_score = 0.5
          AND captured_at > NOW() - INTERVAL '2 days'
        ORDER BY captured_at DESC
        LIMIT 5000
    """)

    if not unscored:
        logger.info("No unscored content found")
        return

    logger.info("Scoring %d content updates", len(unscored))
    scored = 0

    for row in unscored:
        uid, category, subcategory, territory, title, areas = row
        base = CATEGORY_WEIGHTS.get(category, 0.50)
        boost = SUBCATEGORY_BOOST.get(subcategory or "", 0.0)
        territory_boost = TERRITORY_WEIGHTS.get(territory or "", 0.0)

        title_upper = (title or "").upper()
        if any(k in title_upper for k in ["REVOGA", "REVOGAÇÃO", "REVOGACAO"]):
            boost += 0.10
        if any(k in title_upper for k in ["ALTERA", "NOVA REDAÇÃO", "NOVA REDACAO"]):
            boost += 0.05
        if "URGENTE" in title_upper or "MEDIDA PROVISÓRIA" in title_upper:
            boost += 0.05

        score = min(1.0, base + boost + territory_boost)

        hook.run(
            "UPDATE content_updates SET relevance_score = %s WHERE id = %s",
            parameters=(round(score, 3), uid),
        )
        scored += 1

    logger.info("Scored %d content updates", scored)


def cross_reference_cases(**kwargs: Any) -> None:
    """
    Cross-reference new high-relevance content with active cases.
    When a match is found, create an alert for the case owner.
    """
    hook = PostgresHook(postgres_conn_id="jurisai_db")

    active_cases = hook.get_records("""
        SELECT id, tenant_id, title, area, cnj_number, created_by
        FROM cases
        WHERE status = 'active'
        LIMIT 1000
    """)

    if not active_cases:
        logger.info("No active cases to cross-reference")
        return

    recent_high = hook.get_records("""
        SELECT id, title, category, subcategory, areas, court_or_organ, source_url
        FROM content_updates
        WHERE relevance_score >= 0.8
          AND captured_at > NOW() - INTERVAL '1 day'
        ORDER BY relevance_score DESC
        LIMIT 200
    """)

    if not recent_high:
        logger.info("No high-relevance content to cross-reference")
        return

    alerts_created = 0

    for case_row in active_cases:
        case_id, tenant_id, case_title, case_area, cnj, created_by = case_row
        case_area = (case_area or "").lower()
        case_keywords = set((case_title or "").upper().split())

        for update_row in recent_high:
            update_id, update_title, cat, subcat, update_areas, organ, url = update_row
            update_areas_set = set(a.lower() for a in (update_areas or []))

            has_area_match = case_area and case_area in update_areas_set
            has_keyword_overlap = len(
                case_keywords.intersection(set((update_title or "").upper().split()))
            ) >= 3

            if has_area_match or has_keyword_overlap:
                try:
                    hook.run("""
                        INSERT INTO alerts (change_type, title, description, areas, severity, source_url, tenant_id, user_id, metadata)
                        VALUES ('content_impact', %s, %s, %s::text[], 'medium', %s, %s::uuid, %s::uuid, %s::jsonb)
                        ON CONFLICT DO NOTHING
                    """, parameters=(
                        f"Conteúdo relevante para caso: {case_title[:100]}",
                        f"Novo conteúdo '{update_title[:200]}' pode impactar o caso '{case_title[:200]}'.",
                        "{" + ",".join(f'"{a}"' for a in (update_areas or [])) + "}",
                        url,
                        tenant_id,
                        created_by,
                        f'{{"case_id": "{case_id}", "content_update_id": "{update_id}", "category": "{cat}"}}',
                    ))
                    alerts_created += 1
                except Exception as e:
                    logger.warning("Failed to create case-impact alert: %s", e)

    logger.info("Created %d case-impact alerts", alerts_created)


def verify_sources(**kwargs: Any) -> None:
    """
    Basic verification: mark content_updates as verified if source_url is present
    and returns HTTP 200.  Only checks a sample to avoid excessive requests.
    """
    hook = PostgresHook(postgres_conn_id="jurisai_db")

    unverified = hook.get_records("""
        SELECT id, source_url
        FROM content_updates
        WHERE is_verified = FALSE
          AND source_url IS NOT NULL
          AND source_url != ''
          AND captured_at > NOW() - INTERVAL '1 day'
        ORDER BY relevance_score DESC
        LIMIT 50
    """)

    if not unverified:
        return

    import httpx

    verified = 0
    with httpx.Client(timeout=10.0, follow_redirects=True) as client:
        for uid, url in unverified:
            try:
                resp = client.head(url)
                if resp.status_code < 400:
                    hook.run(
                        """UPDATE content_updates
                           SET is_verified = TRUE,
                               verification_details = '{"method": "http_head", "status": %s}'::jsonb
                           WHERE id = %s""",
                        parameters=(resp.status_code, uid),
                    )
                    verified += 1
            except Exception:
                continue

    logger.info("Verified %d/%d source URLs", verified, len(unverified))


with DAG(
    dag_id="score_relevance",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 10 * * *",
    start_date=datetime(2026, 2, 1),
    catchup=False,
    tags=["scoring", "relevance", "alerts"],
) as dag:
    t_score = PythonOperator(
        task_id="score_new_content",
        python_callable=score_new_content,
    )
    t_cases = PythonOperator(
        task_id="cross_reference_cases",
        python_callable=cross_reference_cases,
    )
    t_verify = PythonOperator(
        task_id="verify_sources",
        python_callable=verify_sources,
    )
    t_score >> [t_cases, t_verify]
