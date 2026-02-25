"""
DAG: Ingestao semanal de pareceres de orgaos consultivos.

Fontes:
  - AGU (Advocacia-Geral da Uniao) - portal de pareceres
  - PGE (Procuradorias estaduais) - publicacoes no DOU/DOE
  - CGU (Controladoria-Geral da Uniao) - pareceres e enunciados

Executa semanalmente (quarta-feira, 4h UTC).
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any

import httpx
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

from _content_updates_helper import insert_content_update

DEFAULT_ARGS = {
    "owner": "jurisai",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=15),
}

AGU_SEARCH_URL = "https://www.gov.br/agu/pt-br/composicao/consultoria-geral-da-uniao-1/pareceres"
QUERIDO_DIARIO_API = "https://queridodiario.ok.org.br/api/gazettes"

PARECER_KEYWORDS = [
    "PARECER AGU",
    "PARECER PGFN",
    "PARECER CGU",
    "NOTA TÉCNICA AGU",
    "ENUNCIADO AGU",
    "ORIENTAÇÃO NORMATIVA AGU",
    "SÚMULA AGU",
    "PARECER NORMATIVO",
]


def ingest_pareceres_dou(**kwargs: Any) -> None:
    """Search DOU for AGU/PGE/CGU opinions published in the official gazette."""
    hook = PostgresHook(postgres_conn_id="jurisai_db")
    since = (date.today() - timedelta(days=7)).isoformat()
    today = date.today().isoformat()
    total = 0

    with httpx.Client(timeout=60.0) as client:
        for keyword in PARECER_KEYWORDS:
            try:
                resp = client.get(
                    QUERIDO_DIARIO_API,
                    params={
                        "territory_id": "5300108",
                        "querystring": keyword,
                        "published_since": since,
                        "published_until": today,
                        "size": 50,
                    },
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()
            except Exception:
                continue

            for item in data.get("gazettes", []):
                pub_date = item.get("date", today)
                excerpts = item.get("excerpts", [])
                excerpt = excerpts[0] if excerpts else ""

                title_match = re.search(
                    rf"({keyword}\s*[Nn]?[ºo°]?\s*[\d./-]+[^.;]*)",
                    excerpt,
                    re.IGNORECASE,
                )
                title = title_match.group(1).strip()[:500] if title_match else keyword

                organ = "AGU"
                if "PGFN" in keyword:
                    organ = "PGFN"
                elif "CGU" in keyword:
                    organ = "CGU"

                subcategory = "parecer"
                if "ENUNCIADO" in keyword:
                    subcategory = "enunciado"
                elif "ORIENTAÇÃO" in keyword or "ORIENTACAO" in keyword:
                    subcategory = "orientacao_normativa"
                elif "SÚMULA" in keyword or "SUMULA" in keyword:
                    subcategory = "sumula_agu"
                elif "NOTA TÉCNICA" in keyword or "NOTA TECNICA" in keyword:
                    subcategory = "nota_tecnica"

                insert_content_update(
                    hook,
                    source="pareceres",
                    category="parecer",
                    subcategory=subcategory,
                    title=title,
                    summary=excerpt[:2000] if excerpt else None,
                    publication_date=pub_date,
                    source_url=item.get("url", ""),
                    territory="federal",
                    court_or_organ=organ,
                    areas=_infer_areas_from_parecer(excerpt),
                )
                total += 1

    hook.run(
        "INSERT INTO ingestion_log (source, records_count, status) VALUES ('pareceres_agu', %s, 'completed')",
        parameters=(total,),
    )


def ingest_agu_portal(**kwargs: Any) -> None:
    """Scrape AGU portal for recent opinions."""
    hook = PostgresHook(postgres_conn_id="jurisai_db")
    total = 0

    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            resp = client.get(AGU_SEARCH_URL)
            if resp.status_code != 200:
                return

        links = re.findall(
            r'<a[^>]+href="([^"]*parecer[^"]*)"[^>]*>([^<]+)</a>',
            resp.text,
            re.IGNORECASE,
        )

        for href, title in links[:50]:
            if not href.startswith("http"):
                href = f"https://www.gov.br{href}"

            insert_content_update(
                hook,
                source="agu_portal",
                category="parecer",
                subcategory="parecer",
                title=title.strip()[:500],
                source_url=href,
                territory="federal",
                court_or_organ="AGU",
            )
            total += 1

    except Exception as e:
        hook.run(
            "INSERT INTO ingestion_log (source, records_count, status, error_message) VALUES ('agu_portal', 0, 'failed', %s)",
            parameters=(str(e)[:500],),
        )
        return

    hook.run(
        "INSERT INTO ingestion_log (source, records_count, status) VALUES ('agu_portal', %s, 'completed')",
        parameters=(total,),
    )


def _infer_areas_from_parecer(text: str) -> list[str]:
    areas = []
    upper = text.upper()[:1500]
    mapping = {
        "administrativo": ["ADMINISTRAT", "LICITAÇ", "LICITAC", "CONTRATO ADMINISTRAT"],
        "tributario": ["TRIBUT", "FISCAL", "IMPOSTO"],
        "constitucional": ["CONSTITUCIONAL", "CONSTITUCIO"],
        "trabalhista": ["TRABALH", "PREVIDENC", "SERVIDOR"],
        "ambiental": ["AMBIENT", "LICENCIAMENTO"],
        "aduaneiro": ["ADUAN", "COMÉRCIO EXTERIOR", "COMERCIO EXTERIOR"],
    }
    for area, kws in mapping.items():
        if any(k in upper for k in kws):
            areas.append(area)
    return areas or ["administrativo"]


with DAG(
    dag_id="ingest_pareceres",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 4 * * 3",
    start_date=datetime(2026, 2, 1),
    catchup=False,
    tags=["ingestion", "pareceres", "agu", "pgfn"],
) as dag:
    t1 = PythonOperator(
        task_id="ingest_pareceres_dou",
        python_callable=ingest_pareceres_dou,
    )
    t2 = PythonOperator(
        task_id="ingest_agu_portal",
        python_callable=ingest_agu_portal,
    )
