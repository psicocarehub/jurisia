"""
DAG: Ingestao semanal de normativos de agencias reguladoras.

Captura resolucoes, consultas publicas e outros normativos de:
  - ANVISA (saude)
  - ANATEL (telecomunicacoes)
  - ANS (saude suplementar)
  - ANEEL (energia)
  - ANP (petroleo)
  - ANTAQ (portos e transportes aquaviarios)
  - ANAC (aviacao)
  - CVM (valores mobiliarios)
  - BACEN (banco central)

Executa semanalmente (quinta-feira, 5h UTC).
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

QUERIDO_DIARIO_API = "https://queridodiario.ok.org.br/api/gazettes"

AGENCIES = [
    {
        "name": "ANVISA",
        "keywords": ["RESOLUÇÃO ANVISA", "RDC ANVISA", "INSTRUÇÃO NORMATIVA ANVISA"],
        "areas": ["saude", "regulatorio"],
        "organ": "ANVISA",
    },
    {
        "name": "ANATEL",
        "keywords": ["RESOLUÇÃO ANATEL", "ATO ANATEL"],
        "areas": ["telecomunicacoes", "regulatorio"],
        "organ": "ANATEL",
    },
    {
        "name": "ANS",
        "keywords": ["RESOLUÇÃO NORMATIVA ANS", "INSTRUÇÃO NORMATIVA ANS"],
        "areas": ["saude_suplementar", "regulatorio"],
        "organ": "ANS",
    },
    {
        "name": "ANEEL",
        "keywords": ["RESOLUÇÃO NORMATIVA ANEEL", "RESOLUÇÃO HOMOLOGATÓRIA ANEEL"],
        "areas": ["energia", "regulatorio"],
        "organ": "ANEEL",
    },
    {
        "name": "ANP",
        "keywords": ["RESOLUÇÃO ANP"],
        "areas": ["petroleo", "regulatorio"],
        "organ": "ANP",
    },
    {
        "name": "ANTAQ",
        "keywords": ["RESOLUÇÃO ANTAQ", "PORTARIA ANTAQ"],
        "areas": ["portuario", "regulatorio"],
        "organ": "ANTAQ",
    },
    {
        "name": "ANAC",
        "keywords": ["RESOLUÇÃO ANAC", "INSTRUÇÃO SUPLEMENTAR ANAC"],
        "areas": ["aviacao", "regulatorio"],
        "organ": "ANAC",
    },
    {
        "name": "CVM",
        "keywords": ["RESOLUÇÃO CVM", "INSTRUÇÃO CVM", "OFÍCIO-CIRCULAR CVM"],
        "areas": ["mercado_financeiro", "regulatorio"],
        "organ": "CVM",
    },
    {
        "name": "BACEN",
        "keywords": ["RESOLUÇÃO BCB", "RESOLUÇÃO CMN", "CIRCULAR BCB", "INSTRUÇÃO NORMATIVA BCB"],
        "areas": ["mercado_financeiro", "bancario", "regulatorio"],
        "organ": "Banco Central do Brasil",
    },
]


def ingest_agency(agency_config: dict, **kwargs: Any) -> None:
    """Fetch recent normatives for a single agency from DOU."""
    hook = PostgresHook(postgres_conn_id="jurisai_db")
    since = (date.today() - timedelta(days=7)).isoformat()
    today = date.today().isoformat()
    total = 0
    name = agency_config["name"]

    with httpx.Client(timeout=60.0) as client:
        for keyword in agency_config["keywords"]:
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

                insert_content_update(
                    hook,
                    source=f"agencia_{name.lower()}",
                    category="normativo",
                    subcategory=f"resolucao_{name.lower()}",
                    title=title,
                    summary=excerpt[:2000] if excerpt else None,
                    publication_date=pub_date,
                    source_url=item.get("url", ""),
                    territory="federal",
                    court_or_organ=agency_config["organ"],
                    areas=agency_config["areas"],
                )
                total += 1

    hook.run(
        "INSERT INTO ingestion_log (source, records_count, status) VALUES (%s, %s, 'completed')",
        parameters=(f"agencia_{name.lower()}", total),
    )


with DAG(
    dag_id="ingest_agencias_reguladoras",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 5 * * 4",
    start_date=datetime(2026, 2, 1),
    catchup=False,
    tags=["ingestion", "agencias", "regulatorio"],
) as dag:
    for agency in AGENCIES:
        PythonOperator(
            task_id=f"ingest_{agency['name'].lower()}",
            python_callable=ingest_agency,
            op_kwargs={"agency_config": agency},
        )
