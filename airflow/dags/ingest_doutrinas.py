"""
DAG: Ingestao diaria de artigos e doutrinas juridicas.

Captura novos artigos, colunas e publicacoes doutrinarias de:
  - Conjur (Consultor Juridico)
  - JOTA Info
  - Migalhas
  - Jusbrasil

Usa RSS feeds e scraping leve para obter titulos, resumos e links.

Executa diariamente as 9h UTC.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
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
    "retry_delay": timedelta(minutes=10),
}

RSS_SOURCES = [
    {
        "name": "conjur",
        "url": "https://www.conjur.com.br/rss.xml",
        "organ": "Conjur",
    },
    {
        "name": "jota",
        "url": "https://www.jota.info/feed",
        "organ": "JOTA",
    },
    {
        "name": "migalhas",
        "url": "https://www.migalhas.com.br/rss",
        "organ": "Migalhas",
    },
]


def _parse_rss_date(date_str: str) -> str | None:
    """Best-effort parse of RSS date strings to YYYY-MM-DD."""
    if not date_str:
        return None
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _strip_html(text: str) -> str:
    """Remove HTML tags."""
    return re.sub(r"<[^>]+>", "", text).strip()


def _infer_areas(text: str) -> list[str]:
    areas = []
    upper = text.upper()[:1500]
    mapping = {
        "constitucional": ["CONSTITUCIONAL", "STF", "CONSTITUIÇÃO"],
        "penal": ["PENAL", "CRIMINAL", "CRIME"],
        "civil": ["CIVIL", "CONTRATO", "RESPONSABILIDADE CIVIL"],
        "trabalhista": ["TRABALH", "CLT", "TST"],
        "tributario": ["TRIBUT", "FISCAL", "IMPOSTO"],
        "administrativo": ["ADMINISTRAT", "LICITAÇÃO"],
        "processual": ["PROCESSUAL", "CPC", "RECURSO"],
        "empresarial": ["EMPRESAR", "SOCIETÁR", "FALÊNCIA"],
        "ambiental": ["AMBIENT", "IBAMA"],
        "digital": ["LGPD", "DADOS PESSOAIS", "TECNOLOGIA", "IA ", "INTELIGÊNCIA ARTIFICIAL"],
    }
    for area, kws in mapping.items():
        if any(k in upper for k in kws):
            areas.append(area)
    return areas or ["geral"]


def ingest_rss_source(source_config: dict, **kwargs: Any) -> None:
    """Fetch and parse a single RSS feed."""
    hook = PostgresHook(postgres_conn_id="jurisai_db")
    name = source_config["name"]
    total = 0

    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            resp = client.get(
                source_config["url"],
                headers={"User-Agent": "JurisAI/1.0 (Legal Content Aggregator)"},
            )
            if resp.status_code != 200:
                hook.run(
                    "INSERT INTO ingestion_log (source, records_count, status, error_message) VALUES (%s, 0, 'failed', %s)",
                    parameters=(f"doutrina_{name}", f"HTTP {resp.status_code}"),
                )
                return
    except Exception as e:
        hook.run(
            "INSERT INTO ingestion_log (source, records_count, status, error_message) VALUES (%s, 0, 'failed', %s)",
            parameters=(f"doutrina_{name}", str(e)[:500]),
        )
        return

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    items = root.findall(".//item") or root.findall(".//atom:entry", ns)

    for item in items[:50]:
        title_el = item.find("title") or item.find("atom:title", ns)
        title = (title_el.text or "").strip() if title_el is not None else ""
        if not title:
            continue

        desc_el = item.find("description") or item.find("atom:summary", ns) or item.find("atom:content", ns)
        description = _strip_html(desc_el.text or "") if desc_el is not None else ""

        link_el = item.find("link") or item.find("atom:link", ns)
        if link_el is not None:
            link = link_el.text or link_el.get("href", "") or ""
        else:
            link = ""

        pub_el = item.find("pubDate") or item.find("atom:published", ns) or item.find("atom:updated", ns)
        pub_date = _parse_rss_date(pub_el.text if pub_el is not None else "")

        full_text = f"{title} {description}"

        insert_content_update(
            hook,
            source=f"doutrina_{name}",
            category="doutrina",
            subcategory="artigo",
            title=title[:500],
            summary=description[:2000] if description else None,
            content_preview=description[:500] if description else None,
            publication_date=pub_date,
            source_url=link.strip(),
            territory="federal",
            court_or_organ=source_config["organ"],
            areas=_infer_areas(full_text),
        )
        total += 1

    hook.run(
        "INSERT INTO ingestion_log (source, records_count, status) VALUES (%s, %s, 'completed')",
        parameters=(f"doutrina_{name}", total),
    )


with DAG(
    dag_id="ingest_doutrinas",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 9 * * *",
    start_date=datetime(2026, 2, 1),
    catchup=False,
    tags=["ingestion", "doutrina", "artigos", "rss"],
) as dag:
    for src in RSS_SOURCES:
        PythonOperator(
            task_id=f"ingest_{src['name']}",
            python_callable=ingest_rss_source,
            op_kwargs={"source_config": src},
        )
