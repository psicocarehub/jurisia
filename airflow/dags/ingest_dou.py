"""
DAG de ingestao diaria do DOU (Diario Oficial da Uniao).

Monitora publicacoes no DOU via API do Querido Diario e scraping
do Planalto para detectar novas leis, decretos, MPs e resolucoes.

Roda as 6h (apos ingestao de tribunais).
"""

from datetime import datetime, timedelta, date
from typing import Any

import httpx
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

from _content_updates_helper import insert_content_update

DEFAULT_ARGS = {
    "owner": "jurisai",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

QUERIDO_DIARIO_API = "https://queridodiario.ok.org.br/api/gazettes"
PLANALTO_SEARCH_URL = "https://legislacao.planalto.gov.br/legisla/legislacao.nsf/fraWeb"

DOU_CATEGORIES = [
    "lei",
    "lei_complementar",
    "decreto",
    "medida_provisoria",
    "resolucao",
    "portaria",
    "emenda_constitucional",
]


def _ensure_tables(hook: PostgresHook) -> None:
    """Create DOU ingestion tables if not present."""
    hook.run("""
        CREATE TABLE IF NOT EXISTS ingestion_dou_state (
            source VARCHAR(100) PRIMARY KEY,
            last_date DATE,
            last_ingested_at TIMESTAMPTZ
        )
    """)
    hook.run("""
        CREATE TABLE IF NOT EXISTS dou_documents (
            id SERIAL PRIMARY KEY,
            doc_type VARCHAR(50) NOT NULL,
            number VARCHAR(100),
            title TEXT NOT NULL,
            ementa TEXT,
            full_text TEXT,
            publication_date DATE NOT NULL,
            source_url TEXT,
            areas TEXT[] DEFAULT '{}',
            status VARCHAR(20) DEFAULT 'active',
            previous_version_id INTEGER REFERENCES dou_documents(id),
            ingested_at TIMESTAMPTZ DEFAULT NOW(),
            metadata JSONB DEFAULT '{}'
        )
    """)
    hook.run("""
        CREATE INDEX IF NOT EXISTS idx_dou_doc_type ON dou_documents(doc_type);
        CREATE INDEX IF NOT EXISTS idx_dou_pub_date ON dou_documents(publication_date);
        CREATE INDEX IF NOT EXISTS idx_dou_status ON dou_documents(status);
    """)


def _get_last_date(hook: PostgresHook, source: str) -> str | None:
    row = hook.get_first(
        "SELECT last_date FROM ingestion_dou_state WHERE source = %s",
        parameters=(source,),
    )
    return row[0].isoformat() if row and row[0] else None


def _update_state(hook: PostgresHook, source: str, last_date: str, count: int) -> None:
    hook.run("""
        INSERT INTO ingestion_dou_state (source, last_date, last_ingested_at)
        VALUES (%s, %s::date, NOW())
        ON CONFLICT (source) DO UPDATE SET
            last_date = GREATEST(
                COALESCE(ingestion_dou_state.last_date, '1970-01-01'::date),
                EXCLUDED.last_date
            ),
            last_ingested_at = NOW()
    """, parameters=(source, last_date))
    hook.run(
        "INSERT INTO ingestion_log (source, records_count, status) VALUES (%s, %s, 'completed')",
        parameters=(source, count),
    )


def ingest_querido_diario_federal(**kwargs: Any) -> None:
    """
    Ingest federal DOU entries via Querido Diario API.
    Focuses on legislative acts (leis, decretos, MPs).
    """
    hook = PostgresHook(postgres_conn_id="jurisai_db")
    _ensure_tables(hook)

    last = _get_last_date(hook, "dou_querido_diario") or (
        date.today() - timedelta(days=7)
    ).isoformat()
    today = date.today().isoformat()

    keywords = [
        "LEI Nº", "LEI COMPLEMENTAR", "DECRETO Nº",
        "MEDIDA PROVISÓRIA", "RESOLUÇÃO", "EMENDA CONSTITUCIONAL",
        "SÚMULA VINCULANTE",
    ]

    total_count = 0
    max_date = last

    for keyword in keywords:
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.get(
                    QUERIDO_DIARIO_API,
                    params={
                        "territory_id": "5300108",  # Brasilia/DF (federal)
                        "querystring": keyword,
                        "published_since": last,
                        "published_until": today,
                        "size": 100,
                    },
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()

            for item in data.get("gazettes", []):
                pub_date = item.get("date", today)
                excerpt = item.get("excerpts", [""])[0] if item.get("excerpts") else ""
                title = _extract_title(excerpt, keyword)

                doc_type = _classify_doc_type(keyword)
                doc_url = item.get("url", "")
                hook.run("""
                    INSERT INTO dou_documents (doc_type, title, ementa, publication_date, source_url, metadata)
                    VALUES (%s, %s, %s, %s::date, %s, %s::jsonb)
                    ON CONFLICT DO NOTHING
                """, parameters=(
                    doc_type,
                    title,
                    excerpt[:2000],
                    pub_date,
                    doc_url,
                    '{"source": "querido_diario", "territory_id": "' + item.get("territory_id", "") + '"}',
                ))
                insert_content_update(
                    hook,
                    source="dou",
                    category="legislacao",
                    subcategory=doc_type,
                    title=title,
                    summary=excerpt[:2000],
                    publication_date=pub_date,
                    source_url=doc_url,
                    territory="federal",
                    court_or_organ="DOU",
                )
                total_count += 1
                if pub_date > max_date:
                    max_date = pub_date

        except Exception as e:
            hook.run(
                "INSERT INTO ingestion_log (source, records_count, status, error_message) VALUES (%s, 0, 'partial', %s)",
                parameters=(f"dou_qd_{keyword[:20]}", str(e)[:500]),
            )

    _update_state(hook, "dou_querido_diario", max_date, total_count)


def ingest_planalto_legislation(**kwargs: Any) -> None:
    """
    Scrape Planalto for recent federal legislation.
    Captures full text of new laws and decrees.
    """
    hook = PostgresHook(postgres_conn_id="jurisai_db")
    _ensure_tables(hook)

    last = _get_last_date(hook, "dou_planalto") or (
        date.today() - timedelta(days=30)
    ).isoformat()

    categories_urls = {
        "lei": "https://www.planalto.gov.br/ccivil_03/_ato2023-2026/leis.htm",
        "lei_complementar": "https://www.planalto.gov.br/ccivil_03/leis/lcp.htm",
        "decreto": "https://www.planalto.gov.br/ccivil_03/_ato2023-2026/decretos.htm",
        "medida_provisoria": "https://www.planalto.gov.br/ccivil_03/_ato2023-2026/mpv.htm",
    }

    total_count = 0
    max_date = last

    for doc_type, url in categories_urls.items():
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.get(url, follow_redirects=True)
                if resp.status_code != 200:
                    continue

            import re
            links = re.findall(
                r'href="([^"]*(?:L\d+|Lcp\d+|D\d+|Mpv\d+)[^"]*\.htm)"',
                resp.text,
                re.IGNORECASE,
            )

            for link in links[:50]:
                if not link.startswith("http"):
                    link = f"https://www.planalto.gov.br/ccivil_03/{link}"

                try:
                    with httpx.Client(timeout=30.0) as client:
                        page = client.get(link, follow_redirects=True)
                        if page.status_code != 200:
                            continue

                    text = _extract_law_text(page.text)
                    title = _extract_law_title(page.text) or link.split("/")[-1]
                    number = _extract_law_number(title)
                    ementa = _extract_ementa(text)
                    pub_date = _extract_publication_date(text) or date.today().isoformat()

                    if pub_date < last:
                        continue

                    hook.run("""
                        INSERT INTO dou_documents (doc_type, number, title, ementa, full_text, publication_date, source_url)
                        VALUES (%s, %s, %s, %s, %s, %s::date, %s)
                        ON CONFLICT DO NOTHING
                    """, parameters=(
                        doc_type, number, title, ementa, text[:50000], pub_date, link,
                    ))
                    insert_content_update(
                        hook,
                        source="planalto",
                        category="legislacao",
                        subcategory=doc_type,
                        title=title,
                        summary=ementa,
                        content_preview=text[:500],
                        publication_date=pub_date,
                        source_url=link,
                        territory="federal",
                        court_or_organ="Planalto",
                    )
                    total_count += 1
                    if pub_date > max_date:
                        max_date = pub_date

                except Exception:
                    continue

        except Exception as e:
            hook.run(
                "INSERT INTO ingestion_log (source, records_count, status, error_message) VALUES (%s, 0, 'partial', %s)",
                parameters=(f"dou_planalto_{doc_type}", str(e)[:500]),
            )

    _update_state(hook, "dou_planalto", max_date, total_count)


def detect_law_changes(**kwargs: Any) -> None:
    """
    Compare newly ingested laws against existing ones to detect
    amendments, revocations, and new legislation.
    Triggers alert pipeline when relevant changes found.
    """
    hook = PostgresHook(postgres_conn_id="jurisai_db")

    new_docs = hook.get_records("""
        SELECT id, doc_type, title, ementa, full_text, publication_date
        FROM dou_documents
        WHERE ingested_at > NOW() - INTERVAL '1 day'
          AND status = 'active'
        ORDER BY publication_date DESC
    """)

    import re

    for row in new_docs:
        doc_id, doc_type, title, ementa, full_text, pub_date = row
        text_to_check = (ementa or "") + " " + (full_text or "")

        revokes = re.findall(
            r"(?:revoga|fica\s+revogad[oa])\s+.*?(Lei|Decreto|Resolução)\s+n[ºo°]?\s*([\d.]+(?:/\d{4})?)",
            text_to_check,
            re.IGNORECASE,
        )

        amends = re.findall(
            r"(?:altera|dá nova redação|acrescenta)\s+.*?(Lei|Decreto|Código)\s+n[ºo°]?\s*([\d.]+(?:/\d{4})?)",
            text_to_check,
            re.IGNORECASE,
        )

        for law_type, law_number in revokes:
            hook.run("""
                INSERT INTO ingestion_log (source, records_count, status, error_message)
                VALUES (%s, 1, 'change_detected', %s)
            """, parameters=(
                "law_change",
                f"REVOGACAO: {law_type} {law_number} revogada por DOU doc {doc_id} ({title})",
            ))

        for law_type, law_number in amends:
            hook.run("""
                INSERT INTO ingestion_log (source, records_count, status, error_message)
                VALUES (%s, 1, 'change_detected', %s)
            """, parameters=(
                "law_change",
                f"ALTERACAO: {law_type} {law_number} alterada por DOU doc {doc_id} ({title})",
            ))


def _classify_doc_type(keyword: str) -> str:
    mapping = {
        "LEI Nº": "lei",
        "LEI COMPLEMENTAR": "lei_complementar",
        "DECRETO Nº": "decreto",
        "MEDIDA PROVISÓRIA": "medida_provisoria",
        "RESOLUÇÃO": "resolucao",
        "EMENDA CONSTITUCIONAL": "emenda_constitucional",
        "SÚMULA VINCULANTE": "sumula_vinculante",
    }
    return mapping.get(keyword, "outro")


def _extract_title(text: str, keyword: str) -> str:
    import re
    match = re.search(
        rf"({keyword}\s*[\d./]+.*?)(?:\.|;|\n)",
        text,
        re.IGNORECASE,
    )
    return match.group(1).strip()[:500] if match else keyword


def _extract_law_text(html: str) -> str:
    """Strip HTML tags to extract raw law text."""
    import re
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "\n", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()


def _extract_law_title(html: str) -> str:
    import re
    match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()[:500]
    return ""


def _extract_law_number(title: str) -> str:
    import re
    match = re.search(r"[NnºÀ°]\s*([\d.]+(?:/\d{4})?)", title)
    return match.group(1) if match else ""


def _extract_ementa(text: str) -> str:
    import re
    match = re.search(
        r"((?:Dispõe|Altera|Institui|Regulamenta|Dá nova redação).*?\.)",
        text[:3000],
        re.DOTALL,
    )
    return match.group(1).strip()[:2000] if match else ""


def _extract_publication_date(text: str) -> str:
    import re
    months = {
        "janeiro": "01", "fevereiro": "02", "março": "03", "abril": "04",
        "maio": "05", "junho": "06", "julho": "07", "agosto": "08",
        "setembro": "09", "outubro": "10", "novembro": "11", "dezembro": "12",
    }
    match = re.search(
        r"(\d{1,2})\s+de\s+(janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\s+de\s+(\d{4})",
        text[:5000],
        re.IGNORECASE,
    )
    if match:
        day = match.group(1).zfill(2)
        month = months.get(match.group(2).lower(), "01")
        year = match.group(3)
        return f"{year}-{month}-{day}"
    return ""


with DAG(
    dag_id="ingest_dou_daily",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 6 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ingestion", "dou", "legislation"],
) as dag:
    task_querido_diario = PythonOperator(
        task_id="ingest_querido_diario_federal",
        python_callable=ingest_querido_diario_federal,
    )

    task_planalto = PythonOperator(
        task_id="ingest_planalto_legislation",
        python_callable=ingest_planalto_legislation,
    )

    task_detect = PythonOperator(
        task_id="detect_law_changes",
        python_callable=detect_law_changes,
    )

    [task_querido_diario, task_planalto] >> task_detect
