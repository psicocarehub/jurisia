"""
DAG de ingestão diária do DataJud (API CNJ).

Percorre todos os tribunais com search_after pagination.
Rastreia o último @timestamp ingerido por tribunal para buscar apenas novos registros.
Indexa em Elasticsearch + Qdrant com embeddings Voyage AI.

Executa diariamente às 3h UTC.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

from _content_updates_helper import insert_content_update

TRIBUNAIS_PRIORITARIOS = [
    "stj", "tst", "tse",
    "trf1", "trf2", "trf3", "trf4", "trf5",
    "tjsp", "tjrj", "tjmg", "tjrs", "tjpr", "tjsc", "tjba", "tjdft",
    "tjpe", "tjce", "tjgo", "tjpa", "tjma", "tjes",
    "trt1", "trt2", "trt3", "trt4", "trt5", "trt15",
]

TRIBUNAL_ALIASES = {t: f"api_publica_{t}" for t in TRIBUNAIS_PRIORITARIOS}

DEFAULT_ARGS = {
    "owner": "jurisai",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


def _ensure_state_table(hook: PostgresHook) -> None:
    hook.run("""
        CREATE TABLE IF NOT EXISTS ingestion_datajud_state (
            source VARCHAR(100) PRIMARY KEY,
            last_timestamp BIGINT DEFAULT 0,
            last_ingested_at TIMESTAMPTZ,
            total_ingested BIGINT DEFAULT 0
        )
    """)


def _get_last_timestamp(hook: PostgresHook, tribunal: str) -> int:
    source = f"datajud_{tribunal}"
    row = hook.get_first(
        "SELECT last_timestamp FROM ingestion_datajud_state WHERE source = %s",
        parameters=(source,),
    )
    return row[0] if row and row[0] else 0


def _update_state(hook: PostgresHook, tribunal: str, last_ts: int, count: int) -> None:
    source = f"datajud_{tribunal}"
    hook.run("""
        INSERT INTO ingestion_datajud_state (source, last_timestamp, last_ingested_at, total_ingested)
        VALUES (%s, %s, NOW(), %s)
        ON CONFLICT (source) DO UPDATE SET
            last_timestamp = GREATEST(ingestion_datajud_state.last_timestamp, EXCLUDED.last_timestamp),
            last_ingested_at = NOW(),
            total_ingested = ingestion_datajud_state.total_ingested + EXCLUDED.total_ingested
    """, parameters=(source, last_ts, count))

    hook.run(
        "INSERT INTO ingestion_log (source, records_count, status) VALUES (%s, %s, 'completed')",
        parameters=(source, count),
    )


def ingest_tribunal(tribunal: str, **kwargs: Any) -> None:
    """Fetch new processes from DataJud and index into ES + Qdrant."""
    import asyncio
    import hashlib
    import logging
    import sys
    import uuid
    from pathlib import Path

    import httpx
    from elasticsearch import AsyncElasticsearch
    from elasticsearch.helpers import async_bulk
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import PointStruct

    logger = logging.getLogger(f"datajud.{tribunal}")

    root = Path(__file__).resolve().parents[2]
    if str(root / "apps" / "api") not in sys.path:
        sys.path.insert(0, str(root / "apps" / "api"))
    from app.config import settings

    api_key = Variable.get("datajud_api_key", default_var=settings.DATAJUD_API_KEY)
    if not api_key:
        raise ValueError("DataJud API key not configured")

    base_url = Variable.get("datajud_base_url", default_var=settings.DATAJUD_BASE_URL)
    max_pages = int(Variable.get("datajud_max_pages", default_var="50"))
    page_size = int(Variable.get("datajud_page_size", default_var="100"))

    hook = PostgresHook(postgres_conn_id="jurisai_db")
    _ensure_state_table(hook)
    last_ts = _get_last_timestamp(hook, tribunal)

    alias = TRIBUNAL_ALIASES.get(tribunal, f"api_publica_{tribunal}")
    url = f"{base_url}/{alias}/_search"

    async def _run():
        es_kwargs = {"hosts": [settings.ELASTICSEARCH_URL]}
        if settings.ES_API_KEY:
            es_kwargs["api_key"] = settings.ES_API_KEY
        es = AsyncElasticsearch(**es_kwargs)

        qd_kwargs = {"url": settings.QDRANT_URL}
        if settings.QDRANT_API_KEY:
            qd_kwargs["api_key"] = settings.QDRANT_API_KEY
        qdrant = AsyncQdrantClient(**qd_kwargs)

        total = 0
        search_after = [last_ts] if last_ts > 0 else None
        max_ts_seen = last_ts

        async with httpx.AsyncClient(timeout=120.0) as http:
            try:
                for page in range(max_pages):
                    body: dict[str, Any] = {
                        "size": page_size,
                        "query": {"match_all": {}},
                        "sort": [{"@timestamp": {"order": "asc"}}],
                    }
                    if search_after:
                        body["search_after"] = search_after

                    resp = await http.post(
                        url,
                        headers={
                            "Authorization": f"APIKey {api_key}",
                            "Content-Type": "application/json",
                        },
                        json=body,
                    )

                    if resp.status_code == 429:
                        logger.warning("Rate limited, stopping for %s", tribunal)
                        break
                    resp.raise_for_status()

                    hits = resp.json().get("hits", {}).get("hits", [])
                    if not hits:
                        break

                    texts, metas = [], []
                    for hit in hits:
                        src = hit.get("_source", {})
                        if src.get("nivelSigilo", 0) > 0:
                            continue

                        parts = [f"TRIBUNAL: {src.get('tribunal', tribunal.upper())}"]
                        classe = src.get("classe", {})
                        if isinstance(classe, dict) and classe.get("nome"):
                            parts.append(f"CLASSE: {classe['nome']}")
                        orgao = src.get("orgaoJulgador", {})
                        if isinstance(orgao, dict) and orgao.get("nome"):
                            parts.append(f"ÓRGÃO: {orgao['nome']}")

                        assuntos = src.get("assuntos", [])
                        subs = []
                        for a in assuntos:
                            if isinstance(a, dict) and a.get("nome"):
                                subs.append(a["nome"])
                            elif isinstance(a, list):
                                for s in a:
                                    if isinstance(s, dict) and s.get("nome"):
                                        subs.append(s["nome"])
                        if subs:
                            parts.append(f"ASSUNTOS: {'; '.join(subs)}")

                        text = "\n".join(parts)
                        if len(text) < 60:
                            continue

                        texts.append(text)
                        metas.append({
                            "court": src.get("tribunal", tribunal.upper()),
                            "numero": src.get("numeroProcesso", ""),
                            "classe": classe.get("nome", "") if isinstance(classe, dict) else "",
                            "date": (src.get("dataAjuizamento", "") or "")[:10] or None,
                        })

                    if not texts:
                        search_after = hits[-1].get("sort")
                        continue

                    emb_resp = await http.post(
                        "https://api.voyageai.com/v1/embeddings",
                        headers={"Authorization": f"Bearer {settings.VOYAGE_API_KEY}"},
                        json={
                            "model": settings.EMBEDDING_MODEL,
                            "input": texts[:8],
                            "input_type": "document",
                        },
                    )
                    emb_resp.raise_for_status()
                    embeddings = [d["embedding"] for d in emb_resp.json()["data"]]

                    for batch_start in range(0, len(texts), 8):
                        batch_texts = texts[batch_start:batch_start + 8]
                        if batch_start > 0:
                            er = await http.post(
                                "https://api.voyageai.com/v1/embeddings",
                                headers={"Authorization": f"Bearer {settings.VOYAGE_API_KEY}"},
                                json={
                                    "model": settings.EMBEDDING_MODEL,
                                    "input": batch_texts,
                                    "input_type": "document",
                                },
                            )
                            er.raise_for_status()
                            embeddings.extend([d["embedding"] for d in er.json()["data"]])

                    es_idx = f"{settings.ES_INDEX_PREFIX}_chunks"
                    es_actions = []
                    qd_points = []

                    for text, emb, meta in zip(texts, embeddings, metas):
                        cid = str(uuid.uuid4())
                        h = hashlib.sha256(text.lower().encode()).hexdigest()[:16]
                        doc = {
                            "content": text,
                            "ementa": "",
                            "title": f"{meta['court']} - {meta['classe']} - {meta['numero']}"[:200],
                            "embedding": emb,
                            "document_id": meta["numero"],
                            "document_title": f"{meta['court']} - {meta['classe']}"[:200],
                            "tenant_id": "__system__",
                            "doc_type": "processo",
                            "court": meta["court"],
                            "area": "geral",
                            "source": f"datajud_{tribunal}",
                            "indexed_at": datetime.utcnow().isoformat() + "Z",
                            "status": "active",
                            "content_hash": h,
                        }
                        if meta.get("date"):
                            doc["date"] = meta["date"]
                        es_actions.append({"_index": es_idx, "_id": cid, "_source": doc})
                        payload = {k: v for k, v in doc.items() if k != "embedding"}
                        qd_points.append(PointStruct(id=cid, vector=emb, payload=payload))

                    if es_actions:
                        await async_bulk(es, es_actions, raise_on_error=False)
                    if qd_points:
                        await qdrant.upsert(collection_name=settings.QDRANT_COLLECTION, points=qd_points)

                    for text, meta in zip(texts, metas):
                        insert_content_update(
                            hook,
                            source=f"datajud_{tribunal}",
                            category="jurisprudencia",
                            subcategory=meta.get("classe", "processo"),
                            title=f"{meta['court']} - {meta['classe']} - {meta['numero']}"[:500],
                            summary=text[:500],
                            court_or_organ=meta.get("court", tribunal.upper()),
                            territory="federal",
                            publication_date=meta.get("date"),
                        )

                    total += len(es_actions)
                    search_after = hits[-1].get("sort")
                    if search_after and search_after[0] > max_ts_seen:
                        max_ts_seen = search_after[0]

                    logger.info("[%s] page %d: +%d docs (total=%d)", tribunal, page, len(es_actions), total)

            finally:
                await es.close()
                await qdrant.close()

        return total, max_ts_seen

    count, new_ts = asyncio.run(_run())
    _update_state(hook, tribunal, new_ts, count)
    logger.info("[%s] Done: %d new docs indexed", tribunal, count)


with DAG(
    dag_id="ingest_datajud_daily",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 3 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_tasks=5,
    tags=["ingestion", "datajud", "cnj"],
    doc_md="""
    ## DataJud CNJ Daily Ingestion

    Busca novos processos de 30+ tribunais via API pública do CNJ.
    Usa `search_after` para paginação eficiente.
    Rastreia `@timestamp` para buscar apenas novos registros.
    Indexa em Elasticsearch + Qdrant com embeddings Voyage AI.
    """,
) as dag:
    for tribunal in TRIBUNAIS_PRIORITARIOS:
        PythonOperator(
            task_id=f"ingest_{tribunal}",
            python_callable=ingest_tribunal,
            op_kwargs={"tribunal": tribunal},
            pool="datajud_pool",
        )
