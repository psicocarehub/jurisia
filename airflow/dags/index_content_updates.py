"""
DAG: Index high-relevance content updates into Elasticsearch and Qdrant.

Selects content_updates with relevance_score >= 0.7 that haven't been indexed yet
(document_id IS NULL), generates embeddings via Voyage AI, and indexes them.

Runs daily at 11h UTC, after score_relevance.py.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

logger = logging.getLogger("jurisai.dag.index_updates")

DEFAULT_ARGS = {
    "owner": "jurisai",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

ES_INDEX_PREFIX = os.getenv("ES_INDEX_PREFIX", "jurisai")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "jurisai_chunks")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
VOYAGE_MODEL = os.getenv("VOYAGE_EMBED_MODEL", "voyage-3")
BATCH_SIZE = 32


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings via Voyage AI."""
    if not VOYAGE_API_KEY:
        logger.warning("VOYAGE_API_KEY not set, skipping embedding generation")
        return []

    import httpx

    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        resp = httpx.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {VOYAGE_API_KEY}"},
            json={"model": VOYAGE_MODEL, "input": batch, "input_type": "document"},
            timeout=30.0,
        )
        if resp.status_code != 200:
            logger.error("Voyage API error: %d %s", resp.status_code, resp.text[:200])
            continue
        data = resp.json()
        for item in data.get("data", []):
            all_embeddings.append(item["embedding"])

    return all_embeddings


def index_updates(**kwargs: Any) -> None:
    """Select and index high-relevance content_updates."""
    hook = PostgresHook(postgres_conn_id="jurisai_db")

    rows = hook.get_records("""
        SELECT id, title, summary, category, subcategory, areas,
               court_or_organ, territory, publication_date, source_url
        FROM content_updates
        WHERE relevance_score >= 0.7
          AND document_id IS NULL
          AND captured_at > NOW() - INTERVAL '2 days'
        ORDER BY relevance_score DESC
        LIMIT 500
    """)

    if not rows:
        logger.info("No content updates to index")
        return

    logger.info("Indexing %d content updates", len(rows))

    texts = []
    records = []
    for row in rows:
        uid, title, summary, category, subcategory, areas, organ, territory, pub_date, url = row
        text = f"{title or ''} {summary or ''}"
        texts.append(text[:2000])
        records.append({
            "id": uid,
            "title": title,
            "summary": summary,
            "category": category,
            "subcategory": subcategory,
            "areas": areas or [],
            "court_or_organ": organ,
            "territory": territory,
            "publication_date": str(pub_date) if pub_date else None,
            "source_url": url,
        })

    embeddings = _embed_texts(texts)
    if len(embeddings) != len(records):
        logger.error("Embedding count mismatch: %d texts vs %d embeddings", len(records), len(embeddings))
        return

    indexed_es = 0
    indexed_qdrant = 0

    try:
        from elasticsearch import Elasticsearch

        es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
        es = Elasticsearch([es_url])
        es_index = f"{ES_INDEX_PREFIX}_content_updates"

        for rec, emb in zip(records, embeddings):
            doc = {
                "content": f"{rec['title']} {rec['summary'] or ''}",
                "document_id": str(rec["id"]),
                "document_title": rec["title"],
                "doc_type": rec["category"],
                "court": rec.get("court_or_organ", ""),
                "date": rec.get("publication_date", ""),
                "area": ",".join(rec.get("areas", [])),
                "embedding": emb,
                "metadata": {
                    "source": "content_updates",
                    "subcategory": rec.get("subcategory"),
                    "territory": rec.get("territory"),
                },
            }
            try:
                es.index(index=es_index, id=str(rec["id"]), document=doc)
                indexed_es += 1
            except Exception as e:
                logger.warning("ES indexing failed for %s: %s", rec["id"], e)

    except ImportError:
        logger.warning("elasticsearch package not available, skipping ES indexing")
    except Exception as e:
        logger.error("ES connection failed: %s", e)

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct

        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant = QdrantClient(url=qdrant_url)

        points = []
        for rec, emb in zip(records, embeddings):
            points.append(PointStruct(
                id=str(rec["id"]),
                vector=emb,
                payload={
                    "content": f"{rec['title']} {rec['summary'] or ''}",
                    "document_id": str(rec["id"]),
                    "document_title": rec["title"],
                    "doc_type": rec["category"],
                    "court": rec.get("court_or_organ", ""),
                    "date": rec.get("publication_date", ""),
                    "area": ",".join(rec.get("areas", [])),
                    "source": "content_updates",
                },
            ))

        if points:
            qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
            indexed_qdrant = len(points)

    except ImportError:
        logger.warning("qdrant_client package not available, skipping Qdrant indexing")
    except Exception as e:
        logger.error("Qdrant connection failed: %s", e)

    for rec in records:
        doc_id = str(uuid.uuid4())
        hook.run(
            "UPDATE content_updates SET document_id = %s WHERE id = %s AND document_id IS NULL",
            parameters=(doc_id, rec["id"]),
        )

    logger.info("Indexed %d to ES, %d to Qdrant out of %d total", indexed_es, indexed_qdrant, len(records))


with DAG(
    dag_id="index_content_updates",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 11 * * *",
    start_date=datetime(2026, 2, 1),
    catchup=False,
    tags=["indexing", "content_updates", "rag"],
) as dag:
    PythonOperator(
        task_id="index_updates",
        python_callable=index_updates,
    )
