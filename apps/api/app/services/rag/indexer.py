"""
Incremental indexer for legal documents.

Receives raw documents, chunks them, generates embeddings,
deduplicates, and indexes into both Elasticsearch and Qdrant.
"""

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from elasticsearch import AsyncElasticsearch
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app.config import settings
from app.services.clients import create_es_client, create_qdrant_client
from app.services.ingestion.deduplicator import Deduplicator
from app.services.rag.chunker import Chunk, LegalChunker
from app.services.rag.embeddings import EmbeddingService

logger = logging.getLogger("jurisai.indexer")

ES_INDEX_SETTINGS = {
    "settings": {
        "analysis": {
            "analyzer": {
                "brazilian_legal": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "asciifolding",
                    ],
                }
            }
        },
    },
    "mappings": {
        "properties": {
            "content": {
                "type": "text",
                "analyzer": "brazilian_legal",
            },
            "ementa": {
                "type": "text",
                "analyzer": "brazilian_legal",
            },
            "title": {
                "type": "text",
                "analyzer": "brazilian_legal",
            },
            "embedding": {
                "type": "dense_vector",
                "dims": settings.EMBEDDING_DIM,
                "index": True,
                "similarity": "cosine",
            },
            "document_id": {"type": "keyword"},
            "document_title": {"type": "keyword"},
            "tenant_id": {"type": "keyword"},
            "doc_type": {"type": "keyword"},
            "court": {"type": "keyword"},
            "area": {"type": "keyword"},
            "date": {"type": "date", "format": "yyyy-MM-dd||epoch_millis"},
            "source": {"type": "keyword"},
            "source_date": {"type": "date", "format": "yyyy-MM-dd||epoch_millis"},
            "indexed_at": {"type": "date"},
            "status": {"type": "keyword"},
            "content_type": {"type": "keyword"},
            "metadata": {"type": "object", "enabled": False},
            "content_hash": {"type": "keyword"},
        }
    },
}


class Indexer:
    """Incremental indexer for Elasticsearch + Qdrant."""

    EMBED_BATCH_SIZE = 32

    def __init__(
        self,
        tenant_id: str = "__system__",
        es_client: Optional[AsyncElasticsearch] = None,
        qdrant_client: Optional[AsyncQdrantClient] = None,
    ) -> None:
        self.tenant_id = tenant_id
        self.es = es_client or create_es_client()
        self.qdrant = qdrant_client or create_qdrant_client()
        self.chunker = LegalChunker()
        self.embedder = EmbeddingService()
        self.dedup = Deduplicator()
        self._known_hashes: set[str] = set()

    async def ensure_indices(self) -> None:
        """Create ES index and Qdrant collection if they don't exist."""
        index_name = f"{settings.ES_INDEX_PREFIX}_chunks"
        if not await self.es.indices.exists(index=index_name):
            await self.es.indices.create(index=index_name, body=ES_INDEX_SETTINGS)
            logger.info("Created ES index: %s", index_name)

        collections = await self.qdrant.get_collections()
        existing = {c.name for c in collections.collections}
        if settings.QDRANT_COLLECTION not in existing:
            await self.qdrant.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection: %s", settings.QDRANT_COLLECTION)

    async def index_document(
        self,
        text: str,
        *,
        doc_type: str = "generic",
        document_id: Optional[str] = None,
        document_title: str = "",
        court: str = "",
        area: str = "",
        date: str = "",
        source: str = "",
        metadata: Optional[dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> int:
        """
        Index a single document: chunk -> embed -> deduplicate -> store.

        Returns the number of chunks indexed.
        """
        tid = tenant_id or self.tenant_id
        doc_id = document_id or str(uuid.uuid4())
        meta = metadata or {}
        now = datetime.now(timezone.utc).isoformat()

        chunks = self.chunker.chunk_document(text, doc_type=doc_type)
        if not chunks:
            return 0

        unique_chunks: list[Chunk] = []
        for chunk in chunks:
            h = _content_hash(chunk.content)
            if h in self._known_hashes:
                continue
            if self.dedup.is_duplicate(chunk.content, self._known_hashes):
                continue
            self._known_hashes.add(h)
            unique_chunks.append(chunk)

        if not unique_chunks:
            logger.debug("All chunks duplicated for doc %s", doc_id)
            return 0

        texts = [c.content for c in unique_chunks]
        embeddings = await self._embed_batched(texts)

        es_actions: list[dict[str, Any]] = []
        qdrant_points: list[PointStruct] = []

        for chunk, embedding in zip(unique_chunks, embeddings):
            chunk_id = str(uuid.uuid4())
            ementa = chunk.content if chunk.content_type == "ementa" else ""

            es_doc = {
                "content": chunk.content,
                "ementa": ementa,
                "title": document_title,
                "embedding": embedding,
                "document_id": doc_id,
                "document_title": document_title,
                "tenant_id": tid,
                "doc_type": doc_type,
                "court": court,
                "area": area,
                "date": date or None,
                "source": source,
                "source_date": date or None,
                "indexed_at": now,
                "status": "active",
                "content_type": chunk.content_type,
                "metadata": meta,
                "content_hash": _content_hash(chunk.content),
            }
            es_actions.append({"_index": f"{settings.ES_INDEX_PREFIX}_chunks", "_id": chunk_id, **es_doc})

            payload = {
                "content": chunk.content,
                "document_id": doc_id,
                "document_title": document_title,
                "tenant_id": tid,
                "doc_type": doc_type,
                "court": court,
                "area": area,
                "date": date,
                "source": source,
                "indexed_at": now,
                "status": "active",
                "content_type": chunk.content_type,
                "metadata": meta,
            }
            qdrant_points.append(
                PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        await self._bulk_index_es(es_actions)
        await self._bulk_index_qdrant(qdrant_points)

        logger.info("Indexed %d chunks for doc %s", len(unique_chunks), doc_id)
        return len(unique_chunks)

    async def index_process(self, process_data: dict[str, Any]) -> int:
        """Index a DataJud/STJ process dict (convenience wrapper)."""
        cnj = process_data.get("numeroProcesso", "")
        court = process_data.get("orgaoJulgador", {}).get("nome", "")
        subject_list = process_data.get("assuntos", [{}])
        subject = subject_list[0].get("nome", "") if subject_list else ""
        class_name = process_data.get("classe", {}).get("nome", "")
        filing_date = str(process_data.get("dataAjuizamento", ""))[:10]

        text_parts = [
            f"Processo: {cnj}",
            f"Classe: {class_name}",
            f"Assunto: {subject}",
            f"Órgão: {court}",
        ]
        for mov in process_data.get("movimentos", []):
            text_parts.append(f"- {mov.get('nome', '')} ({str(mov.get('dataHora', ''))[:10]})")

        full_text = "\n".join(text_parts)

        return await self.index_document(
            text=full_text,
            doc_type="decisao",
            document_id=cnj,
            document_title=f"{class_name} - {cnj}",
            court=court,
            area=subject,
            date=filing_date,
            source="datajud",
            metadata={"cnj_number": cnj, "class_name": class_name},
        )

    async def mark_superseded(self, document_id: str) -> None:
        """Mark all chunks of a document as superseded (reformed/overturned)."""
        index_name = f"{settings.ES_INDEX_PREFIX}_chunks"
        await self.es.update_by_query(
            index=index_name,
            body={
                "query": {"term": {"document_id": document_id}},
                "script": {"source": "ctx._source.status = 'superseded'"},
            },
        )
        logger.info("Marked document %s as superseded", document_id)

    async def reindex_all(self) -> None:
        """Full reindex — drop and recreate indices. Use with care."""
        index_name = f"{settings.ES_INDEX_PREFIX}_chunks"
        if await self.es.indices.exists(index=index_name):
            await self.es.indices.delete(index=index_name)

        collections = await self.qdrant.get_collections()
        existing = {c.name for c in collections.collections}
        if settings.QDRANT_COLLECTION in existing:
            await self.qdrant.delete_collection(settings.QDRANT_COLLECTION)

        await self.ensure_indices()
        self._known_hashes.clear()
        logger.info("Reindex complete — indices recreated")

    async def _embed_batched(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches to respect API limits."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.EMBED_BATCH_SIZE):
            batch = texts[i : i + self.EMBED_BATCH_SIZE]
            embeddings = await self.embedder.embed_documents(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    async def _bulk_index_es(self, actions: list[dict[str, Any]]) -> None:
        """Bulk index into Elasticsearch."""
        if not actions:
            return
        from elasticsearch.helpers import async_bulk

        docs = []
        for action in actions:
            index = action.pop("_index")
            doc_id = action.pop("_id")
            docs.append({"_index": index, "_id": doc_id, "_source": action})

        success, errors = await async_bulk(self.es, docs, raise_on_error=False)
        if errors:
            logger.warning("ES bulk errors: %d failures out of %d", len(errors), len(docs))

    async def _bulk_index_qdrant(self, points: list[PointStruct]) -> None:
        """Bulk upsert into Qdrant."""
        if not points:
            return
        await self.qdrant.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=points,
        )


def _content_hash(text: str) -> str:
    """SHA-256 hash of normalized content for deduplication."""
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
