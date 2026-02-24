"""
Hybrid RAG Retriever: BM25 (Elasticsearch) + Dense (Qdrant) with RRF fusion.

Uses ES 8.16+ native RRF when available, fallback to manual RRF merge.
"""

from typing import List, Optional

from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.config import settings
from app.services.clients import create_es_client, create_qdrant_client
from app.services.rag.embeddings import EmbeddingService
from app.services.rag.models import RetrievedChunk


class HybridRetriever:
    """Hybrid retriever combining Elasticsearch BM25 and Qdrant vector search."""

    RRF_K = 60

    def __init__(self) -> None:
        self.es = create_es_client()
        self.qdrant = create_qdrant_client()
        self.embedder = EmbeddingService()
        from app.services.rag.reranker import RerankerService
        self.reranker = RerankerService()

    async def retrieve(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 10,
        filters: Optional[dict] = None,
        use_reranker: bool = True,
    ) -> List[RetrievedChunk]:
        """Retrieve relevant chunks via hybrid search with optional reranking."""
        # 1. Generate query embedding
        query_embedding = await self.embedder.embed_query(query)

        # 2. ES hybrid search with RRF (k=60)
        es_results = await self._es_hybrid_search(
            query=query,
            embedding=query_embedding,
            tenant_id=tenant_id,
            top_k=top_k * 3,  # Over-retrieve for reranking
            filters=filters,
        )

        # 3. Qdrant vector search
        qdrant_results = await self._qdrant_search(
            embedding=query_embedding,
            tenant_id=tenant_id,
            top_k=top_k * 2,
            filters=filters,
        )

        # 4. Merge and deduplicate
        all_results = self._merge_results(es_results, qdrant_results)

        # 5. Rerank
        if use_reranker and all_results:
            all_results = await self.reranker.rerank(
                query=query,
                chunks=all_results,
                top_k=top_k,
            )
        else:
            all_results = all_results[:top_k]

        return all_results

    def _rrf_merge(
        self,
        bm25_hits: List[tuple[str, dict]],
        knn_hits: List[tuple[str, dict]],
        k: int = 60,
    ) -> List[tuple[str, dict]]:
        """Reciprocal Rank Fusion: merge two ranked lists with RRF (k=60)."""
        scores: dict[str, float] = {}
        all_hits: dict[str, tuple[str, dict]] = {}

        for rank, (hit_id, source) in enumerate(bm25_hits, start=1):
            key = f"{hit_id}_{source.get('content', '')[:80]}"
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            all_hits[key] = (hit_id, source)

        for rank, (hit_id, source) in enumerate(knn_hits, start=1):
            key = f"{hit_id}_{source.get('content', '')[:80]}"
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            all_hits[key] = (hit_id, source)

        sorted_keys = sorted(scores.keys(), key=lambda x: -scores[x])
        return [all_hits[k] for k in sorted_keys]

    async def _es_hybrid_search(
        self,
        query: str,
        embedding: List[float],
        tenant_id: str,
        top_k: int,
        filters: Optional[dict] = None,
    ) -> List[RetrievedChunk]:
        """ES 8.16+ hybrid: BM25 + kNN, merge with RRF (k=60)."""
        must_clauses: List[dict] = [
            {"term": {"tenant_id": tenant_id}},
        ]
        if filters:
            if filters.get("area"):
                must_clauses.append({"term": {"area": filters["area"]}})
            if filters.get("court"):
                must_clauses.append({"term": {"court": filters["court"]}})
            if filters.get("date_from"):
                must_clauses.append(
                    {"range": {"date": {"gte": filters["date_from"]}}}
                )
            if filters.get("date_to"):
                must_clauses.append(
                    {"range": {"date": {"lte": filters["date_to"]}}}
                )

        source = [
            "content", "document_id", "document_title",
            "doc_type", "court", "date", "metadata",
        ]

        # BM25 search
        bm25_body = {
            "query": {
                "bool": {
                    "must": must_clauses,
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["content^1", "ementa^3", "title^2"],
                                "type": "best_fields",
                            }
                        },
                    ],
                }
            },
            "size": top_k,
            "_source": source,
        }

        # kNN search
        knn_body = {
            "query": {
                "bool": {"must": must_clauses},
            },
            "knn": {
                "field": "embedding",
                "query_vector": embedding,
                "k": top_k,
                "num_candidates": top_k * 5,
            },
            "size": top_k,
            "_source": source,
        }

        index = f"{settings.ES_INDEX_PREFIX}_chunks"

        bm25_resp = await self.es.search(index=index, body=bm25_body)
        knn_resp = await self.es.search(index=index, body=knn_body)

        bm25_hits = [
            (h["_id"], h["_source"])
            for h in bm25_resp.get("hits", {}).get("hits", [])
        ]
        knn_hits = [
            (h["_id"], h["_source"])
            for h in knn_resp.get("hits", {}).get("hits", [])
        ]

        merged = self._rrf_merge(bm25_hits, knn_hits, k=self.RRF_K)
        hits = [{"_id": hit_id, "_source": source} for hit_id, source in merged]
        return [
            RetrievedChunk(
                id=hit["_id"],
                content=hit["_source"].get("content", ""),
                score=float(hit.get("_score", 0.0)),
                document_id=hit["_source"].get("document_id", ""),
                document_title=hit["_source"].get("document_title", ""),
                doc_type=hit["_source"].get("doc_type", ""),
                court=hit["_source"].get("court", ""),
                date=hit["_source"].get("date", ""),
                metadata=hit["_source"].get("metadata", {}),
            )
            for hit in hits
        ]

    async def _qdrant_search(
        self,
        embedding: List[float],
        tenant_id: str,
        top_k: int,
        filters: Optional[dict] = None,
    ) -> List[RetrievedChunk]:
        """Qdrant vector search with tenant_id filter."""
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="tenant_id", match=MatchValue(value=tenant_id)
                )
            ]
        )
        if filters:
            if filters.get("area"):
                qdrant_filter.must.append(
                    FieldCondition(key="area", match=MatchValue(value=filters["area"]))
                )
            if filters.get("court"):
                qdrant_filter.must.append(
                    FieldCondition(key="court", match=MatchValue(value=filters["court"]))
                )

        results = await self.qdrant.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=embedding,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )

        return [
            RetrievedChunk(
                id=str(r.id),
                content=r.payload.get("content", ""),
                score=float(r.score) if r.score else 0.0,
                document_id=r.payload.get("document_id", ""),
                document_title=r.payload.get("document_title", ""),
                doc_type=r.payload.get("doc_type", ""),
                court=r.payload.get("court", ""),
                date=r.payload.get("date", ""),
                metadata=r.payload.get("metadata", {}),
            )
            for r in results.points
        ]

    def _merge_results(
        self,
        es_results: List[RetrievedChunk],
        qdrant_results: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """Merge and deduplicate by document_id + content prefix."""
        seen: set[str] = set()
        merged: List[RetrievedChunk] = []
        for chunk in es_results + qdrant_results:
            key = f"{chunk.document_id}_{chunk.content[:100]}"
            if key not in seen:
                seen.add(key)
                merged.append(chunk)
        # Sort by score descending
        merged.sort(key=lambda c: c.score, reverse=True)
        return merged
