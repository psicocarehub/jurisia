"""
Reranker service: trained cross-encoder (local) with Voyage AI fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import httpx

from app.config import settings
from app.services.rag.models import RetrievedChunk

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[5] / "training" / "models" / "reranker" / "model"


class RerankerService:
    """Reranks chunks using trained cross-encoder, falling back to Voyage AI."""

    BASE_URL = "https://api.voyageai.com/v1"

    def __init__(self) -> None:
        self.model = settings.RERANK_MODEL
        self.api_key = settings.VOYAGE_API_KEY
        self._local_model = None
        self._local_tokenizer = None
        self._local_loaded = False

    def _load_local(self) -> bool:
        if self._local_loaded:
            return self._local_model is not None
        self._local_loaded = True
        if not MODEL_DIR.exists():
            return False
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self._local_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
            self._local_model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
            self._local_model.eval()
            logger.info("Local cross-encoder reranker loaded from %s", MODEL_DIR)
            return True
        except Exception as e:
            logger.warning("Failed to load local reranker: %s", e)
            return False

    def _local_rerank(self, query: str, documents: List[str]) -> List[float]:
        import torch
        pairs = [(query, d) for d in documents]
        enc = self._local_tokenizer(
            [p[0] for p in pairs], [p[1] for p in pairs],
            truncation=True, padding=True, max_length=512, return_tensors="pt",
        )
        with torch.no_grad():
            scores = self._local_model(**enc).logits.squeeze(-1).tolist()
        if isinstance(scores, float):
            scores = [scores]
        return scores

    async def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = 10,
    ) -> List[RetrievedChunk]:
        if not chunks:
            return []

        documents = [c.content for c in chunks]

        if self._load_local():
            try:
                scores = self._local_rerank(query, documents)
                scored = list(zip(chunks, scores))
                scored.sort(key=lambda x: x[1], reverse=True)
                for chunk, score in scored[:top_k]:
                    chunk.score = float(score)
                return [c for c, _ in scored[:top_k]]
            except Exception as e:
                logger.warning("Local reranker failed, falling back: %s", e)

        if not self.api_key:
            sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
            return sorted_chunks[:top_k]

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.BASE_URL}/rerank",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "query": query,
                        "documents": documents,
                        "top_k": top_k,
                    },
                )
                response.raise_for_status()
                data = response.json()

            reranked = []
            for result in data.get("data", []):
                idx = result["index"]
                chunk = chunks[idx]
                chunk.score = result["relevance_score"]
                reranked.append(chunk)

            return reranked

        except (httpx.HTTPError, KeyError) as e:
            logger.warning("Voyage reranker API failed, using original order: %s", e)
            sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
            return sorted_chunks[:top_k]
