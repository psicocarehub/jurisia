"""
Citation verification for Brazilian legal text.
Uses ML classifier for type detection with regex extraction + API verification.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

CITATION_CLF_PATH = Path(__file__).resolve().parents[5] / "training" / "models" / "citation_classifier" / "citation_classifier.joblib"
CITATION_CONFIG_PATH = CITATION_CLF_PATH.parent / "config.json"


class CitationStatus(str, Enum):
    VERIFIED = "verified"
    NOT_FOUND = "not_found"
    REVOKED = "revoked"
    OUTDATED = "outdated"
    UNCHECKED = "unchecked"


class Citation(BaseModel):
    """Verified or unverified citation."""

    text: str
    type: str  # legislacao, jurisprudencia, doutrina, sumula
    status: CitationStatus
    verified_text: Optional[str] = None
    source_url: Optional[str] = None
    confidence: float = 0.0


class CitationVerifier:
    """Extract and verify citations in legal text."""

    _clf = None
    _clf_loaded = False
    _clf_labels = ["legislacao", "jurisprudencia", "sumula", "doutrina", "nao_citacao"]

    @classmethod
    def _load_clf(cls) -> bool:
        if cls._clf_loaded:
            return cls._clf is not None
        cls._clf_loaded = True
        if CITATION_CLF_PATH.exists():
            try:
                import joblib, json
                cls._clf = joblib.load(CITATION_CLF_PATH)
                if CITATION_CONFIG_PATH.exists():
                    with open(CITATION_CONFIG_PATH) as f:
                        cls._clf_labels = json.load(f).get("labels", cls._clf_labels)
                logger.info("Citation classifier loaded")
                return True
            except Exception as e:
                logger.warning("Failed to load citation classifier: %s", e)
        return False

    def classify_citation(self, text: str) -> str:
        """Classify citation type using ML model."""
        if self._load_clf():
            try:
                pred = self._clf.predict([text])[0]
                return self._clf_labels[pred] if isinstance(pred, int) else str(pred)
            except Exception as e:
                logger.debug("Citation ML classification failed for %r: %s", text[:50], e)
        return "legislacao"

    async def verify_all(self, text: str) -> List[Citation]:
        """Extract and verify all citations."""
        citations: List[Citation] = []

        for ref in self._extract_legislation_refs(text):
            result = await self._verify_legislation(ref)
            citations.append(result)

        for ref in self._extract_jurisprudence_refs(text):
            result = await self._verify_jurisprudence(ref)
            citations.append(result)

        for ref in self._extract_sumula_refs(text):
            result = await self._verify_sumula(ref)
            citations.append(result)

        return citations

    def _extract_legislation_refs(self, text: str) -> List[str]:
        pattern = (
            r"(?:Art(?:igo)?\.?\s*\d+(?:[°ºª])?"
            r"(?:\s*(?:,|e)\s*(?:§\s*\d+[°ºª]?|inciso\s+[IVXLCDM]+|alínea\s+[\"']?[a-z][\"']?))*"
            r"\s*(?:,?\s*d[aoe]\s+)?(?:Lei|Decreto|Resolução|CF|Código\s+\w+)"
            r"(?:\s+(?:n[ºo°]?\s*)?[\d.,/]+)?(?:\s*/\s*\d{4})?)"
        )
        return re.findall(pattern, text, re.IGNORECASE)

    def _extract_jurisprudence_refs(self, text: str) -> List[str]:
        cnj = r"\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}"
        informal = r"(?:RE|REsp|AgRg|AI|HC|MS|ADI|ADPF|RMS)\s+[\d.,]+"
        return re.findall(f"({cnj}|{informal})", text)

    def _extract_sumula_refs(self, text: str) -> List[str]:
        pattern = (
            r"Súmula\s+(?:Vinculante\s+)?(?:n[ºo°]?\s*)?\d+"
            r"(?:\s+d[oe]\s+(?:STF|STJ|TST|TSE))?"
        )
        return re.findall(pattern, text, re.IGNORECASE)

    async def _verify_legislation(self, ref: str) -> Citation:
        """Verify legislation against LexML SRU and Planalto normas.leg.br."""
        import httpx

        ref_lower = ref.lower()
        search_term = re.sub(r"[^\w\s/]", "", ref)

        # Try LexML SRU search
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://www.lexml.gov.br/busca/SRU",
                    params={
                        "operation": "searchRetrieve",
                        "query": search_term,
                        "maximumRecords": "3",
                    },
                )
                if resp.status_code == 200 and search_term.split()[0].lower() in resp.text.lower():
                    return Citation(
                        text=ref,
                        type="legislacao",
                        status=CitationStatus.VERIFIED,
                        source_url=f"https://www.lexml.gov.br/busca/SRU?query={search_term}",
                        confidence=0.85,
                    )
        except Exception as e:
            logger.debug("LexML SRU legislation verification failed for %r: %s", ref[:50], e)

        # Try normas.leg.br (Planalto)
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                resp = await client.get(
                    "https://normas.leg.br/api/normas",
                    params={"busca": search_term, "limit": "3"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    normas = data if isinstance(data, list) else data.get("normas", [])
                    if normas:
                        norma = normas[0]
                        url = norma.get("url_texto_original", norma.get("url", ""))
                        situacao = norma.get("situacao", "").lower()
                        if "revogad" in situacao:
                            return Citation(
                                text=ref, type="legislacao",
                                status=CitationStatus.REVOKED,
                                source_url=url, confidence=0.9,
                                verified_text=f"Situação: {norma.get('situacao', '')}",
                            )
                        return Citation(
                            text=ref, type="legislacao",
                            status=CitationStatus.VERIFIED,
                            source_url=url, confidence=0.9,
                        )
        except Exception as e:
            logger.debug("normas.leg.br legislation verification failed for %r: %s", ref[:50], e)

        # Fallback: search in Elasticsearch
        try:
            from app.services.rag.retriever import HybridRetriever
            retriever = HybridRetriever()
            chunks = await retriever.retrieve(
                query=ref, tenant_id="__global__", top_k=3, use_reranker=False,
            )
            for c in chunks:
                if c.doc_type in ("legislacao", "lei", "decreto") and c.score > 0.5:
                    return Citation(
                        text=ref, type="legislacao",
                        status=CitationStatus.VERIFIED,
                        confidence=round(c.score, 2),
                        verified_text=c.content[:200],
                    )
        except Exception as e:
            logger.debug("Elasticsearch legislation verification failed for %r: %s", ref[:50], e)

        return Citation(
            text=ref, type="legislacao",
            status=CitationStatus.NOT_FOUND, confidence=0.0,
        )

    async def _verify_jurisprudence(self, ref: str) -> Citation:
        """Verify jurisprudence against STF/STJ APIs and Elasticsearch."""
        import httpx

        # Try STJ API
        try:
            from app.services.ingestion.stj import STJClient
            stj = STJClient()
            results = await stj.search_decisions(query=ref, max_results=3)
            if results:
                first = results[0]
                return Citation(
                    text=ref, type="jurisprudencia",
                    status=CitationStatus.VERIFIED,
                    source_url=first.get("url", ""),
                    confidence=0.9,
                    verified_text=first.get("ementa", "")[:200],
                )
        except Exception as e:
            logger.debug("STJ jurisprudence verification failed for %r: %s", ref[:50], e)

        # Try STF API
        try:
            from app.services.ingestion.stf import STFClient
            stf = STFClient()
            results = await stf.search_decisions(query=ref, max_results=3)
            if results:
                first = results[0]
                return Citation(
                    text=ref, type="jurisprudencia",
                    status=CitationStatus.VERIFIED,
                    source_url=first.get("url", ""),
                    confidence=0.9,
                    verified_text=first.get("ementa", "")[:200],
                )
        except Exception as e:
            logger.debug("STF jurisprudence verification failed for %r: %s", ref[:50], e)

        # Fallback: Elasticsearch
        try:
            from app.services.rag.retriever import HybridRetriever
            retriever = HybridRetriever()
            chunks = await retriever.retrieve(
                query=ref, tenant_id="__global__", top_k=3, use_reranker=False,
            )
            for c in chunks:
                if c.doc_type in ("acordao", "decisao", "jurisprudencia") and c.score > 0.5:
                    return Citation(
                        text=ref, type="jurisprudencia",
                        status=CitationStatus.VERIFIED,
                        confidence=round(c.score, 2),
                        verified_text=c.content[:200],
                    )
        except Exception as e:
            logger.debug("Elasticsearch jurisprudence verification failed for %r: %s", ref[:50], e)

        return Citation(
            text=ref, type="jurisprudencia",
            status=CitationStatus.NOT_FOUND, confidence=0.0,
        )

    async def _verify_sumula(self, ref: str) -> Citation:
        """Verify súmula against STF/STJ APIs."""
        import httpx

        ref_lower = ref.lower()
        number_match = re.search(r"\d+", ref)
        sumula_number = number_match.group() if number_match else ""
        is_vinculante = "vinculante" in ref_lower
        is_stf = "stf" in ref_lower or is_vinculante
        is_stj = "stj" in ref_lower

        # Try STF for súmulas vinculantes
        if is_stf or is_vinculante:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        "https://portal.stf.jus.br/textos/verTexto.asp",
                        params={"servico": "jurisprudenciaSumula", "pagina": "sumula_001_100"},
                    )
                    if resp.status_code == 200 and sumula_number and f"Súmula {sumula_number}" in resp.text:
                        return Citation(
                            text=ref, type="jurisprudencia",
                            status=CitationStatus.VERIFIED,
                            source_url="https://portal.stf.jus.br/jurisprudencia/sumariosumulas.asp",
                            confidence=0.9,
                        )
            except Exception as e:
                logger.debug("STF sumula verification failed for %r: %s", ref[:50], e)

        # Try STJ
        if is_stj or not is_stf:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        "https://scon.stj.jus.br/SCON/sumstj/toc.jsp",
                        params={"livre": f"@num={sumula_number}" if sumula_number else ref},
                    )
                    if resp.status_code == 200 and sumula_number and sumula_number in resp.text:
                        return Citation(
                            text=ref, type="jurisprudencia",
                            status=CitationStatus.VERIFIED,
                            source_url=f"https://scon.stj.jus.br/SCON/sumstj/toc.jsp?livre=@num={sumula_number}",
                            confidence=0.85,
                        )
            except Exception as e:
                logger.debug("STJ sumula verification failed for %r: %s", ref[:50], e)

        # Fallback: Elasticsearch
        try:
            from app.services.rag.retriever import HybridRetriever
            retriever = HybridRetriever()
            chunks = await retriever.retrieve(
                query=ref, tenant_id="__global__", top_k=3, use_reranker=False,
            )
            for c in chunks:
                if "súmula" in c.content.lower() or "sumula" in c.content.lower():
                    return Citation(
                        text=ref, type="jurisprudencia",
                        status=CitationStatus.VERIFIED,
                        confidence=round(c.score, 2),
                        verified_text=c.content[:200],
                    )
        except Exception as e:
            logger.debug("Elasticsearch sumula verification failed for %r: %s", ref[:50], e)

        return Citation(
            text=ref, type="jurisprudencia",
            status=CitationStatus.NOT_FOUND, confidence=0.0,
        )
