"""
Case Analyzer — "Raio-X do Processo"

Orchestrates existing services to produce a comprehensive case analysis:
  1. Fetch case + documents from Supabase
  2. Retrieve indexed content from Elasticsearch
  3. RAG: find similar jurisprudence
  4. Judge profile (if judge_name available)
  5. Outcome prediction
  6. LLM analysis with specialized prompt
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from pydantic import BaseModel

from app.config import settings
from app.services.cache import CacheService

logger = logging.getLogger(__name__)
_cache = CacheService()


class Vulnerability(BaseModel):
    type: str
    title: str
    description: str
    severity: str
    legal_basis: str = ""
    recommendation: str = ""


class Strategy(BaseModel):
    type: str
    title: str
    description: str
    legal_basis: str = ""
    success_likelihood: str = ""
    risks: str = ""


class TimelineEvent(BaseModel):
    date: str
    event: str


class CaseAnalysis(BaseModel):
    case_id: str
    summary: str
    timeline: list[TimelineEvent] = []
    legal_framework: str = ""
    vulnerabilities: list[Vulnerability] = []
    strategies: list[Strategy] = []
    judge_profile: Optional[dict[str, Any]] = None
    prediction: Optional[dict[str, Any]] = None
    similar_cases: list[dict[str, Any]] = []
    risk_level: str = "médio"
    risk_assessment: str = ""
    model_used: str = ""
    generated_at: str = ""


def _headers() -> dict[str, str]:
    return {
        "apikey": settings.SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
    }


def _base_url() -> str:
    return f"{settings.SUPABASE_URL}/rest/v1"


class CaseAnalyzer:
    """Orchestrates a full case analysis using all available services."""

    async def analyze(self, case_id: str, tenant_id: str, *, force: bool = False) -> CaseAnalysis:
        cache_key = f"case_analysis:{tenant_id}:{case_id}"
        if not force:
            cached = await _cache.get(cache_key)
            if cached:
                logger.info("Cache hit for case analysis %s", case_id)
                return CaseAnalysis(**cached)

        case_data = await self._fetch_case(case_id, tenant_id)
        if not case_data:
            raise ValueError(f"Caso {case_id} não encontrado")

        documents = await self._fetch_case_documents(case_id, tenant_id)
        doc_content = await self._fetch_indexed_content(documents)

        query_text = self._build_search_query(case_data, doc_content)
        similar_cases = await self._retrieve_similar(query_text, tenant_id)

        judge_profile = None
        judge_name = case_data.get("judge_name")
        if judge_name:
            judge_profile = await self._get_judge_profile(
                judge_name, case_data.get("court")
            )

        prediction = await self._predict_outcome(case_data, doc_content)

        llm_result = await self._run_llm_analysis(
            case_data=case_data,
            doc_content=doc_content,
            similar_cases=similar_cases,
            judge_profile=judge_profile,
            prediction=prediction,
        )

        similar_cases_summary = [
            {
                "title": c.get("title", c.get("document_title", "")),
                "court": c.get("court", ""),
                "date": c.get("date", ""),
                "doc_type": c.get("doc_type", ""),
                "snippet": c.get("content", "")[:300],
            }
            for c in similar_cases
        ]

        analysis = CaseAnalysis(
            case_id=case_id,
            summary=llm_result.get("summary", ""),
            timeline=[
                TimelineEvent(**e) for e in llm_result.get("timeline", [])
            ],
            legal_framework=llm_result.get("legal_framework", ""),
            vulnerabilities=[
                Vulnerability(**v) for v in llm_result.get("vulnerabilities", [])
            ],
            strategies=[
                Strategy(**s) for s in llm_result.get("strategies", [])
            ],
            judge_profile=judge_profile,
            prediction=prediction,
            similar_cases=similar_cases_summary,
            risk_level=llm_result.get("risk_level", "médio"),
            risk_assessment=llm_result.get("risk_assessment", ""),
            model_used=llm_result.get("_model_used", ""),
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

        await _cache.set(cache_key, analysis.model_dump(), ttl=CacheService.TTL_ANALYSIS)
        return analysis

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    async def _fetch_case(
        self, case_id: str, tenant_id: str
    ) -> Optional[dict[str, Any]]:
        if not settings.SUPABASE_URL:
            return None
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{_base_url()}/cases",
                    headers={**_headers(), "Accept": "application/vnd.pgrst.object+json"},
                    params={"id": f"eq.{case_id}", "tenant_id": f"eq.{tenant_id}"},
                )
                if resp.status_code == 406:
                    return None
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.error("Failed to fetch case %s: %s", case_id, e)
            return None

    async def _fetch_case_documents(
        self, case_id: str, tenant_id: str
    ) -> list[dict[str, Any]]:
        if not settings.SUPABASE_URL:
            return []
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{_base_url()}/documents",
                    headers=_headers(),
                    params={
                        "case_id": f"eq.{case_id}",
                        "tenant_id": f"eq.{tenant_id}",
                        "select": "id,title,doc_type,classification_label,ocr_status",
                    },
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.error("Failed to fetch documents for case %s: %s", case_id, e)
            return []

    async def _fetch_indexed_content(
        self, documents: list[dict[str, Any]]
    ) -> str:
        """Retrieve full text of case documents from Elasticsearch."""
        doc_ids = [d["id"] for d in documents if d.get("ocr_status") == "completed"]
        if not doc_ids:
            return ""

        try:
            from app.services.clients import create_es_client

            es = create_es_client()
            index = f"{settings.ES_INDEX_PREFIX}_chunks"
            body = {
                "size": 100,
                "query": {"terms": {"document_id": doc_ids}},
                "sort": [{"metadata.chunk_index": {"order": "asc"}}],
                "_source": ["content", "document_id", "document_title"],
            }
            resp = await es.search(index=index, body=body)
            hits = resp.get("hits", {}).get("hits", [])
            parts = [h["_source"]["content"] for h in hits if h.get("_source", {}).get("content")]
            await es.close()
            return "\n\n".join(parts)
        except Exception as e:
            logger.warning("Failed to fetch indexed content: %s", e)
            return ""

    # ------------------------------------------------------------------
    # Service orchestration
    # ------------------------------------------------------------------

    def _build_search_query(
        self, case_data: dict[str, Any], doc_content: str
    ) -> str:
        parts = []
        if case_data.get("title"):
            parts.append(case_data["title"])
        if case_data.get("description"):
            parts.append(case_data["description"])
        if case_data.get("area"):
            parts.append(f"direito {case_data['area']}")
        if doc_content:
            parts.append(doc_content[:2000])
        return " ".join(parts) if parts else "caso jurídico"

    async def _retrieve_similar(
        self, query: str, tenant_id: str
    ) -> list[dict[str, Any]]:
        try:
            from app.services.rag.retriever import HybridRetriever

            retriever = HybridRetriever()
            chunks = await retriever.retrieve(
                query=query[:1000],
                tenant_id=tenant_id,
                top_k=8,
            )
            return [
                {
                    "document_title": c.document_title,
                    "court": c.court,
                    "date": c.date,
                    "doc_type": c.doc_type,
                    "content": c.content,
                    "score": c.score,
                }
                for c in chunks
            ]
        except Exception as e:
            logger.warning("RAG retrieval failed: %s", e)
            return []

    async def _get_judge_profile(
        self, judge_name: str, court: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        try:
            from app.services.jurimetrics.judge_profile import JurimetricsService

            service = JurimetricsService()
            profile = await service.get_judge_profile(judge_name, court=court)
            if profile:
                return profile.model_dump()
        except Exception as e:
            logger.warning("Judge profile failed for %s: %s", judge_name, e)
        return None

    async def _predict_outcome(
        self, case_data: dict[str, Any], doc_content: str
    ) -> Optional[dict[str, Any]]:
        try:
            from app.services.jurimetrics.predictor import OutcomePredictor

            predictor = OutcomePredictor()
            enriched = {
                **case_data,
                "content": doc_content[:5000],
                "tribunal": case_data.get("court", ""),
            }
            result = predictor.predict(enriched, area=case_data.get("area", ""))
            return result.model_dump()
        except Exception as e:
            logger.warning("Outcome prediction failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # LLM analysis
    # ------------------------------------------------------------------

    async def _run_llm_analysis(
        self,
        case_data: dict[str, Any],
        doc_content: str,
        similar_cases: list[dict[str, Any]],
        judge_profile: Optional[dict[str, Any]],
        prediction: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        from app.services.llm.prompts import CASE_ANALYSIS_SYSTEM_PROMPT
        from app.services.llm.router import LLMRouter

        context_parts = []

        context_parts.append("## DADOS DO CASO")
        context_parts.append(f"Título: {case_data.get('title', 'N/A')}")
        context_parts.append(f"Número CNJ: {case_data.get('cnj_number', 'N/A')}")
        context_parts.append(f"Área: {case_data.get('area', 'N/A')}")
        context_parts.append(f"Tribunal/Vara: {case_data.get('court', 'N/A')}")
        context_parts.append(f"Juiz: {case_data.get('judge_name', 'N/A')}")
        context_parts.append(f"Cliente: {case_data.get('client_name', 'N/A')}")
        context_parts.append(f"Parte Contrária: {case_data.get('opposing_party', 'N/A')}")
        context_parts.append(f"Valor da Causa: R$ {case_data.get('estimated_value', 'N/A')}")
        context_parts.append(f"Descrição: {case_data.get('description', 'N/A')}")

        if doc_content:
            truncated = doc_content[:12000]
            context_parts.append(f"\n## CONTEÚDO DOS DOCUMENTOS DO CASO\n{truncated}")

        if judge_profile:
            fav = judge_profile.get("favorability", {})
            context_parts.append("\n## PERFIL DO JUIZ (dados reais)")
            context_parts.append(f"Nome: {judge_profile.get('name', 'N/A')}")
            context_parts.append(f"Tribunal: {judge_profile.get('court', 'N/A')}")
            context_parts.append(f"Total de decisões: {judge_profile.get('total_decisions', 0)}")
            if fav.get("geral"):
                context_parts.append(f"Favorabilidade ao autor: {fav['geral'].get('autor', 'N/A')}%")
                context_parts.append(f"Favorabilidade ao réu: {fav['geral'].get('reu', 'N/A')}%")
            top_cites = judge_profile.get("top_citations", [])[:5]
            if top_cites:
                cite_strs = [f"{c.get('law', '')} ({c.get('count', 0)}x)" for c in top_cites]
                context_parts.append(f"Leis mais citadas: {', '.join(cite_strs)}")

        if prediction and not prediction.get("warning"):
            context_parts.append("\n## PREDIÇÃO ESTATÍSTICA (XGBoost)")
            context_parts.append(f"Resultado mais provável: {prediction.get('outcome', 'N/A')}")
            context_parts.append(f"Confiança: {prediction.get('confidence', 0):.1%}")
            probs = prediction.get("probabilities", {})
            if probs:
                prob_strs = [f"{k}: {v:.1%}" for k, v in probs.items()]
                context_parts.append(f"Probabilidades: {', '.join(prob_strs)}")

        if similar_cases:
            context_parts.append("\n## JURISPRUDÊNCIA SIMILAR (via RAG)")
            for i, sc in enumerate(similar_cases[:5], 1):
                context_parts.append(
                    f"\n### Caso {i}: {sc.get('document_title', 'N/A')} "
                    f"({sc.get('court', '')}, {sc.get('date', '')})"
                )
                context_parts.append(sc.get("content", "")[:800])

        user_message = "\n".join(context_parts)

        router = LLMRouter()
        response = await router.generate(
            messages=[
                {"role": "system", "content": CASE_ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            stream=False,
            tier="high",
            max_tokens=8192,
            temperature=0.3,
        )

        raw = response.get("content", "")
        model_used = response.get("model", "")

        parsed = self._parse_llm_response(raw)
        parsed["_model_used"] = model_used
        return parsed

    def _parse_llm_response(self, raw: str) -> dict[str, Any]:
        """Parse structured JSON from LLM response, with fallback."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # remove ```json
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the response
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse LLM analysis as JSON, returning raw text")
        return {
            "summary": text[:2000],
            "timeline": [],
            "legal_framework": "",
            "vulnerabilities": [],
            "strategies": [],
            "risk_level": "médio",
            "risk_assessment": "Análise não pôde ser estruturada automaticamente.",
        }
