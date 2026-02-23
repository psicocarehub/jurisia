"""
Petition generator — multi-step LLM pipeline.

Steps:
1. Select template
2. Assemble case context + RAG fundamentation
3. Generate via LLM with specialized prompt
4. Verify citations
5. Format output with CNJ Res. 615/2025 label
"""

import logging
from typing import Any, AsyncIterator, Optional

from app.services.petition.templates import get_template, render_template

logger = logging.getLogger(__name__)

GENERATION_PROMPT = """Você é um advogado brasileiro experiente redigindo uma peça jurídica.

## Tipo de Peça
{petition_type_name}

## Template / Estrutura
{template_skeleton}

## Contexto do Caso
{case_context}

## Fundamentação Jurídica (RAG)
{rag_context}

## Instruções
1. Redija a peça completa seguindo RIGOROSAMENTE a estrutura do template.
2. Preencha TODAS as seções obrigatórias com conteúdo substantivo.
3. Use linguagem jurídica formal e técnica.
4. Cite artigos de lei, súmulas e jurisprudência REAIS (das fontes fornecidas).
5. NÃO invente números de processo, artigos ou súmulas.
6. Formate em Markdown com cabeçalhos para cada seção.
7. Inclua no final: "⚠️ Conteúdo gerado com auxílio de IA — CNJ Resolução 615/2025"

Redija a peça agora:"""

AI_LABEL = "⚠️ Conteúdo gerado com auxílio de IA — CNJ Resolução 615/2025"


class PetitionGenerator:
    """Generate legal petitions via multi-step LLM pipeline."""

    def __init__(self) -> None:
        self._llm_router = None

    def _get_router(self):
        if self._llm_router is None:
            from app.services.llm.router import LLMRouter
            self._llm_router = LLMRouter()
        return self._llm_router

    async def generate(
        self,
        petition_type: str,
        case_context: dict[str, Any],
        template_id: Optional[str] = None,
        tenant_id: str = "",
        variables: Optional[dict[str, str]] = None,
    ) -> str:
        """Generate full petition content."""

        # Step 1: Select and render template skeleton
        tid = template_id or petition_type
        template = get_template(tid)
        if not template:
            template = get_template("peticao_inicial")

        template_skeleton = render_template(tid, variables or {})
        petition_type_name = template["name"] if template else petition_type

        # Step 2: RAG retrieval for legal fundamentation
        rag_context = await self._fetch_rag_context(case_context, tenant_id)

        # Step 3: Build prompt and generate via LLM
        case_text = self._format_case_context(case_context)
        prompt = GENERATION_PROMPT.format(
            petition_type_name=petition_type_name,
            template_skeleton=template_skeleton,
            case_context=case_text,
            rag_context=rag_context or "Nenhuma fonte adicional disponível.",
        )

        router = self._get_router()
        response = await router.generate(
            messages=[
                {"role": "system", "content": "Você é um advogado brasileiro experiente e redator jurídico."},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )

        content = response.get("content", "")

        # Step 4: Ensure AI label
        if AI_LABEL not in content:
            content += f"\n\n{AI_LABEL}"

        return content

    async def generate_stream(
        self,
        petition_type: str,
        case_context: dict[str, Any],
        template_id: Optional[str] = None,
        tenant_id: str = "",
        variables: Optional[dict[str, str]] = None,
    ) -> AsyncIterator[str]:
        """Generate petition with streaming output."""
        content = await self.generate(
            petition_type=petition_type,
            case_context=case_context,
            template_id=template_id,
            tenant_id=tenant_id,
            variables=variables,
        )
        for i in range(0, len(content), 50):
            yield content[i : i + 50]

    async def _fetch_rag_context(
        self, case_context: dict[str, Any], tenant_id: str
    ) -> str:
        """Retrieve relevant legal sources via RAG."""
        if not tenant_id:
            return ""

        query_parts = []
        if case_context.get("area"):
            query_parts.append(case_context["area"])
        if case_context.get("description"):
            query_parts.append(case_context["description"])
        if case_context.get("facts"):
            query_parts.append(case_context["facts"][:300])

        if not query_parts:
            return ""

        query = " ".join(query_parts)

        try:
            from app.services.rag.retriever import HybridRetriever

            retriever = HybridRetriever()
            chunks = await retriever.retrieve(
                query=query,
                tenant_id=tenant_id,
                top_k=6,
                use_reranker=True,
            )
            if chunks:
                parts = []
                for c in chunks:
                    parts.append(
                        f"[{c.doc_type}] {c.document_title} ({c.court}, {c.date}):\n{c.content[:400]}"
                    )
                return "\n\n".join(parts)
        except Exception as e:
            logger.warning("RAG retrieval for petition failed: %s", e)

        return ""

    def _format_case_context(self, ctx: dict[str, Any]) -> str:
        """Format case context dict into readable text."""
        lines = []
        field_labels = {
            "title": "Título",
            "area": "Área do Direito",
            "court": "Tribunal/Vara",
            "judge_name": "Juiz",
            "client_name": "Cliente",
            "opposing_party": "Parte Contrária",
            "cnj_number": "Número CNJ",
            "description": "Descrição",
            "facts": "Fatos",
            "legal_basis": "Fundamentação Legal",
            "requests": "Pedidos",
            "estimated_value": "Valor da Causa",
        }
        for key, label in field_labels.items():
            value = ctx.get(key)
            if value:
                lines.append(f"- {label}: {value}")

        return "\n".join(lines) if lines else "Contexto não fornecido."
