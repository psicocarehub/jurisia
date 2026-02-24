"""
Research agent: RAG retrieval + LLM generation.
"""

from app.services.rag.models import RetrievedChunk
from app.services.rag.retriever import HybridRetriever
from app.services.llm.router import LLMRouter
from langchain_core.messages import AIMessage


async def research_node(state: dict) -> dict:
    """Legal research agent with RAG."""
    retriever = HybridRetriever()
    router = LLMRouter()

    messages = state.get("messages", [])
    # Use last user message as query
    query = ""
    for m in reversed(messages):
        role = m.type if hasattr(m, "type") else m.get("role", "")
        if role in ("human", "user"):
            query = m.content if hasattr(m, "content") else m.get("content", "")
            break

    tenant_id = state.get("tenant_id", "")
    use_rag = state.get("use_rag", True)

    chunks: list[RetrievedChunk] = []
    if use_rag and tenant_id:
        chunks = await retriever.retrieve(
            query=query,
            tenant_id=tenant_id,
            top_k=10,
        )

    context = "\n\n---\n\n".join(
        [
            f"[{c.doc_type} | {c.court} | {c.date}]\n{c.content}"
            for c in chunks
        ]
    ) if chunks else ""

    memory_section = state.get("memory_context", "")

    system_prompt = f"""Você é um assistente jurídico especializado no direito brasileiro.
Use as fontes fornecidas para fundamentar sua resposta.
SEMPRE cite a fonte (tribunal, número do processo, data) quando referenciar jurisprudência.
Se não encontrar informação nas fontes, diga explicitamente.

{memory_section}

## Fontes Recuperadas
{context if context else "Nenhuma fonte encontrada para esta consulta."}

## Regras
- Cite artigos de lei com precisão (Art. X, Lei Y/Z)
- Indique súmulas relevantes
- Diferencie jurisprudência dominante de posições isoladas
- Se houver divergência entre tribunais, mencione
- ⚠️ Este conteúdo é gerado com auxílio de IA (CNJ Res. 615/2025)"""

    def to_role(m):
        t = (m.type if hasattr(m, "type") else m.get("role", "user"))
        return "user" if t == "human" else "assistant" if t == "ai" else str(t)

    llm_messages = [
        {"role": "system", "content": system_prompt},
        *[
            {"role": to_role(m), "content": m.content if hasattr(m, "content") else m.get("content", "")}
            for m in messages
        ],
    ]

    response = await router.generate(messages=llm_messages, stream=False)

    return {
        "messages": [AIMessage(content=response.get("content", ""))],
        "rag_results": [{"id": c.id, "title": c.document_title} for c in chunks],
        "thinking": response.get("thinking", ""),
    }
