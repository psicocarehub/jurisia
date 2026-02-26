import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.dependencies import get_current_user_optional, get_tenant_id_optional

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/demo-token")
async def demo_token():
    """Issue a demo JWT for unauthenticated frontend sessions. Disabled in production."""
    from app.config import settings
    from app.core.auth import create_access_token

    if not settings.DEBUG:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Demo tokens are disabled in production")

    token = create_access_token({
        "sub": f"demo-{uuid.uuid4().hex[:8]}",
        "tenant_id": "__system__",
        "role": "demo",
        "email": "demo@jurisai.com.br",
    })
    return {"token": token}


LEGAL_SYSTEM_PROMPT = """\
Você é o **Juris.AI**, assistente jurídico especializado no direito brasileiro.

## Estilo de resposta
- Seja **conversacional e acessível**, mas tecnicamente preciso.
- Antes de dar uma resposta completa, avalie se precisa de mais informações do usuário. \
Se a pergunta for vaga ou ambígua, faça 1-3 perguntas objetivas para entender melhor o caso \
(ex.: "Você é o autor ou réu?", "Em qual estado?", "Já houve citação?").
- Quando tiver contexto suficiente, responda de forma **estruturada em seções** usando Markdown:
  - Use `###` para títulos de seção
  - Use **negrito** para conceitos-chave e artigos de lei
  - Use listas numeradas para requisitos e passos processuais
  - Use `>` blockquote para transcrever trechos de lei ou súmula
  - Ao final, inclua uma seção `### Conclusão e Recomendações` com ações práticas

## Fundamentação
- Cite **artigos de lei** com precisão (ex.: Art. 319, CPC/2015)
- Indique **súmulas** relevantes (ex.: Súmula 331, TST)
- Referencie **jurisprudência** com tribunal e número quando disponível nas fontes
- Se houver **divergência entre tribunais**, explique as duas posições
- Indique o **grau de consolidação** da tese (pacificada, majoritária, divergente)

## Interatividade
- Ao final de respostas completas, sugira 2-3 perguntas de aprofundamento na seção:
  `### Perguntas para aprofundar`
  - Liste como bullets com frases completas

## Avisos obrigatórios
- Sempre encerre com: **⚠️ Aviso:** Este conteúdo é gerado com auxílio de IA (CNJ Res. 615/2025). \
Para um caso real, consulte um advogado.
"""


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    case_id: Optional[str] = None
    session_id: Optional[str] = None
    stream: bool = True
    use_rag: bool = True
    use_memory: bool = True


class ChatResponse(BaseModel):
    message: ChatMessage
    sources: list[dict] = []
    thinking: Optional[str] = None
    model_used: str = ""


_NOVIDADES_KEYWORDS = {
    "novidades", "novo", "nova", "novos", "novas", "recente", "recentes",
    "atualização", "atualizacao", "atualizações", "atualizacoes",
    "última", "ultima", "últimas", "ultimas", "publicado", "publicada",
    "publicados", "publicadas", "saiu", "mudança", "mudanca", "alteração",
    "alteracao",
}


async def _fetch_recent_updates(query: str, areas: list[str] | None = None, limit: int = 5) -> list[dict]:
    """Fetch recent high-relevance content_updates when query mentions new content."""
    query_lower = query.lower()
    if not any(kw in query_lower for kw in _NOVIDADES_KEYWORDS):
        return []

    try:
        import httpx
        from app.config import settings

        base_url = f"{settings.SUPABASE_URL}/rest/v1"
        headers = {
            "apikey": settings.SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_ANON_KEY}",
        }
        params = {
            "select": "title,category,subcategory,summary,court_or_organ,publication_date,source_url,relevance_score,areas",
            "relevance_score": "gte.0.7",
            "order": "captured_at.desc",
            "limit": str(limit),
            "captured_at": f"gte.{(datetime.now(timezone.utc) - timedelta(days=7)).strftime('%Y-%m-%dT00:00:00Z')}",
        }
        if areas:
            for area in areas:
                params["areas"] = f"cs.{{{area}}}"
                break

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{base_url}/content_updates", headers=headers, params=params)
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        logger.warning("Failed to fetch recent updates for chat: %s", e)
    return []


async def _build_context(
    request: ChatRequest, tenant_id: str, user_id: str
) -> tuple[str, list[dict]]:
    """Build RAG + Memory context to inject into the LLM system prompt."""
    context_parts: list[str] = []
    sources: list[dict] = []
    user_query = request.messages[-1].content if request.messages else ""

    if request.use_rag and user_query:
        try:
            from app.services.rag.retriever import HybridRetriever

            retriever = HybridRetriever()
            chunks = await retriever.retrieve(
                query=user_query,
                tenant_id=tenant_id,
                top_k=8,
                use_reranker=True,
            )
            if chunks:
                rag_section = "## Fontes Jurídicas Recuperadas\n"
                for i, c in enumerate(chunks, 1):
                    rag_section += (
                        f"\n### Fonte {i}: {c.document_title}\n"
                        f"Tribunal: {c.court} | Data: {c.date} | Tipo: {c.doc_type}\n"
                        f"{c.content[:600]}\n"
                    )
                    sources.append({
                        "title": c.document_title,
                        "court": c.court,
                        "date": c.date,
                        "doc_type": c.doc_type,
                        "score": round(c.score, 3),
                        "snippet": c.content[:200],
                    })
                context_parts.append(rag_section)
        except Exception as e:
            logger.warning("RAG retrieval failed: %s", e)

    try:
        updates = await _fetch_recent_updates(user_query)
        if updates:
            updates_section = "## Conteúdo Novo Relevante (últimos 7 dias)\n"
            for i, u in enumerate(updates, 1):
                updates_section += (
                    f"\n### Atualização {i}: {u.get('title', '')}\n"
                    f"Categoria: {u.get('category', '')} | Órgão: {u.get('court_or_organ', 'N/A')} | "
                    f"Publicação: {u.get('publication_date', 'N/A')}\n"
                    f"{(u.get('summary') or '')[:400]}\n"
                )
                if u.get('source_url'):
                    sources.append({
                        "title": u.get("title", ""),
                        "court": u.get("court_or_organ", ""),
                        "date": u.get("publication_date", ""),
                        "doc_type": u.get("category", ""),
                        "score": u.get("relevance_score", 0.0),
                        "snippet": (u.get("summary") or "")[:200],
                    })
            context_parts.append(updates_section)
    except Exception as e:
        logger.warning("Failed to add recent updates to context: %s", e)

    if request.use_memory and user_query:
        try:
            from app.services.memory.manager import MemoryManager
            from app.services.memory.graphiti import GraphitiClient
            from app.services.memory.mem0 import Mem0Client

            manager = MemoryManager(
                graphiti_client=GraphitiClient(namespace=tenant_id),
                mem0_client=Mem0Client(),
            )
            mem_ctx = await manager.assemble_context(
                tenant_id=tenant_id,
                user_id=user_id,
                case_id=request.case_id,
                session_id=request.session_id,
                query=user_query,
            )
            if not mem_ctx.is_empty:
                context_parts.append(mem_ctx.to_system_prompt_section())
        except Exception as e:
            logger.warning("Memory assembly failed: %s", e)

    return "\n\n".join(context_parts), sources


@router.post("/completions")
async def chat_completions(
    request: ChatRequest,
    user: dict = Depends(get_current_user_optional),
    tenant_id: str = Depends(get_tenant_id_optional),
):
    if request.stream:
        return StreamingResponse(
            stream_chat(request, user, tenant_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        result = await run_chat(request, user, tenant_id)
        return result


async def stream_chat(request: ChatRequest, user: dict, tenant_id: str):
    """SSE streaming with RAG + Memory context."""
    from app.services.llm.router import LLMRouter

    user_id = user.get("id", "")
    context_text, sources = await _build_context(request, tenant_id, user_id)

    system_prompt = LEGAL_SYSTEM_PROMPT
    if context_text:
        system_prompt += f"\n\n{context_text}"

    messages = [{"role": "system", "content": system_prompt}]
    messages += [{"role": m.role, "content": m.content} for m in request.messages]

    if sources:
        sources_data = json.dumps({"type": "sources", "sources": sources})
        yield f"data: {sources_data}\n\n"

    llm_router = LLMRouter()
    try:
        response = await llm_router.generate(messages=messages, stream=False)
        content = response.get("content", "")

        for i in range(0, len(content), 20):
            chunk = content[i : i + 20]
            data = json.dumps({"type": "token", "content": chunk})
            yield f"data: {data}\n\n"

    except Exception as e:
        data = json.dumps({"type": "error", "content": str(e)})
        yield f"data: {data}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


async def run_chat(request: ChatRequest, user: dict, tenant_id: str) -> ChatResponse:
    """Non-streaming chat with RAG + Memory context."""
    from app.services.llm.router import LLMRouter

    user_id = user.get("id", "")
    context_text, sources = await _build_context(request, tenant_id, user_id)

    system_prompt = LEGAL_SYSTEM_PROMPT
    if context_text:
        system_prompt += f"\n\n{context_text}"

    messages = [{"role": "system", "content": system_prompt}]
    messages += [{"role": m.role, "content": m.content} for m in request.messages]

    llm_router = LLMRouter()
    response = await llm_router.generate(messages=messages, stream=False)

    return ChatResponse(
        message=ChatMessage(role="assistant", content=response.get("content", "")),
        sources=sources,
        thinking=response.get("thinking"),
        model_used=response.get("model", ""),
    )
