import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.dependencies import get_current_user, get_tenant_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

LEGAL_SYSTEM_PROMPT = (
    "Você é o Juris.AI, assistente jurídico especializado no direito brasileiro. "
    "Responda com precisão, citando artigos de lei, súmulas e jurisprudência quando pertinente. "
    "Sempre indique quando houver divergência entre tribunais. "
    "⚠️ Conteúdo gerado com auxílio de IA — CNJ Resolução 615/2025."
)


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
    user: dict = Depends(get_current_user),
    tenant_id: str = Depends(get_tenant_id),
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
