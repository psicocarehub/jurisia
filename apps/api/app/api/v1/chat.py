import json
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.dependencies import get_current_user, get_tenant_id

router = APIRouter(prefix="/chat", tags=["chat"])


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
    """SSE streaming via LangGraph agent."""
    from app.services.llm.router import LLMRouter

    llm_router = LLMRouter()
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        response = await llm_router.generate(messages=messages, stream=False)

        content = response.get("content", "")
        for i in range(0, len(content), 10):
            chunk = content[i : i + 10]
            data = json.dumps({"type": "token", "content": chunk})
            yield f"data: {data}\n\n"

    except Exception as e:
        data = json.dumps({"type": "error", "content": str(e)})
        yield f"data: {data}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


async def run_chat(request: ChatRequest, user: dict, tenant_id: str) -> ChatResponse:
    """Non-streaming chat completion."""
    from app.services.llm.router import LLMRouter

    llm_router = LLMRouter()
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    response = await llm_router.generate(messages=messages, stream=False)

    return ChatResponse(
        message=ChatMessage(role="assistant", content=response.get("content", "")),
        sources=[],
        thinking=response.get("thinking"),
        model_used=response.get("model", ""),
    )
