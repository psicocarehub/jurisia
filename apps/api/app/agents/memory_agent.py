"""
Memory agent: query and update cross-session memory.
"""

from app.services.memory.manager import MemoryManager
from app.services.llm.router import LLMRouter
from langchain_core.messages import AIMessage


async def memory_node(state: dict) -> dict:
    """Memory management agent: query facts, store new facts."""
    manager = MemoryManager()  # Stub: no graphiti/mem0
    router = LLMRouter()

    messages = state.get("messages", [])
    query = messages[-1].content if messages else ""

    tenant_id = state.get("tenant_id", "")
    case_id = state.get("case_id") or ""
    user_id = state.get("user_id", "")

    context = await manager.assemble_context(
        tenant_id=tenant_id,
        user_id=user_id,
        case_id=case_id,
        query=query,
    )

    memory_section = context.to_system_prompt_section()

    system_prompt = f"""Você é um assistente que consulta a memória do escritório.
Use as informações de memória fornecidas para responder à consulta.
Se não houver informações relevantes, informe isso ao usuário.

{memory_section}

⚠️ Este conteúdo é gerado com auxílio de IA (CNJ Res. 615/2025)"""

    def to_role(m):
        t = m.type if hasattr(m, "type") else m.get("role", "user")
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
        "memory_context": memory_section,
    }
