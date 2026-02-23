"""
Supervisor agent: routes to research, drafting, analysis, memory, or chat.
"""

from app.services.llm.router import LLMRouter


async def supervisor_node(state: dict) -> dict:
    """Route to the appropriate agent based on user intent."""
    router = LLMRouter()
    messages = state.get("messages", [])
    last_message = messages[-1].content if messages else ""

    classification_prompt = f"""Classifique a intenção do usuário em uma das categorias:
- research: pesquisa jurídica, busca de jurisprudência, legislação
- drafting: redação de petição, documento jurídico
- analysis: análise de caso, risk assessment, timeline
- memory: consulta sobre casos/clientes anteriores
- chat: conversa geral, dúvidas simples

Mensagem: {last_message}

Responda APENAS com a categoria (research, drafting, analysis, memory ou chat)."""

    intent = await router.quick_classify(classification_prompt)
    intent = intent.strip().lower().split()[0] if intent else "chat"

    # Map variations to graph node names
    if "draft" in intent or "redac" in intent:
        intent = "drafting"
    elif "analy" in intent or "análise" in intent:
        intent = "analysis"
    elif "memory" in intent or "memór" in intent:
        intent = "memory"
    elif "research" in intent or "pesquis" in intent:
        intent = "research"
    else:
        intent = "chat"

    return {"current_agent": intent}
