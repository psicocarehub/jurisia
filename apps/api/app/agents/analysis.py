"""
Analysis agent: case analysis, timeline, risk assessment.
"""

from app.services.llm.router import LLMRouter
from app.services.llm.prompts import ANALYSIS_SYSTEM_PROMPT
from langchain_core.messages import AIMessage


async def analysis_node(state: dict) -> dict:
    """Case analysis agent."""
    router = LLMRouter()

    messages = state.get("messages", [])

    def to_role(m):
        t = m.type if hasattr(m, "type") else m.get("role", "user")
        return "user" if t == "human" else "assistant" if t == "ai" else str(t)

    llm_messages = [
        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
        *[
            {"role": to_role(m), "content": m.content if hasattr(m, "content") else m.get("content", "")}
            for m in messages
        ],
    ]

    response = await router.generate(
        messages=llm_messages,
        stream=False,
        tier="medium",
    )

    return {
        "messages": [AIMessage(content=response.get("content", ""))],
    }
