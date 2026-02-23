"""
Drafting agent: petition generation with citation verification.
"""

from app.services.llm.router import LLMRouter
from app.services.petition.citation_verifier import CitationVerifier
from app.services.llm.prompts import DRAFTING_SYSTEM_PROMPT
from langchain_core.messages import AIMessage


async def drafting_node(state: dict) -> dict:
    """Petition drafting agent."""
    router = LLMRouter()
    verifier = CitationVerifier()

    messages = state.get("messages", [])

    def to_role(m):
        t = m.type if hasattr(m, "type") else m.get("role", "user")
        return "user" if t == "human" else "assistant" if t == "ai" else str(t)

    llm_messages = [
        {"role": "system", "content": DRAFTING_SYSTEM_PROMPT},
        *[
            {"role": to_role(m), "content": m.content if hasattr(m, "content") else m.get("content", "")}
            for m in messages
        ],
    ]

    response = await router.generate(
        messages=llm_messages,
        stream=False,
        tier="high",
    )

    content = response.get("content", "")

    # Verify citations in generated text
    citations = await verifier.verify_all(content)

    return {
        "messages": [AIMessage(content=content)],
        "citations": [c.model_dump() for c in citations],
    }
