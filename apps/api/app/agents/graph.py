"""
LangGraph Multi-Agent orchestration for Juris.AI.
"""

from typing import Annotated, Literal, TypedDict

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config import settings
from app.agents.supervisor import supervisor_node
from app.agents.research import research_node
from app.agents.drafting import drafting_node
from app.agents.analysis import analysis_node
from app.agents.memory_agent import memory_node


# ============================================
# STATE DEFINITION
# ============================================


class AgentState(TypedDict):
    """State for the legal agent graph."""

    messages: Annotated[list, add_messages]
    tenant_id: str
    user_id: str
    case_id: str | None
    use_rag: bool
    use_memory: bool
    current_agent: str
    rag_results: list
    memory_context: str
    citations: list
    thinking: str


# ============================================
# ROUTING
# ============================================


def route_by_intent(
    state: AgentState,
) -> Literal["research", "drafting", "analysis", "memory", "chat"]:
    """Route from supervisor to the appropriate agent."""
    agent = state.get("current_agent", "chat")
    if agent in ("research", "drafting", "analysis", "memory"):
        return agent
    return "chat"


# ============================================
# GRAPH CONSTRUCTION
# ============================================


def build_legal_agent_graph(checkpointer: AsyncPostgresSaver | None = None):
    """Build the legal agent StateGraph with Postgres checkpointing."""
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("research", research_node)
    builder.add_node("drafting", drafting_node)
    builder.add_node("analysis", analysis_node)
    builder.add_node("memory", memory_node)
    builder.add_node("chat", research_node)  # Default: same as research

    builder.add_edge(START, "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_by_intent,
        {
            "research": "research",
            "drafting": "drafting",
            "analysis": "analysis",
            "memory": "memory",
            "chat": "chat",
        },
    )

    builder.add_edge("research", END)
    builder.add_edge("drafting", END)
    builder.add_edge("analysis", END)
    builder.add_edge("memory", END)
    builder.add_edge("chat", END)

    return builder.compile(checkpointer=checkpointer)


# Initialize graph with PostgreSQL checkpointer
_legal_agent_graph = None


async def get_legal_agent_graph():
    """Get or create the legal agent graph with Postgres checkpointing."""
    global _legal_agent_graph
    if _legal_agent_graph is None:
        # PostgresSaver expects postgresql:// (psycopg), not postgresql+asyncpg
        conn_str = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
        checkpointer = AsyncPostgresSaver.from_conn_string(conn_str)
        await checkpointer.setup()
        _legal_agent_graph = build_legal_agent_graph(checkpointer)
    return _legal_agent_graph
