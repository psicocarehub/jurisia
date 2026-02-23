"""
LangGraph debate pipeline for high-quality CoT trace generation.

Orchestrates 5 agents:
  Jurist -> Verifier -> Judge -> Critic -> Refiner
with a conditional loop (max 2 iterations) when critic finds severe issues.

Usage:
    from training.agents.debate_graph import run_debate
    trace = await run_debate(question="...", area="civil")
"""

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from training.agents.critic_agent import critic_node
from training.agents.judge_agent import judge_node
from training.agents.jurist_agent import jurist_node
from training.agents.refiner_agent import refiner_node
from training.agents.state import DebateState
from training.agents.verifier_agent import verifier_node


def _should_retry(state: DebateState) -> Literal["jurist", "end"]:
    """Decide whether to loop back to jurist or finish."""
    if state.get("should_retry", False):
        return "jurist"
    return "end"


def build_debate_graph() -> Any:
    """Build the LangGraph StateGraph for multi-agent debate."""
    builder = StateGraph(DebateState)

    builder.add_node("jurist", jurist_node)
    builder.add_node("verifier", verifier_node)
    builder.add_node("judge", judge_node)
    builder.add_node("critic", critic_node)
    builder.add_node("refiner", refiner_node)

    builder.add_edge(START, "jurist")
    builder.add_edge("jurist", "verifier")
    builder.add_edge("verifier", "judge")
    builder.add_edge("judge", "critic")
    builder.add_edge("critic", "refiner")

    builder.add_conditional_edges(
        "refiner",
        _should_retry,
        {
            "jurist": "jurist",
            "end": END,
        },
    )

    return builder.compile()


_graph = None


def get_debate_graph():
    """Get or create the debate graph singleton."""
    global _graph
    if _graph is None:
        _graph = build_debate_graph()
    return _graph


async def run_debate(
    question: str,
    options: dict[str, str] | None = None,
    correct_answer: str | None = None,
    area: str = "geral",
    difficulty: str = "medium",
    max_iterations: int = 2,
) -> dict[str, Any]:
    """
    Run the full debate pipeline for a single question.

    Returns the final trace dict ready for JSONL output.
    """
    graph = get_debate_graph()

    initial_state: DebateState = {
        "question": question,
        "options": options,
        "correct_answer": correct_answer,
        "area": area,
        "difficulty": difficulty,
        "jurist_response": "",
        "jurist_thinking": "",
        "jurist_answer": "",
        "citation_checks": [],
        "citation_score": 0.0,
        "citation_feedback": "",
        "tribunal_context": "",
        "tribunal_patterns": {},
        "critic_feedback": "",
        "critic_issues": [],
        "critic_severity": "none",
        "refined_response": "",
        "refined_thinking": "",
        "refined_answer": "",
        "iteration": 0,
        "max_iterations": max_iterations,
        "should_retry": False,
        "final_trace": {},
    }

    result = await graph.ainvoke(initial_state)
    return result.get("final_trace", {})
