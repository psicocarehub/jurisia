"""
Shared state definition for the multi-agent debate pipeline.
All agents read from and write to this TypedDict.
"""

from typing import Any, TypedDict


class CitationCheck(TypedDict):
    text: str
    type: str  # legislacao, jurisprudencia, sumula
    valid: bool
    issue: str  # empty if valid


class DebateState(TypedDict):
    """State shared across all debate agents."""

    question: str
    options: dict[str, str] | None
    correct_answer: str | None
    area: str
    difficulty: str

    jurist_response: str
    jurist_thinking: str
    jurist_answer: str

    citation_checks: list[CitationCheck]
    citation_score: float  # 0.0 to 1.0
    citation_feedback: str

    tribunal_context: str
    tribunal_patterns: dict[str, Any]

    critic_feedback: str
    critic_issues: list[str]
    critic_severity: str  # none, low, medium, high

    refined_response: str
    refined_thinking: str
    refined_answer: str

    iteration: int
    max_iterations: int
    should_retry: bool

    final_trace: dict[str, Any]
