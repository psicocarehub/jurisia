"""
4-Tier Memory System orchestrator.
Tier 1: Active context | Tier 2: Session | Tier 3: Knowledge graph | Tier 4: Documents
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class MemoryContext:
    """Assembled memory context for LLM injection."""

    active_memory: str = ""  # Tier 1
    session_summary: str = ""  # Tier 2
    knowledge_facts: List[str] = field(default_factory=list)  # Tier 3
    relevant_docs: List[dict] = field(default_factory=list)  # Tier 4

    def to_system_prompt_section(self) -> str:
        """Format memory for injection into system prompt."""
        sections = []

        if self.active_memory:
            sections.append(
                f"## Memória Ativa (Caso/Cliente Atual)\n{self.active_memory}"
            )

        if self.session_summary:
            sections.append(f"## Contexto da Sessão\n{self.session_summary}")

        if self.knowledge_facts:
            facts = "\n".join(f"- {f}" for f in self.knowledge_facts[:20])
            sections.append(f"## Fatos Relevantes do Knowledge Graph\n{facts}")

        if self.relevant_docs:
            docs = "\n".join(
                f"- {d.get('title', 'Doc')}: {d.get('snippet', '')[:200]}..."
                for d in self.relevant_docs[:5]
            )
            sections.append(f"## Documentos Relevantes\n{docs}")

        return "\n\n".join(sections)


class MemoryManager:
    """Orchestrates 4-tier memory. Stub when Graphiti/Mem0 not configured."""

    def __init__(
        self,
        graphiti_client: Any = None,
        mem0_client: Any = None,
        checkpointer: Any = None,
    ) -> None:
        self.graphiti = graphiti_client
        self.mem0 = mem0_client
        self.checkpointer = checkpointer

    async def assemble_context(
        self,
        tenant_id: str,
        user_id: str,
        case_id: Optional[str] = None,
        query: str = "",
    ) -> MemoryContext:
        """Assemble memory context from all 4 tiers."""
        context = MemoryContext()

        # Tier 1: Active memory (Mem0)
        if self.mem0 and case_id:
            try:
                facts = await self.mem0.search(
                    query=query,
                    user_id=f"{tenant_id}_{case_id}",
                    limit=10,
                )
                context.active_memory = "\n".join(
                    f"- {f['memory']}" for f in facts.get("results", [])
                )
            except Exception:
                pass

        # Tier 3: Knowledge graph
        if self.graphiti and query:
            try:
                kg_results = await self.graphiti.search(
                    query=query,
                    namespace=tenant_id,
                    limit=15,
                )
                context.knowledge_facts = [
                    f"{r['fact']} (fonte: {r.get('source', 'N/A')})"
                    for r in kg_results
                ]
            except Exception:
                pass

        return context

    async def store_fact(
        self,
        tenant_id: str,
        case_id: str,
        fact: str,
        source: str = "conversation",
    ) -> None:
        """Store a new fact in Mem0 + Graphiti."""
        if self.mem0:
            try:
                await self.mem0.add(
                    messages=[{"role": "assistant", "content": fact}],
                    user_id=f"{tenant_id}_{case_id}",
                    metadata={"source": source, "tenant_id": tenant_id},
                )
            except Exception:
                pass

        if self.graphiti:
            try:
                await self.graphiti.add_episode(
                    name=f"fact_{case_id}",
                    episode_body=fact,
                    source_description=source,
                    namespace=tenant_id,
                )
            except Exception:
                pass
