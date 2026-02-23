"""
4-Tier Memory System orchestrator.
Tier 1: Active context (Mem0 / SuperMemory)
Tier 2: Session state (LangGraph Checkpointer)
Tier 3: Knowledge graph (Graphiti / PostgreSQL)
Tier 4: Relevant documents (RAG retrieval)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryContext:
    """Assembled memory context for LLM injection."""

    active_memory: str = ""
    session_summary: str = ""
    knowledge_facts: List[str] = field(default_factory=list)
    relevant_docs: List[dict] = field(default_factory=list)

    def to_system_prompt_section(self) -> str:
        sections = []

        if self.active_memory:
            sections.append(
                f"## Memória Ativa (Caso/Cliente Atual)\n{self.active_memory}"
            )

        if self.session_summary:
            sections.append(f"## Contexto da Sessão\n{self.session_summary}")

        if self.knowledge_facts:
            facts = "\n".join(f"- {f}" for f in self.knowledge_facts[:20])
            sections.append(f"## Fatos Relevantes\n{facts}")

        if self.relevant_docs:
            docs = "\n".join(
                f"- {d.get('title', 'Doc')}: {d.get('snippet', '')[:200]}"
                for d in self.relevant_docs[:5]
            )
            sections.append(f"## Documentos Relevantes\n{docs}")

        return "\n\n".join(sections)

    @property
    def is_empty(self) -> bool:
        return not (
            self.active_memory
            or self.session_summary
            or self.knowledge_facts
            or self.relevant_docs
        )


class MemoryManager:
    """Orchestrates 4-tier memory system."""

    def __init__(
        self,
        graphiti_client: Any = None,
        mem0_client: Any = None,
        checkpointer: Any = None,
        retriever: Any = None,
    ) -> None:
        self.graphiti = graphiti_client
        self.mem0 = mem0_client
        self.checkpointer = checkpointer
        self.retriever = retriever

    async def assemble_context(
        self,
        tenant_id: str,
        user_id: str,
        case_id: Optional[str] = None,
        session_id: Optional[str] = None,
        query: str = "",
    ) -> MemoryContext:
        """Assemble memory context from all 4 tiers."""
        context = MemoryContext()

        # Tier 1: Active memory (Mem0 / SuperMemory) -- case-scoped facts
        if self.mem0 and query:
            try:
                scope_id = f"{tenant_id}_{case_id}" if case_id else f"{tenant_id}_{user_id}"
                facts = await self.mem0.search(
                    query=query,
                    user_id=scope_id,
                    top_k=10,
                )
                if facts:
                    context.active_memory = "\n".join(
                        f"- {f.content}" for f in facts
                    )
            except Exception as e:
                logger.debug("Mem0 tier 1 failed: %s", e)

        # Tier 2: Session summary (LangGraph checkpointer)
        if self.checkpointer and session_id:
            try:
                config = {"configurable": {"thread_id": session_id}}
                checkpoint = await self.checkpointer.aget(config)
                if checkpoint and checkpoint.get("channel_values"):
                    messages = checkpoint["channel_values"].get("messages", [])
                    if messages:
                        recent = messages[-10:]
                        summary_lines = []
                        for msg in recent:
                            role = getattr(msg, "type", "unknown")
                            content = getattr(msg, "content", "")
                            if content:
                                summary_lines.append(
                                    f"[{role}] {content[:150]}"
                                )
                        context.session_summary = "\n".join(summary_lines)
            except Exception as e:
                logger.debug("Checkpointer tier 2 failed: %s", e)

        # Tier 3: Knowledge graph (Graphiti)
        if self.graphiti and query:
            try:
                kg_results = await self.graphiti.search(
                    query=query,
                    namespace=tenant_id,
                    top_k=15,
                )
                context.knowledge_facts = [
                    f"{r.fact} (fonte: {r.source or 'N/A'})"
                    for r in kg_results
                    if r.fact
                ]
            except Exception as e:
                logger.debug("Graphiti tier 3 failed: %s", e)

        # Tier 4: Relevant documents (RAG)
        if self.retriever and query:
            try:
                chunks = await self.retriever.retrieve(
                    query=query,
                    tenant_id=tenant_id,
                    top_k=5,
                    use_reranker=False,
                )
                context.relevant_docs = [
                    {
                        "title": c.document_title,
                        "snippet": c.content[:300],
                        "court": c.court,
                        "date": c.date,
                        "doc_type": c.doc_type,
                    }
                    for c in chunks
                ]
            except Exception as e:
                logger.debug("RAG tier 4 failed: %s", e)

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
                    content=fact,
                    user_id=f"{tenant_id}_{case_id}",
                    metadata={"source": source, "tenant_id": tenant_id},
                )
            except Exception as e:
                logger.debug("Mem0 store failed: %s", e)

        if self.graphiti:
            try:
                await self.graphiti.add_episode(
                    name=f"fact_{case_id}",
                    content=fact,
                    source_description=source,
                    namespace=tenant_id,
                )
            except Exception as e:
                logger.debug("Graphiti store failed: %s", e)

    async def search_facts(
        self,
        tenant_id: str,
        query: str,
        case_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Search across Mem0 and Graphiti for relevant facts."""
        mem0_results: list = []
        kg_results: list = []

        if self.mem0:
            try:
                scope = f"{tenant_id}_{case_id}" if case_id else tenant_id
                mem0_results = await self.mem0.search(
                    query=query, user_id=scope, top_k=10
                )
            except Exception:
                pass

        if self.graphiti:
            try:
                kg_results = await self.graphiti.search(
                    query=query, namespace=tenant_id, top_k=10
                )
            except Exception:
                pass

        return {
            "facts": [
                {"content": f.content, "source": "mem0"} for f in mem0_results
            ],
            "knowledge": [
                {"content": r.fact, "source": r.source, "node_id": r.node_id}
                for r in kg_results
            ],
            "query": query,
        }
