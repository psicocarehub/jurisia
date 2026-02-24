from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    """Chunk retrieved from RAG search."""

    id: str
    content: str
    score: float
    document_id: str
    document_title: str
    doc_type: str
    court: str
    date: str
    metadata: dict
