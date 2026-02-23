"""
Indexer service (stub). Indexes documents and process metadata to ES + Qdrant.
"""

from typing import Any


class Indexer:
    """Index documents and process metadata to Elasticsearch and Qdrant."""

    async def index_document(
        self,
        document_id: str,
        tenant_id: str,
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        """Index a document and its chunks. Stub implementation."""
        # TODO: chunk, embed, index to ES + Qdrant
        pass

    async def index_process(
        self,
        process: Any,
    ) -> None:
        """Index process metadata (DataJud etc). Stub implementation."""
        # TODO: index process metadata to ES
        pass
