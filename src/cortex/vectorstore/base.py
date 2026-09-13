"""Vector store abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

from cortex.vectorstore.models import ChunkVectorRecord, VectorSearchResult


class VectorStore(ABC):
    """Persist and query chunk embedding vectors.

    Relational chunk text remains in PostgreSQL; vector stores hold embeddings
    and the minimum metadata required to map hits back to SQL rows.
    """

    @abstractmethod
    async def add_chunks(self, chunks: list[ChunkVectorRecord]) -> None:
        """Upsert chunk vectors and identifying metadata."""

    @abstractmethod
    async def delete_document(self, document_id: str) -> None:
        """Delete all vectors associated with a document."""

    @abstractmethod
    async def similarity_search(
        self,
        *,
        query_embedding: list[float],
        limit: int = 10,
        document_id: str | None = None,
    ) -> list[VectorSearchResult]:
        """Return the most similar chunk vectors to ``query_embedding``."""
