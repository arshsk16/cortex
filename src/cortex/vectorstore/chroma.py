"""ChromaDB-backed vector store implementation."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from cortex.core.exceptions import DocumentProcessingError
from cortex.vectorstore.base import VectorStore
from cortex.vectorstore.models import ChunkVectorRecord, VectorSearchResult

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection

    from cortex.core.config import Settings

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """Persistent vector store using ChromaDB."""

    def __init__(self, settings: Settings) -> None:
        import chromadb

        self._collection_name = settings.chroma_collection_name
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
        self._collection: Collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def add_chunks(self, chunks: list[ChunkVectorRecord]) -> None:
        """Upsert chunk vectors and identifying metadata."""
        if not chunks:
            return
        try:
            await asyncio.to_thread(self._add_chunks_sync, chunks)
        except DocumentProcessingError:
            raise
        except Exception as exc:
            logger.exception("Failed to add chunk vectors to Chroma")
            raise DocumentProcessingError(
                "Vector storage failed",
                details={"reason": str(exc), "chunk_count": len(chunks)},
            ) from exc

    async def delete_document(self, document_id: str) -> None:
        """Delete all vectors associated with a document."""
        try:
            await asyncio.to_thread(self._delete_document_sync, document_id)
        except Exception as exc:
            logger.exception(
                "Failed to delete vectors for document_id=%s",
                document_id,
            )
            raise DocumentProcessingError(
                "Vector deletion failed",
                details={"document_id": document_id, "reason": str(exc)},
            ) from exc

    async def similarity_search(
        self,
        *,
        query_embedding: list[float],
        limit: int = 10,
        document_id: str | None = None,
    ) -> list[VectorSearchResult]:
        """Return the most similar chunk vectors to ``query_embedding``."""
        try:
            return await asyncio.to_thread(
                self._similarity_search_sync,
                query_embedding,
                limit,
                document_id,
            )
        except Exception as exc:
            logger.exception("Vector similarity search failed")
            raise DocumentProcessingError(
                "Vector similarity search failed",
                details={"reason": str(exc)},
            ) from exc

    def _add_chunks_sync(self, chunks: list[ChunkVectorRecord]) -> None:
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=[chunk.embedding for chunk in chunks],
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in chunks
            ],
        )
        logger.debug("Upserted %d chunk vectors into Chroma", len(chunks))

    def _delete_document_sync(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})
        logger.debug("Deleted Chroma vectors for document_id=%s", document_id)

    def _similarity_search_sync(
        self,
        query_embedding: list[float],
        limit: int,
        document_id: str | None,
    ) -> list[VectorSearchResult]:
        where = {"document_id": document_id} if document_id is not None else None
        response = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where,
            include=["metadatas", "distances"],
        )

        ids = response.get("ids", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: list[VectorSearchResult] = []
        for chunk_id, metadata, distance in zip(ids, metadatas, distances, strict=True):
            if metadata is None:
                continue
            results.append(
                VectorSearchResult(
                    chunk_id=chunk_id,
                    document_id=str(metadata["document_id"]),
                    chunk_index=int(metadata["chunk_index"]),
                    score=max(0.0, 1.0 - float(distance)),
                )
            )
        return results
