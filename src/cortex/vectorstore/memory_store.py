"""MemoryVectorStore -- Chroma-backed semantic index for memory embeddings.

This is a *concrete* helper class (not an ABC) that wraps a single Chroma
collection dedicated to memory vectors.  It is separate from the document
``ChromaVectorStore`` so the two indexes never intermingle.

Key design
----------
* Each vector is stored with ``user_id`` in its metadata so that all queries
  can be filtered to the requesting user.  Cross-user leakage is impossible
  because Chroma only returns records whose ``user_id`` matches the caller.
* All blocking Chroma calls are executed via ``asyncio.to_thread`` so the
  event loop is never blocked.
* Errors propagate as ``DocumentProcessingError`` (reused for simplicity).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from cortex.core.exceptions import DocumentProcessingError

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)


class MemoryVectorStore:
    """Wraps a Chroma collection to provide async memory vector operations."""

    def __init__(self, collection: Collection) -> None:
        self._collection = collection

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def upsert(
        self,
        *,
        memory_id: str,
        user_id: str,
        embedding: list[float],
    ) -> None:
        """Store or replace the embedding for a memory entry."""
        try:
            await asyncio.to_thread(
                self._upsert_sync,
                memory_id=memory_id,
                user_id=user_id,
                embedding=embedding
            )
            logger.debug("MemoryVectorStore: upserted memory_id=%s", memory_id)
        except Exception as exc:
            logger.exception(
                "MemoryVectorStore: failed to upsert memory_id=%s", memory_id
            )
            raise DocumentProcessingError(
                "Memory vector upsert failed",
                details={"memory_id": memory_id, "reason": str(exc)},
            ) from exc

    async def delete(self, memory_id: str) -> None:
        """Delete the vector for a single memory entry."""
        try:
            await asyncio.to_thread(self._delete_sync, memory_id)
            logger.debug("MemoryVectorStore: deleted memory_id=%s", memory_id)
        except Exception as exc:
            logger.exception(
                "MemoryVectorStore: failed to delete memory_id=%s", memory_id
            )
            raise DocumentProcessingError(
                "Memory vector deletion failed",
                details={"memory_id": memory_id, "reason": str(exc)},
            ) from exc

    async def search(
        self,
        *,
        user_id: str,
        query_embedding: list[float],
        limit: int,
    ) -> list[tuple[str, float]]:
        """Return ``(memory_id, score)`` pairs ordered by descending similarity.

        Results are restricted to memories owned by ``user_id``.
        """
        try:
            return await asyncio.to_thread(
                self._search_sync,
                user_id=user_id,
                query_embedding=query_embedding,
                limit=limit,
            )
        except Exception as exc:
            logger.exception("MemoryVectorStore: similarity search failed")
            raise DocumentProcessingError(
                "Memory similarity search failed",
                details={"reason": str(exc)},
            ) from exc

    async def delete_all_for_user(self, user_id: str) -> None:
        """Delete all memory vectors belonging to ``user_id``."""
        try:
            await asyncio.to_thread(self._delete_user_sync, user_id)
            logger.debug(
                "MemoryVectorStore: deleted all vectors for user_id=%s", user_id
            )
        except Exception as exc:
            logger.exception(
                "MemoryVectorStore: failed to delete vectors for user_id=%s", user_id
            )
            raise DocumentProcessingError(
                "Memory vector user-deletion failed",
                details={"user_id": user_id, "reason": str(exc)},
            ) from exc

    # ------------------------------------------------------------------
    # Sync helpers (run inside asyncio.to_thread)
    # ------------------------------------------------------------------

    def _upsert_sync(
        self, *, memory_id: str, user_id: str, embedding: list[float]
    ) -> None:
        self._collection.upsert(
            ids=[memory_id],
            embeddings=[embedding],
            metadatas=[{"user_id": user_id}],
        )

    def _delete_sync(self, memory_id: str) -> None:
        self._collection.delete(ids=[memory_id])

    def _search_sync(
        self,
        *,
        user_id: str,
        query_embedding: list[float],
        limit: int,
    ) -> list[tuple[str, float]]:
        response = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where={"user_id": user_id},
            include=["distances"],
        )
        ids: list[str] = response.get("ids", [[]])[0]
        distances: list[float] = response.get("distances", [[]])[0]
        return [
            (mid, max(0.0, 1.0 - float(dist)))
            for mid, dist in zip(ids, distances, strict=True)
        ]

    def _delete_user_sync(self, user_id: str) -> None:
        self._collection.delete(where={"user_id": user_id})
