"""Shared internal data models for retrieval operations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A ranked chunk returned by a :class:`Retriever`.

    ``text`` is fetched from PostgreSQL; ``score`` comes from the vector
    store.  Both are always populated — callers never have to fall back to
    a secondary lookup.
    """

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    score: float
