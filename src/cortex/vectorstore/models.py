"""Shared data models for vector store operations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChunkVectorRecord:
    """Vector payload stored for a persisted document chunk."""

    chunk_id: str
    document_id: str
    chunk_index: int
    embedding: list[float]


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    """Similarity search hit referencing a stored chunk vector."""

    chunk_id: str
    document_id: str
    chunk_index: int
    score: float
