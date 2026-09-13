"""Pydantic schemas for semantic retrieval requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    """Query payload for the semantic retrieval endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language search query",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum number of chunks to return",
    )
    document_id: str | None = Field(
        default=None,
        description=(
            "Optional document UUID to restrict search to a single owned "
            "document.  The document must have status 'ready'."
        ),
    )


class RetrievalChunkResult(BaseModel):
    """A single ranked chunk in the retrieval response."""

    chunk_id: str = Field(..., description="UUID of the document chunk")
    document_id: str = Field(..., description="UUID of the parent document")
    chunk_index: int = Field(
        ..., ge=0, description="Zero-based position within the document"
    )
    text: str = Field(..., description="Raw chunk text from PostgreSQL")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score (1 = identical, 0 = orthogonal)",
    )


class RetrievalResponse(BaseModel):
    """Response envelope for the semantic retrieval endpoint."""

    query: str = Field(..., description="The original search query")
    top_k: int = Field(..., ge=1, description="Requested result limit")
    results: list[RetrievalChunkResult] = Field(
        default_factory=list,
        description="Ranked list of matching chunks (highest score first)",
    )
    result_count: int = Field(
        ...,
        ge=0,
        description="Number of results actually returned",
    )
