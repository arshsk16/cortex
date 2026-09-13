"""Pydantic schemas for memory resources."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MemoryCreate(BaseModel):
    """Request body for creating a new memory."""

    content: str = Field(
        ..., min_length=1, max_length=10_000, description="Memory text content"
    )
    memory_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional caller-supplied metadata (tags, source, etc.)",
    )


class MemoryRead(BaseModel):
    """Public memory record returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    content: str
    memory_metadata: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime


class MemoryResponse(BaseModel):
    """Single-memory API response envelope."""

    memory: MemoryRead


class MemoryList(BaseModel):
    """Paginated list of memories owned by the authenticated user."""

    items: list[MemoryRead]
    total: int = Field(..., ge=0)
    skip: int = Field(..., ge=0)
    limit: int = Field(..., ge=1)


class MemorySearchRequest(BaseModel):
    """Request body for semantic memory search."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2_000,
        description="Query text for semantic search"
    )
    limit: int = Field(
        default=5, ge=1, le=50, description="Maximum number of results to return"
    )


class MemorySearchResult(BaseModel):
    """A single semantic search hit."""

    memory: MemoryRead
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Cosine similarity score (0-1)"
    )


class MemorySearchResponse(BaseModel):
    """Response envelope for semantic search."""

    results: list[MemorySearchResult]
    query: str
