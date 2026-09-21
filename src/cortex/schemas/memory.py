"""Pydantic schemas for memory resources."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitise_text(value: str) -> str:
    """Strip null bytes and control characters from user-supplied text."""
    return _CONTROL_CHAR_RE.sub("", value)


class MemoryCreate(BaseModel):
    """Request body for creating a new memory."""

    content: str = Field(
        ..., min_length=1, max_length=10_000, description="Memory text content"
    )
    memory_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional caller-supplied metadata (tags, source, etc.)",
    )

    @field_validator("content", mode="before")
    @classmethod
    def sanitise_content(cls, value: object) -> object:
        if isinstance(value, str):
            return _sanitise_text(value)
        return value



class MemoryUpdate(BaseModel):
    """Request body for updating an existing memory."""

    content: str = Field(
        ..., min_length=1, max_length=10_000, description="Updated memory text content"
    )

    @field_validator("content", mode="before")
    @classmethod
    def sanitise_content(cls, value: object) -> object:
        if isinstance(value, str):
            return _sanitise_text(value)
        return value

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
        description="Query text for semantic search",
    )
    limit: int = Field(
        default=5, ge=1, le=50, description="Maximum number of results to return"
    )

    @field_validator("query", mode="before")
    @classmethod
    def sanitise_query(cls, value: object) -> object:
        if isinstance(value, str):
            return _sanitise_text(value)
        return value


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


class MemoryExtractionItem(BaseModel):
    """A single memory extraction action from the LLM."""

    action: str = Field(
        ..., description="Action to perform: 'create' or 'update'"
    )
    content: str = Field(
        ..., description="The new or updated fact to save"
    )
    target_memory_id: str | None = Field(
        default=None, description="The ID of the memory to update, if applicable"
    )


class MemoryExtractionResult(BaseModel):
    """Structured output from the memory extractor LLM."""

    items: list[MemoryExtractionItem]
