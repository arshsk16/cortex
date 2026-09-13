"""Pydantic schemas for conversation endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class ConversationCreate(BaseModel):
    """Payload for POST /api/v1/conversations."""

    title: str | None = Field(
        default=None,
        max_length=255,
        description=(
            "Optional initial title.  If omitted the title will be generated "
            "automatically from the first user message."
        ),
    )


class ConversationRename(BaseModel):
    """Payload for PATCH /api/v1/conversations/{id}."""

    title: str = Field(..., min_length=1, max_length=255)


class ChatRequest(BaseModel):
    """Payload for POST /api/v1/conversations/{id}/chat."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="User's message for the current turn",
    )
    document_id: str | None = Field(
        default=None,
        description="Optional document UUID to restrict retrieval context",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve as context",
    )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class CitationRead(BaseModel):
    """A single richer citation returned in chat responses."""

    document_id: str
    chunk_id: str
    chunk_index: int
    document_name: str | None = Field(
        default=None,
        description="Human-readable document title, if available",
    )


class MessageRead(BaseModel):
    """A single message turn in a conversation."""

    id: str
    conversation_id: str
    role: str = Field(..., description="'user' or 'assistant'")
    content: str
    citations: list[Any] | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationRead(BaseModel):
    """Conversation summary (without messages)."""

    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConversationDetail(BaseModel):
    """Conversation with its full message history."""

    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageRead] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class ConversationList(BaseModel):
    """Paginated list of conversations."""

    conversations: list[ConversationRead]
    total: int


class ChatResponse(BaseModel):
    """Response for POST /api/v1/conversations/{id}/chat."""

    conversation_id: str
    message_id: str
    answer: str
    citations: list[CitationRead] = Field(default_factory=list)
