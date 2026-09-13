"""Pydantic schemas for the agent endpoint."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    """Request payload for POST /api/v1/agent/run."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="User question or instruction for the agent",
    )
    conversation_id: str | None = Field(
        default=None,
        description=(
            "Optional conversation UUID.  When supplied the agent enforces "
            "ownership and persists the message pair."
        ),
    )


class AgentToolCallRead(BaseModel):
    """A single tool invocation record included in the agent response."""

    tool_name: str = Field(..., description="Name of the tool that was called")
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments that were passed to the tool",
    )
    observation: str = Field(
        ...,
        description="Plain-text result returned by the tool",
    )


class AgentCitationRead(BaseModel):
    """Source citation from an agent RAG search."""

    document_id: str
    chunk_id: str
    chunk_index: int


class AgentResponse(BaseModel):
    """Response envelope for POST /api/v1/agent/run."""

    answer: str = Field(
        ...,
        description="The agent's final grounded answer",
    )
    tool_calls_made: list[AgentToolCallRead] = Field(
        default_factory=list,
        description="Ordered list of tool invocations made during this run",
    )
    citations: list[AgentCitationRead] = Field(
        default_factory=list,
        description="Source citations from document retrieval",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Conversation ID if a conversation was supplied in the request",
    )
