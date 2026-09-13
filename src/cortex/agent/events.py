"""Provider-neutral SSE event types for the streaming agent endpoint.

Each event is a frozen dataclass serialisable to a single SSE ``data:`` line.
The :func:`format_sse` helper converts any event to the wire format::

    data: {"type": "...", ...}\\n\\n

Event types
-----------
``tool_call``
    Emitted immediately after the LLM selects a tool (before execution).
    Lets the client display a "calling tool …" spinner.

``tool_result``
    Emitted after a tool returns its observation.
    ``observation_preview`` is bounded to :data:`_PREVIEW_MAX_CHARS` so that
    large RAG excerpts do not flood the SSE stream with unformatted text.
    The full observation is still recorded in
    :class:`~cortex.agent.result.ToolCallRecord`
    and included in the ``done`` event.

``token``
    One text chunk of the final grounded-answer LLM response.

``done``
    Signals successful completion.  Fields mirror the non-streaming
    :class:`~cortex.schemas.agent.AgentResponse` so clients can share
    handling code.

``error``
    Unrecoverable failure.  The stream closes immediately after this event.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Maximum characters sent in a ``tool_result`` observation_preview.
# Keeps SSE events bounded regardless of RAG chunk size.
_PREVIEW_MAX_CHARS = 500


# ---------------------------------------------------------------------------
# Event dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolCallEvent:
    """Agent selected a tool — emitted before tool execution."""

    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    type: str = field(default="tool_call", init=False)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "tool_name": self.tool_name, "args": self.args}


@dataclass(frozen=True, slots=True)
class ToolResultEvent:
    """Tool returned its observation — emitted after tool execution.

    ``observation_preview`` is the first :data:`_PREVIEW_MAX_CHARS` characters
    of the raw observation.  The full text is preserved in the agent's state
    and returned in the :class:`DoneEvent`.
    """

    tool_name: str
    observation_preview: str
    type: str = field(default="tool_result", init=False)

    @classmethod
    def from_observation(
        cls, tool_name: str, observation: str
    ) -> ToolResultEvent:
        """Construct from a full observation, truncating if necessary."""
        preview = observation[:_PREVIEW_MAX_CHARS]
        if len(observation) > _PREVIEW_MAX_CHARS:
            preview += "…"
        return cls(tool_name=tool_name, observation_preview=preview)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "tool_name": self.tool_name,
            "observation_preview": self.observation_preview,
        }


@dataclass(frozen=True, slots=True)
class TokenEvent:
    """One text chunk from the final streamed LLM answer."""

    text: str
    type: str = field(default="token", init=False)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "text": self.text}


@dataclass(frozen=True, slots=True)
class DoneEvent:
    """Signals successful completion.

    Field names mirror :class:`~cortex.schemas.agent.AgentResponse` so
    clients can share deserialisation code between the streaming and
    non-streaming paths.
    """

    answer: str
    tool_calls_made: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    conversation_id: str | None = None
    type: str = field(default="done", init=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "answer": self.answer,
            "tool_calls_made": self.tool_calls_made,
            "citations": self.citations,
            "conversation_id": self.conversation_id,
        }


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    """Unrecoverable failure.  The stream closes after this event."""

    message: str
    type: str = field(default="error", init=False)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "message": self.message}


# ---------------------------------------------------------------------------
# Union type alias
# ---------------------------------------------------------------------------

AgentEvent = ToolCallEvent | ToolResultEvent | TokenEvent | DoneEvent | ErrorEvent


# ---------------------------------------------------------------------------
# SSE serialiser
# ---------------------------------------------------------------------------


def format_sse(event: AgentEvent) -> str:
    """Serialise an agent event to an SSE ``data:`` line.

    Returns a string of the form::

        data: {"type": "...", ...}\\n\\n

    The double newline is required by the SSE specification to delimit events.
    """
    return f"data: {json.dumps(event.to_dict(), ensure_ascii=False)}\n\n"
