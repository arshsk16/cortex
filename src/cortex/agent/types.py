"""Provider-neutral agent types for Phase 9 native tool calling.

These dataclasses are the contract between AgentService and any
LLMProvider that supports structured tool calling.  They contain
no provider-specific types so AgentService remains independent of
the Gemini SDK (or any other SDK).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolCallRequest:
    """A single tool call requested by the LLM.

    Attributes
    ----------
    tool_name:
        Name of the tool to invoke.  Must match a registered tool.
    args:
        Argument dict to pass to the tool.
    call_id:
        Provider-assigned identifier linking this request to its response.
        Used in multi-turn message history so the provider can correlate
        function responses to the function calls that triggered them.
    """

    tool_name: str
    args: dict[str, Any]
    call_id: str


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The result of executing a tool, ready to be sent back to the LLM.

    Attributes
    ----------
    tool_name:
        Name of the tool that produced the result.
    output:
        Plain-text observation returned by the tool.
    call_id:
        Must match the ``call_id`` of the originating
        :class:`ToolCallRequest`.
    """

    tool_name: str
    output: str
    call_id: str


@dataclass(frozen=True)
class AgentMessage:
    """A single turn in the agent's multi-turn conversation history.

    Only one of ``text``, ``tool_call``, or ``tool_result`` should be
    set per message.

    Attributes
    ----------
    role:
        ``"user"``  — a user or tool-result turn.
        ``"model"`` — a model turn, optionally including a tool call.
        ``"tool"``  — a tool-result turn (mapped by the provider).
    text:
        Plain-text content for ``"user"`` or final ``"model"`` messages.
    tool_call:
        Set on ``"model"`` messages when the model requests a tool.
    tool_result:
        Set on ``"tool"`` messages when a tool execution result is
        being sent back to the model.
    """

    role: str
    text: str | None = None
    tool_call: ToolCallRequest | None = None
    tool_result: ToolResult | None = None


@dataclass
class GenerateWithToolsResult:
    """The outcome of a single :meth:`SupportsToolCalling.generate_with_tools`
    call.

    Exactly one of ``tool_call`` or ``text`` will be set.

    Attributes
    ----------
    tool_call:
        Set when the LLM requests a tool invocation.
    text:
        Set when the LLM produces a final text answer (no tool needed).
    """

    tool_call: ToolCallRequest | None = None
    text: str | None = None

    @property
    def is_tool_call(self) -> bool:
        """True when the LLM has requested a tool invocation."""
        return self.tool_call is not None

    @property
    def is_text(self) -> bool:
        """True when the LLM has produced a text response."""
        return self.text is not None
