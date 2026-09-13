"""LLM provider abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cortex.agent.types import AgentMessage, GenerateWithToolsResult


class LLMProvider(ABC):
    """Generate text completions from a language model.

    Application code (RAGService, PromptBuilder, endpoints) depends only on
    this interface.  Concrete implementations — Gemini, OpenAI, Anthropic —
    can be swapped in ``deps.py`` without touching any business logic.

    Only ``generate`` is required for Phase 6.  ``generate_stream`` is defined
    here so future providers can expose incremental streaming without a new
    abstraction layer.
    """

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Return the model's full text completion for ``prompt``.

        Parameters
        ----------
        prompt:
            The fully-assembled prompt string.  The caller is responsible
            for structuring the prompt (system instructions, context, question).

        Returns
        -------
        str
            The model's complete response text, stripped of leading/trailing
            whitespace.
        """

    @abstractmethod
    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        """Yield successive chunks of the model's response for ``prompt``.

        Implementations must yield non-empty strings.  The caller can
        concatenate chunks to reconstruct the full response.

        Note: The Phase 6 RAG endpoint uses ``generate`` (non-streaming).
        This method exists so future streaming endpoints do not require a
        new interface.
        """
        raise NotImplementedError
        # This makes the method an async generator in type stubs without
        # requiring a real yield in the abstract base.
        # Subclasses must return an AsyncIterator[str] directly.


class SupportsToolCalling(ABC):
    """Mixin for LLM providers that support native structured tool calling.

    Providers that implement this mixin allow
    :class:`~cortex.agent.service.AgentService`
    to use native function-calling APIs instead of prompt-based JSON parsing.

    The interface is intentionally provider-agnostic: it works entirely with
    :mod:`cortex.agent.types` dataclasses so ``AgentService`` remains
    decoupled from any specific SDK.

    Usage in ``AgentService``::

        if isinstance(self._llm, SupportsToolCalling):
            # Use native path
        else:
            # Fallback to prompt-based JSON path
    """

    @abstractmethod
    async def generate_with_tools(
        self,
        messages: list[AgentMessage],
        tool_schemas: list[dict[str, Any]],
    ) -> GenerateWithToolsResult:
        """Run one turn of a tool-calling conversation.

        Parameters
        ----------
        messages:
            Ordered conversation history in provider-neutral form.
            The caller maintains and extends this list between turns.
        tool_schemas:
            List of provider-neutral tool descriptors in the format::

                {
                    "name": "tool_name",
                    "description": "What the tool does.",
                    "parameters": { <JSON Schema object> },
                }

        Returns
        -------
        GenerateWithToolsResult
            Either a :class:`~cortex.agent.types.ToolCallRequest` (model
            wants to call a tool) or a ``text`` response (model is done).
        """
