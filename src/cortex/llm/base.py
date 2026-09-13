"""LLM provider abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


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
