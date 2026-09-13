"""Factory helper for LLM providers."""

from __future__ import annotations

from cortex.core.config import Settings
from cortex.llm.base import LLMProvider
from cortex.llm.gemini import GeminiProvider


def create_llm_provider(settings: Settings) -> LLMProvider:
    """Construct the configured LLM provider implementation.

    Adding a new provider requires only:
    1. Create ``cortex/llm/<provider>.py`` implementing :class:`LLMProvider`.
    2. Add a ``gemini_provider`` (or similar) settings field.
    3. Add the branch here.

    No other application code changes are needed.
    """
    return GeminiProvider(settings)
