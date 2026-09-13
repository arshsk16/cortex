"""Embedding provider abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Generate dense vector representations for text.

    Application services depend on this interface so providers such as OpenAI,
    Gemini, or Voyage can be swapped without changing ingestion logic.
    """

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for a single text input."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for multiple texts in model batches."""
