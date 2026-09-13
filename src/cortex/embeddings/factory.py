"""Factory helpers for embedding providers."""

from __future__ import annotations

from cortex.core.config import Settings
from cortex.embeddings.base import EmbeddingProvider
from cortex.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Construct the configured embedding provider implementation."""
    return SentenceTransformerEmbeddingProvider(settings)
