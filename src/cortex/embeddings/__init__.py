"""Embedding provider abstractions and implementations."""

from cortex.embeddings.base import EmbeddingProvider
from cortex.embeddings.factory import create_embedding_provider
from cortex.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "create_embedding_provider",
]
