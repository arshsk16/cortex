"""Factory helpers for vector store implementations."""

from __future__ import annotations

from cortex.core.config import Settings
from cortex.vectorstore.base import VectorStore
from cortex.vectorstore.chroma import ChromaVectorStore


def create_vector_store(settings: Settings) -> VectorStore:
    """Construct the configured vector store implementation."""
    return ChromaVectorStore(settings)
