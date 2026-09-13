"""Unit tests for embedding providers."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from cortex.core.exceptions import DocumentProcessingError
from cortex.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider


@pytest.mark.asyncio
async def test_embed_batch_returns_list_vectors(test_settings) -> None:
    """SentenceTransformer provider converts numpy output to Python lists."""
    provider = SentenceTransformerEmbeddingProvider(test_settings)
    fake_vectors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    with patch.object(
        provider,
        "_encode_batch",
        return_value=[vector.tolist() for vector in fake_vectors],
    ):
        vectors = await provider.embed_batch(["hello", "world"])

    assert len(vectors) == 2
    assert vectors[0] == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_embed_single_delegates_to_batch(test_settings) -> None:
    """Single-text embed uses the batch code path."""
    provider = SentenceTransformerEmbeddingProvider(test_settings)

    with patch.object(
        provider,
        "embed_batch",
        new=AsyncMock(return_value=[[0.1, 0.2, 0.3]]),
    ) as mock_batch:
        vector = await provider.embed("hello")

    mock_batch.assert_awaited_once_with(["hello"])
    assert vector == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_embed_batch_empty_input(test_settings) -> None:
    """Empty input returns an empty list without calling the model."""
    provider = SentenceTransformerEmbeddingProvider(test_settings)
    assert await provider.embed_batch([]) == []


@pytest.mark.asyncio
async def test_embed_batch_wraps_unexpected_errors(test_settings) -> None:
    """Unexpected model failures raise DocumentProcessingError."""
    provider = SentenceTransformerEmbeddingProvider(test_settings)

    with (
        patch.object(
            provider,
            "_encode_batch",
            side_effect=RuntimeError("model unavailable"),
        ),
        pytest.raises(DocumentProcessingError) as exc_info,
    ):
        await provider.embed_batch(["hello"])

    assert exc_info.value.code == "document_processing_error"
