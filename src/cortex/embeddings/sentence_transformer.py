"""Local sentence-transformers embedding provider."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from cortex.core.exceptions import DocumentProcessingError
from cortex.embeddings.base import EmbeddingProvider

if TYPE_CHECKING:
    from cortex.core.config import Settings

logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by ``sentence-transformers``."""

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.embedding_model_name
        self._batch_size = settings.embedding_batch_size
        self._model: object | None = None

    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for a single text input."""
        vectors = await self.embed_batch([text])
        return vectors[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for multiple texts in model batches."""
        if not texts:
            return []
        try:
            return await asyncio.to_thread(self._encode_batch, texts)
        except DocumentProcessingError:
            raise
        except Exception as exc:
            logger.exception(
                "Embedding generation failed for model=%s", self._model_name
            )
            raise DocumentProcessingError(
                "Embedding generation failed",
                details={"model": self._model_name, "reason": str(exc)},
            ) from exc

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        from sentence_transformers import SentenceTransformer

        if self._model is None:
            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)

        model = self._model
        assert isinstance(model, SentenceTransformer)

        vectors = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]
