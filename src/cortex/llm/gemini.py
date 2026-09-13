"""Google Gemini LLM provider implementation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from cortex.core.exceptions import ServiceUnavailableError
from cortex.llm.base import LLMProvider

if TYPE_CHECKING:
    from cortex.core.config import Settings

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """LLM provider backed by the Google GenAI SDK (``google-genai``).

    The client is lazy-initialised on first call so that tests that mock
    this provider never touch the network.

    Configuration is read from :class:`~cortex.core.config.Settings`:

    * ``gemini_api_key``     — Google AI Studio / Vertex API key.
    * ``gemini_model_name``  — Model identifier, e.g. ``gemini-2.0-flash``.
    * ``gemini_temperature`` — Sampling temperature (0.0 – 2.0).
    * ``gemini_max_output_tokens`` — Upper bound on response length.
    """

    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.gemini_api_key
        self._model_name = settings.gemini_model_name
        self._temperature = settings.gemini_temperature
        self._max_output_tokens = settings.gemini_max_output_tokens
        self._client: object | None = None  # lazy-init

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> object:
        """Return (and cache) the google.genai Client."""
        if self._client is None:
            import google.genai as genai  # type: ignore[import-untyped]

            self._client = genai.Client(api_key=self._api_key)
            logger.info("Initialised Gemini client (model=%s)", self._model_name)
        return self._client

    def _build_config(self) -> object:
        """Build a GenerateContentConfig for the configured parameters."""
        from google.genai import types  # type: ignore[import-untyped]

        return types.GenerateContentConfig(
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
        )

    def _generate_sync(self, prompt: str) -> str:
        """Synchronous Gemini call executed in a thread pool."""
        client = self._get_client()
        config = self._build_config()

        # google.genai ≥ 1.0  API surface
        response = client.models.generate_content(  # type: ignore[attr-defined]
            model=self._model_name,
            contents=prompt,
            config=config,
        )
        return response.text.strip()

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    async def generate(self, prompt: str) -> str:
        """Return the full Gemini completion for ``prompt``."""
        try:
            text = await asyncio.to_thread(self._generate_sync, prompt)
            logger.debug(
                "Gemini generation complete (model=%s, chars=%d)",
                self._model_name,
                len(text),
            )
            return text
        except ServiceUnavailableError:
            raise
        except Exception as exc:
            logger.exception("Gemini generation failed (model=%s)", self._model_name)
            raise ServiceUnavailableError(
                "LLM generation failed",
                details={"model": self._model_name, "reason": str(exc)},
            ) from exc

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        """Yield successive text chunks from Gemini streaming."""

        async def _stream() -> AsyncIterator[str]:
            try:
                client = self._get_client()
                config = self._build_config()

                # google.genai streaming runs synchronously; run in thread.
                def _iter_sync() -> list[str]:
                    chunks: list[str] = []
                    for chunk in client.models.generate_content_stream(  # type: ignore[attr-defined]
                        model=self._model_name,
                        contents=prompt,
                        config=config,
                    ):
                        if chunk.text:
                            chunks.append(chunk.text)
                    return chunks

                parts = await asyncio.to_thread(_iter_sync)
                for part in parts:
                    yield part
            except ServiceUnavailableError:
                raise
            except Exception as exc:
                logger.exception(
                    "Gemini streaming failed (model=%s)", self._model_name
                )
                raise ServiceUnavailableError(
                    "LLM streaming failed",
                    details={"model": self._model_name, "reason": str(exc)},
                ) from exc

        return _stream()
