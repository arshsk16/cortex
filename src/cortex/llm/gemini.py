"""Google Gemini LLM provider implementation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from cortex.agent.types import (
    AgentMessage,
    GenerateWithToolsResult,
    ToolCallRequest,
)
from cortex.core.exceptions import ServiceUnavailableError
from cortex.llm.base import LLMProvider, SupportsToolCalling

if TYPE_CHECKING:
    from cortex.core.config import Settings

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider, SupportsToolCalling):
    """LLM provider backed by the Google GenAI SDK (``google-genai``).

    Implements both :class:`~cortex.llm.base.LLMProvider` (plain text
    generation) and :class:`~cortex.llm.base.SupportsToolCalling` (native
    structured function calling).

    The client is lazy-initialised on first call so that tests that mock
    this provider never touch the network.

    Configuration is read from :class:`~cortex.core.config.Settings`:

    * ``gemini_api_key``            — Google AI Studio / Vertex API key.
    * ``gemini_model_name``         — Model identifier, e.g. ``gemini-2.0-flash``.
    * ``gemini_temperature``        — Sampling temperature (0.0 – 2.0).
    * ``gemini_max_output_tokens``  — Upper bound on response length.
    """

    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.gemini_api_key
        self._model_name = settings.gemini_model_name
        self._temperature = settings.gemini_temperature
        self._max_output_tokens = settings.gemini_max_output_tokens
        self._client: object | None = None  # lazy-init

    # ------------------------------------------------------------------
    # Internal helpers — shared
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

    # ------------------------------------------------------------------
    # SupportsToolCalling interface
    # ------------------------------------------------------------------

    async def generate_with_tools(
        self,
        messages: list[AgentMessage],
        tool_schemas: list[dict[str, Any]],
    ) -> GenerateWithToolsResult:
        """Run one turn of native Gemini function calling.

        Converts provider-neutral :class:`~cortex.agent.types.AgentMessage`
        objects into ``google.genai.types.Content`` objects, attaches
        :class:`~google.genai.types.FunctionDeclaration` schemas, calls the
        API, and returns a provider-neutral
        :class:`~cortex.agent.types.GenerateWithToolsResult`.

        Parameters
        ----------
        messages:
            Ordered conversation history (user turns, model function calls,
            tool results).
        tool_schemas:
            Provider-neutral tool descriptors — each has ``name``,
            ``description``, and ``parameters`` (JSON Schema).

        Returns
        -------
        GenerateWithToolsResult
            Either a :class:`~cortex.agent.types.ToolCallRequest` or a
            plain-text answer.
        """
        try:
            result = await asyncio.to_thread(
                self._generate_with_tools_sync, messages, tool_schemas
            )
            return result
        except ServiceUnavailableError:
            raise
        except Exception as exc:
            logger.exception(
                "Gemini generate_with_tools failed (model=%s)", self._model_name
            )
            raise ServiceUnavailableError(
                "LLM tool-calling generation failed",
                details={"model": self._model_name, "reason": str(exc)},
            ) from exc

    def _generate_with_tools_sync(
        self,
        messages: list[AgentMessage],
        tool_schemas: list[dict[str, Any]],
    ) -> GenerateWithToolsResult:
        """Synchronous implementation of native Gemini function calling."""
        from google.genai import types  # type: ignore[import-untyped]

        client = self._get_client()
        contents = self._messages_to_contents(messages)
        tools = self._build_gemini_tools(tool_schemas)

        config = types.GenerateContentConfig(
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
            tools=tools,
        )

        response = client.models.generate_content(  # type: ignore[attr-defined]
            model=self._model_name,
            contents=contents,
            config=config,
        )

        return self._parse_tool_response(response)

    # ------------------------------------------------------------------
    # Private helpers — tool calling
    # ------------------------------------------------------------------

    @staticmethod
    def _build_gemini_tools(
        tool_schemas: list[dict[str, Any]],
    ) -> list[object]:
        """Convert provider-neutral tool schemas to a Gemini Tool object list.

        Each schema is converted to a ``FunctionDeclaration`` using
        ``parameters_json_schema`` so the raw JSON Schema dict is passed
        directly to the SDK without manual Schema object construction.
        """
        from google.genai import types  # type: ignore[import-untyped]

        declarations = []
        for schema in tool_schemas:
            declarations.append(
                types.FunctionDeclaration(
                    name=schema["name"],
                    description=schema.get("description", ""),
                    parameters_json_schema=schema.get("parameters", {}),
                )
            )

        return [types.Tool(function_declarations=declarations)]

    @staticmethod
    def _messages_to_contents(messages: list[AgentMessage]) -> list[object]:
        """Convert provider-neutral AgentMessages to Gemini Content objects.

        Message role mapping
        --------------------
        * ``role="user"`` with ``text``       → Content(role="user", text)
        * ``role="model"`` with ``tool_call`` → Content(role="model", function_call)
        * ``role="tool"`` with ``tool_result``→ Content(role="user", function_response)
        """
        from google.genai import types  # type: ignore[import-untyped]

        contents = []
        for msg in messages:
            if msg.role == "user" and msg.text is not None:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=msg.text)],
                    )
                )
            elif msg.role == "model" and msg.tool_call is not None:
                tc = msg.tool_call
                contents.append(
                    types.Content(
                        role="model",
                        parts=[
                            types.Part(
                                function_call=types.FunctionCall(
                                    name=tc.tool_name,
                                    args=tc.args,
                                    id=tc.call_id,
                                )
                            )
                        ],
                    )
                )
            elif msg.role == "model" and msg.text is not None:
                # Final model text turn (e.g., when replaying history)
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part(text=msg.text)],
                    )
                )
            elif msg.role == "tool" and msg.tool_result is not None:
                tr = msg.tool_result
                # Gemini expects function responses in a "user" role Content
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=tr.tool_name,
                                    response={"output": tr.output},
                                    id=tr.call_id,
                                )
                            )
                        ],
                    )
                )
        return contents

    @staticmethod
    def _parse_tool_response(response: object) -> GenerateWithToolsResult:
        """Extract a GenerateWithToolsResult from a Gemini response object.

        Checks the first candidate's parts for a ``function_call``.  If found,
        returns a :class:`~cortex.agent.types.ToolCallRequest`.  Otherwise
        returns the response text.
        """
        # Safely access candidates
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            # Safety-filtered or empty — return empty text so the loop exits
            logger.warning("Gemini returned no candidates in tool-calling response")
            return GenerateWithToolsResult(text="")

        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []

        for i, part in enumerate(parts):
            fc = getattr(part, "function_call", None)
            if fc is not None and getattr(fc, "name", None):
                call_id = getattr(fc, "id", None) or f"call_{fc.name}_{i}"
                raw_args = getattr(fc, "args", None) or {}
                return GenerateWithToolsResult(
                    tool_call=ToolCallRequest(
                        tool_name=fc.name,
                        args=dict(raw_args),
                        call_id=call_id,
                    )
                )

        # No function call found — extract text answer
        text = getattr(response, "text", None) or ""
        return GenerateWithToolsResult(text=text.strip())
