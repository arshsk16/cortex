"""Tests for Phase 9 native tool-calling path.

Covers:
1. GeminiProvider.generate_with_tools — function call response
2. GeminiProvider.generate_with_tools — text response
3. GeminiProvider.generate_with_tools — no candidates (safety filter)
4. GeminiProvider._build_gemini_tools — schema conversion
5. GeminiProvider._messages_to_contents — message history conversion
6. AgentService: native path taken when provider implements SupportsToolCalling
7. AgentService: prompt-based fallback when provider does not
8. AgentService: single tool call → observation → final answer (native)
9. AgentService: multiple tool call iterations (native)
10. AgentService: max_tool_calls limit respected (native)
11. AgentService: unknown/erroring tool — error fed back, loop continues (native)
12. AgentService: RAG tool chunks accumulated for citations (native)
13. AgentService: conversation security still enforced (native path)
14. AgentService: conversation persistence after native run
15. Regression: Phase 8 behavior unchanged when using non-tool-calling provider
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from cortex.agent.registry import ToolRegistry
from cortex.agent.service import AgentService
from cortex.agent.tools.rag_search import RAGSearchTool
from cortex.agent.types import (
    AgentMessage,
    GenerateWithToolsResult,
    ToolCallRequest,
    ToolResult,
)
from cortex.core.exceptions import (
    BadRequestError,
    ForbiddenError,
    ServiceUnavailableError,
)
from cortex.llm.base import LLMProvider, SupportsToolCalling
from cortex.retrieval.models import RetrievalResult
from cortex.services.prompt_builder import PromptBuilder

# ---------------------------------------------------------------------------
# Helpers — fake LLM providers
# ---------------------------------------------------------------------------


class _FakeNativeProvider(LLMProvider, SupportsToolCalling):
    """Provider that records generate_with_tools calls and returns preset results."""

    def __init__(
        self,
        responses: list[GenerateWithToolsResult],
        final_text: str = "Final answer.",
    ) -> None:
        self._responses = list(responses)
        self._final_text = final_text
        self.tool_calls_made: list[tuple[list[AgentMessage], list[dict]]] = []

    async def generate(self, prompt: str) -> str:
        return self._final_text

    async def generate_stream(self, prompt: str):  # type: ignore[override]
        raise NotImplementedError

    async def generate_with_tools(
        self,
        messages: list[AgentMessage],
        tool_schemas: list[dict[str, Any]],
    ) -> GenerateWithToolsResult:
        self.tool_calls_made.append((list(messages), list(tool_schemas)))
        if self._responses:
            return self._responses.pop(0)
        # Default: text response once responses exhausted
        return GenerateWithToolsResult(text="Fallback answer.")


class _FakePromptProvider(LLMProvider):
    """Plain text provider — no SupportsToolCalling — triggers prompt-based fallback."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def generate(self, prompt: str) -> str:
        if self._responses:
            return self._responses.pop(0)
        return "Prompt fallback answer."

    async def generate_stream(self, prompt: str):  # type: ignore[override]
        raise NotImplementedError


def _make_user() -> MagicMock:
    user = MagicMock()
    user.id = uuid4()
    return user


def _make_prompt_builder() -> PromptBuilder:
    pb = MagicMock(spec=PromptBuilder)
    pb.build.return_value = "grounded prompt"
    return pb


def _make_registry(*tool_names: str) -> ToolRegistry:
    """Return a registry with stub tools that return 'observation:<name>'."""
    reg = ToolRegistry()
    for name in tool_names:
        tool = MagicMock()
        tool.name = name
        tool.description = f"Stub {name}"
        tool.parameters_schema = {"type": "object", "properties": {}, "required": []}
        tool.execute = AsyncMock(return_value=f"observation:{name}")
        reg.register(tool)
    return reg


def _make_service(
    provider: LLMProvider,
    registry: ToolRegistry | None = None,
    max_tool_calls: int = 5,
    conversation_service: object = None,
) -> AgentService:
    return AgentService(
        llm_provider=provider,
        tool_registry=registry or ToolRegistry(),
        prompt_builder=_make_prompt_builder(),
        conversation_service=conversation_service,
        max_tool_calls=max_tool_calls,
    )


def _tool_call_result(
    tool_name: str,
    args: dict | None = None,
    call_id: str = "c1",
) -> GenerateWithToolsResult:
    return GenerateWithToolsResult(
        tool_call=ToolCallRequest(
            tool_name=tool_name,
            args=args or {},
            call_id=call_id,
        )
    )


def _text_result(text: str = "Native answer.") -> GenerateWithToolsResult:
    return GenerateWithToolsResult(text=text)


# ---------------------------------------------------------------------------
# 1–5: GeminiProvider unit tests (mocked SDK)
# ---------------------------------------------------------------------------


class TestGeminiProviderGenerateWithTools:
    """Unit tests for GeminiProvider.generate_with_tools with mocked SDK."""

    def _make_provider(self) -> object:
        from cortex.core.config import Settings
        from cortex.llm.gemini import GeminiProvider

        settings = Settings(
            database_url="postgresql+asyncpg://u:p@localhost/db",
            jwt_secret_key="test-secret-key-that-is-at-least-32-chars",
            gemini_api_key="fake-key",
            gemini_model_name="gemini-2.0-flash",
        )
        return GeminiProvider(settings)

    def _make_fc_response(
        self, tool_name: str, args: dict, call_id: str = "c1"
    ) -> MagicMock:
        """Build a mock Gemini response with a function_call part."""
        fc = MagicMock()
        fc.name = tool_name
        fc.args = args
        fc.id = call_id

        part = MagicMock()
        part.function_call = fc

        content = MagicMock()
        content.parts = [part]

        candidate = MagicMock()
        candidate.content = content

        response = MagicMock()
        response.candidates = [candidate]
        response.text = None
        return response

    def _make_text_response(self, text: str) -> MagicMock:
        """Build a mock Gemini response with a text part only."""
        part = MagicMock()
        part.function_call = None

        content = MagicMock()
        content.parts = [part]

        candidate = MagicMock()
        candidate.content = content

        response = MagicMock()
        response.candidates = [candidate]
        response.text = text
        return response

    def _make_empty_response(self) -> MagicMock:
        """Build a mock Gemini response with no candidates."""
        response = MagicMock()
        response.candidates = []
        response.text = None
        return response

    async def test_function_call_response_returns_tool_call_request(self) -> None:
        provider = self._make_provider()
        mock_response = self._make_fc_response(
            "calculator", {"expression": "2+2"}, "call_1"
        )

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        provider._client = mock_client

        messages = [AgentMessage(role="user", text="What is 2+2?")]
        schemas = [{"name": "calculator", "description": "calc", "parameters": {}}]

        result = await provider.generate_with_tools(
            messages=messages, tool_schemas=schemas
        )

        assert result.is_tool_call
        assert result.tool_call.tool_name == "calculator"
        assert result.tool_call.args == {"expression": "2+2"}
        assert result.tool_call.call_id == "call_1"

    async def test_text_response_returns_text_result(self) -> None:
        provider = self._make_provider()
        mock_response = self._make_text_response("  The answer is 4.  ")

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        provider._client = mock_client

        messages = [AgentMessage(role="user", text="What is 2+2?")]
        result = await provider.generate_with_tools(messages=messages, tool_schemas=[])

        assert result.is_text
        assert result.text == "The answer is 4."
        assert not result.is_tool_call

    async def test_no_candidates_returns_empty_text(self) -> None:
        provider = self._make_provider()
        mock_response = self._make_empty_response()

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        provider._client = mock_client

        messages = [AgentMessage(role="user", text="Hello")]
        result = await provider.generate_with_tools(messages=messages, tool_schemas=[])

        assert result.is_text
        assert result.text == ""

    async def test_sdk_error_raises_service_unavailable(self) -> None:
        provider = self._make_provider()

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API down")
        provider._client = mock_client

        messages = [AgentMessage(role="user", text="Hello")]
        with pytest.raises(ServiceUnavailableError):
            await provider.generate_with_tools(messages=messages, tool_schemas=[])

    def test_messages_to_contents_user_text(self) -> None:
        from cortex.llm.gemini import GeminiProvider

        msgs = [AgentMessage(role="user", text="Hello")]
        contents = GeminiProvider._messages_to_contents(msgs)
        assert len(contents) == 1
        assert contents[0].role == "user"

    def test_messages_to_contents_model_tool_call(self) -> None:
        from cortex.llm.gemini import GeminiProvider

        tc = ToolCallRequest(tool_name="calc", args={"expression": "1+1"}, call_id="c1")
        msgs = [AgentMessage(role="model", tool_call=tc)]
        contents = GeminiProvider._messages_to_contents(msgs)
        assert len(contents) == 1
        assert contents[0].role == "model"

    def test_messages_to_contents_tool_result(self) -> None:
        from cortex.llm.gemini import GeminiProvider

        tr = ToolResult(tool_name="calc", output="2", call_id="c1")
        msgs = [AgentMessage(role="tool", tool_result=tr)]
        contents = GeminiProvider._messages_to_contents(msgs)
        assert len(contents) == 1
        # Gemini maps tool results to "user" role
        assert contents[0].role == "user"

    def test_build_gemini_tools_creates_function_declarations(self) -> None:
        from cortex.llm.gemini import GeminiProvider

        schemas = [
            {
                "name": "rag_search",
                "description": "Search docs",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]
        tools = GeminiProvider._build_gemini_tools(schemas)
        assert len(tools) == 1
        # tools[0] is a types.Tool with function_declarations
        assert tools[0].function_declarations is not None
        assert len(tools[0].function_declarations) == 1
        assert tools[0].function_declarations[0].name == "rag_search"

    async def test_implements_supports_tool_calling(self) -> None:
        provider = self._make_provider()
        assert isinstance(provider, SupportsToolCalling)


# ---------------------------------------------------------------------------
# 6–7: Provider capability detection
# ---------------------------------------------------------------------------


class TestAgentServiceProviderDetection:
    async def test_native_path_taken_for_supports_tool_calling_provider(self) -> None:
        """AgentService uses _native_tool_loop when provider implements SupportsToolCalling."""  # noqa: E501
        provider = _FakeNativeProvider(
            responses=[_text_result("Native direct answer.")],
            final_text="Grounded answer.",
        )
        service = _make_service(provider)
        user = _make_user()

        result = await service.run(question="Hello", user=user)

        # generate_with_tools must have been called (native path)
        assert len(provider.tool_calls_made) == 1
        assert result.answer == "Grounded answer."

    async def test_prompt_path_taken_for_plain_provider(self) -> None:
        """AgentService falls back to prompt loop when provider has no tool calling."""
        responses = [
            '{"action": "final_answer", "answer": "Prompt answer."}',
            "Grounded prompt answer.",
        ]
        provider = _FakePromptProvider(responses)
        service = _make_service(provider)
        user = _make_user()

        result = await service.run(question="Hello", user=user)

        # Plain provider — generate_with_tools never called (it doesn't exist)
        assert not isinstance(provider, SupportsToolCalling)
        assert result.answer == "Grounded prompt answer."


# ---------------------------------------------------------------------------
# 8: Single tool call — native path
# ---------------------------------------------------------------------------


class TestNativeSingleToolCall:
    async def test_single_tool_call_then_final_answer(self) -> None:
        registry = _make_registry("calculator")
        provider = _FakeNativeProvider(
            responses=[
                _tool_call_result("calculator", {"expression": "10 * 5"}, "c1"),
                _text_result("The answer is 50."),
            ],
            final_text="Grounded: the answer is 50.",
        )
        service = _make_service(provider, registry=registry)
        user = _make_user()

        result = await service.run(question="What is 10 * 5?", user=user)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "calculator"
        assert result.tool_calls[0].args == {"expression": "10 * 5"}
        assert "observation:calculator" in result.tool_calls[0].observation
        assert result.answer == "Grounded: the answer is 50."

    async def test_message_history_includes_tool_result(self) -> None:
        """After tool dispatch, history must contain model tool-call + tool result."""
        registry = _make_registry("calculator")
        provider = _FakeNativeProvider(
            responses=[
                _tool_call_result("calculator", {"expression": "1+1"}, "c1"),
                _text_result("Done."),
            ],
            final_text="Answer.",
        )
        service = _make_service(provider, registry=registry)
        user = _make_user()

        await service.run(question="Calc", user=user)

        # Second call's messages should contain: user, model(tool_call), tool(result)
        second_call_messages = provider.tool_calls_made[1][0]
        roles = [m.role for m in second_call_messages]
        assert "user" in roles
        assert "model" in roles
        assert "tool" in roles


# ---------------------------------------------------------------------------
# 9: Multiple iterations — native path
# ---------------------------------------------------------------------------


class TestNativeMultipleIterations:
    async def test_two_tool_calls_before_text(self) -> None:
        registry = _make_registry("rag_search", "calculator")
        provider = _FakeNativeProvider(
            responses=[
                _tool_call_result("rag_search", {"query": "Paris"}, "c1"),
                _tool_call_result("calculator", {"expression": "2+2"}, "c2"),
                _text_result("Both tools used."),
            ],
            final_text="Final grounded answer.",
        )
        service = _make_service(provider, registry=registry)
        user = _make_user()

        result = await service.run(question="Multi-tool question", user=user)

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].tool_name == "rag_search"
        assert result.tool_calls[1].tool_name == "calculator"
        # generate_with_tools called 3 times: 2 tool responses + 1 text
        assert len(provider.tool_calls_made) == 3


# ---------------------------------------------------------------------------
# 10: max_tool_calls limit — native path
# ---------------------------------------------------------------------------


class TestNativeMaxToolCalls:
    async def test_loop_stops_at_max_tool_calls(self) -> None:
        """Loop must stop after max_tool_calls even if provider keeps returning tool calls."""  # noqa: E501
        registry = _make_registry("calculator")
        # Always returns a tool call — never returns text
        endless_responses = [
            _tool_call_result("calculator", {"expression": str(i)}, f"c{i}")
            for i in range(10)
        ]
        provider = _FakeNativeProvider(
            responses=endless_responses,
            final_text="Capped answer.",
        )
        service = _make_service(provider, registry=registry, max_tool_calls=3)
        user = _make_user()

        result = await service.run(question="Loop test", user=user)

        # Must have exactly max_tool_calls tool calls recorded
        assert len(result.tool_calls) == 3
        # generate_with_tools called at most max_tool_calls times
        assert len(provider.tool_calls_made) == 3


# ---------------------------------------------------------------------------
# 11: Unknown / erroring tool — native path
# ---------------------------------------------------------------------------


class TestNativeToolErrors:
    async def test_unknown_tool_error_fed_back_as_observation(self) -> None:
        """Unknown tool → BadRequestError is caught and observation carries the error."""  # noqa: E501
        registry = ToolRegistry()  # empty — no tools registered
        provider = _FakeNativeProvider(
            responses=[
                _tool_call_result("ghost_tool", {}, "c1"),
                _text_result("I cannot use that tool."),
            ],
            final_text="Error answer.",
        )
        service = _make_service(provider, registry=registry)
        user = _make_user()

        result = await service.run(question="Use unknown tool", user=user)

        assert len(result.tool_calls) == 1
        assert "error" in result.tool_calls[0].observation.lower()
        assert "ghost_tool" in result.tool_calls[0].observation.lower()

    async def test_tool_exception_does_not_crash_agent(self) -> None:
        """A tool that raises BadRequestError must not propagate — loop continues."""
        registry = _make_registry("bad_tool")
        # Override execute to raise
        tool = registry.get_tool("bad_tool")
        tool.execute = AsyncMock(side_effect=BadRequestError("Tool exploded"))

        provider = _FakeNativeProvider(
            responses=[
                _tool_call_result("bad_tool", {}, "c1"),
                _text_result("Recovered."),
            ],
            final_text="Recovered answer.",
        )
        service = _make_service(provider, registry=registry)
        user = _make_user()

        result = await service.run(question="Run bad tool", user=user)

        assert result.answer == "Recovered answer."
        assert "error" in result.tool_calls[0].observation.lower()


# ---------------------------------------------------------------------------
# 12: RAG tool citation accumulation — native path
# ---------------------------------------------------------------------------


class TestNativeRAGCitationAccumulation:
    async def test_rag_chunks_accumulated_across_iterations(self) -> None:
        """RetrievalResult objects from RAGSearchTool must be collected for citations."""  # noqa: E501
        # Build a real RAGSearchTool stub
        chunk = RetrievalResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            chunk_index=0,
            text="Paris is the capital of France.",
            score=0.9,
        )

        rag_tool = MagicMock(spec=RAGSearchTool)
        rag_tool.name = "rag_search"
        rag_tool.description = "Search"
        rag_tool.parameters_schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        rag_tool.execute = AsyncMock(return_value="Paris is the capital of France.")
        rag_tool.last_results = [chunk]

        registry = ToolRegistry()
        registry.register(rag_tool)

        provider = _FakeNativeProvider(
            responses=[
                _tool_call_result("rag_search", {"query": "Paris"}, "c1"),
                _text_result("Paris is the capital."),
            ],
            final_text="Paris is the capital of France.",
        )
        service = _make_service(provider, registry=registry)
        user = _make_user()

        result = await service.run(question="Capital of France?", user=user)

        assert len(result.retrieved_chunks) == 1
        assert result.retrieved_chunks[0].chunk_id == "chunk-1"


# ---------------------------------------------------------------------------
# 13: Conversation security — native path
# ---------------------------------------------------------------------------


class TestNativeConversationSecurity:
    async def test_ownership_check_before_any_tool_call(self) -> None:
        """Security gate must fire before generate_with_tools is ever called."""
        conv_service = MagicMock()
        conv_service.get_conversation = AsyncMock(
            side_effect=ForbiddenError("Not your conversation")
        )

        provider = _FakeNativeProvider(responses=[])
        service = _make_service(
            provider,
            conversation_service=conv_service,
        )
        user = _make_user()

        with pytest.raises(ForbiddenError):
            await service.run(question="Hello", user=user, conversation_id="conv-1")

        # generate_with_tools must NOT have been called
        assert len(provider.tool_calls_made) == 0


# ---------------------------------------------------------------------------
# 14: Conversation persistence — native path
# ---------------------------------------------------------------------------


class TestNativeConversationPersistence:
    async def test_messages_persisted_after_native_run(self) -> None:
        conv_service = MagicMock()
        conv_service.get_conversation = AsyncMock(return_value=MagicMock())
        conv_service.add_message = AsyncMock()
        conv_service.set_auto_title_if_needed = AsyncMock()
        conv_service.record_token_usage = AsyncMock()

        provider = _FakeNativeProvider(
            responses=[_text_result("Answer.")],
            final_text="Grounded.",
        )
        service = _make_service(provider, conversation_service=conv_service)
        user = _make_user()

        await service.run(question="Hello", user=user, conversation_id="conv-99")

        # add_message called twice: user + assistant
        assert conv_service.add_message.call_count == 2
        calls = conv_service.add_message.call_args_list
        assert calls[0].kwargs["role"] == "user"
        assert calls[1].kwargs["role"] == "assistant"


# ---------------------------------------------------------------------------
# 15: Regression — Phase 8 prompt-based behavior unchanged
# ---------------------------------------------------------------------------


class TestPromptBasedRegression:
    async def test_prompt_loop_tool_call_and_final_answer(self) -> None:
        """Phase 8 prompt-based path still works exactly as before."""
        tool_decision = (
            '{"action": "tool_call", "tool": "calculator",'
            ' "args": {"expression": "3*7"}}'
        )
        final_decision = '{"action": "final_answer", "answer": "The answer is 21."}'
        grounded = "Grounded: 21."

        provider = _FakePromptProvider([tool_decision, final_decision, grounded])
        registry = _make_registry("calculator")
        pb = MagicMock(spec=PromptBuilder)
        pb.build.return_value = "grounded prompt"

        service = AgentService(
            llm_provider=provider,
            tool_registry=registry,
            prompt_builder=pb,
            max_tool_calls=5,
        )
        user = _make_user()
        result = await service.run(question="What is 3*7?", user=user)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "calculator"
        assert result.answer == grounded

    async def test_prompt_loop_max_tool_calls_respected(self) -> None:
        """Prompt-based path also respects max_tool_calls."""
        # Always return tool_call, never final_answer
        endless = ['{"action": "tool_call", "tool": "calculator", "args": {}}'] * 10
        endless.append("Grounded.")
        provider = _FakePromptProvider(endless)
        registry = _make_registry("calculator")

        service = _make_service(provider, registry=registry, max_tool_calls=2)
        user = _make_user()
        result = await service.run(question="Loop test", user=user)

        assert len(result.tool_calls) == 2

    async def test_non_json_response_exits_loop_gracefully(self) -> None:
        """Non-JSON LLM response in prompt path exits gracefully."""
        provider = _FakePromptProvider(["This is not JSON at all.", "Grounded."])
        service = _make_service(provider)
        user = _make_user()
        result = await service.run(question="Test", user=user)
        assert result.answer == "Grounded."
