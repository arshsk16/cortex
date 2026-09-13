"""Tests for Phase 10: AgentState & Conversation Memory.

Covers:
1. AgentState construction — fields, defaults, properties
2. AgentState.elapsed_ms and has_history
3. History loading — correct limit applied, ownership enforced
4. History loaded BEFORE user message is persisted (so it excludes current turn)
5. History passed to PromptBuilder.build() in grounded answer step
6. History NOT passed to tool-calling loop (memory boundary preserved)
7. Multi-turn context: second question can see first answer in history
8. History limit prevents unbounded growth
9. Stateless run (no conversation_id): no history loaded, PromptBuilder gets None
10. conversation_history_limit=0: no history loaded even with conversation
11. AgentState.retrieved_chunks accumulates across tool calls
12. AgentState.tool_calls accumulates across tool calls
13. Ownership check fires before any tool call (security preserved)
14. Conversation persistence still works (regression)
15. Native tool-calling path still writes to state (regression)
16. Prompt-based fallback still writes to state (regression)
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from cortex.agent.registry import ToolRegistry
from cortex.agent.result import ToolCallRecord
from cortex.agent.service import AgentService
from cortex.agent.state import AgentState
from cortex.agent.tools.rag_search import RAGSearchTool
from cortex.agent.types import GenerateWithToolsResult, ToolCallRequest
from cortex.core.exceptions import ForbiddenError
from cortex.llm.base import LLMProvider, SupportsToolCalling
from cortex.retrieval.models import RetrievalResult
from cortex.services.prompt_builder import PromptBuilder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user() -> MagicMock:
    user = MagicMock()
    user.id = str(uuid4())
    return user


def _make_message(role: str, content: str) -> MagicMock:
    msg = MagicMock()
    msg.role = role
    msg.content = content
    return msg


def _make_conv_service(
    history: list | None = None,
    forbidden: bool = False,
) -> MagicMock:
    """Build a mock ConversationService."""
    svc = MagicMock()
    if forbidden:
        svc.get_history = AsyncMock(
            side_effect=ForbiddenError("Not your conversation")
        )
    else:
        svc.get_history = AsyncMock(return_value=history or [])
    svc.get_conversation = AsyncMock(return_value=MagicMock())
    svc.add_message = AsyncMock()
    svc.set_auto_title_if_needed = AsyncMock()
    svc.record_token_usage = AsyncMock()
    return svc


def _make_prompt_builder(captured: list | None = None) -> PromptBuilder:
    """Return a PromptBuilder mock that captures build() kwargs."""
    pb = MagicMock(spec=PromptBuilder)

    def _capture(**kwargs: Any) -> str:
        if captured is not None:
            captured.append(kwargs)
        return "grounded prompt"

    pb.build.side_effect = _capture
    return pb


class _FakeNativeProvider(LLMProvider, SupportsToolCalling):
    def __init__(
        self,
        responses: list[GenerateWithToolsResult],
        final_text: str = "Final answer.",
    ) -> None:
        self._responses = list(responses)
        self._final_text = final_text

    async def generate(self, prompt: str) -> str:
        return self._final_text

    async def generate_stream(self, prompt: str):  # type: ignore[override]
        raise NotImplementedError

    async def generate_with_tools(self, messages, tool_schemas):
        if self._responses:
            return self._responses.pop(0)
        return GenerateWithToolsResult(text="Fallback.")


class _FakePromptProvider(LLMProvider):
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def generate(self, prompt: str) -> str:
        if self._responses:
            return self._responses.pop(0)
        return "Prompt answer."

    async def generate_stream(self, prompt: str):  # type: ignore[override]
        raise NotImplementedError


def _make_service(
    provider: LLMProvider,
    conv_service: object = None,
    history_limit: int = 10,
    prompt_builder: PromptBuilder | None = None,
) -> AgentService:
    return AgentService(
        llm_provider=provider,
        tool_registry=ToolRegistry(),
        prompt_builder=prompt_builder or _make_prompt_builder(),
        conversation_service=conv_service,
        max_tool_calls=5,
        conversation_history_limit=history_limit,
    )


def _text_result(text: str = "Answer.") -> GenerateWithToolsResult:
    return GenerateWithToolsResult(text=text)


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


# ---------------------------------------------------------------------------
# 1–2: AgentState unit tests
# ---------------------------------------------------------------------------


class TestAgentState:
    def test_default_fields(self) -> None:
        user = _make_user()
        state = AgentState(question="Hello?", user=user, conversation_id="c1")

        assert state.question == "Hello?"
        assert state.user is user
        assert state.conversation_id == "c1"
        assert state.history == []
        assert state.tool_calls == []
        assert state.retrieved_chunks == []
        assert state.extra == {}

    def test_has_history_false_when_empty(self) -> None:
        state = AgentState(question="Q", user=_make_user(), conversation_id=None)
        assert not state.has_history

    def test_has_history_true_when_populated(self) -> None:
        state = AgentState(
            question="Q",
            user=_make_user(),
            conversation_id="c1",
            history=[_make_message("user", "Hi")],
        )
        assert state.has_history

    def test_total_tool_calls_counts_list(self) -> None:
        state = AgentState(question="Q", user=_make_user(), conversation_id=None)
        assert state.total_tool_calls == 0
        state.tool_calls.append(
            ToolCallRecord(tool_name="calc", args={}, observation="2")
        )
        assert state.total_tool_calls == 1

    def test_elapsed_ms_is_positive(self) -> None:
        state = AgentState(question="Q", user=_make_user(), conversation_id=None)
        time.sleep(0.01)  # 10ms
        assert state.elapsed_ms >= 0.0

    def test_extra_dict_is_extensible(self) -> None:
        state = AgentState(question="Q", user=_make_user(), conversation_id=None)
        state.extra["semantic_hits"] = [1, 2, 3]
        assert state.extra["semantic_hits"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# 3: History loading with correct limit
# ---------------------------------------------------------------------------


class TestHistoryLoading:
    async def test_history_loaded_with_configured_limit(self) -> None:
        """get_history must be called with the configured limit."""
        prev = [_make_message("user", "Q1"), _make_message("assistant", "A1")]
        conv_svc = _make_conv_service(history=prev)

        provider = _FakeNativeProvider(responses=[_text_result()])
        service = _make_service(provider, conv_service=conv_svc, history_limit=6)
        user = _make_user()

        await service.run(question="Q2", user=user, conversation_id="conv-1")

        conv_svc.get_history.assert_awaited_once_with(
            conversation_id="conv-1",
            user_id=user.id,
            limit=6,
        )

    async def test_no_history_loaded_without_conversation_id(self) -> None:
        """Stateless runs must not call get_history."""
        conv_svc = _make_conv_service()
        provider = _FakeNativeProvider(responses=[_text_result()])
        service = _make_service(provider, conv_service=conv_svc)
        user = _make_user()

        await service.run(question="Hello", user=user, conversation_id=None)

        conv_svc.get_history.assert_not_awaited()

    async def test_history_limit_zero_loads_nothing(self) -> None:
        """history_limit=0 must call get_history with limit=0, returning empty."""
        conv_svc = _make_conv_service(history=[])
        provider = _FakeNativeProvider(responses=[_text_result()])
        service = _make_service(
            provider, conv_service=conv_svc, history_limit=0
        )
        user = _make_user()

        await service.run(question="Q", user=user, conversation_id="c1")

        conv_svc.get_history.assert_awaited_once_with(
            conversation_id="c1",
            user_id=user.id,
            limit=0,
        )


# ---------------------------------------------------------------------------
# 4: History loaded BEFORE current message persisted
# ---------------------------------------------------------------------------


class TestHistoryLoadedBeforeUserMessage:
    async def test_get_history_called_before_add_message(self) -> None:
        """Ordering: get_history → add_message(user) — never the reverse."""
        call_order: list[str] = []

        conv_svc = MagicMock()
        conv_svc.get_history = AsyncMock(
            side_effect=lambda **kw: call_order.append("get_history") or []
        )
        conv_svc.add_message = AsyncMock(
            side_effect=lambda **kw: call_order.append(f"add_message:{kw['role']}")
        )
        conv_svc.set_auto_title_if_needed = AsyncMock()
        conv_svc.record_token_usage = AsyncMock()

        provider = _FakeNativeProvider(responses=[_text_result()])
        service = _make_service(provider, conv_service=conv_svc)
        user = _make_user()

        await service.run(question="Q", user=user, conversation_id="c1")

        assert call_order[0] == "get_history"
        assert call_order[1] == "add_message:user"


# ---------------------------------------------------------------------------
# 5: History passed to PromptBuilder
# ---------------------------------------------------------------------------


class TestHistoryPassedToPromptBuilder:
    async def test_history_passed_to_prompt_builder_build(self) -> None:
        """PromptBuilder.build() must receive the loaded history."""
        prev = [_make_message("user", "Q1"), _make_message("assistant", "A1")]
        conv_svc = _make_conv_service(history=prev)

        captured: list[dict] = []
        pb = _make_prompt_builder(captured=captured)

        provider = _FakeNativeProvider(responses=[_text_result()])
        service = _make_service(provider, conv_service=conv_svc, prompt_builder=pb)
        user = _make_user()

        await service.run(question="Q2", user=user, conversation_id="c1")

        assert len(captured) == 1
        assert captured[0]["history"] == prev

    async def test_no_history_passes_none_to_prompt_builder(self) -> None:
        """When no history, PromptBuilder.build() must receive history=None."""
        captured: list[dict] = []
        pb = _make_prompt_builder(captured=captured)

        provider = _FakeNativeProvider(responses=[_text_result()])
        service = _make_service(provider, prompt_builder=pb)
        user = _make_user()

        await service.run(question="Hello", user=user)

        assert captured[0].get("history") is None

    async def test_empty_history_passes_none_to_prompt_builder(self) -> None:
        """Empty history list (first message in conv) must pass history=None."""
        conv_svc = _make_conv_service(history=[])  # no prior messages
        captured: list[dict] = []
        pb = _make_prompt_builder(captured=captured)

        provider = _FakeNativeProvider(responses=[_text_result()])
        service = _make_service(
            provider, conv_service=conv_svc, prompt_builder=pb
        )
        user = _make_user()

        await service.run(question="First question", user=user, conversation_id="c1")

        assert captured[0].get("history") is None


# ---------------------------------------------------------------------------
# 6: History NOT injected into tool-calling loop
# ---------------------------------------------------------------------------


class TestHistoryNotInToolLoop:
    async def test_generate_with_tools_does_not_receive_history(self) -> None:
        """Native loop messages must NOT include conversation history turns."""
        prev = [_make_message("user", "Q1"), _make_message("assistant", "A1")]
        conv_svc = _make_conv_service(history=prev)

        received_messages: list = []

        class _TracingProvider(LLMProvider, SupportsToolCalling):
            async def generate(self, prompt: str) -> str:
                return "Final."

            async def generate_stream(self, prompt: str):  # type: ignore[override]
                raise NotImplementedError

            async def generate_with_tools(self, messages, tool_schemas):
                received_messages.extend(messages)
                return GenerateWithToolsResult(text="Done.")

        provider = _TracingProvider()
        service = _make_service(provider, conv_service=conv_svc)
        user = _make_user()

        await service.run(question="Q2", user=user, conversation_id="c1")

        # The tool loop should only see the current question, not Q1/A1
        roles_seen = [m.role for m in received_messages]
        assert "user" in roles_seen
        texts_seen = [m.text for m in received_messages if m.text]
        assert "Q2" in texts_seen
        assert "Q1" not in texts_seen
        assert "A1" not in texts_seen


# ---------------------------------------------------------------------------
# 7: Multi-turn context
# ---------------------------------------------------------------------------


class TestMultiTurnContext:
    async def test_second_turn_receives_first_turn_in_history(self) -> None:
        """Simulate two turns; second turn's PromptBuilder call gets first turn."""
        turn1_history: list = []
        turn2_history = [
            _make_message("user", "What is Python?"),
            _make_message("assistant", "Python is a programming language."),
        ]

        captured: list[dict] = []
        pb = _make_prompt_builder(captured=captured)

        # Turn 1: no prior history
        conv_svc_1 = _make_conv_service(history=turn1_history)
        provider_1 = _FakeNativeProvider(responses=[_text_result()])
        service_1 = _make_service(
            provider_1, conv_service=conv_svc_1, prompt_builder=pb
        )
        user = _make_user()
        await service_1.run(
            question="What is Python?", user=user, conversation_id="c1"
        )

        # Turn 2: prior history present
        conv_svc_2 = _make_conv_service(history=turn2_history)
        provider_2 = _FakeNativeProvider(responses=[_text_result()])
        service_2 = _make_service(
            provider_2, conv_service=conv_svc_2, prompt_builder=pb
        )
        await service_2.run(
            question="Give me an example.", user=user, conversation_id="c1"
        )

        # Second call to PromptBuilder must have turn2_history
        assert captured[1]["history"] == turn2_history


# ---------------------------------------------------------------------------
# 8: History limit prevents unbounded growth
# ---------------------------------------------------------------------------


class TestHistoryLimit:
    async def test_history_limit_passed_as_kwarg_to_get_history(self) -> None:
        """history_limit is forwarded verbatim to get_history's limit kwarg."""
        conv_svc = _make_conv_service(history=[])
        provider = _FakeNativeProvider(responses=[_text_result()])
        service = _make_service(
            provider, conv_service=conv_svc, history_limit=3
        )
        user = _make_user()

        await service.run(question="Q", user=user, conversation_id="c1")

        call_kwargs = conv_svc.get_history.call_args.kwargs
        assert call_kwargs["limit"] == 3


# ---------------------------------------------------------------------------
# 9: Stateless run
# ---------------------------------------------------------------------------


class TestStatelessRun:
    async def test_stateless_run_uses_no_conv_service(self) -> None:
        """AgentService with no conv_service should not call get_history."""
        captured: list[dict] = []
        pb = _make_prompt_builder(captured=captured)
        provider = _FakeNativeProvider(responses=[_text_result()])

        service = AgentService(
            llm_provider=provider,
            tool_registry=ToolRegistry(),
            prompt_builder=pb,
            conversation_service=None,
        )
        user = _make_user()
        result = await service.run(question="Stateless Q", user=user)

        assert result.answer is not None
        assert captured[0].get("history") is None


# ---------------------------------------------------------------------------
# 10: AgentState.retrieved_chunks accumulation
# ---------------------------------------------------------------------------


class TestRetrievedChunksAccumulation:
    async def test_chunks_from_rag_tool_added_to_state(self) -> None:
        """RAGSearchTool results must be collected into state.retrieved_chunks."""
        chunk = RetrievalResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            chunk_index=0,
            text="Python is a language.",
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
        rag_tool.execute = AsyncMock(return_value="Python is a language.")
        rag_tool.last_results = [chunk]

        registry = ToolRegistry()
        registry.register(rag_tool)

        provider = _FakeNativeProvider(
            responses=[
                _tool_call_result("rag_search", {"query": "Python"}, "c1"),
                _text_result("Python is great."),
            ],
            final_text="Answer.",
        )

        service = AgentService(
            llm_provider=provider,
            tool_registry=registry,
            prompt_builder=_make_prompt_builder(),
        )
        user = _make_user()
        result = await service.run(question="What is Python?", user=user)

        assert len(result.retrieved_chunks) == 1
        assert result.retrieved_chunks[0].chunk_id == "chunk-1"


# ---------------------------------------------------------------------------
# 11: Ownership / security
# ---------------------------------------------------------------------------


class TestOwnershipSecurity:
    async def test_forbidden_history_aborts_before_tool_call(self) -> None:
        """ForbiddenError from get_history must abort before any tool call."""
        conv_svc = _make_conv_service(forbidden=True)

        tool_called = False

        class _TrackingProvider(LLMProvider, SupportsToolCalling):
            async def generate(self, prompt: str) -> str:
                return "answer"

            async def generate_stream(self, prompt: str):  # type: ignore[override]
                raise NotImplementedError

            async def generate_with_tools(self, messages, tool_schemas):
                nonlocal tool_called
                tool_called = True
                return GenerateWithToolsResult(text="Done.")

        provider = _TrackingProvider()
        service = _make_service(provider, conv_service=conv_svc)
        user = _make_user()

        with pytest.raises(ForbiddenError):
            await service.run(question="Q", user=user, conversation_id="c1")

        assert not tool_called

    async def test_ownership_check_uses_requesting_user_id(self) -> None:
        """get_history must be called with the requesting user's ID."""
        conv_svc = _make_conv_service()
        provider = _FakeNativeProvider(responses=[_text_result()])
        service = _make_service(provider, conv_service=conv_svc)
        user = _make_user()

        await service.run(question="Q", user=user, conversation_id="c1")

        call_kwargs = conv_svc.get_history.call_args.kwargs
        assert call_kwargs["user_id"] == user.id


# ---------------------------------------------------------------------------
# 12: Conversation persistence regression
# ---------------------------------------------------------------------------


class TestPersistenceRegression:
    async def test_user_and_assistant_messages_persisted(self) -> None:
        """add_message must be called for both user and assistant roles."""
        conv_svc = _make_conv_service()
        provider = _FakeNativeProvider(responses=[_text_result()], final_text="Ans.")
        service = _make_service(provider, conv_service=conv_svc)
        user = _make_user()

        await service.run(question="Q", user=user, conversation_id="c1")

        calls = conv_svc.add_message.call_args_list
        roles = [c.kwargs["role"] for c in calls]
        assert "user" in roles
        assert "assistant" in roles


# ---------------------------------------------------------------------------
# 13: Native path writes to state — regression
# ---------------------------------------------------------------------------


class TestNativePathStateRegression:
    async def test_native_tool_call_recorded_in_result(self) -> None:
        registry = ToolRegistry()
        tool = MagicMock()
        tool.name = "calculator"
        tool.description = "calc"
        tool.parameters_schema = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        tool.execute = AsyncMock(return_value="4")
        registry.register(tool)

        provider = _FakeNativeProvider(
            responses=[
                _tool_call_result("calculator", {"expression": "2+2"}, "c1"),
                _text_result("The answer is 4."),
            ],
            final_text="Grounded: 4.",
        )
        service = AgentService(
            llm_provider=provider,
            tool_registry=registry,
            prompt_builder=_make_prompt_builder(),
        )
        result = await service.run(question="What is 2+2?", user=_make_user())

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "calculator"

    async def test_prompt_fallback_tool_call_recorded(self) -> None:
        """Prompt-based path also writes tool calls to AgentResult."""
        registry = ToolRegistry()
        tool = MagicMock()
        tool.name = "calculator"
        tool.description = "calc"
        tool.parameters_schema = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        tool.execute = AsyncMock(return_value="21")
        registry.register(tool)

        responses = [
            '{"action": "tool_call", "tool": "calculator", "args": {}}',
            '{"action": "final_answer", "answer": "21"}',
            "Grounded: 21.",
        ]
        provider = _FakePromptProvider(responses)
        service = AgentService(
            llm_provider=provider,
            tool_registry=registry,
            prompt_builder=_make_prompt_builder(),
        )
        result = await service.run(question="What is 3*7?", user=_make_user())

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "calculator"
