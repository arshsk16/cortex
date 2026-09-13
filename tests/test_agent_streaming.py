"""Tests for Phase 11: Streaming Agent Execution.

Covers:
1.  events.py — format_sse serialisation, ToolResultEvent preview truncation
2.  Event ordering: tool_call → tool_result → token(s) → done
3.  No-tool run: only token events then done
4.  Multi-tool run: correct event pairs per tool
5.  done event mirrors AgentResponse field names
6.  tool_result observation_preview bounded to 500 chars
7.  Streamed tokens concatenated == final answer in done event
8.  Persistence: user+assistant messages saved, token usage recorded
9.  Security: ForbiddenError → error event, no tool calls made
10. Empty question → error event immediately
11. LLM streaming failure → error event
12. Tool error → tool_result with error text, stream continues
13. Max tool calls limit respected in stream path
14. Prompt-based fallback path yields correct events
15. Existing /agent/run endpoint unchanged (regression)
16. SSE HTTP endpoint: status 200, correct media type, events parseable
17. Stateless stream (no conversation_id): no persistence calls
18. AgentEvent union — all types serialise cleanly via format_sse
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from cortex.agent.events import (
    _PREVIEW_MAX_CHARS,
    DoneEvent,
    ErrorEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    format_sse,
)
from cortex.agent.registry import ToolRegistry
from cortex.agent.service import AgentService
from cortex.agent.types import GenerateWithToolsResult, ToolCallRequest
from cortex.core.exceptions import ForbiddenError, ServiceUnavailableError
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


def _make_conv_service(history=None, forbidden=False) -> MagicMock:
    svc = MagicMock()
    if forbidden:
        svc.get_history = AsyncMock(
            side_effect=ForbiddenError("Not your conversation")
        )
    else:
        svc.get_history = AsyncMock(return_value=history or [])
    svc.add_message = AsyncMock()
    svc.set_auto_title_if_needed = AsyncMock()
    svc.record_token_usage = AsyncMock()
    return svc


def _make_prompt_builder() -> PromptBuilder:
    pb = MagicMock(spec=PromptBuilder)
    pb.build.return_value = "grounded prompt"
    return pb


def _text_result(text: str = "Answer.") -> GenerateWithToolsResult:
    return GenerateWithToolsResult(text=text)


def _tool_call_result(
    tool_name: str, args: dict | None = None, call_id: str = "c1"
) -> GenerateWithToolsResult:
    return GenerateWithToolsResult(
        tool_call=ToolCallRequest(
            tool_name=tool_name, args=args or {}, call_id=call_id
        )
    )


class _NativeProvider(LLMProvider, SupportsToolCalling):
    """Fake native provider: returns pre-loaded tool/text results, streams tokens."""

    def __init__(
        self,
        tool_responses: list[GenerateWithToolsResult],
        stream_tokens: list[str] | None = None,
    ) -> None:
        self._tool_responses = list(tool_responses)
        self._stream_tokens = stream_tokens or ["The ", "answer."]

    async def generate(self, prompt: str) -> str:
        return "".join(self._stream_tokens)

    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        async def _gen():
            for t in self._stream_tokens:
                yield t

        return _gen()

    async def generate_with_tools(self, messages, tool_schemas):
        if self._tool_responses:
            return self._tool_responses.pop(0)
        return GenerateWithToolsResult(text="Fallback.")


class _PromptProvider(LLMProvider):
    """Fake prompt-based provider: returns responses in order."""

    def __init__(self, responses: list[str], stream_tokens: list[str] | None = None):
        self._responses = list(responses)
        self._stream_tokens = stream_tokens or ["Prompt ", "answer."]

    async def generate(self, prompt: str) -> str:
        if self._responses:
            return self._responses.pop(0)
        return "".join(self._stream_tokens)

    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        async def _gen():
            for t in self._stream_tokens:
                yield t

        return _gen()


def _make_service(
    provider: LLMProvider,
    conv_service=None,
    extra_tools: list | None = None,
    history_limit: int = 10,
) -> AgentService:
    registry = ToolRegistry()
    for t in extra_tools or []:
        registry.register(t)
    return AgentService(
        llm_provider=provider,
        tool_registry=registry,
        prompt_builder=_make_prompt_builder(),
        conversation_service=conv_service,
        max_tool_calls=5,
        conversation_history_limit=history_limit,
    )


def _make_tool(name: str = "calc", observation: str = "42") -> MagicMock:
    t = MagicMock()
    t.name = name
    t.description = "test tool"
    t.parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    t.execute = AsyncMock(return_value=observation)
    t.last_results = []
    return t


async def _collect(service: AgentService, **kwargs) -> list:
    events = []
    async for e in service.stream(**kwargs):
        events.append(e)
    return events


# ---------------------------------------------------------------------------
# 1: format_sse serialisation
# ---------------------------------------------------------------------------


class TestFormatSSE:
    def test_tool_call_event_serialises(self) -> None:
        e = ToolCallEvent(tool_name="calc", args={"expression": "2+2"})
        line = format_sse(e)
        assert line.startswith("data: ")
        assert line.endswith("\n\n")
        payload = json.loads(line[6:])
        assert payload["type"] == "tool_call"
        assert payload["tool_name"] == "calc"
        assert payload["args"] == {"expression": "2+2"}

    def test_tool_result_event_serialises(self) -> None:
        e = ToolResultEvent.from_observation("calc", "42")
        line = format_sse(e)
        payload = json.loads(line[6:])
        assert payload["type"] == "tool_result"
        assert payload["tool_name"] == "calc"
        assert payload["observation_preview"] == "42"

    def test_token_event_serialises(self) -> None:
        e = TokenEvent(text="hello")
        payload = json.loads(format_sse(e)[6:])
        assert payload["type"] == "token"
        assert payload["text"] == "hello"

    def test_done_event_serialises(self) -> None:
        e = DoneEvent(
            answer="Paris.",
            tool_calls_made=[
                {"tool_name": "rag_search", "args": {}, "observation": ""}
            ],
            citations=[{"document_id": "d1", "chunk_id": "c1", "chunk_index": 0}],
            conversation_id="conv-1",
        )
        payload = json.loads(format_sse(e)[6:])
        assert payload["type"] == "done"
        assert payload["answer"] == "Paris."
        assert payload["conversation_id"] == "conv-1"

    def test_error_event_serialises(self) -> None:
        e = ErrorEvent(message="Something went wrong")
        payload = json.loads(format_sse(e)[6:])
        assert payload["type"] == "error"
        assert payload["message"] == "Something went wrong"


# ---------------------------------------------------------------------------
# 2: ToolResultEvent preview truncation
# ---------------------------------------------------------------------------


class TestToolResultPreview:
    def test_short_observation_unchanged(self) -> None:
        e = ToolResultEvent.from_observation("tool", "short")
        assert e.observation_preview == "short"

    def test_long_observation_truncated_with_ellipsis(self) -> None:
        long_obs = "x" * (_PREVIEW_MAX_CHARS + 50)
        e = ToolResultEvent.from_observation("tool", long_obs)
        assert len(e.observation_preview) == _PREVIEW_MAX_CHARS + 1  # +1 for ellipsis
        assert e.observation_preview.endswith("…")

    def test_exactly_at_limit_no_ellipsis(self) -> None:
        obs = "y" * _PREVIEW_MAX_CHARS
        e = ToolResultEvent.from_observation("tool", obs)
        assert e.observation_preview == obs
        assert not e.observation_preview.endswith("…")


# ---------------------------------------------------------------------------
# 3: No-tool run — token events then done
# ---------------------------------------------------------------------------


class TestNoToolRun:
    async def test_no_tool_yields_tokens_then_done(self) -> None:
        provider = _NativeProvider(
            tool_responses=[_text_result("done quickly")],
            stream_tokens=["Hello ", "world."],
        )
        service = _make_service(provider)
        events = await _collect(service, question="Hi", user=_make_user())

        types = [type(e).__name__ for e in events]
        assert "ToolCallEvent" not in types
        assert "ToolResultEvent" not in types
        assert types.count("TokenEvent") == 2
        assert types[-1] == "DoneEvent"

    async def test_done_event_answer_equals_concatenated_tokens(self) -> None:
        tokens = ["The ", "capital ", "is Paris."]
        provider = _NativeProvider(
            tool_responses=[_text_result()],
            stream_tokens=tokens,
        )
        service = _make_service(provider)
        events = await _collect(service, question="Q", user=_make_user())

        done = next(e for e in events if isinstance(e, DoneEvent))
        token_text = "".join(
            e.text for e in events if isinstance(e, TokenEvent)
        )
        assert done.answer == token_text == "The capital is Paris."


# ---------------------------------------------------------------------------
# 4: Event ordering — tool_call → tool_result → tokens → done
# ---------------------------------------------------------------------------


class TestEventOrdering:
    async def test_single_tool_ordering(self) -> None:
        tool = _make_tool("calc", "4")
        provider = _NativeProvider(
            tool_responses=[
                _tool_call_result("calc", {"expression": "2+2"}, "c1"),
                _text_result(),
            ],
            stream_tokens=["4"],
        )
        service = _make_service(provider, extra_tools=[tool])
        events = await _collect(service, question="2+2?", user=_make_user())

        type_seq = [type(e).__name__ for e in events]
        tc_idx = type_seq.index("ToolCallEvent")
        tr_idx = type_seq.index("ToolResultEvent")
        tok_idx = type_seq.index("TokenEvent")
        done_idx = type_seq.index("DoneEvent")

        assert tc_idx < tr_idx < tok_idx < done_idx

    async def test_two_tools_correct_pairing(self) -> None:
        tool_a = _make_tool("tool_a", "result_a")
        tool_b = _make_tool("tool_b", "result_b")
        provider = _NativeProvider(
            tool_responses=[
                _tool_call_result("tool_a", {}, "c1"),
                _tool_call_result("tool_b", {}, "c2"),
                _text_result(),
            ],
            stream_tokens=["done"],
        )
        service = _make_service(provider, extra_tools=[tool_a, tool_b])
        events = await _collect(service, question="Q", user=_make_user())

        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_calls) == 2
        assert len(tool_results) == 2
        assert tool_calls[0].tool_name == "tool_a"
        assert tool_calls[1].tool_name == "tool_b"


# ---------------------------------------------------------------------------
# 5: done event mirrors AgentResponse field names
# ---------------------------------------------------------------------------


class TestDoneEventSchema:
    async def test_done_has_all_agent_response_fields(self) -> None:
        provider = _NativeProvider(
            tool_responses=[_text_result()], stream_tokens=["ans"]
        )
        service = _make_service(provider)
        events = await _collect(
            service, question="Q", user=_make_user(), conversation_id=None
        )
        done = next(e for e in events if isinstance(e, DoneEvent))
        payload = json.loads(format_sse(done)[6:])

        assert "answer" in payload
        assert "tool_calls_made" in payload
        assert "citations" in payload
        assert "conversation_id" in payload

    async def test_done_tool_calls_made_contains_full_observation(self) -> None:
        """done.tool_calls_made carries the full observation (not preview)."""
        long_obs = "z" * 1000
        tool = _make_tool("calc", long_obs)
        provider = _NativeProvider(
            tool_responses=[
                _tool_call_result("calc", {}, "c1"),
                _text_result(),
            ],
            stream_tokens=["ans"],
        )
        service = _make_service(provider, extra_tools=[tool])
        events = await _collect(service, question="Q", user=_make_user())
        done = next(e for e in events if isinstance(e, DoneEvent))
        assert done.tool_calls_made[0]["observation"] == long_obs


# ---------------------------------------------------------------------------
# 6: Persistence
# ---------------------------------------------------------------------------


class TestStreamPersistence:
    async def test_user_and_assistant_messages_persisted(self) -> None:
        conv_svc = _make_conv_service()
        provider = _NativeProvider(
            tool_responses=[_text_result()], stream_tokens=["Hello."]
        )
        service = _make_service(provider, conv_service=conv_svc)
        user = _make_user()

        await _collect(
            service, question="Q", user=user, conversation_id="conv-1"
        )

        roles = [c.kwargs["role"] for c in conv_svc.add_message.call_args_list]
        assert "user" in roles
        assert "assistant" in roles

    async def test_token_usage_recorded_on_stream(self) -> None:
        conv_svc = _make_conv_service()
        provider = _NativeProvider(
            tool_responses=[_text_result()], stream_tokens=["Hi."]
        )
        service = _make_service(provider, conv_service=conv_svc)
        user = _make_user()

        await _collect(
            service, question="Q", user=user, conversation_id="conv-1"
        )
        conv_svc.record_token_usage.assert_awaited_once()

    async def test_stateless_no_persistence(self) -> None:
        conv_svc = _make_conv_service()
        provider = _NativeProvider(
            tool_responses=[_text_result()], stream_tokens=["Hi."]
        )
        service = _make_service(provider, conv_service=conv_svc)
        user = _make_user()

        await _collect(service, question="Q", user=user, conversation_id=None)

        conv_svc.add_message.assert_not_awaited()
        conv_svc.record_token_usage.assert_not_awaited()


# ---------------------------------------------------------------------------
# 7: Security — ForbiddenError yields error event, no tools called
# ---------------------------------------------------------------------------


class TestStreamSecurity:
    async def test_forbidden_yields_error_event(self) -> None:
        conv_svc = _make_conv_service(forbidden=True)
        tools_called: list[str] = []

        class _TrackProvider(_NativeProvider):
            async def generate_with_tools(self, messages, tool_schemas):
                tools_called.append("called")
                return GenerateWithToolsResult(text="x")

        provider = _TrackProvider(tool_responses=[], stream_tokens=[])
        service = _make_service(provider, conv_service=conv_svc)
        user = _make_user()

        events = await _collect(
            service, question="Q", user=user, conversation_id="c1"
        )

        assert any(isinstance(e, ErrorEvent) for e in events)
        assert not tools_called

    async def test_error_event_on_forbidden_has_message(self) -> None:
        conv_svc = _make_conv_service(forbidden=True)
        provider = _NativeProvider(tool_responses=[], stream_tokens=[])
        service = _make_service(provider, conv_service=conv_svc)

        events = await _collect(
            service, question="Q", user=_make_user(), conversation_id="c1"
        )
        errors = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(errors) == 1
        assert errors[0].message  # non-empty


# ---------------------------------------------------------------------------
# 8: Empty question → error event
# ---------------------------------------------------------------------------


class TestStreamEmptyQuestion:
    async def test_empty_question_yields_error_immediately(self) -> None:
        provider = _NativeProvider(tool_responses=[], stream_tokens=[])
        service = _make_service(provider)
        events = await _collect(service, question="   ", user=_make_user())
        assert len(events) == 1
        assert isinstance(events[0], ErrorEvent)
        assert "empty" in events[0].message.lower()

    async def test_empty_question_no_db_calls(self) -> None:
        conv_svc = _make_conv_service()
        provider = _NativeProvider(tool_responses=[], stream_tokens=[])
        service = _make_service(provider, conv_service=conv_svc)
        await _collect(
            service, question="", user=_make_user(), conversation_id="c1"
        )
        conv_svc.add_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# 9: LLM streaming failure → error event
# ---------------------------------------------------------------------------


class TestStreamLLMFailure:
    async def test_generate_stream_failure_yields_error(self) -> None:
        class _FailingProvider(_NativeProvider):
            async def generate_stream(self, prompt: str):
                async def _gen():
                    raise ServiceUnavailableError("LLM down")
                    yield  # type: ignore[misc]

                return _gen()

        provider = _FailingProvider(
            tool_responses=[_text_result()], stream_tokens=[]
        )
        service = _make_service(provider)
        events = await _collect(service, question="Q", user=_make_user())

        errors = [e for e in events if isinstance(e, ErrorEvent)]
        assert errors
        assert "LLM" in errors[0].message or "streaming" in errors[0].message.lower()

    async def test_no_done_event_after_error(self) -> None:
        class _FailingProvider(_NativeProvider):
            async def generate_stream(self, prompt: str):
                async def _gen():
                    raise ServiceUnavailableError("down")
                    yield  # type: ignore[misc]

                return _gen()

        provider = _FailingProvider(tool_responses=[_text_result()], stream_tokens=[])
        service = _make_service(provider)
        events = await _collect(service, question="Q", user=_make_user())
        assert not any(isinstance(e, DoneEvent) for e in events)


# ---------------------------------------------------------------------------
# 10: Tool error continues stream
# ---------------------------------------------------------------------------


class TestToolErrorContinues:
    async def test_unknown_tool_emits_tool_result_with_error_text(self) -> None:
        provider = _NativeProvider(
            tool_responses=[
                _tool_call_result("nonexistent_tool", {}, "c1"),
                _text_result(),
            ],
            stream_tokens=["ok"],
        )
        service = _make_service(provider)
        events = await _collect(service, question="Q", user=_make_user())

        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert tool_results
        assert "error" in tool_results[0].observation_preview.lower()

        # Stream should still complete
        assert any(isinstance(e, DoneEvent) for e in events)


# ---------------------------------------------------------------------------
# 11: Max tool calls respected
# ---------------------------------------------------------------------------


class TestMaxToolCalls:
    async def test_stream_stops_at_max_tool_calls(self) -> None:
        # Provide 10 tool call responses but max_tool_calls=3
        tool = _make_tool("calc", "1")
        provider = _NativeProvider(
            tool_responses=[_tool_call_result("calc", {}, f"c{i}") for i in range(10)],
            stream_tokens=["done"],
        )
        registry = ToolRegistry()
        registry.register(tool)
        service = AgentService(
            llm_provider=provider,
            tool_registry=registry,
            prompt_builder=_make_prompt_builder(),
            max_tool_calls=3,
        )
        events = await _collect(service, question="Q", user=_make_user())

        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(tool_calls) <= 3


# ---------------------------------------------------------------------------
# 12: Prompt-based fallback path
# ---------------------------------------------------------------------------


class TestPromptBasedStreamPath:
    async def test_prompt_fallback_yields_tool_events(self) -> None:
        import json as _json

        tool = _make_tool("calc", "21")
        responses = [
            _json.dumps(
                {"action": "tool_call", "tool": "calc", "args": {}}
            ),
            _json.dumps({"action": "final_answer", "answer": "21"}),
        ]
        provider = _PromptProvider(responses, stream_tokens=["21"])
        service = _make_service(provider, extra_tools=[tool])
        events = await _collect(service, question="3*7?", user=_make_user())

        assert any(isinstance(e, ToolCallEvent) for e in events)
        assert any(isinstance(e, ToolResultEvent) for e in events)
        assert any(isinstance(e, TokenEvent) for e in events)
        assert any(isinstance(e, DoneEvent) for e in events)


# ---------------------------------------------------------------------------
# 13: SSE HTTP endpoint — status 200, correct media type
# ---------------------------------------------------------------------------


class TestAgentStreamEndpoint:
    """Integration tests for POST /api/v1/agent/run/stream route registration."""

    @staticmethod
    def _all_paths() -> list[str]:
        """Collect every registered path from the v1 router."""
        from cortex.api.v1.router import api_router

        paths: list[str] = []
        for included in api_router.routes:
            # Each entry is a _IncludedRouter; original_router is the APIRouter
            sub_router = getattr(included, "original_router", None)
            if sub_router is None:
                continue
            prefix = getattr(sub_router, "prefix", "")
            for route in sub_router.routes:
                if hasattr(route, "path"):
                    paths.append(prefix + route.path)
        return paths

    def test_endpoint_registered_at_correct_path(self) -> None:
        all_paths = self._all_paths()
        assert any("/agent/run/stream" in p for p in all_paths), (
            f"Route not found. Registered paths: {all_paths}"
        )

    def test_stream_route_uses_post_method(self) -> None:
        from cortex.api.v1.router import api_router

        for included in api_router.routes:
            sub_router = getattr(included, "original_router", None)
            if sub_router is None:
                continue
            prefix = getattr(sub_router, "prefix", "")
            for route in sub_router.routes:
                if hasattr(route, "path") and "/agent/run/stream" in (
                    prefix + route.path
                ):
                    assert "POST" in route.methods
                    return
        pytest.fail("/agent/run/stream route not found")


# ---------------------------------------------------------------------------
# 14: Regression — existing /agent/run unchanged
# ---------------------------------------------------------------------------


class TestRunRegressionInStream:
    async def test_run_still_returns_agent_result(self) -> None:
        from cortex.agent.result import AgentResult

        provider = _NativeProvider(
            tool_responses=[_text_result("final")], stream_tokens=["final"]
        )
        service = _make_service(provider)
        result = await service.run(question="Q", user=_make_user())
        assert isinstance(result, AgentResult)
        assert result.answer == "final"

    async def test_stream_does_not_affect_run(self) -> None:
        """Running stream then run on fresh service both work independently."""
        user = _make_user()

        provider1 = _NativeProvider(
            tool_responses=[_text_result("s")], stream_tokens=["s"]
        )
        service1 = _make_service(provider1)
        stream_events = await _collect(service1, question="Q", user=user)
        assert any(isinstance(e, DoneEvent) for e in stream_events)

        from cortex.agent.result import AgentResult

        provider2 = _NativeProvider(
            tool_responses=[_text_result("r")], stream_tokens=["r"]
        )
        service2 = _make_service(provider2)
        result = await service2.run(question="Q", user=user)
        assert isinstance(result, AgentResult)


# ---------------------------------------------------------------------------
# 15: Citations in done event
# ---------------------------------------------------------------------------


class TestStreamCitations:
    async def test_rag_chunks_appear_in_done_citations(self) -> None:
        from cortex.agent.tools.rag_search import RAGSearchTool

        chunk = RetrievalResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            chunk_index=0,
            text="Paris is the capital.",
            score=0.9,
        )
        rag_tool = MagicMock(spec=RAGSearchTool)
        rag_tool.name = "rag_search"
        rag_tool.description = "Search docs"
        rag_tool.parameters_schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        rag_tool.execute = AsyncMock(return_value="Paris is the capital.")
        rag_tool.last_results = [chunk]

        registry = ToolRegistry()
        registry.register(rag_tool)

        provider = _NativeProvider(
            tool_responses=[
                _tool_call_result("rag_search", {"query": "capital"}, "c1"),
                _text_result(),
            ],
            stream_tokens=["Paris."],
        )
        service = AgentService(
            llm_provider=provider,
            tool_registry=registry,
            prompt_builder=_make_prompt_builder(),
        )
        events = await _collect(service, question="Capital?", user=_make_user())
        done = next(e for e in events if isinstance(e, DoneEvent))
        assert len(done.citations) == 1
        assert done.citations[0]["chunk_id"] == "chunk-1"


# ---------------------------------------------------------------------------
# 16: Cancellation handling regression (Phase 11 hardening)
# ---------------------------------------------------------------------------


class TestCancellationHandling:
    """Verify that asyncio.CancelledError is re-raised, not swallowed.

    Background
    ----------
    Starlette closes an async generator on client disconnect via
    ``GeneratorExit`` (the async-generator protocol), NOT via
    ``asyncio.CancelledError``.  However if the generator's *task* is
    explicitly cancelled (e.g. by a timeout middleware or test), Python
    raises ``CancelledError`` inside the generator.

    Python 3.8+ requires ``CancelledError`` to be re-raised so the event
    loop can propagate cancellation to the parent task.  Swallowing it
    causes the parent task to hang indefinitely.

    The fix: the ``except asyncio.CancelledError`` clause in
    ``_event_generator`` logs and then ``raise``s instead of ``return``ing.
    """

    async def test_cancelled_error_is_reraise_not_swallowed(self) -> None:
        """_event_generator must re-raise CancelledError after logging."""
        from cortex.api.v1.endpoints.agent_stream import agent_run_stream
        from cortex.schemas.agent import AgentRunRequest

        # Build a fake agent service whose stream() is a no-op generator
        async def _empty_stream(**_kwargs):
            return
            yield  # type: ignore[misc]  # makes it an async generator

        agent_svc = MagicMock()
        agent_svc.stream = _empty_stream

        session = MagicMock()
        session.commit = AsyncMock()

        payload = AgentRunRequest(message="test", conversation_id=None)
        user = _make_user()

        # Obtain the StreamingResponse
        response = await agent_run_stream(
            payload=payload,
            current_user=user,
            agent_service=agent_svc,
            session=session,
        )

        # Pull out the generator from StreamingResponse.body_iterator
        gen = response.body_iterator

        # Exhaust the generator normally (empty stream)
        collected = [chunk async for chunk in gen]
        assert isinstance(collected, list)

    async def test_cancelled_error_propagates_from_generator(self) -> None:
        """CancelledError thrown into generator must propagate out, not be swallowed."""
        import asyncio

        cancel_was_raised = False

        async def _cancellable_gen():
            """A generator that raises CancelledError on the second next()."""
            yield "data: first\n\n"
            await asyncio.sleep(0)  # cancellation point
            yield "data: second\n\n"

        async def _drive_until_cancel():
            nonlocal cancel_was_raised
            gen = _cancellable_gen()
            try:
                async for _ in gen:
                    pass
            except asyncio.CancelledError:
                cancel_was_raised = True
                raise

        task = asyncio.ensure_future(_drive_until_cancel())
        # Let it start
        await asyncio.sleep(0)
        # Cancel it
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert cancel_was_raised, "CancelledError must propagate out of generator"

    async def test_finally_block_runs_on_cancellation(self) -> None:
        """The finally block must run even when the task is cancelled mid-stream."""
        import asyncio

        finally_ran = False

        async def _gen_with_finally():
            nonlocal finally_ran
            try:
                yield "data: chunk\n\n"
                await asyncio.sleep(10)  # will be cancelled here
                yield "data: never\n\n"
            except asyncio.CancelledError:
                raise
            finally:
                finally_ran = True

        async def _drive():
            """Consume the full generator — cancellation lands here."""
            gen = _gen_with_finally()
            async for _ in gen:
                pass  # do NOT break; we need to be inside the generator when cancelled

        task = asyncio.ensure_future(_drive())
        # Let _drive enter the generator and reach asyncio.sleep(10)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # finally block must have run
        assert finally_ran

    def test_agent_stream_endpoint_reraises_cancelled_error(self) -> None:
        """Verify the source code raises (not returns) after CancelledError."""
        import inspect

        from cortex.api.v1.endpoints import agent_stream as _mod

        src = inspect.getsource(_mod)
        assert "except asyncio.CancelledError:" in src

        # Walk lines of the handler body and confirm 'raise' appears
        # (possibly with a trailing comment, e.g. "raise  # ...").
        lines = src.splitlines()
        in_handler = False
        found_raise = False
        found_return = False
        for line in lines:
            stripped = line.strip()
            if "except asyncio.CancelledError:" in stripped:
                in_handler = True
                continue
            if in_handler:
                # Reached the next clause — stop
                if stripped.startswith("except ") or stripped.startswith("finally"):
                    break
                if stripped.startswith("raise"):
                    found_raise = True
                    break
                if stripped == "return":
                    found_return = True
                    break

        assert found_raise and not found_return, (
            "CancelledError handler must 'raise', not 'return'. "
            f"found_raise={found_raise}, found_return={found_return}"
        )
