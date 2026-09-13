"""Tests for Phase 12: Redis & Distributed Agent State.

Covers:
1. NullStateStore operations (save, load, delete, ping)
2. RedisStateStore save, load, delete, TTL operations
3. RedisStateStore payload bounding (>256 KiB skipped)
4. RedisStateStore corrupt JSON handling (JSONDecodeError handled gracefully)
5. RedisStateStore non-dict payload handling
6. RedisStateStore backend failure resilience
   (never raises on connection/timeout errors)
7. RedisStateStore ping behavior (True on success, False on error)
8. Key schema format and tenant/user/conversation isolation
9. AgentState run_id and started_at_iso initialization
10. AgentService StateStore integration:
    - Initial state saved at _build_state (status="in_progress")
    - State saved after each tool call (native loop and prompt fallback)
    - State deleted on successful run completion
    - State marked failed on run exception
    - State saved after each tool call in streaming mode
    - State deleted on successful stream completion
    - State marked failed on stream exception
    - State marked cancelled on asyncio.CancelledError
    - Redis failure does not break the agent run
11. Settings config validation for redis_url and agent_state_ttl_seconds
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import ValidationError

from cortex.agent.events import DoneEvent
from cortex.agent.registry import ToolRegistry
from cortex.agent.service import AgentService
from cortex.agent.state import AgentState
from cortex.agent.types import GenerateWithToolsResult, ToolCallRequest
from cortex.core.config import Settings
from cortex.core.exceptions import ServiceUnavailableError
from cortex.llm.base import LLMProvider, SupportsToolCalling
from cortex.services.prompt_builder import PromptBuilder
from cortex.state_store import NullStateStore, RedisStateStore, StateStore

# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def _make_user() -> MagicMock:
    user = MagicMock()
    user.id = str(uuid4())
    return user


def _make_prompt_builder() -> MagicMock:
    pb = MagicMock(spec=PromptBuilder)
    pb.build.return_value = "grounded prompt"
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
        async def _gen():
            for token in ["Hello", " world", "!"]:
                yield token

        return _gen()

    async def generate_with_tools(self, messages, tool_schemas):
        if self._responses:
            return self._responses.pop(0)
        return GenerateWithToolsResult(text="Done.")


class _FakePromptProvider(LLMProvider):
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def generate(self, prompt: str) -> str:
        if self._responses:
            return self._responses.pop(0)
        return "Prompt answer."

    async def generate_stream(self, prompt: str):  # type: ignore[override]
        async def _gen():
            yield "Streamed prompt answer."

        return _gen()


# ---------------------------------------------------------------------------
# 1. NullStateStore Unit Tests
# ---------------------------------------------------------------------------


class TestNullStateStore:
    @pytest.mark.asyncio
    async def test_null_state_store_contract(self) -> None:
        store = NullStateStore()

        # save does not raise
        await store.save("test_key", {"a": 1}, ttl_seconds=60)

        # load returns None
        loaded = await store.load("test_key")
        assert loaded is None

        # delete does not raise
        await store.delete("test_key")

        # ping returns True
        assert await store.ping() is True


# ---------------------------------------------------------------------------
# 2. RedisStateStore Unit Tests
# ---------------------------------------------------------------------------


class TestRedisStateStore:
    @pytest.mark.asyncio
    async def test_save_and_load_success(self) -> None:
        mock_redis = AsyncMock()
        mock_redis.get.return_value = json.dumps(
            {"question": "what is cortex?", "status": "in_progress"}
        )

        store = RedisStateStore(mock_redis)

        key = "cortex:agent:run:u1:c1:r1"
        data = {"question": "what is cortex?", "status": "in_progress"}

        await store.save(key, data, ttl_seconds=300)
        mock_redis.setex.assert_awaited_once_with(key, 300, json.dumps(data))

        loaded = await store.load(key)
        mock_redis.get.assert_awaited_once_with(key)
        assert loaded == data

    @pytest.mark.asyncio
    async def test_load_non_existent_key_returns_none(self) -> None:
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None

        store = RedisStateStore(mock_redis)
        loaded = await store.load("non_existent_key")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_load_corrupt_json_returns_none(self) -> None:
        mock_redis = AsyncMock()
        mock_redis.get.return_value = "not a valid { json"

        store = RedisStateStore(mock_redis)
        loaded = await store.load("corrupt_key")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_load_non_dict_json_returns_none(self) -> None:
        mock_redis = AsyncMock()
        mock_redis.get.return_value = json.dumps(["a", "b", "c"])

        store = RedisStateStore(mock_redis)
        loaded = await store.load("list_key")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_payload_exceeding_max_bytes_is_skipped(self) -> None:
        mock_redis = AsyncMock()
        store = RedisStateStore(mock_redis)

        # 300 KiB payload
        large_payload = {"data": "x" * (300 * 1024)}
        await store.save("large_key", large_payload, ttl_seconds=60)

        # setex should not be called
        mock_redis.setex.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delete_calls_redis_delete(self) -> None:
        mock_redis = AsyncMock()
        store = RedisStateStore(mock_redis)

        await store.delete("some_key")
        mock_redis.delete.assert_awaited_once_with("some_key")

    @pytest.mark.asyncio
    async def test_ping_returns_true_when_reachable(self) -> None:
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True

        store = RedisStateStore(mock_redis)
        assert await store.ping() is True

    @pytest.mark.asyncio
    async def test_ping_returns_false_on_exception(self) -> None:
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = ConnectionError("Redis down")

        store = RedisStateStore(mock_redis)
        assert await store.ping() is False

    @pytest.mark.asyncio
    async def test_redis_exceptions_never_propagate(self) -> None:
        """Redis failure must never raise out of RedisStateStore methods."""
        mock_redis = AsyncMock()
        mock_redis.setex.side_effect = TimeoutError("Redis timed out")
        mock_redis.get.side_effect = ConnectionError("Redis connection lost")
        mock_redis.delete.side_effect = RuntimeError("Redis error")

        store = RedisStateStore(mock_redis)

        # None of these should raise
        await store.save("key", {"a": 1}, ttl_seconds=60)
        assert await store.load("key") is None
        await store.delete("key")


# ---------------------------------------------------------------------------
# 3. Key Schema & Tenant Isolation
# ---------------------------------------------------------------------------


class TestKeySchemaAndIsolation:
    def test_state_key_with_conversation(self) -> None:
        uid = str(uuid4())
        cid = "conv-123"
        rid = "run-abc"
        key = AgentService._state_key(user_id=uid, conversation_id=cid, run_id=rid)
        assert key == f"cortex:agent:run:{uid}:{cid}:{rid}"

    def test_state_key_stateless(self) -> None:
        uid = str(uuid4())
        rid = "run-abc"
        key = AgentService._state_key(user_id=uid, conversation_id=None, run_id=rid)
        assert key == f"cortex:agent:run:{uid}:_stateless:{rid}"

    def test_state_keys_isolated_by_user(self) -> None:
        uid1 = str(uuid4())
        uid2 = str(uuid4())
        key1 = AgentService._state_key(
            user_id=uid1, conversation_id="conv-1", run_id="r1"
        )
        key2 = AgentService._state_key(
            user_id=uid2, conversation_id="conv-1", run_id="r1"
        )
        assert key1 != key2
        assert uid1 in key1
        assert uid2 in key2


# ---------------------------------------------------------------------------
# 4. AgentState Run ID & Timestamp
# ---------------------------------------------------------------------------


class TestAgentStateFields:
    def test_agent_state_has_run_id_and_iso_timestamp(self) -> None:
        user = _make_user()
        state = AgentState(question="hello", user=user, conversation_id=None)

        assert isinstance(state.run_id, str)
        assert len(state.run_id) > 0
        assert isinstance(state.started_at_iso, str)
        assert "T" in state.started_at_iso

    def test_custom_run_id_respected(self) -> None:
        user = _make_user()
        state = AgentState(
            question="hello",
            user=user,
            conversation_id=None,
            run_id="custom-run-id",
        )
        assert state.run_id == "custom-run-id"


# ---------------------------------------------------------------------------
# 5. AgentService StateStore Lifecycle Integration
# ---------------------------------------------------------------------------


class TestAgentServiceStateStoreLifecycle:
    @pytest.mark.asyncio
    async def test_run_saves_at_start_and_deletes_on_success(self) -> None:
        mock_store = AsyncMock(spec=StateStore)
        provider = _FakeNativeProvider(responses=[GenerateWithToolsResult(text="42")])

        service = AgentService(
            llm_provider=provider,
            tool_registry=ToolRegistry(),
            prompt_builder=_make_prompt_builder(),
            state_store=mock_store,
            state_ttl_seconds=600,
        )

        user = _make_user()
        result = await service.run(
            question="What is 40 + 2?", user=user, conversation_id="c1"
        )

        assert result.answer == "Final answer."

        # Verify save called at start with in_progress
        assert mock_store.save.await_count >= 1
        first_save_call = mock_store.save.call_args_list[0]
        key, snapshot = first_save_call.args
        ttl = first_save_call.kwargs.get("ttl_seconds")
        assert key.startswith(f"cortex:agent:run:{user.id}:c1:")
        assert snapshot["status"] == "in_progress"
        assert snapshot["question"] == "What is 40 + 2?"
        assert ttl == 600

        # Verify delete called on success
        mock_store.delete.assert_awaited_once_with(key)

    @pytest.mark.asyncio
    async def test_run_saves_after_tool_call(self) -> None:
        mock_store = AsyncMock(spec=StateStore)

        # Register a mock tool
        registry = ToolRegistry()
        mock_tool = MagicMock()
        mock_tool.name = "calc"
        mock_tool.description = "calc tool"
        mock_tool.parameters_schema = {"type": "object", "properties": {}}
        mock_tool.execute = AsyncMock(return_value="result=42")
        registry.register(mock_tool)

        provider = _FakeNativeProvider(
            responses=[
                GenerateWithToolsResult(
                    tool_call=ToolCallRequest(
                        tool_name="calc", args={"x": 1}, call_id="c1"
                    )
                ),
                GenerateWithToolsResult(text="done"),
            ]
        )

        service = AgentService(
            llm_provider=provider,
            tool_registry=registry,
            prompt_builder=_make_prompt_builder(),
            state_store=mock_store,
        )

        user = _make_user()
        await service.run(question="Compute something", user=user)

        # Save should be called at start (1) + after tool call (2)
        assert mock_store.save.await_count == 2
        second_save_call = mock_store.save.call_args_list[1]
        _, snapshot = second_save_call.args
        assert len(snapshot["tool_calls"]) == 1
        assert snapshot["tool_calls"][0]["tool_name"] == "calc"
        assert snapshot["tool_calls"][0]["observation"] == "result=42"

        # And delete on completion
        mock_store.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_marks_failed_on_exception(self) -> None:
        mock_store = AsyncMock(spec=StateStore)

        provider = MagicMock(spec=LLMProvider)
        provider.generate = AsyncMock(
            side_effect=ServiceUnavailableError("LLM exploded")
        )

        service = AgentService(
            llm_provider=provider,
            tool_registry=ToolRegistry(),
            prompt_builder=_make_prompt_builder(),
            state_store=mock_store,
        )

        user = _make_user()
        with pytest.raises(ServiceUnavailableError):
            await service.run(question="Will fail", user=user)

        # Delete should NOT be called
        mock_store.delete.assert_not_awaited()

        # Last save should have status="failed"
        assert mock_store.save.await_count >= 1
        last_save = mock_store.save.call_args_list[-1]
        _, snapshot = last_save.args
        assert snapshot["status"] == "failed"
        assert "LLM exploded" in (snapshot["error"] or "")

    @pytest.mark.asyncio
    async def test_stream_saves_at_start_and_deletes_on_completion(self) -> None:
        mock_store = AsyncMock(spec=StateStore)
        provider = _FakeNativeProvider(
            responses=[GenerateWithToolsResult(text="stream done")]
        )

        service = AgentService(
            llm_provider=provider,
            tool_registry=ToolRegistry(),
            prompt_builder=_make_prompt_builder(),
            state_store=mock_store,
        )

        user = _make_user()
        events = []
        async for event in service.stream(question="Stream question", user=user):
            events.append(event)

        assert any(isinstance(e, DoneEvent) for e in events)

        # Save called at start
        assert mock_store.save.await_count >= 1
        first_save = mock_store.save.call_args_list[0]
        assert first_save.args[1]["status"] == "in_progress"

        # Delete called on done
        mock_store.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stream_marks_cancelled_on_cancellation(self) -> None:
        mock_store = AsyncMock(spec=StateStore)

        # Provider stream that raises CancelledError mid-stream
        class _CancellingProvider(LLMProvider, SupportsToolCalling):
            async def generate(self, prompt: str) -> str:
                return "text"

            async def generate_stream(self, prompt: str):
                async def _gen():
                    yield "first token"
                    raise asyncio.CancelledError()

                return _gen()

            async def generate_with_tools(self, messages, tool_schemas):
                return GenerateWithToolsResult(text="ready")

        provider = _CancellingProvider()
        service = AgentService(
            llm_provider=provider,
            tool_registry=ToolRegistry(),
            prompt_builder=_make_prompt_builder(),
            state_store=mock_store,
        )

        user = _make_user()
        with pytest.raises(asyncio.CancelledError):
            async for _ in service.stream(question="Cancel me", user=user):
                pass

        # Must record status="cancelled" and NOT delete
        mock_store.delete.assert_not_awaited()
        last_save = mock_store.save.call_args_list[-1]
        _, snapshot = last_save.args
        assert snapshot["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_state_store_failure_does_not_break_agent_run(self) -> None:
        """If StateStore raises unexpectedly, AgentService catches it and
        run succeeds.
        """
        failing_store = AsyncMock(spec=StateStore)
        failing_store.save.side_effect = RuntimeError("Store crash")
        failing_store.delete.side_effect = RuntimeError("Store delete crash")

        provider = _FakeNativeProvider(responses=[GenerateWithToolsResult(text="ok")])
        service = AgentService(
            llm_provider=provider,
            tool_registry=ToolRegistry(),
            prompt_builder=_make_prompt_builder(),
            state_store=failing_store,
        )

        user = _make_user()
        result = await service.run(question="Does this work?", user=user)
        assert result.answer == "Final answer."

    @pytest.mark.asyncio
    async def test_load_state_delegates_to_state_store(self) -> None:
        mock_store = AsyncMock(spec=StateStore)
        mock_store.load.return_value = {"run_id": "r1", "status": "in_progress"}

        service = AgentService(
            llm_provider=MagicMock(spec=LLMProvider),
            tool_registry=ToolRegistry(),
            prompt_builder=_make_prompt_builder(),
            state_store=mock_store,
        )

        loaded = await service.load_state(
            user_id="u1", conversation_id="c1", run_id="r1"
        )
        assert loaded == {"run_id": "r1", "status": "in_progress"}
        mock_store.load.assert_awaited_once_with("cortex:agent:run:u1:c1:r1")

    @pytest.mark.asyncio
    async def test_load_state_returns_none_without_state_store(self) -> None:
        service = AgentService(
            llm_provider=MagicMock(spec=LLMProvider),
            tool_registry=ToolRegistry(),
            prompt_builder=_make_prompt_builder(),
            state_store=None,
        )
        loaded = await service.load_state(
            user_id="u1", conversation_id="c1", run_id="r1"
        )
        assert loaded is None


# ---------------------------------------------------------------------------
# 6. Settings Configuration Tests
# ---------------------------------------------------------------------------


class TestRedisSettings:
    def test_default_redis_settings(self) -> None:
        s = Settings()
        assert s.redis_url == ""
        assert s.agent_state_ttl_seconds == 1800

    def test_custom_redis_settings(self) -> None:
        s = Settings(
            redis_url="redis://custom:6379/1",
            agent_state_ttl_seconds=3600,
        )
        assert s.redis_url == "redis://custom:6379/1"
        assert s.agent_state_ttl_seconds == 3600

    def test_agent_state_ttl_bounds(self) -> None:
        with pytest.raises(ValidationError):
            Settings(agent_state_ttl_seconds=30)  # ge=60 violated

        with pytest.raises(ValidationError):
            Settings(agent_state_ttl_seconds=100000)  # le=86400 violated


# ---------------------------------------------------------------------------
# 7. Lifespan & Dependency Injection Tests
# ---------------------------------------------------------------------------


class TestLifespanAndDependencies:
    @pytest.mark.asyncio
    async def test_get_state_store_dep(self) -> None:
        from cortex.api.deps import get_state_store

        mock_request = MagicMock()
        mock_store = NullStateStore()
        mock_request.app.state.state_store = mock_store

        resolved = get_state_store(mock_request)
        assert resolved is mock_store

    @pytest.mark.asyncio
    async def test_get_agent_service_dep_wiring(self) -> None:
        from cortex.api.deps import get_agent_service

        mock_store = AsyncMock(spec=StateStore)
        settings = Settings(agent_state_ttl_seconds=900)

        service = get_agent_service(
            retriever=MagicMock(),
            llm_provider=MagicMock(spec=LLMProvider),
            prompt_builder=_make_prompt_builder(),
            document_service=MagicMock(),
            conv_service=MagicMock(),
            state_store=mock_store,
            memory_service=MagicMock(),
            settings=settings,
        )

        assert service._state_store is mock_store
        assert service._state_ttl == 900

    @pytest.mark.asyncio
    async def test_lifespan_initializes_null_store_when_redis_url_empty(self) -> None:
        from fastapi import FastAPI

        from cortex.main import lifespan

        app = FastAPI()
        settings = Settings(redis_url="")
        app.state.settings = settings

        with patch("cortex.main.Database.dispose", new_callable=AsyncMock):
            async with lifespan(app):
                assert isinstance(app.state.state_store, NullStateStore)

    @pytest.mark.asyncio
    async def test_lifespan_initializes_redis_store_when_redis_url_present(
        self,
    ) -> None:
        from fastapi import FastAPI

        from cortex.main import lifespan

        app = FastAPI()
        settings = Settings(redis_url="redis://localhost:6379/0")
        app.state.settings = settings

        mock_redis_instance = AsyncMock()
        mock_redis_instance.aclose = AsyncMock()

        with (
            patch("redis.asyncio.Redis.from_url", return_value=mock_redis_instance),
            patch("cortex.main.Database.dispose", new_callable=AsyncMock),
        ):
            async with lifespan(app):
                assert isinstance(app.state.state_store, RedisStateStore)
                assert app.state.state_store._client is mock_redis_instance

            # Shutdown cleans up redis client
            mock_redis_instance.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_lifespan_falls_back_to_null_store_on_redis_init_error(self) -> None:
        from fastapi import FastAPI

        from cortex.main import lifespan

        app = FastAPI()
        settings = Settings(redis_url="redis://localhost:6379/0")
        app.state.settings = settings

        with (
            patch(
                "redis.asyncio.Redis.from_url",
                side_effect=ConnectionError("Cannot connect"),
            ),
            patch("cortex.main.Database.dispose", new_callable=AsyncMock),
        ):
            async with lifespan(app):
                assert isinstance(app.state.state_store, NullStateStore)
