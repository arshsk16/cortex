"""Tests for Phase 13B -- agent memory integration.

Covers:
- Memory retrieval injected into AgentState during _build_state
- PromptBuilder includes LONG-TERM MEMORY section when hits are present
- PromptBuilder omits memory section when no hits
- Memory retrieval failure is non-fatal (agent continues without it)
- Ownership isolation: memory_service.search is called with the correct user
- Streaming run includes memory_hits in prompt
- AgentState.has_memory_hits property
- memory_retrieval_limit=0 disables retrieval entirely
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from cortex.agent.state import AgentState
from cortex.retrieval.models import RetrievalResult
from cortex.schemas.memory import MemoryRead, MemorySearchResult
from cortex.services.prompt_builder import PromptBuilder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _memory_result(content: str, score: float = 0.85) -> MemorySearchResult:
    now = datetime.now(UTC)
    return MemorySearchResult(
        memory=MemoryRead(
            id=str(uuid4()),
            user_id=str(uuid4()),
            content=content,
            memory_metadata=None,
            created_at=now,
            updated_at=now,
        ),
        score=score,
    )


def _retrieval_result() -> RetrievalResult:
    return RetrievalResult(
        chunk_id=str(uuid4()),
        document_id=str(uuid4()),
        chunk_index=0,
        text="A relevant document excerpt.",
        score=0.9,
    )


# ---------------------------------------------------------------------------
# AgentState.has_memory_hits
# ---------------------------------------------------------------------------


def test_agent_state_has_memory_hits_false_by_default() -> None:
    """AgentState.has_memory_hits is False when memory_hits is empty."""
    from cortex.db.models.user import User, UserRole

    now = datetime.now(UTC)
    user = User(
        id=str(uuid4()),
        email="u@example.com",
        username="u",
        full_name="U",
        hashed_password="x",
        role=UserRole.USER,
        is_active=True,
        is_verified=False,
        created_at=now,
        updated_at=now,
    )
    state = AgentState(question="q", user=user, conversation_id=None)
    assert not state.has_memory_hits
    assert state.memory_hits == []


def test_agent_state_has_memory_hits_true_when_populated() -> None:
    """AgentState.has_memory_hits is True when memory_hits list is non-empty."""
    from cortex.db.models.user import User, UserRole

    now = datetime.now(UTC)
    user = User(
        id=str(uuid4()),
        email="u@example.com",
        username="u",
        full_name="U",
        hashed_password="x",
        role=UserRole.USER,
        is_active=True,
        is_verified=False,
        created_at=now,
        updated_at=now,
    )
    hit = _memory_result("Paris is the capital of France")
    state = AgentState(question="q", user=user, conversation_id=None, memory_hits=[hit])
    assert state.has_memory_hits
    assert len(state.memory_hits) == 1


# ---------------------------------------------------------------------------
# PromptBuilder memory section
# ---------------------------------------------------------------------------


def test_prompt_builder_includes_memory_section() -> None:
    """Build() includes LONG-TERM MEMORY section when memory_hits is provided."""
    pb = PromptBuilder()
    hits = [
        _memory_result("Alice prefers Python over Java.", score=0.9),
        _memory_result("Bob is allergic to peanuts.", score=0.75),
    ]
    prompt = pb.build(
        question="What does Alice prefer?",
        retrieved_chunks=[],
        memory_hits=hits,
    )
    assert "=== LONG-TERM MEMORY ===" in prompt
    assert "=== END OF LONG-TERM MEMORY ===" in prompt
    assert "Alice prefers Python" in prompt
    assert "Bob is allergic" in prompt
    # Memory section appears BEFORE document context
    mem_pos = prompt.index("=== LONG-TERM MEMORY ===")
    ctx_pos = prompt.index("=== DOCUMENT CONTEXT ===")
    assert mem_pos < ctx_pos


def test_prompt_builder_omits_memory_section_when_none() -> None:
    """Build() omits memory section when memory_hits is None."""
    pb = PromptBuilder()
    prompt = pb.build(
        question="What is 2+2?",
        retrieved_chunks=[],
        memory_hits=None,
    )
    assert "LONG-TERM MEMORY" not in prompt


def test_prompt_builder_omits_memory_section_when_empty_list() -> None:
    """Build() omits memory section when memory_hits is empty list."""
    pb = PromptBuilder()
    prompt = pb.build(
        question="What is 2+2?",
        retrieved_chunks=[],
        memory_hits=[],
    )
    assert "LONG-TERM MEMORY" not in prompt


def test_prompt_builder_memory_section_between_history_and_context() -> None:
    """Memory section sits between HISTORY and DOCUMENT CONTEXT."""
    from cortex.db.models.conversation import Message

    pb = PromptBuilder()
    now = datetime.now(UTC)
    msg = Message(
        id=str(uuid4()),
        conversation_id=str(uuid4()),
        role="user",
        content="Hello",
        created_at=now,
    )
    hits = [_memory_result("Some remembered fact.", score=0.8)]
    chunks = [_retrieval_result()]
    prompt = pb.build(
        question="test",
        retrieved_chunks=chunks,
        history=[msg],
        memory_hits=hits,
    )
    hist_pos = prompt.index("=== CONVERSATION HISTORY ===")
    mem_pos = prompt.index("=== LONG-TERM MEMORY ===")
    ctx_pos = prompt.index("=== DOCUMENT CONTEXT ===")
    assert hist_pos < mem_pos < ctx_pos


def test_prompt_builder_memory_block_format() -> None:
    """Memory entries include relevance percentage."""
    pb = PromptBuilder()
    hit = _memory_result("Cats are mammals.", score=0.92)
    prompt = pb.build(
        question="What are cats?",
        retrieved_chunks=[],
        memory_hits=[hit],
    )
    assert "[Memory 1 | relevance=92%]" in prompt
    assert "Cats are mammals." in prompt


def test_prompt_builder_backward_compat_no_memory_arg() -> None:
    """build() without memory_hits arg still works (backward compatibility)."""
    pb = PromptBuilder()
    prompt = pb.build(
        question="What is the capital of France?",
        retrieved_chunks=[],
    )
    assert "LONG-TERM MEMORY" not in prompt
    assert "=== QUESTION ===" in prompt


# ---------------------------------------------------------------------------
# AgentService memory integration (unit tests with mocked service)
# ---------------------------------------------------------------------------


def _make_mock_agent_service(
    *,
    memory_service=None,
    memory_limit: int = 5,
    return_hits: list[MemorySearchResult] | None = None,
):
    """Build an AgentService with mocked dependencies."""
    from cortex.agent.registry import ToolRegistry
    from cortex.agent.service import AgentService

    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value="The answer.")
    mock_llm.generate_stream = AsyncMock(
        return_value=_async_iter(["Token1", " Token2"])
    )

    mock_pb = MagicMock(spec=PromptBuilder)
    mock_pb.build = MagicMock(return_value="prompt text")

    registry = ToolRegistry()

    if memory_service is None and return_hits is not None:
        memory_service = MagicMock()
        memory_service.search = AsyncMock(return_value=return_hits)

    svc = AgentService(
        llm_provider=mock_llm,
        tool_registry=registry,
        prompt_builder=mock_pb,
        conversation_service=None,
        state_store=None,
        memory_service=memory_service,
        memory_retrieval_limit=memory_limit,
    )
    return svc, mock_pb


async def _async_iter(items):
    for item in items:
        yield item


def _make_user():
    from cortex.db.models.user import User, UserRole

    now = datetime.now(UTC)
    return User(
        id=str(uuid4()),
        email="alice@example.com",
        username="alice",
        full_name="Alice",
        hashed_password="hashed",
        role=UserRole.USER,
        is_active=True,
        is_verified=False,
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_agent_service_calls_memory_search_on_run() -> None:
    """AgentService.run() calls memory_service.search with user and question."""
    user = _make_user()
    hits = [_memory_result("Remembered fact")]
    svc, mock_pb = _make_mock_agent_service(return_hits=hits)

    await svc.run(question="What do I know?", user=user)

    svc._memory_service.search.assert_awaited_once_with(
        user=user,
        query="What do I know?",
        limit=5,
    )
    # Prompt builder receives memory_hits
    call_kwargs = mock_pb.build.call_args.kwargs
    assert call_kwargs["memory_hits"] == hits


@pytest.mark.asyncio
async def test_agent_service_no_memory_when_limit_zero() -> None:
    """memory_retrieval_limit=0 disables retrieval even when service is present."""
    user = _make_user()
    mock_mem_svc = MagicMock()
    mock_mem_svc.search = AsyncMock(return_value=[])

    svc, mock_pb = _make_mock_agent_service(
        memory_service=mock_mem_svc,
        memory_limit=0,
    )
    await svc.run(question="test question", user=user)

    mock_mem_svc.search.assert_not_awaited()
    call_kwargs = mock_pb.build.call_args.kwargs
    # memory_hits should be None (empty list -> falsy -> passed as None)
    assert not call_kwargs.get("memory_hits")


@pytest.mark.asyncio
async def test_agent_service_memory_failure_is_nonfatal() -> None:
    """Memory retrieval errors must not abort the agent run."""
    user = _make_user()
    mock_mem_svc = MagicMock()
    mock_mem_svc.search = AsyncMock(side_effect=RuntimeError("Chroma down"))

    svc, mock_pb = _make_mock_agent_service(memory_service=mock_mem_svc)

    # Should complete without raising
    result = await svc.run(question="Will this crash?", user=user)
    assert result.answer == "The answer."

    # Prompt builder called with empty/None memory_hits
    call_kwargs = mock_pb.build.call_args.kwargs
    assert not call_kwargs.get("memory_hits")


@pytest.mark.asyncio
async def test_agent_service_no_memory_service() -> None:
    """AgentService without a memory_service runs normally (no retrieval)."""
    user = _make_user()
    svc, mock_pb = _make_mock_agent_service(memory_service=None)

    result = await svc.run(question="No memory question", user=user)
    assert result.answer == "The answer."
    call_kwargs = mock_pb.build.call_args.kwargs
    assert not call_kwargs.get("memory_hits")


@pytest.mark.asyncio
async def test_agent_service_ownership_isolation() -> None:
    """Memory retrieval is called with the exact requesting user, not any other."""
    alice = _make_user()
    bob = _make_user()
    bob.email = "bob@example.com"
    bob.username = "bob"

    hits_for_alice = [_memory_result("Alice fact")]
    svc, _ = _make_mock_agent_service(return_hits=hits_for_alice)

    await svc.run(question="Alice question", user=alice)
    # search was called with alice only
    svc._memory_service.search.assert_awaited_once()
    call_user = svc._memory_service.search.call_args.kwargs["user"]
    assert call_user.id == alice.id
    assert call_user.id != bob.id


@pytest.mark.asyncio
async def test_agent_service_stream_includes_memory_hits() -> None:
    """AgentService.stream() also retrieves memories and injects them into prompt."""
    from cortex.agent.events import DoneEvent, TokenEvent

    user = _make_user()
    hits = [_memory_result("Streamed memory")]
    svc, mock_pb = _make_mock_agent_service(return_hits=hits)

    events = []
    async for event in svc.stream(question="Stream question", user=user):
        events.append(event)

    svc._memory_service.search.assert_awaited_once_with(
        user=user,
        query="Stream question",
        limit=5,
    )
    call_kwargs = mock_pb.build.call_args.kwargs
    assert call_kwargs["memory_hits"] == hits

    token_events = [e for e in events if isinstance(e, TokenEvent)]
    done_events = [e for e in events if isinstance(e, DoneEvent)]
    assert token_events
    assert done_events


@pytest.mark.asyncio
async def test_agent_service_stream_memory_failure_nonfatal() -> None:
    """Memory failure during streaming does not abort the stream."""
    from cortex.agent.events import DoneEvent

    user = _make_user()
    mock_mem_svc = MagicMock()
    mock_mem_svc.search = AsyncMock(side_effect=RuntimeError("Redis timeout"))

    svc, _ = _make_mock_agent_service(memory_service=mock_mem_svc)

    events = []
    async for event in svc.stream(question="Stream without memory", user=user):
        events.append(event)

    done_events = [e for e in events if isinstance(e, DoneEvent)]
    assert done_events


# ---------------------------------------------------------------------------
# _snapshot includes memory_hits_count
# ---------------------------------------------------------------------------


def test_snapshot_includes_memory_hits_count() -> None:
    """The Redis state snapshot records how many memory hits were retrieved."""
    from cortex.agent.registry import ToolRegistry
    from cortex.agent.service import AgentService
    from cortex.db.models.user import User, UserRole

    now = datetime.now(UTC)
    user = User(
        id=str(uuid4()),
        email="u@example.com",
        username="u",
        full_name="U",
        hashed_password="x",
        role=UserRole.USER,
        is_active=True,
        is_verified=False,
        created_at=now,
        updated_at=now,
    )
    svc = AgentService(
        llm_provider=MagicMock(),
        tool_registry=ToolRegistry(),
        prompt_builder=PromptBuilder(),
    )
    hits = [_memory_result("fact one"), _memory_result("fact two")]
    state = AgentState(
        question="q", user=user, conversation_id=None, memory_hits=hits
    )
    snap = svc._snapshot(state, status="in_progress")
    assert snap["memory_hits_count"] == 2
