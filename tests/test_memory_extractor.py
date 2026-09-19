"""Tests for automatic memory extraction and reconciliation service."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from cortex.schemas.memory import (
    MemoryRead,
    MemorySearchResult,
)
from cortex.services.memory_extractor import (
    ExtractedFacts,
    FactEvaluation,
    MemoryExtractorService,
)


@pytest.fixture
def mock_llm():
    return AsyncMock()


@pytest.fixture
def mock_memory_service():
    return AsyncMock()


@pytest.fixture
def user():
    user_mock = MagicMock()
    user_mock.id = "test-user-id"
    return user_mock


@pytest.mark.asyncio
async def test_extractor_no_facts(mock_llm, mock_memory_service, user):
    mock_llm.generate_structured.return_value = ExtractedFacts(facts=[])

    extractor = MemoryExtractorService(
        llm_provider=mock_llm,
        memory_service=mock_memory_service,
    )
    await extractor.extract_and_store(
        user=user,
        question="hello",
        answer="hi",
        memory_hits=[],
    )

    mock_memory_service.search.assert_not_called()
    mock_memory_service.create.assert_not_called()


@pytest.mark.asyncio
async def test_extractor_creates_new_fact(mock_llm, mock_memory_service, user):
    mock_llm.generate_structured.side_effect = [
        ExtractedFacts(facts=["User likes Python"]),
        FactEvaluation(action="create", target_memory_id=None),
    ]
    mock_memory_service.search.return_value = []

    extractor = MemoryExtractorService(
        llm_provider=mock_llm,
        memory_service=mock_memory_service,
    )
    await extractor.extract_and_store(
        user=user,
        question="I like Python",
        answer="Noted.",
        memory_hits=[],
    )

    mock_memory_service.create.assert_called_once()
    call_args = mock_memory_service.create.call_args
    assert call_args.kwargs["user"] == user
    assert call_args.kwargs["payload"].content == "User likes Python"


@pytest.mark.asyncio
async def test_extractor_updates_existing_fact(
    mock_llm,
    mock_memory_service,
    user,
):
    mock_llm.generate_structured.side_effect = [
        ExtractedFacts(facts=["User likes Rust now"]),
        FactEvaluation(action="update", target_memory_id="mem-123"),
    ]

    now = datetime.now(UTC)
    existing_mem = MemoryRead(
        id="mem-123",
        user_id=user.id,
        content="User likes Python",
        memory_metadata={},
        created_at=now,
        updated_at=now,
    )
    mock_memory_service.search.return_value = [
        MemorySearchResult(memory=existing_mem, score=0.9)
    ]

    extractor = MemoryExtractorService(
        llm_provider=mock_llm,
        memory_service=mock_memory_service,
    )
    await extractor.extract_and_store(
        user=user,
        question="I like Rust now",
        answer="Okay.",
        memory_hits=[],
    )

    mock_memory_service.update.assert_called_once()
    call_args = mock_memory_service.update.call_args
    assert call_args.kwargs["memory_id"] == "mem-123"
    assert call_args.kwargs["user"] == user
    assert call_args.kwargs["payload"].content == "User likes Rust now"


@pytest.mark.asyncio
async def test_extractor_ignores_duplicate(
    mock_llm,
    mock_memory_service,
    user,
):
    mock_llm.generate_structured.side_effect = [
        ExtractedFacts(facts=["User likes Python"]),
        FactEvaluation(action="ignore", target_memory_id=None),
    ]

    now = datetime.now(UTC)
    existing_mem = MemoryRead(
        id="mem-123",
        user_id=user.id,
        content="User likes Python",
        memory_metadata={},
        created_at=now,
        updated_at=now,
    )
    mock_memory_service.search.return_value = [
        MemorySearchResult(memory=existing_mem, score=1.0)
    ]

    extractor = MemoryExtractorService(
        llm_provider=mock_llm,
        memory_service=mock_memory_service,
    )
    await extractor.extract_and_store(
        user=user,
        question="I still like Python",
        answer="Cool.",
        memory_hits=[],
    )

    mock_memory_service.update.assert_not_called()
    mock_memory_service.create.assert_not_called()


@pytest.mark.asyncio
async def test_extractor_target_id_not_in_candidates_falls_back_to_create(
    mock_llm,
    mock_memory_service,
    user,
):
    """If LLM hallucinates an arbitrary ID not in candidates, creates new."""
    mock_llm.generate_structured.side_effect = [
        ExtractedFacts(facts=["User is a designer"]),
        FactEvaluation(action="update", target_memory_id="alien-id-999"),
    ]

    now = datetime.now(UTC)
    existing_mem = MemoryRead(
        id="mem-123",
        user_id=user.id,
        content="User is a developer",
        memory_metadata={},
        created_at=now,
        updated_at=now,
    )
    mock_memory_service.search.return_value = [
        MemorySearchResult(memory=existing_mem, score=0.8)
    ]

    extractor = MemoryExtractorService(
        llm_provider=mock_llm,
        memory_service=mock_memory_service,
    )
    await extractor.extract_and_store(
        user=user,
        question="I design interfaces now",
        answer="Great.",
        memory_hits=[],
    )

    # Should NOT call update on alien-id-999
    mock_memory_service.update.assert_not_called()
    # Should safely create
    mock_memory_service.create.assert_called_once()
    assert (
        mock_memory_service.create.call_args.kwargs["payload"].content
        == "User is a designer"
    )


@pytest.mark.asyncio
async def test_extractor_combines_turn_hits_and_search_hits(
    mock_llm,
    mock_memory_service,
    user,
):
    """Reconciliation candidate pool includes both turn memory_hits and search."""
    mock_llm.generate_structured.side_effect = [
        ExtractedFacts(facts=["User lives in London"]),
        FactEvaluation(action="update", target_memory_id="hit-1"),
    ]

    now = datetime.now(UTC)
    mem1 = MemoryRead(
        id="hit-1",
        user_id=user.id,
        content="User lives in Paris",
        memory_metadata={},
        created_at=now,
        updated_at=now,
    )
    mem2 = MemoryRead(
        id="hit-2",
        user_id=user.id,
        content="User loves travel",
        memory_metadata={},
        created_at=now,
        updated_at=now,
    )
    # hit-1 came from turn hits, hit-2 came from search
    turn_hits = [MemorySearchResult(memory=mem1, score=0.85)]
    mock_memory_service.search.return_value = [
        MemorySearchResult(memory=mem2, score=0.75)
    ]

    extractor = MemoryExtractorService(
        llm_provider=mock_llm,
        memory_service=mock_memory_service,
    )
    await extractor.extract_and_store(
        user=user,
        question="I moved to London",
        answer="Noted.",
        memory_hits=turn_hits,
    )

    mock_memory_service.update.assert_called_once()
    assert (
        mock_memory_service.update.call_args.kwargs["memory_id"] == "hit-1"
    )


@pytest.mark.asyncio
async def test_extractor_handles_exceptions_gracefully(
    mock_llm,
    mock_memory_service,
    user,
    caplog,
):
    mock_llm.generate_structured.side_effect = Exception("LLM Error")

    extractor = MemoryExtractorService(
        llm_provider=mock_llm,
        memory_service=mock_memory_service,
    )
    await extractor.extract_and_store(
        user=user,
        question="fail",
        answer="fail",
        memory_hits=[],
    )

    assert "Background memory extraction failed" in caplog.text
