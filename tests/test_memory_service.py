"""Unit tests for MemoryService.

All external I/O (database, embedding, Chroma) is mocked so these tests
run without any network or database access.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from cortex.core.exceptions import NotFoundError
from cortex.db.models.memory import Memory
from cortex.db.models.user import User, UserRole
from cortex.schemas.memory import (
    MemoryCreate,
    MemoryList,
    MemoryRead,
    MemorySearchResult,
    MemoryUpdate,
)
from cortex.services.memory import MemoryService
from cortex.vectorstore.memory_store import MemoryVectorStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 384


def _make_user(user_id: str | None = None) -> User:
    now = datetime.now(UTC)
    return User(
        id=user_id or str(uuid4()),
        email="alice@example.com",
        username="alice",
        full_name="Alice Example",
        hashed_password="hashed",
        role=UserRole.USER,
        is_active=True,
        is_verified=False,
        created_at=now,
        updated_at=now,
    )


def _make_memory(user_id: str, content: str = "Test memory content") -> Memory:
    now = datetime.now(UTC)
    return Memory(
        id=str(uuid4()),
        user_id=user_id,
        content=content,
        memory_metadata=None,
        created_at=now,
        updated_at=now,
    )


def _make_embedding() -> list[float]:
    return [0.1] * EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.add = MagicMock()
    session.delete = AsyncMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def mock_embedding_provider():
    provider = MagicMock()
    provider.embed = AsyncMock(return_value=_make_embedding())
    return provider


@pytest.fixture
def mock_memory_vector_store():
    store = MagicMock(spec=MemoryVectorStore)
    store.upsert = AsyncMock()
    store.delete = AsyncMock()
    store.search = AsyncMock(return_value=[])
    store.delete_all_for_user = AsyncMock()
    return store


@pytest.fixture
def memory_service(mock_session, mock_embedding_provider, mock_memory_vector_store):
    return MemoryService(
        session=mock_session,
        embedding_provider=mock_embedding_provider,
        memory_vector_store=mock_memory_vector_store,
    )


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_memory_success(
    memory_service: MemoryService,
    mock_session,
    mock_embedding_provider,
    mock_memory_vector_store,
) -> None:
    """create() persists to PG, embeds, then upserts to Chroma."""
    user = _make_user()
    mem = _make_memory(user.id)
    mock_session.refresh = AsyncMock(side_effect=lambda obj: None)

    # Simulate ORM attaching id/timestamps
    def _side_add(obj: Any) -> None:
        if isinstance(obj, Memory):
            obj.id = mem.id
            obj.created_at = mem.created_at
            obj.updated_at = mem.updated_at

    mock_session.add = MagicMock(side_effect=_side_add)
    mock_session.refresh = AsyncMock()

    payload = MemoryCreate(content="Test memory content")
    result = await memory_service.create(user=user, payload=payload)

    mock_session.flush.assert_awaited_once()
    mock_embedding_provider.embed.assert_awaited_once_with("Test memory content")
    mock_memory_vector_store.upsert.assert_awaited_once()
    assert isinstance(result, MemoryRead)


@pytest.mark.asyncio
async def test_create_memory_with_metadata(
    memory_service: MemoryService,
    mock_session,
    mock_memory_vector_store,
) -> None:
    """create() stores supplied metadata in the Memory model."""
    user = _make_user()

    def _side_add(obj: Any) -> None:
        if isinstance(obj, Memory):
            obj.id = str(uuid4())
            obj.created_at = datetime.now(UTC)
            obj.updated_at = datetime.now(UTC)

    mock_session.add = MagicMock(side_effect=_side_add)

    payload = MemoryCreate(
        content="Meeting with Alice",
        memory_metadata={"source": "calendar", "tag": "work"},
    )
    result = await memory_service.create(user=user, payload=payload)
    assert isinstance(result, MemoryRead)
    mock_memory_vector_store.upsert.assert_awaited_once()


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_memory_success(
    memory_service: MemoryService,
    mock_session,
) -> None:
    """get() returns MemoryRead for an owned memory."""
    user = _make_user()
    mem = _make_memory(user.id)

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mem
    mock_session.execute = AsyncMock(return_value=mock_result)

    result = await memory_service.get(memory_id=mem.id, user=user)

    assert result.id == mem.id
    assert result.content == mem.content


@pytest.mark.asyncio
async def test_get_memory_not_found(
    memory_service: MemoryService,
    mock_session,
) -> None:
    """get() raises NotFoundError when memory doesn't belong to user."""
    user = _make_user()

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)

    with pytest.raises(NotFoundError, match="Memory not found"):
        await memory_service.get(memory_id=str(uuid4()), user=user)


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_memories_empty(
    memory_service: MemoryService,
    mock_session,
) -> None:
    """list() returns an empty MemoryList when user has no memories."""
    user = _make_user()

    # First execute call: count query
    count_result = MagicMock()
    count_result.scalar_one.return_value = 0

    # Second execute call: rows query
    rows_result = MagicMock()
    rows_result.scalars.return_value.all.return_value = []

    mock_session.execute = AsyncMock(side_effect=[count_result, rows_result])

    result = await memory_service.list(user=user)

    assert isinstance(result, MemoryList)
    assert result.total == 0
    assert result.items == []


@pytest.mark.asyncio
async def test_list_memories_with_results(
    memory_service: MemoryService,
    mock_session,
) -> None:
    """list() returns paginated MemoryList."""
    user = _make_user()
    mems = [_make_memory(user.id, content=f"Memory {i}") for i in range(3)]

    count_result = MagicMock()
    count_result.scalar_one.return_value = 3

    rows_result = MagicMock()
    rows_result.scalars.return_value.all.return_value = mems

    mock_session.execute = AsyncMock(side_effect=[count_result, rows_result])

    result = await memory_service.list(user=user, skip=0, limit=50)

    assert result.total == 3
    assert len(result.items) == 3


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_memory_success(
    memory_service: MemoryService,
    mock_session,
    mock_embedding_provider,
    mock_memory_vector_store,
) -> None:
    """update() updates PG record, flushes, re-embeds, and upserts Chroma."""
    user = _make_user()
    mem = _make_memory(user.id, content="Old memory content")

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mem
    mock_session.execute = AsyncMock(return_value=mock_result)

    payload = MemoryUpdate(content="New updated content")
    result = await memory_service.update(
        memory_id=mem.id,
        user=user,
        payload=payload,
    )

    assert mem.content == "New updated content"
    mock_session.flush.assert_awaited_once()
    mock_session.refresh.assert_awaited_once_with(mem)
    mock_embedding_provider.embed.assert_awaited_once_with("New updated content")
    mock_memory_vector_store.upsert.assert_awaited_once()
    assert result.content == "New updated content"


@pytest.mark.asyncio
async def test_update_memory_not_found(
    memory_service: MemoryService,
    mock_session,
    mock_memory_vector_store,
) -> None:
    """update() raises NotFoundError when memory is not found or not owned."""
    user = _make_user()

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)

    payload = MemoryUpdate(content="New content")
    with pytest.raises(NotFoundError, match="Memory not found"):
        await memory_service.update(
            memory_id=str(uuid4()),
            user=user,
            payload=payload,
        )

    mock_memory_vector_store.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_update_memory_chroma_failure_bubbles_up(
    memory_service: MemoryService,
    mock_session,
    mock_memory_vector_store,
) -> None:
    """update() surfaces Chroma upsert failure to ensure failure awareness."""
    user = _make_user()
    mem = _make_memory(user.id, content="Old content")

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mem
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_memory_vector_store.upsert = AsyncMock(
        side_effect=RuntimeError("Chroma down")
    )

    payload = MemoryUpdate(content="New content")
    with pytest.raises(RuntimeError, match="Chroma down"):
        await memory_service.update(
            memory_id=mem.id,
            user=user,
            payload=payload,
        )


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_memory_success(
    memory_service: MemoryService,
    mock_session,
    mock_memory_vector_store,
) -> None:
    """delete() removes from Chroma then from PostgreSQL."""
    user = _make_user()
    mem = _make_memory(user.id)

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mem
    mock_session.execute = AsyncMock(return_value=mock_result)

    await memory_service.delete(memory_id=mem.id, user=user)

    mock_memory_vector_store.delete.assert_awaited_once_with(mem.id)
    mock_session.delete.assert_awaited_once_with(mem)
    mock_session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_memory_not_found(
    memory_service: MemoryService,
    mock_session,
) -> None:
    """delete() raises NotFoundError if memory not owned by user."""
    user = _make_user()

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)

    with pytest.raises(NotFoundError):
        await memory_service.delete(memory_id=str(uuid4()), user=user)


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_no_results(
    memory_service: MemoryService,
    mock_memory_vector_store,
) -> None:
    """search() returns empty list when Chroma returns no hits."""
    user = _make_user()
    mock_memory_vector_store.search = AsyncMock(return_value=[])

    results = await memory_service.search(user=user, query="anything")
    assert results == []


@pytest.mark.asyncio
async def test_search_returns_results(
    memory_service: MemoryService,
    mock_session,
    mock_memory_vector_store,
) -> None:
    """search() maps Chroma hits back to MemorySearchResult objects."""
    user = _make_user()
    mem = _make_memory(user.id, content="Paris is the capital of France")
    hit_score = 0.92

    mock_memory_vector_store.search = AsyncMock(return_value=[(mem.id, hit_score)])

    rows_result = MagicMock()
    rows_result.scalars.return_value.all.return_value = [mem]
    mock_session.execute = AsyncMock(return_value=rows_result)

    results = await memory_service.search(user=user, query="France capital")

    assert len(results) == 1
    assert isinstance(results[0], MemorySearchResult)
    assert results[0].score == hit_score
    assert results[0].memory.id == mem.id


@pytest.mark.asyncio
async def test_search_skips_stale_chroma_entries(
    memory_service: MemoryService,
    mock_session,
    mock_memory_vector_store,
) -> None:
    """search() skips hits whose memory_id is not found in PostgreSQL (stale index)."""
    user = _make_user()
    stale_id = str(uuid4())
    mock_memory_vector_store.search = AsyncMock(return_value=[(stale_id, 0.88)])

    rows_result = MagicMock()
    rows_result.scalars.return_value.all.return_value = []  # PG returned nothing
    mock_session.execute = AsyncMock(return_value=rows_result)

    results = await memory_service.search(user=user, query="stale")
    assert results == []


# ---------------------------------------------------------------------------
# User isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_enforces_ownership(
    memory_service: MemoryService,
    mock_session,
) -> None:
    """get() must not return a memory belonging to a different user."""
    owner = _make_user()
    attacker = _make_user()
    mem = _make_memory(owner.id)

    # Simulating PG WHERE user_id = attacker.id => nothing found
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)

    with pytest.raises(NotFoundError):
        await memory_service.get(memory_id=mem.id, user=attacker)
