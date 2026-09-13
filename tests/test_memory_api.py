"""API-layer tests for the /memories endpoints.

All service I/O is mocked via FastAPI dependency overrides.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from cortex.api.deps import get_current_active_user, get_memory_service
from cortex.core.exceptions import NotFoundError
from cortex.db.models.user import User, UserRole
from cortex.schemas.memory import (
    MemoryList,
    MemoryRead,
    MemorySearchResult,
)
from cortex.services.memory import MemoryService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user() -> User:
    now = datetime.now(UTC)
    return User(
        id=str(uuid4()),
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


def _memory_read(
    user_id: str,
    content: str = "Test content",
    memory_metadata: dict | None = None,
) -> MemoryRead:
    now = datetime.now(UTC)
    return MemoryRead(
        id=str(uuid4()),
        user_id=user_id,
        content=content,
        memory_metadata=memory_metadata,
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# POST /memories
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_memory_success(app: FastAPI, client: AsyncClient) -> None:
    """POST /api/v1/memories returns 201 with memory envelope."""
    user = _make_user()
    read = _memory_read(user.id, content="Paris is in France")

    async def _user() -> User:
        return user

    async def _svc() -> MemoryService:
        svc = AsyncMock(spec=MemoryService)
        svc.create = AsyncMock(return_value=read)
        return svc

    app.dependency_overrides[get_current_active_user] = _user
    app.dependency_overrides[get_memory_service] = _svc
    try:
        response = await client.post(
            "/api/v1/memories",
            json={"content": "Paris is in France"},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["memory"]["content"] == "Paris is in France"
        assert body["memory"]["user_id"] == user.id
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_create_memory_with_metadata(app: FastAPI, client: AsyncClient) -> None:
    """POST /api/v1/memories stores memory_metadata."""
    user = _make_user()
    meta = {"source": "note", "tag": "personal"}
    read = _memory_read(user.id, content="Meeting notes", memory_metadata=meta)

    async def _user() -> User:
        return user

    async def _svc() -> MemoryService:
        svc = AsyncMock(spec=MemoryService)
        svc.create = AsyncMock(return_value=read)
        return svc

    app.dependency_overrides[get_current_active_user] = _user
    app.dependency_overrides[get_memory_service] = _svc
    try:
        response = await client.post(
            "/api/v1/memories",
            json={"content": "Meeting notes", "memory_metadata": meta},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["memory"]["memory_metadata"] == meta
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_create_memory_empty_content(app: FastAPI, client: AsyncClient) -> None:
    """POST /api/v1/memories rejects empty content with 422."""
    user = _make_user()

    async def _user() -> User:
        return user

    async def _svc() -> MemoryService:
        return AsyncMock(spec=MemoryService)

    app.dependency_overrides[get_current_active_user] = _user
    app.dependency_overrides[get_memory_service] = _svc
    try:
        response = await client.post("/api/v1/memories", json={"content": ""})
        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_create_memory_unauthenticated(app: FastAPI, client: AsyncClient) -> None:
    """POST /api/v1/memories without auth returns 401/403."""
    response = await client.post("/api/v1/memories", json={"content": "hello"})
    assert response.status_code in (401, 403)


# ---------------------------------------------------------------------------
# GET /memories
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_memories_empty(app: FastAPI, client: AsyncClient) -> None:
    """GET /api/v1/memories returns empty list."""
    user = _make_user()

    async def _user() -> User:
        return user

    async def _svc() -> MemoryService:
        svc = AsyncMock(spec=MemoryService)
        svc.list = AsyncMock(
            return_value=MemoryList(items=[], total=0, skip=0, limit=50)
        )
        return svc

    app.dependency_overrides[get_current_active_user] = _user
    app.dependency_overrides[get_memory_service] = _svc
    try:
        response = await client.get("/api/v1/memories")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 0
        assert body["items"] == []
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_list_memories_with_results(app: FastAPI, client: AsyncClient) -> None:
    """GET /api/v1/memories returns paginated list."""
    user = _make_user()
    items = [_memory_read(user.id, content=f"Memory {i}") for i in range(3)]

    async def _user() -> User:
        return user

    async def _svc() -> MemoryService:
        svc = AsyncMock(spec=MemoryService)
        svc.list = AsyncMock(
            return_value=MemoryList(items=items, total=3, skip=0, limit=50)
        )
        return svc

    app.dependency_overrides[get_current_active_user] = _user
    app.dependency_overrides[get_memory_service] = _svc
    try:
        response = await client.get("/api/v1/memories")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 3
        assert len(body["items"]) == 3
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_list_memories_pagination(app: FastAPI, client: AsyncClient) -> None:
    """GET /api/v1/memories respects skip/limit query params."""
    user = _make_user()

    async def _user() -> User:
        return user

    async def _svc() -> MemoryService:
        svc = AsyncMock(spec=MemoryService)
        svc.list = AsyncMock(
            return_value=MemoryList(items=[], total=100, skip=20, limit=10)
        )
        return svc

    app.dependency_overrides[get_current_active_user] = _user
    app.dependency_overrides[get_memory_service] = _svc
    try:
        response = await client.get("/api/v1/memories?skip=20&limit=10")
        assert response.status_code == 200
        body = response.json()
        assert body["skip"] == 20
        assert body["limit"] == 10
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# GET /memories/{memory_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_memory_success(app: FastAPI, client: AsyncClient) -> None:
    """GET /api/v1/memories/{id} returns the memory record."""
    user = _make_user()
    read = _memory_read(user.id)

    async def _user() -> User:
        return user

    async def _svc() -> MemoryService:
        svc = AsyncMock(spec=MemoryService)
        svc.get = AsyncMock(return_value=read)
        return svc

    app.dependency_overrides[get_current_active_user] = _user
    app.dependency_overrides[get_memory_service] = _svc
    try:
        response = await client.get(f"/api/v1/memories/{read.id}")
        assert response.status_code == 200
        body = response.json()
        assert body["memory"]["id"] == read.id
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_memory_not_found(app: FastAPI, client: AsyncClient) -> None:
    """GET /api/v1/memories/{id} returns 404 when memory not owned by user."""
    user = _make_user()

    async def _user() -> User:
        return user

    async def _svc() -> MemoryService:
        svc = AsyncMock(spec=MemoryService)
        svc.get = AsyncMock(
            side_effect=NotFoundError("Memory not found", details={"memory_id": "x"})
        )
        return svc

    app.dependency_overrides[get_current_active_user] = _user
    app.dependency_overrides[get_memory_service] = _svc
    try:
        response = await client.get("/api/v1/memories/non-existent-id")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# DELETE /memories/{memory_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_memory_success(app: FastAPI, client: AsyncClient) -> None:
    """DELETE /api/v1/memories/{id} returns 204."""
    user = _make_user()
    memory_id = str(uuid4())

    async def _user() -> User:
        return user

    async def _svc() -> MemoryService:
        svc = AsyncMock(spec=MemoryService)
        svc.delete = AsyncMock()
        return svc

    app.dependency_overrides[get_current_active_user] = _user
    app.dependency_overrides[get_memory_service] = _svc
    try:
        response = await client.delete(f"/api/v1/memories/{memory_id}")
        assert response.status_code == 204
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_delete_memory_not_found(app: FastAPI, client: AsyncClient) -> None:
    """DELETE /api/v1/memories/{id} returns 404 when memory not found."""
    user = _make_user()

    async def _user() -> User:
        return user

    async def _svc() -> MemoryService:
        svc = AsyncMock(spec=MemoryService)
        svc.delete = AsyncMock(
            side_effect=NotFoundError("Memory not found", details={"memory_id": "x"})
        )
        return svc

    app.dependency_overrides[get_current_active_user] = _user
    app.dependency_overrides[get_memory_service] = _svc
    try:
        response = await client.delete("/api/v1/memories/no-such-id")
        assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# POST /memories/search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_memories_empty(app: FastAPI, client: AsyncClient) -> None:
    """POST /api/v1/memories/search returns empty results."""
    user = _make_user()

    async def _user() -> User:
        return user

    async def _svc() -> MemoryService:
        svc = AsyncMock(spec=MemoryService)
        svc.search = AsyncMock(return_value=[])
        return svc

    app.dependency_overrides[get_current_active_user] = _user
    app.dependency_overrides[get_memory_service] = _svc
    try:
        response = await client.post(
            "/api/v1/memories/search",
            json={"query": "capital of France", "limit": 5},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["results"] == []
        assert body["query"] == "capital of France"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_search_memories_with_results(app: FastAPI, client: AsyncClient) -> None:
    """POST /api/v1/memories/search returns scored results."""
    user = _make_user()
    read = _memory_read(user.id, content="Paris is the capital of France")
    result = MemorySearchResult(memory=read, score=0.92)

    async def _user() -> User:
        return user

    async def _svc() -> MemoryService:
        svc = AsyncMock(spec=MemoryService)
        svc.search = AsyncMock(return_value=[result])
        return svc

    app.dependency_overrides[get_current_active_user] = _user
    app.dependency_overrides[get_memory_service] = _svc
    try:
        response = await client.post(
            "/api/v1/memories/search",
            json={"query": "France capital", "limit": 5},
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["results"]) == 1
        assert body["results"][0]["score"] == pytest.approx(0.92)
        assert "Paris" in body["results"][0]["memory"]["content"]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_search_memories_empty_query(app: FastAPI, client: AsyncClient) -> None:
    """POST /api/v1/memories/search rejects empty query with 422."""
    user = _make_user()

    async def _user() -> User:
        return user

    async def _svc() -> MemoryService:
        return AsyncMock(spec=MemoryService)

    app.dependency_overrides[get_current_active_user] = _user
    app.dependency_overrides[get_memory_service] = _svc
    try:
        response = await client.post(
            "/api/v1/memories/search",
            json={"query": "", "limit": 5},
        )
        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_search_memories_unauthenticated(
    app: FastAPI, client: AsyncClient
) -> None:
    """POST /api/v1/memories/search without auth returns 401/403."""
    response = await client.post(
        "/api/v1/memories/search",
        json={"query": "test", "limit": 5},
    )
    assert response.status_code in (401, 403)


# ---------------------------------------------------------------------------
# MemoryVectorStore unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_vector_store_upsert_calls_collection() -> None:
    """MemoryVectorStore.upsert delegates to chroma collection.upsert."""
    from cortex.vectorstore.memory_store import MemoryVectorStore

    coll = MagicMock()
    store = MemoryVectorStore(coll)
    mem_id = str(uuid4())
    uid = str(uuid4())
    embedding = [0.1] * 10

    await store.upsert(memory_id=mem_id, user_id=uid, embedding=embedding)

    coll.upsert.assert_called_once()
    call_kwargs = coll.upsert.call_args
    ids = call_kwargs.kwargs.get("ids", call_kwargs.args[0] if call_kwargs.args else [])
    assert mem_id in ids


@pytest.mark.asyncio
async def test_memory_vector_store_delete_calls_collection() -> None:
    """MemoryVectorStore.delete delegates to chroma collection.delete."""
    from cortex.vectorstore.memory_store import MemoryVectorStore

    coll = MagicMock()
    store = MemoryVectorStore(coll)
    mem_id = str(uuid4())

    await store.delete(mem_id)
    coll.delete.assert_called_once()

