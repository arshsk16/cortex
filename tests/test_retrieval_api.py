"""Integration tests for POST /api/v1/retrieval/search."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from cortex.api.deps import get_current_active_user, get_retriever
from cortex.core.exceptions import BadRequestError, NotFoundError
from cortex.db.models.user import User, UserRole
from cortex.retrieval.base import Retriever
from cortex.retrieval.models import RetrievalResult

BASE_URL = "http://test"
SEARCH_URL = "/api/v1/retrieval/search"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


@pytest.fixture
def sample_user() -> User:
    return _make_user()


@pytest.fixture
def mock_retriever() -> Retriever:
    retriever = MagicMock(spec=Retriever)
    retriever.retrieve = AsyncMock(return_value=[])
    return retriever


@pytest.fixture
def api_client(
    app: FastAPI,
    sample_user: User,
    mock_retriever: Retriever,
) -> AsyncClient:
    """AsyncClient with auth and retriever overridden via FastAPI DI."""
    app.dependency_overrides[get_current_active_user] = lambda: sample_user
    app.dependency_overrides[get_retriever] = lambda: mock_retriever
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url=BASE_URL)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_returns_200_with_empty_results(
    api_client: AsyncClient,
    mock_retriever: Retriever,
) -> None:
    """When the retriever returns no results, the endpoint responds 200 with []."""
    async with api_client as client:
        response = await client.post(
            SEARCH_URL,
            json={"query": "machine learning", "top_k": 5},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "machine learning"
    assert body["top_k"] == 5
    assert body["results"] == []
    assert body["result_count"] == 0


@pytest.mark.asyncio
async def test_search_returns_ranked_results(
    app: FastAPI,
    sample_user: User,
) -> None:
    """Ranked results from the retriever are serialised correctly."""
    doc_id = str(uuid4())
    chunk_id_a = str(uuid4())
    chunk_id_b = str(uuid4())

    fake_results = [
        RetrievalResult(
            chunk_id=chunk_id_a,
            document_id=doc_id,
            chunk_index=0,
            text="First result text.",
            score=0.95,
        ),
        RetrievalResult(
            chunk_id=chunk_id_b,
            document_id=doc_id,
            chunk_index=1,
            text="Second result text.",
            score=0.80,
        ),
    ]

    retriever = MagicMock(spec=Retriever)
    retriever.retrieve = AsyncMock(return_value=fake_results)

    app.dependency_overrides[get_current_active_user] = lambda: sample_user
    app.dependency_overrides[get_retriever] = lambda: retriever

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url=BASE_URL) as client:
        response = await client.post(
            SEARCH_URL,
            json={"query": "retrieval test", "top_k": 10},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["result_count"] == 2
    assert body["results"][0]["chunk_id"] == chunk_id_a
    assert body["results"][0]["text"] == "First result text."
    assert body["results"][0]["score"] == pytest.approx(0.95)
    assert body["results"][1]["chunk_id"] == chunk_id_b
    assert body["results"][1]["score"] == pytest.approx(0.80)

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_search_with_document_id_filter(
    app: FastAPI,
    sample_user: User,
) -> None:
    """document_id is forwarded to the retriever.retrieve call."""
    doc_id = str(uuid4())

    retriever = MagicMock(spec=Retriever)
    retriever.retrieve = AsyncMock(return_value=[])

    app.dependency_overrides[get_current_active_user] = lambda: sample_user
    app.dependency_overrides[get_retriever] = lambda: retriever

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url=BASE_URL) as client:
        response = await client.post(
            SEARCH_URL,
            json={"query": "filter test", "top_k": 3, "document_id": doc_id},
        )

    assert response.status_code == 200
    retriever.retrieve.assert_called_once_with(
        query="filter test",
        user=sample_user,
        top_k=3,
        document_id=doc_id,
    )

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_search_uses_default_top_k(
    api_client: AsyncClient,
    mock_retriever: Retriever,
) -> None:
    """top_k defaults to 5 when omitted from the request payload."""
    async with api_client as client:
        response = await client.post(
            SEARCH_URL,
            json={"query": "default top_k"},
        )

    assert response.status_code == 200
    assert response.json()["top_k"] == 5


# ---------------------------------------------------------------------------
# Authentication tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_requires_authentication(app: FastAPI) -> None:
    """Requests without a Bearer token receive 401."""
    # Remove any DI overrides so the real auth dependency runs.
    app.dependency_overrides.clear()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url=BASE_URL) as client:
        response = await client.post(
            SEARCH_URL,
            json={"query": "unauthed"},
        )

    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_rejects_empty_query(
    api_client: AsyncClient,
) -> None:
    """The Pydantic schema rejects a query shorter than min_length=1."""
    async with api_client as client:
        response = await client.post(
            SEARCH_URL,
            json={"query": ""},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_search_rejects_top_k_zero(
    api_client: AsyncClient,
) -> None:
    """top_k=0 violates ge=1 and returns 422."""
    async with api_client as client:
        response = await client.post(
            SEARCH_URL,
            json={"query": "test", "top_k": 0},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_search_rejects_top_k_above_100(
    api_client: AsyncClient,
) -> None:
    """top_k=101 violates le=100 and returns 422."""
    async with api_client as client:
        response = await client.post(
            SEARCH_URL,
            json={"query": "test", "top_k": 101},
        )

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Domain error propagation tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_propagates_not_found_error(
    app: FastAPI,
    sample_user: User,
) -> None:
    """NotFoundError raised by the retriever maps to HTTP 404."""
    retriever = MagicMock(spec=Retriever)
    retriever.retrieve = AsyncMock(
        side_effect=NotFoundError("Document not found")
    )

    app.dependency_overrides[get_current_active_user] = lambda: sample_user
    app.dependency_overrides[get_retriever] = lambda: retriever

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url=BASE_URL) as client:
        response = await client.post(
            SEARCH_URL,
            json={"query": "missing doc", "document_id": str(uuid4())},
        )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_search_propagates_bad_request_error(
    app: FastAPI,
    sample_user: User,
) -> None:
    """BadRequestError raised by the retriever maps to HTTP 400."""
    retriever = MagicMock(spec=Retriever)
    retriever.retrieve = AsyncMock(
        side_effect=BadRequestError("Document has not been processed yet")
    )

    app.dependency_overrides[get_current_active_user] = lambda: sample_user
    app.dependency_overrides[get_retriever] = lambda: retriever

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url=BASE_URL) as client:
        response = await client.post(
            SEARCH_URL,
            json={"query": "not ready doc", "document_id": str(uuid4())},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bad_request"

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_search_response_schema_fields(
    app: FastAPI,
    sample_user: User,
) -> None:
    """The response always contains query, top_k, results, and result_count."""
    retriever = MagicMock(spec=Retriever)
    retriever.retrieve = AsyncMock(return_value=[])

    app.dependency_overrides[get_current_active_user] = lambda: sample_user
    app.dependency_overrides[get_retriever] = lambda: retriever

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url=BASE_URL) as client:
        response = await client.post(
            SEARCH_URL,
            json={"query": "schema test", "top_k": 7},
        )

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) >= {"query", "top_k", "results", "result_count"}
    assert body["query"] == "schema test"
    assert body["top_k"] == 7

    app.dependency_overrides.clear()
