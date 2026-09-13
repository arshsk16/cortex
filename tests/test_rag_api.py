"""Integration tests for POST /api/v1/rag/query."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from cortex.api.deps import get_current_active_user, get_rag_service
from cortex.core.exceptions import (
    BadRequestError,
    NotFoundError,
    ServiceUnavailableError,
)
from cortex.db.models.user import User, UserRole
from cortex.retrieval.models import RetrievalResult
from cortex.services.rag import RAGResult, RAGService

BASE_URL = "http://test"
QUERY_URL = "/api/v1/rag/query"


# ---------------------------------------------------------------------------
# Helpers
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


def _make_chunk(
    *,
    chunk_id: str | None = None,
    document_id: str | None = None,
    chunk_index: int = 0,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id or str(uuid4()),
        document_id=document_id or str(uuid4()),
        chunk_index=chunk_index,
        text="Some relevant text.",
        score=0.9,
    )


def _make_rag_result(
    *,
    answer: str = "The answer is X.",
    chunks: list[RetrievalResult] | None = None,
) -> RAGResult:
    return RAGResult(answer=answer, retrieved_chunks=chunks or [])


@pytest.fixture
def sample_user() -> User:
    return _make_user()


@pytest.fixture
def mock_rag_service() -> RAGService:
    svc = MagicMock(spec=RAGService)
    svc.answer = AsyncMock(return_value=_make_rag_result())
    return svc


@pytest.fixture
def api_client(
    app: FastAPI,
    sample_user: User,
    mock_rag_service: RAGService,
) -> AsyncClient:
    """AsyncClient with auth and RAGService overridden via DI."""
    app.dependency_overrides[get_current_active_user] = lambda: sample_user
    app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url=BASE_URL)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_returns_200_with_answer(
    api_client: AsyncClient,
    mock_rag_service: RAGService,
) -> None:
    """A valid request returns 200 with the answer from RAGService."""
    mock_rag_service.answer = AsyncMock(
        return_value=_make_rag_result(answer="AI stands for Artificial Intelligence.")
    )
    async with api_client as client:
        response = await client.post(
            QUERY_URL,
            json={"question": "What is AI?"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "AI stands for Artificial Intelligence."


@pytest.mark.asyncio
async def test_query_returns_empty_citations_for_no_chunks(
    api_client: AsyncClient,
    mock_rag_service: RAGService,
) -> None:
    """When no chunks are retrieved, citations is an empty list."""
    mock_rag_service.answer = AsyncMock(
        return_value=_make_rag_result(chunks=[])
    )
    async with api_client as client:
        response = await client.post(QUERY_URL, json={"question": "Q?"})

    assert response.status_code == 200
    assert response.json()["citations"] == []


@pytest.mark.asyncio
async def test_query_returns_citations_from_chunks(
    app: FastAPI,
    sample_user: User,
) -> None:
    """Citations in the response correspond to the retrieved chunks."""
    doc_id = str(uuid4())
    chunk_id_a = str(uuid4())
    chunk_id_b = str(uuid4())

    chunks = [
        _make_chunk(chunk_id=chunk_id_a, document_id=doc_id, chunk_index=0),
        _make_chunk(chunk_id=chunk_id_b, document_id=doc_id, chunk_index=1),
    ]
    svc = MagicMock(spec=RAGService)
    svc.answer = AsyncMock(
        return_value=_make_rag_result(answer="Answer text.", chunks=chunks)
    )

    app.dependency_overrides[get_current_active_user] = lambda: sample_user
    app.dependency_overrides[get_rag_service] = lambda: svc

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url=BASE_URL) as client:
        response = await client.post(QUERY_URL, json={"question": "Q?"})

    assert response.status_code == 200
    citations = response.json()["citations"]
    assert len(citations) == 2
    assert citations[0]["chunk_id"] == chunk_id_a
    assert citations[0]["document_id"] == doc_id
    assert citations[0]["chunk_index"] == 0
    assert citations[1]["chunk_id"] == chunk_id_b
    assert citations[1]["chunk_index"] == 1

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_query_with_document_id_forwarded_to_service(
    app: FastAPI,
    sample_user: User,
) -> None:
    """document_id from the request is forwarded to RAGService.answer."""
    doc_id = str(uuid4())
    svc = MagicMock(spec=RAGService)
    svc.answer = AsyncMock(return_value=_make_rag_result())

    app.dependency_overrides[get_current_active_user] = lambda: sample_user
    app.dependency_overrides[get_rag_service] = lambda: svc

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url=BASE_URL) as client:
        await client.post(
            QUERY_URL,
            json={"question": "Q?", "document_id": doc_id, "top_k": 3},
        )

    svc.answer.assert_called_once_with(
        question="Q?",
        user=sample_user,
        top_k=3,
        document_id=doc_id,
    )
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_query_uses_default_top_k_5(
    api_client: AsyncClient,
    mock_rag_service: RAGService,
) -> None:
    """top_k defaults to 5 when omitted from the request payload."""
    async with api_client as client:
        await client.post(QUERY_URL, json={"question": "Default top_k?"})

    call_kwargs = mock_rag_service.answer.call_args.kwargs
    assert call_kwargs["top_k"] == 5


@pytest.mark.asyncio
async def test_query_response_schema_fields(
    api_client: AsyncClient,
) -> None:
    """The response always contains 'answer' and 'citations' keys."""
    async with api_client as client:
        response = await client.post(QUERY_URL, json={"question": "Schema test?"})

    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "citations" in body


# ---------------------------------------------------------------------------
# Authentication tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_requires_authentication(app: FastAPI) -> None:
    """Requests without a Bearer token receive 401."""
    app.dependency_overrides.clear()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url=BASE_URL) as client:
        response = await client.post(QUERY_URL, json={"question": "unauthed"})

    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_rejects_empty_question(api_client: AsyncClient) -> None:
    """Empty question string returns 422."""
    async with api_client as client:
        response = await client.post(QUERY_URL, json={"question": ""})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_rejects_top_k_zero(api_client: AsyncClient) -> None:
    """top_k=0 violates ge=1 and returns 422."""
    async with api_client as client:
        response = await client.post(
            QUERY_URL, json={"question": "Q?", "top_k": 0}
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_rejects_top_k_above_20(api_client: AsyncClient) -> None:
    """top_k=21 violates le=20 and returns 422."""
    async with api_client as client:
        response = await client.post(
            QUERY_URL, json={"question": "Q?", "top_k": 21}
        )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Domain error propagation tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_propagates_bad_request_from_service(
    app: FastAPI,
    sample_user: User,
) -> None:
    """BadRequestError from RAGService maps to HTTP 400."""
    svc = MagicMock(spec=RAGService)
    svc.answer = AsyncMock(
        side_effect=BadRequestError("Document not ready")
    )

    app.dependency_overrides[get_current_active_user] = lambda: sample_user
    app.dependency_overrides[get_rag_service] = lambda: svc

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url=BASE_URL) as client:
        response = await client.post(QUERY_URL, json={"question": "Q?"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bad_request"
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_query_propagates_not_found_from_service(
    app: FastAPI,
    sample_user: User,
) -> None:
    """NotFoundError from RAGService maps to HTTP 404."""
    svc = MagicMock(spec=RAGService)
    svc.answer = AsyncMock(side_effect=NotFoundError("Document not found"))

    app.dependency_overrides[get_current_active_user] = lambda: sample_user
    app.dependency_overrides[get_rag_service] = lambda: svc

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url=BASE_URL) as client:
        response = await client.post(
            QUERY_URL,
            json={"question": "Q?", "document_id": str(uuid4())},
        )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_query_propagates_service_unavailable_from_llm(
    app: FastAPI,
    sample_user: User,
) -> None:
    """ServiceUnavailableError from RAGService maps to HTTP 503."""
    svc = MagicMock(spec=RAGService)
    svc.answer = AsyncMock(
        side_effect=ServiceUnavailableError("LLM unavailable")
    )

    app.dependency_overrides[get_current_active_user] = lambda: sample_user
    app.dependency_overrides[get_rag_service] = lambda: svc

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url=BASE_URL) as client:
        response = await client.post(QUERY_URL, json={"question": "Q?"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_unavailable"
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_query_openapi_includes_rag_route(
    app: FastAPI,
    sample_user: User,
    mock_rag_service: RAGService,
) -> None:
    """The OpenAPI schema exposes /api/v1/rag/query."""
    app.dependency_overrides[get_current_active_user] = lambda: sample_user
    app.dependency_overrides[get_rag_service] = lambda: mock_rag_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url=BASE_URL) as client:
        response = await client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/v1/rag/query" in paths
    app.dependency_overrides.clear()
