"""Integration tests for conversation endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from cortex.api.deps import (
    get_conversation_rag_service,
    get_conversation_service,
    get_current_active_user,
)
from cortex.core.exceptions import (
    ForbiddenError,
    NotFoundError,
    ServiceUnavailableError,
)
from cortex.db.models.conversation import Conversation, Message
from cortex.db.models.user import User, UserRole
from cortex.retrieval.models import RetrievalResult
from cortex.services.conversation import ConversationService
from cortex.services.rag import RAGResult, RAGService

BASE_URL = "http://test"
CONV_URL = "/api/v1/conversations"


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


def _make_conv(user_id: str, title: str = "Test Conv") -> Conversation:
    now = datetime.now(UTC)
    return Conversation(
        id=str(uuid4()),
        user_id=user_id,
        title=title,
        created_at=now,
        updated_at=now,
    )


def _make_msg(
    conv_id: str, role: str = "user", content: str = "Hello"
) -> Message:
    return Message(
        id=str(uuid4()),
        conversation_id=conv_id,
        role=role,
        content=content,
        citations=None,
        created_at=datetime.now(UTC),
    )


def _make_chunk() -> RetrievalResult:
    return RetrievalResult(
        chunk_id=str(uuid4()),
        document_id=str(uuid4()),
        chunk_index=0,
        text="Relevant excerpt.",
        score=0.9,
    )


@pytest.fixture
def sample_user() -> User:
    return _make_user()


@pytest.fixture
def mock_conv_service() -> ConversationService:
    svc = MagicMock(spec=ConversationService)
    svc.create_conversation = AsyncMock()
    svc.get_conversation = AsyncMock()
    svc.list_conversations = AsyncMock(return_value=[])
    svc.rename_conversation = AsyncMock()
    svc.delete_conversation = AsyncMock()
    svc.get_history = AsyncMock(return_value=[])
    svc.add_message = AsyncMock()
    svc.record_token_usage = AsyncMock()
    svc.set_auto_title_if_needed = AsyncMock()
    return svc


@pytest.fixture
def mock_rag_service() -> RAGService:
    svc = MagicMock(spec=RAGService)
    svc.answer = AsyncMock(
        return_value=RAGResult(answer="Mocked answer.", retrieved_chunks=[])
    )

    svc._history_limit = 10
    svc._retriever = MagicMock()
    svc._retriever.retrieve = AsyncMock(return_value=[])
    svc._prompt_builder = MagicMock()
    svc._prompt_builder.build = MagicMock(return_value="built prompt")
    svc._llm = MagicMock()
    svc._llm.generate_stream = AsyncMock(return_value=aiter(["Hello ", "world!"]))
    return svc


async def aiter(items: list[str]):
    """Async iterator helper for tests."""
    for item in items:
        yield item


@pytest.fixture
def api_client(
    app: FastAPI,
    sample_user: User,
    mock_conv_service: ConversationService,
    mock_rag_service: RAGService,
) -> AsyncClient:
    app.dependency_overrides[get_current_active_user] = lambda: sample_user
    app.dependency_overrides[get_conversation_service] = lambda: mock_conv_service
    app.dependency_overrides[get_conversation_rag_service] = lambda: mock_rag_service
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url=BASE_URL)


# ---------------------------------------------------------------------------
# POST /conversations — create
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_conversation_returns_201(
    api_client: AsyncClient,
    sample_user: User,
    mock_conv_service: ConversationService,
) -> None:
    """Creating a conversation returns 201 with conversation data."""
    conv = _make_conv(user_id=sample_user.id, title="New Conversation")
    mock_conv_service.create_conversation = AsyncMock(return_value=conv)

    async with api_client as client:
        response = await client.post(CONV_URL, json={})

    assert response.status_code == 201
    body = response.json()
    assert body["id"] == conv.id
    assert body["title"] == "New Conversation"


@pytest.mark.asyncio
async def test_create_conversation_with_title(
    api_client: AsyncClient,
    sample_user: User,
    mock_conv_service: ConversationService,
) -> None:
    """Custom title is forwarded to ConversationService."""
    conv = _make_conv(user_id=sample_user.id, title="My RAG Chat")
    mock_conv_service.create_conversation = AsyncMock(return_value=conv)

    async with api_client as client:
        response = await client.post(CONV_URL, json={"title": "My RAG Chat"})

    assert response.status_code == 201
    mock_conv_service.create_conversation.assert_called_once_with(
        user_id=sample_user.id, title="My RAG Chat"
    )


@pytest.mark.asyncio
async def test_create_conversation_requires_auth(app: FastAPI) -> None:
    """Unauthenticated request returns 401."""
    app.dependency_overrides.clear()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url=BASE_URL) as client:
        response = await client.post(CONV_URL, json={})
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# GET /conversations — list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_conversations_returns_200(
    api_client: AsyncClient,
    sample_user: User,
    mock_conv_service: ConversationService,
) -> None:
    """List endpoint returns 200 with conversations array."""
    convs = [_make_conv(user_id=sample_user.id) for _ in range(2)]
    mock_conv_service.list_conversations = AsyncMock(return_value=convs)

    async with api_client as client:
        response = await client.get(CONV_URL)

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert len(body["conversations"]) == 2


@pytest.mark.asyncio
async def test_list_conversations_empty(
    api_client: AsyncClient,
    mock_conv_service: ConversationService,
) -> None:
    """Empty conversation list returns total=0."""
    mock_conv_service.list_conversations = AsyncMock(return_value=[])

    async with api_client as client:
        response = await client.get(CONV_URL)

    assert response.status_code == 200
    assert response.json()["total"] == 0


# ---------------------------------------------------------------------------
# GET /conversations/{id} — get
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_conversation_returns_detail(
    api_client: AsyncClient,
    sample_user: User,
    mock_conv_service: ConversationService,
) -> None:
    """Get endpoint returns conversation with messages."""
    conv = _make_conv(user_id=sample_user.id)
    msgs = [
        _make_msg(conv.id, "user", "Hello"),
        _make_msg(conv.id, "assistant", "Hi there!"),
    ]
    mock_conv_service.get_conversation = AsyncMock(return_value=conv)
    mock_conv_service.get_history = AsyncMock(return_value=msgs)

    async with api_client as client:
        response = await client.get(f"{CONV_URL}/{conv.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == conv.id
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_get_conversation_not_found(
    api_client: AsyncClient,
    mock_conv_service: ConversationService,
) -> None:
    """NotFoundError from service maps to 404."""
    mock_conv_service.get_conversation = AsyncMock(
        side_effect=NotFoundError("Not found")
    )

    async with api_client as client:
        response = await client.get(f"{CONV_URL}/{uuid4()}")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_conversation_forbidden(
    api_client: AsyncClient,
    mock_conv_service: ConversationService,
) -> None:
    """ForbiddenError from service maps to 403."""
    mock_conv_service.get_conversation = AsyncMock(
        side_effect=ForbiddenError("Forbidden")
    )

    async with api_client as client:
        response = await client.get(f"{CONV_URL}/{uuid4()}")

    assert response.status_code == 403


# ---------------------------------------------------------------------------
# PATCH /conversations/{id} — rename
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rename_conversation_returns_200(
    api_client: AsyncClient,
    sample_user: User,
    mock_conv_service: ConversationService,
) -> None:
    """Rename returns 200 with updated conversation."""
    conv = _make_conv(user_id=sample_user.id, title="Renamed Title")
    mock_conv_service.rename_conversation = AsyncMock(return_value=conv)

    async with api_client as client:
        response = await client.patch(
            f"{CONV_URL}/{conv.id}",
            json={"title": "Renamed Title"},
        )

    assert response.status_code == 200
    assert response.json()["title"] == "Renamed Title"


@pytest.mark.asyncio
async def test_rename_conversation_requires_non_empty_title(
    api_client: AsyncClient,
) -> None:
    """Empty title returns 422."""
    async with api_client as client:
        response = await client.patch(
            f"{CONV_URL}/{uuid4()}",
            json={"title": ""},
        )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# DELETE /conversations/{id} — delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_conversation_returns_204(
    api_client: AsyncClient,
    mock_conv_service: ConversationService,
) -> None:
    """Delete returns 204 No Content."""
    mock_conv_service.delete_conversation = AsyncMock()
    async with api_client as client:
        response = await client.delete(f"{CONV_URL}/{uuid4()}")
    assert response.status_code == 204


@pytest.mark.asyncio
async def test_delete_conversation_not_owned_returns_403(
    api_client: AsyncClient,
    mock_conv_service: ConversationService,
) -> None:
    """ForbiddenError maps to 403."""
    mock_conv_service.delete_conversation = AsyncMock(
        side_effect=ForbiddenError("Not yours")
    )
    async with api_client as client:
        response = await client.delete(f"{CONV_URL}/{uuid4()}")
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# GET /conversations/{id}/messages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_messages_returns_history(
    api_client: AsyncClient,
    sample_user: User,
    mock_conv_service: ConversationService,
) -> None:
    """List messages returns ordered message list."""
    conv = _make_conv(user_id=sample_user.id)
    msgs = [
        _make_msg(conv.id, "user", "Q1"),
        _make_msg(conv.id, "assistant", "A1"),
    ]
    mock_conv_service.get_conversation = AsyncMock(return_value=conv)
    mock_conv_service.get_history = AsyncMock(return_value=msgs)

    async with api_client as client:
        response = await client.get(f"{CONV_URL}/{conv.id}/messages")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert body[0]["content"] == "Q1"


@pytest.mark.asyncio
async def test_list_messages_empty_conversation(
    api_client: AsyncClient,
    sample_user: User,
    mock_conv_service: ConversationService,
) -> None:
    """Empty conversation returns empty list."""
    conv = _make_conv(user_id=sample_user.id)
    mock_conv_service.get_conversation = AsyncMock(return_value=conv)
    mock_conv_service.get_history = AsyncMock(return_value=[])

    async with api_client as client:
        response = await client.get(f"{CONV_URL}/{conv.id}/messages")

    assert response.status_code == 200
    assert response.json() == []


# ---------------------------------------------------------------------------
# POST /conversations/{id}/chat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_returns_200_with_answer(
    api_client: AsyncClient,
    sample_user: User,
    mock_conv_service: ConversationService,
    mock_rag_service: RAGService,
) -> None:
    """Chat endpoint returns 200 with answer."""
    conv = _make_conv(user_id=sample_user.id)
    assistant_msg = _make_msg(conv.id, "assistant", "Mocked answer.")
    mock_conv_service.get_conversation = AsyncMock(return_value=conv)
    mock_conv_service.get_history = AsyncMock(return_value=[assistant_msg])
    mock_rag_service.answer = AsyncMock(
        return_value=RAGResult(answer="Mocked answer.", retrieved_chunks=[])
    )

    async with api_client as client:
        response = await client.post(
            f"{CONV_URL}/{conv.id}/chat",
            json={"message": "What is AI?"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Mocked answer."
    assert body["conversation_id"] == conv.id


@pytest.mark.asyncio
async def test_chat_returns_citations(
    api_client: AsyncClient,
    sample_user: User,
    mock_conv_service: ConversationService,
    mock_rag_service: RAGService,
) -> None:
    """Chat endpoint includes citations from retrieved chunks."""
    conv = _make_conv(user_id=sample_user.id)
    chunk = _make_chunk()
    mock_conv_service.get_conversation = AsyncMock(return_value=conv)
    msg = _make_msg(conv.id, "assistant", "Answer")
    mock_conv_service.get_history = AsyncMock(return_value=[msg])
    mock_rag_service.answer = AsyncMock(
        return_value=RAGResult(answer="Answer", retrieved_chunks=[chunk])
    )

    async with api_client as client:
        response = await client.post(
            f"{CONV_URL}/{conv.id}/chat",
            json={"message": "Q?"},
        )

    assert response.status_code == 200
    citations = response.json()["citations"]
    assert len(citations) == 1
    assert citations[0]["chunk_id"] == chunk.chunk_id


@pytest.mark.asyncio
async def test_chat_conversation_not_found(
    api_client: AsyncClient,
    mock_conv_service: ConversationService,
) -> None:
    """Chat on missing conversation returns 404."""
    mock_conv_service.get_conversation = AsyncMock(
        side_effect=NotFoundError("Not found")
    )

    async with api_client as client:
        response = await client.post(
            f"{CONV_URL}/{uuid4()}/chat",
            json={"message": "Q?"},
        )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_chat_rejects_empty_message(
    api_client: AsyncClient,
) -> None:
    """Empty message returns 422."""
    async with api_client as client:
        response = await client.post(
            f"{CONV_URL}/{uuid4()}/chat",
            json={"message": ""},
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_propagates_llm_service_unavailable(
    api_client: AsyncClient,
    sample_user: User,
    mock_conv_service: ConversationService,
    mock_rag_service: RAGService,
) -> None:
    """ServiceUnavailableError from LLM maps to 503."""
    conv = _make_conv(user_id=sample_user.id)
    mock_conv_service.get_conversation = AsyncMock(return_value=conv)
    mock_rag_service.answer = AsyncMock(
        side_effect=ServiceUnavailableError("LLM down")
    )

    async with api_client as client:
        response = await client.post(
            f"{CONV_URL}/{conv.id}/chat",
            json={"message": "Q?"},
        )

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_chat_requires_auth(app: FastAPI) -> None:
    """Unauthenticated chat returns 401."""
    app.dependency_overrides.clear()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url=BASE_URL) as client:
        response = await client.post(
            f"{CONV_URL}/{uuid4()}/chat",
            json={"message": "Q?"},
        )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# POST /conversations/{id}/stream — SSE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_returns_200_with_event_stream(
    api_client: AsyncClient,
    sample_user: User,
    mock_conv_service: ConversationService,
    mock_rag_service: RAGService,
) -> None:
    """Stream endpoint returns 200 with text/event-stream content type."""
    conv = _make_conv(user_id=sample_user.id)
    mock_conv_service.get_conversation = AsyncMock(return_value=conv)
    mock_conv_service.get_history = AsyncMock(return_value=[])

    mock_rag_service._retriever.retrieve = AsyncMock(return_value=[])
    mock_rag_service._prompt_builder.build = MagicMock(return_value="prompt")

    async def _fake_stream(prompt: str):
        yield "Hello"
        yield " world"

    mock_rag_service._llm.generate_stream = AsyncMock(
        side_effect=lambda p: _fake_stream(p)
    )

    async with api_client as client:
        response = await client.post(
            f"{CONV_URL}/{conv.id}/stream",
            json={"message": "Q?"},
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_stream_emits_done_event(
    api_client: AsyncClient,
    sample_user: User,
    mock_conv_service: ConversationService,
    mock_rag_service: RAGService,
) -> None:
    """Stream response body contains '[DONE]' sentinel."""
    conv = _make_conv(user_id=sample_user.id)
    mock_conv_service.get_conversation = AsyncMock(return_value=conv)
    mock_conv_service.get_history = AsyncMock(return_value=[])
    mock_conv_service.add_message = AsyncMock()
    mock_conv_service.set_auto_title_if_needed = AsyncMock()
    mock_conv_service.record_token_usage = AsyncMock()

    mock_rag_service._retriever.retrieve = AsyncMock(return_value=[])
    mock_rag_service._prompt_builder.build = MagicMock(return_value="prompt")

    async def _fake_stream(p: str):
        yield "token1"

    mock_rag_service._llm.generate_stream = AsyncMock(
        side_effect=lambda p: _fake_stream(p)
    )

    async with api_client as client:
        response = await client.post(
            f"{CONV_URL}/{conv.id}/stream",
            json={"message": "Q?"},
        )

    assert "[DONE]" in response.text


# ---------------------------------------------------------------------------
# OpenAPI schema registration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conversations_routes_in_openapi(
    api_client: AsyncClient,
) -> None:
    """All conversation routes appear in the OpenAPI schema."""
    async with api_client as client:
        response = await client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/v1/conversations" in paths
    assert "/api/v1/conversations/{conversation_id}" in paths
    assert "/api/v1/conversations/{conversation_id}/chat" in paths
    assert "/api/v1/conversations/{conversation_id}/stream" in paths
    assert "/api/v1/conversations/{conversation_id}/messages" in paths


# ---------------------------------------------------------------------------
# Regression: Phase 6 /rag/query still works
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rag_query_endpoint_still_works(
    app: FastAPI,
    sample_user: User,
) -> None:
    """Existing stateless RAG endpoint is unaffected by Phase 7 changes."""
    from cortex.api.deps import get_rag_service

    mock_rag = MagicMock(spec=RAGService)
    mock_rag.answer = AsyncMock(
        return_value=RAGResult(answer="Still works.", retrieved_chunks=[])
    )

    app.dependency_overrides[get_current_active_user] = lambda: sample_user
    app.dependency_overrides[get_rag_service] = lambda: mock_rag

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url=BASE_URL) as client:
        response = await client.post(
            "/api/v1/rag/query",
            json={"question": "Does this still work?"},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "Still works."
    app.dependency_overrides.clear()
