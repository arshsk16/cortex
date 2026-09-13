"""Security tests for the agent layer.

Verifies:
1. The agent endpoint requires authentication.
2. Conversation ownership is enforced — one user cannot access another's conversation.
3. The RAGSearchTool only returns data belonging to the requesting user.
4. The CalculatorTool never accesses any user data (no cross-user leakage possible).
5. An inactive user account is rejected before the agent runs.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from cortex.agent.registry import ToolRegistry
from cortex.agent.result import AgentResult
from cortex.agent.service import AgentService
from cortex.agent.tools.calculator import CalculatorTool
from cortex.agent.tools.rag_search import RAGSearchTool
from cortex.api.deps import get_agent_service, get_current_active_user, get_current_user
from cortex.core.exceptions import ForbiddenError
from cortex.db.models.user import User, UserRole
from cortex.retrieval.models import RetrievalResult
from cortex.services.prompt_builder import PromptBuilder

BASE_URL = "http://test"
AGENT_URL = "/api/v1/agent/run"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user(
    user_id: str | None = None,
    *,
    is_active: bool = True,
) -> User:
    now = datetime.now(UTC)
    return User(
        id=user_id or str(uuid4()),
        email=f"{uuid4()}@example.com",
        username=f"user-{uuid4().hex[:6]}",
        full_name="Test User",
        hashed_password="hashed",
        role=UserRole.USER,
        is_active=is_active,
        is_verified=False,
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# Test: Authentication required
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_requires_authentication(app: FastAPI) -> None:
    """Request without Bearer token must be rejected with 401."""
    app.dependency_overrides.pop(get_current_active_user, None)
    app.dependency_overrides.pop(get_agent_service, None)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url=BASE_URL) as client:
        response = await client.post(AGENT_URL, json={"message": "Hello"})

    assert response.status_code == 401
    body = response.json()
    assert "error" in body


# ---------------------------------------------------------------------------
# Test: Inactive user rejected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inactive_user_cannot_use_agent(app: FastAPI) -> None:
    """An inactive user account must be rejected with 403."""
    inactive_user = _make_user(is_active=False)
    # Override get_current_user (identity only); let the real
    # get_current_active_user dep run — it raises ForbiddenError for
    # inactive accounts, which the exception handler maps to 403.
    app.dependency_overrides[get_current_user] = lambda: inactive_user
    app.dependency_overrides.pop(get_current_active_user, None)
    app.dependency_overrides.pop(get_agent_service, None)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url=BASE_URL) as client:
        response = await client.post(AGENT_URL, json={"message": "Hello"})

    assert response.status_code == 403


# ---------------------------------------------------------------------------
# Test: Conversation ownership enforced
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_enforces_conversation_ownership(app: FastAPI) -> None:
    """Accessing another user's conversation_id returns 403."""
    user_a = _make_user()
    other_conv_id = str(uuid4())

    svc = MagicMock(spec=AgentService)
    svc.run = AsyncMock(
        side_effect=ForbiddenError("You do not own this conversation")
    )

    app.dependency_overrides[get_current_active_user] = lambda: user_a
    app.dependency_overrides[get_agent_service] = lambda: svc

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url=BASE_URL) as client:
        response = await client.post(
            AGENT_URL,
            json={"message": "Show me something", "conversation_id": other_conv_id},
        )

    assert response.status_code == 403
    body = response.json()
    assert body["error"]["code"] == "forbidden"


# ---------------------------------------------------------------------------
# Test: RAGSearchTool passes user through to retriever
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rag_search_tool_passes_user_to_retriever(sample_user: User) -> None:
    """RAGSearchTool.execute must forward the user to Retriever.retrieve.

    SemanticRetriever enforces ownership via user_id; passing a different
    (or no) user would leak cross-user data.
    """
    mock_retriever = MagicMock()
    mock_retriever.retrieve = AsyncMock(return_value=[])

    tool = RAGSearchTool(mock_retriever)
    await tool.execute({"query": "test query"}, sample_user)

    mock_retriever.retrieve.assert_called_once()
    call_kwargs = mock_retriever.retrieve.call_args.kwargs
    # The user object must be passed through unchanged
    assert call_kwargs["user"] is sample_user


@pytest.mark.asyncio
async def test_rag_search_tool_never_returns_other_users_chunks(
    sample_user: User,
    other_user: User,
) -> None:
    """Chunks belonging to other_user must not appear in sample_user's results.

    The retriever is authoritative for ownership; the tool must not bypass
    the user parameter.  We simulate this by verifying the tool passes the
    correct user to the retriever (which rejects cross-user hits at the DB
    level in the real implementation).
    """
    # Two separate retrievers — one for each user
    retriever_a = MagicMock()
    retriever_b = MagicMock()

    chunk_a = RetrievalResult(
        chunk_id="chunk-a",
        document_id="doc-a",
        chunk_index=0,
        text="User A's private data.",
        score=0.99,
    )
    chunk_b = RetrievalResult(
        chunk_id="chunk-b",
        document_id="doc-b",
        chunk_index=0,
        text="User B's private data.",
        score=0.99,
    )
    retriever_a.retrieve = AsyncMock(return_value=[chunk_a])
    retriever_b.retrieve = AsyncMock(return_value=[chunk_b])

    tool_a = RAGSearchTool(retriever_a)
    tool_b = RAGSearchTool(retriever_b)

    result_a = await tool_a.execute({"query": "data"}, sample_user)
    result_b = await tool_b.execute({"query": "data"}, other_user)

    # User A's tool must only see user A's chunks
    assert "User A's private data." in result_a
    assert "User B's private data." not in result_a

    # User B's tool must only see user B's chunks
    assert "User B's private data." in result_b
    assert "User A's private data." not in result_b

    # Each retriever was called with the correct user
    retriever_a.retrieve.assert_called_once()
    assert retriever_a.retrieve.call_args.kwargs["user"] is sample_user

    retriever_b.retrieve.assert_called_once()
    assert retriever_b.retrieve.call_args.kwargs["user"] is other_user


# ---------------------------------------------------------------------------
# Test: AgentService ownership check happens before tool calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ownership_check_blocks_tool_execution(sample_user: User) -> None:
    """If conversation ownership fails, no tools must be called."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve = AsyncMock(return_value=[])

    # Conversation service that always denies access
    conv_service = MagicMock()
    conv_service.get_history = AsyncMock(
        side_effect=ForbiddenError("Not your conversation")
    )

    registry = ToolRegistry()
    registry.register(RAGSearchTool(mock_retriever))
    registry.register(CalculatorTool())

    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(
        return_value=json.dumps(
            {"action": "tool_call", "tool": "rag_search", "args": {"query": "test"}}
        )
    )

    service = AgentService(
        llm_provider=mock_llm,
        tool_registry=registry,
        prompt_builder=PromptBuilder(),
        conversation_service=conv_service,
        max_tool_calls=5,
    )

    with pytest.raises(ForbiddenError):
        await service.run(
            question="Tell me something",
            user=sample_user,
            conversation_id="not-my-conv",
        )

    # LLM and retriever must never be reached
    mock_llm.generate.assert_not_called()
    mock_retriever.retrieve.assert_not_called()


# ---------------------------------------------------------------------------
# Test: CalculatorTool has no user-data access
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calculator_tool_does_not_access_user_data(sample_user: User) -> None:
    """CalculatorTool accepts user for interface compliance but uses no user data."""
    from cortex.agent.tools.calculator import CalculatorTool

    tool = CalculatorTool()
    # Use two different users — result must be identical (user-independent)
    result_a = await tool.execute({"expression": "100 / 4"}, sample_user)

    other = _make_user()
    result_b = await tool.execute({"expression": "100 / 4"}, other)

    assert result_a == result_b
    assert "25" in result_a
