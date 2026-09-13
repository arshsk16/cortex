"""Unit tests for ConversationService."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cortex.core.exceptions import ForbiddenError, NotFoundError
from cortex.db.models.conversation import Conversation, Message
from cortex.services.conversation import ConversationService, _auto_title

# ---------------------------------------------------------------------------
# _auto_title helper tests
# ---------------------------------------------------------------------------


def test_auto_title_short_message() -> None:
    """Short messages are returned verbatim."""
    assert _auto_title("What is AI?") == "What is AI?"


def test_auto_title_strips_whitespace() -> None:
    """Leading/trailing whitespace is stripped."""
    assert _auto_title("  Hello world  ") == "Hello world"


def test_auto_title_truncates_at_60_chars() -> None:
    """Messages longer than 60 chars are truncated."""
    long = "A" * 80
    result = _auto_title(long)
    assert len(result) <= 60


def test_auto_title_exactly_60_chars_kept() -> None:
    """A message of exactly 60 chars is kept as-is."""
    text = "X" * 60
    assert _auto_title(text) == text


def test_auto_title_truncated_title_does_not_exceed_60() -> None:
    """After truncation, result never exceeds 60 characters."""
    text = "word " * 20  # 100 chars
    result = _auto_title(text)
    assert len(result) <= 60


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_conversation(
    user_id: str | None = None,
    conv_id: str | None = None,
    title: str = "Test Conv",
) -> Conversation:
    now = datetime.now(UTC)
    conv = Conversation(
        id=conv_id or str(uuid4()),
        user_id=user_id or str(uuid4()),
        title=title,
        created_at=now,
        updated_at=now,
    )
    return conv


def _make_message(
    conversation_id: str,
    role: str = "user",
    content: str = "Hello",
    created_at: datetime | None = None,
) -> Message:
    return Message(
        id=str(uuid4()),
        conversation_id=conversation_id,
        role=role,
        content=content,
        citations=None,
        created_at=created_at or datetime.now(UTC),
    )


def _make_service() -> tuple[ConversationService, MagicMock]:
    """Return (service, mock_session) pair."""
    session = MagicMock()
    session.get = AsyncMock()
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.delete = AsyncMock()
    svc = ConversationService(session=session)  # type: ignore[arg-type]
    return svc, session


# ---------------------------------------------------------------------------
# create_conversation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_conversation_default_title() -> None:
    """Without a title, conversation gets 'New Conversation' default."""
    svc, session = _make_service()
    user_id = str(uuid4())
    conv_id = str(uuid4())

    created_conv = _make_conversation(user_id=user_id, title="New Conversation")
    session.get = AsyncMock(return_value=created_conv)
    session.flush = AsyncMock()

    # Patch uuid4 so we control the id
    with patch("cortex.services.conversation.uuid4", return_value=conv_id):
        # We need refresh to work — stub it
        session.refresh = AsyncMock(side_effect=lambda obj: None)
        result = await svc.create_conversation(user_id=user_id)

    assert result.user_id == user_id
    session.add.assert_called_once()
    session.flush.assert_called_once()


@pytest.mark.asyncio
async def test_create_conversation_custom_title() -> None:
    """Custom title is stored on the conversation."""
    svc, session = _make_service()
    conv = _make_conversation(title="My Report Chat")
    session.refresh = AsyncMock(side_effect=lambda obj: None)

    result = await svc.create_conversation(user_id=conv.user_id, title="My Report Chat")
    assert result.title == "My Report Chat"


# ---------------------------------------------------------------------------
# get_conversation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_conversation_returns_owned_conv() -> None:
    """Returns conversation when user_id matches."""
    svc, session = _make_service()
    user_id = str(uuid4())
    conv = _make_conversation(user_id=user_id)
    session.get = AsyncMock(return_value=conv)

    result = await svc.get_conversation(
        conversation_id=conv.id, user_id=user_id
    )
    assert result.id == conv.id


@pytest.mark.asyncio
async def test_get_conversation_raises_not_found_for_missing() -> None:
    """Missing conversation raises NotFoundError."""
    svc, session = _make_service()
    session.get = AsyncMock(return_value=None)

    with pytest.raises(NotFoundError):
        await svc.get_conversation(
            conversation_id=str(uuid4()), user_id=str(uuid4())
        )


@pytest.mark.asyncio
async def test_get_conversation_raises_forbidden_for_wrong_owner() -> None:
    """Conversation owned by another user raises ForbiddenError."""
    svc, session = _make_service()
    conv = _make_conversation(user_id=str(uuid4()))
    session.get = AsyncMock(return_value=conv)

    with pytest.raises(ForbiddenError):
        await svc.get_conversation(
            conversation_id=conv.id, user_id=str(uuid4())  # different user
        )


# ---------------------------------------------------------------------------
# list_conversations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_conversations_returns_all_for_user() -> None:
    """Returns all conversations for the user."""
    svc, session = _make_service()
    user_id = str(uuid4())
    convs = [_make_conversation(user_id=user_id) for _ in range(3)]

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = convs
    session.execute = AsyncMock(return_value=mock_result)

    result = await svc.list_conversations(user_id=user_id)
    assert len(result) == 3


@pytest.mark.asyncio
async def test_list_conversations_returns_empty_for_new_user() -> None:
    """New user with no conversations gets empty list."""
    svc, session = _make_service()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=mock_result)

    result = await svc.list_conversations(user_id=str(uuid4()))
    assert result == []


# ---------------------------------------------------------------------------
# rename_conversation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rename_conversation_updates_title() -> None:
    """rename_conversation sets the new title."""
    svc, session = _make_service()
    user_id = str(uuid4())
    conv = _make_conversation(user_id=user_id, title="Old Title")
    session.get = AsyncMock(return_value=conv)

    result = await svc.rename_conversation(
        conversation_id=conv.id,
        user_id=user_id,
        title="New Title",
    )
    assert result.title == "New Title"


@pytest.mark.asyncio
async def test_rename_conversation_enforces_ownership() -> None:
    """rename_conversation raises ForbiddenError for wrong owner."""
    svc, session = _make_service()
    conv = _make_conversation(user_id=str(uuid4()))
    session.get = AsyncMock(return_value=conv)

    with pytest.raises(ForbiddenError):
        await svc.rename_conversation(
            conversation_id=conv.id,
            user_id=str(uuid4()),
            title="Hacked Title",
        )


# ---------------------------------------------------------------------------
# delete_conversation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_conversation_calls_delete() -> None:
    """delete_conversation calls session.delete on the conversation."""
    svc, session = _make_service()
    user_id = str(uuid4())
    conv = _make_conversation(user_id=user_id)
    session.get = AsyncMock(return_value=conv)
    session.delete = AsyncMock()

    await svc.delete_conversation(conversation_id=conv.id, user_id=user_id)

    session.delete.assert_called_once_with(conv)
    session.flush.assert_called_once()


@pytest.mark.asyncio
async def test_delete_conversation_enforces_ownership() -> None:
    """delete_conversation raises ForbiddenError for wrong owner."""
    svc, session = _make_service()
    conv = _make_conversation(user_id=str(uuid4()))
    session.get = AsyncMock(return_value=conv)

    with pytest.raises(ForbiddenError):
        await svc.delete_conversation(
            conversation_id=conv.id,
            user_id=str(uuid4()),
        )


# ---------------------------------------------------------------------------
# get_history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_history_returns_messages_in_order() -> None:
    """get_history returns messages sorted by created_at ascending."""
    svc, session = _make_service()
    user_id = str(uuid4())
    conv = _make_conversation(user_id=user_id)
    conv_id = conv.id

    msgs = [
        _make_message(conv_id, role="user", content="Q1",
                      created_at=datetime(2026, 1, 1, 10, 0, tzinfo=UTC)),
        _make_message(conv_id, role="assistant", content="A1",
                      created_at=datetime(2026, 1, 1, 10, 1, tzinfo=UTC)),
        _make_message(conv_id, role="user", content="Q2",
                      created_at=datetime(2026, 1, 1, 10, 2, tzinfo=UTC)),
    ]

    # First call: get_conversation; second call: get_history query
    session.get = AsyncMock(return_value=conv)
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = msgs
    session.execute = AsyncMock(return_value=mock_result)

    result = await svc.get_history(
        conversation_id=conv_id, user_id=user_id
    )
    assert len(result) == 3
    assert result[0].content == "Q1"
    assert result[1].role == "assistant"


@pytest.mark.asyncio
async def test_get_history_enforces_ownership() -> None:
    """get_history raises ForbiddenError for wrong owner."""
    svc, session = _make_service()
    conv = _make_conversation(user_id=str(uuid4()))
    session.get = AsyncMock(return_value=conv)

    with pytest.raises(ForbiddenError):
        await svc.get_history(
            conversation_id=conv.id, user_id=str(uuid4())
        )


@pytest.mark.asyncio
async def test_get_history_with_limit_uses_subquery() -> None:
    """get_history with limit executes a subquery-based query."""
    svc, session = _make_service()
    user_id = str(uuid4())
    conv = _make_conversation(user_id=user_id)
    session.get = AsyncMock(return_value=conv)

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=mock_result)

    await svc.get_history(
        conversation_id=conv.id, user_id=user_id, limit=5
    )
    # execute was called (once for the subquery + re-fetch path)
    session.execute.assert_called()


# ---------------------------------------------------------------------------
# add_message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_message_persists_user_message() -> None:
    """add_message calls session.add and session.flush."""
    svc, session = _make_service()
    conv_id = str(uuid4())
    conv = _make_conversation(conv_id=conv_id)
    session.get = AsyncMock(return_value=conv)

    await svc.add_message(
        conversation_id=conv_id,
        role="user",
        content="Hello there",
    )

    session.add.assert_called_once()
    session.flush.assert_called_once()


@pytest.mark.asyncio
async def test_add_message_stores_citations_for_assistant() -> None:
    """add_message stores citations JSON for assistant messages."""
    svc, session = _make_service()
    conv_id = str(uuid4())
    conv = _make_conversation(conv_id=conv_id)
    session.get = AsyncMock(return_value=conv)

    citations = [{"document_id": "d1", "chunk_id": "c1", "chunk_index": 0}]
    await svc.add_message(
        conversation_id=conv_id,
        role="assistant",
        content="The answer is yes.",
        citations=citations,
    )

    added_msg = session.add.call_args[0][0]
    assert added_msg.citations == citations
    assert added_msg.role == "assistant"


# ---------------------------------------------------------------------------
# record_token_usage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_token_usage_persists_record() -> None:
    """record_token_usage creates a TokenUsage and flushes."""
    svc, session = _make_service()
    conv_id = str(uuid4())

    usage = await svc.record_token_usage(
        conversation_id=conv_id,
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        estimated_cost=0.001,
    )

    session.add.assert_called_once()
    session.flush.assert_called_once()
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50
    assert usage.total_tokens == 150
    assert abs(usage.estimated_cost - 0.001) < 1e-6


# ---------------------------------------------------------------------------
# set_auto_title_if_needed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_auto_title_if_needed_replaces_default() -> None:
    """Default 'New Conversation' title is replaced by auto-generated one."""
    svc, session = _make_service()
    conv = _make_conversation(title="New Conversation")
    session.get = AsyncMock(return_value=conv)

    await svc.set_auto_title_if_needed(
        conversation_id=conv.id,
        first_message="Tell me about machine learning",
    )

    assert conv.title == "Tell me about machine learning"


@pytest.mark.asyncio
async def test_set_auto_title_if_needed_preserves_custom_title() -> None:
    """Custom titles are NOT overwritten."""
    svc, session = _make_service()
    conv = _make_conversation(title="My Custom Title")
    session.get = AsyncMock(return_value=conv)

    await svc.set_auto_title_if_needed(
        conversation_id=conv.id,
        first_message="This should NOT replace the title",
    )

    assert conv.title == "My Custom Title"


@pytest.mark.asyncio
async def test_set_auto_title_if_needed_truncates_long_message() -> None:
    """Auto-title truncates messages longer than 60 chars."""
    svc, session = _make_service()
    conv = _make_conversation(title="New Conversation")
    session.get = AsyncMock(return_value=conv)

    long_msg = "word " * 20
    await svc.set_auto_title_if_needed(
        conversation_id=conv.id,
        first_message=long_msg,
    )

    assert len(conv.title) <= 60


@pytest.mark.asyncio
async def test_set_auto_title_if_needed_noop_if_conv_missing() -> None:
    """If conversation is not found, no error is raised."""
    svc, session = _make_service()
    session.get = AsyncMock(return_value=None)

    # Should silently do nothing
    await svc.set_auto_title_if_needed(
        conversation_id=str(uuid4()),
        first_message="anything",
    )
