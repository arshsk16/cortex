"""Extended PromptBuilder tests — conversation history section."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from cortex.retrieval.models import RetrievalResult
from cortex.services.prompt_builder import PromptBuilder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message(role: str, content: str) -> MagicMock:
    """Create a lightweight Message-like object for prompt tests."""
    msg = MagicMock()
    msg.role = role
    msg.content = content
    msg.id = str(uuid4())
    msg.created_at = datetime.now(UTC)
    return msg


def _make_chunk(text: str = "Some context.") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=str(uuid4()),
        document_id=str(uuid4()),
        chunk_index=0,
        text=text,
        score=0.9,
    )


@pytest.fixture
def builder() -> PromptBuilder:
    return PromptBuilder()


# ---------------------------------------------------------------------------
# Backward-compatibility: no history
# ---------------------------------------------------------------------------


def test_build_without_history_unchanged(builder: PromptBuilder) -> None:
    """When history=None, prompt matches Phase 6 behaviour exactly."""
    chunk = _make_chunk("Context A.")
    prompt = builder.build(question="Q?", retrieved_chunks=[chunk])
    assert "Context A." in prompt
    assert "Q?" in prompt
    # No history header
    assert "CONVERSATION HISTORY" not in prompt


def test_build_with_empty_history_list_no_history_section(
    builder: PromptBuilder,
) -> None:
    """history=[] (empty list) produces no history section."""
    prompt = builder.build(question="Q?", retrieved_chunks=[], history=[])
    assert "CONVERSATION HISTORY" not in prompt


# ---------------------------------------------------------------------------
# History section presence and content
# ---------------------------------------------------------------------------


def test_build_with_history_includes_history_header(builder: PromptBuilder) -> None:
    """When history is non-empty, the history section header appears."""
    msgs = [_make_message("user", "Previous question")]
    prompt = builder.build(
        question="Follow-up?", retrieved_chunks=[], history=msgs
    )
    assert "CONVERSATION HISTORY" in prompt


def test_build_history_user_labelled(builder: PromptBuilder) -> None:
    """User messages are labelled [User]."""
    msgs = [_make_message("user", "My earlier question")]
    prompt = builder.build(question="Q?", retrieved_chunks=[], history=msgs)
    assert "[User]" in prompt
    assert "My earlier question" in prompt


def test_build_history_assistant_labelled(builder: PromptBuilder) -> None:
    """Assistant messages are labelled [Assistant]."""
    msgs = [_make_message("assistant", "My earlier answer")]
    prompt = builder.build(question="Q?", retrieved_chunks=[], history=msgs)
    assert "[Assistant]" in prompt
    assert "My earlier answer" in prompt


def test_build_history_multiple_turns_all_present(
    builder: PromptBuilder,
) -> None:
    """All history messages appear in the prompt."""
    msgs = [
        _make_message("user", "First question"),
        _make_message("assistant", "First answer"),
        _make_message("user", "Second question"),
        _make_message("assistant", "Second answer"),
    ]
    prompt = builder.build(question="Third?", retrieved_chunks=[], history=msgs)
    assert "First question" in prompt
    assert "First answer" in prompt
    assert "Second question" in prompt
    assert "Second answer" in prompt


# ---------------------------------------------------------------------------
# Section ordering
# ---------------------------------------------------------------------------


def test_build_section_order_is_sys_hist_ctx_question(
    builder: PromptBuilder,
) -> None:
    """Sections appear in order: system → history → context → question."""
    msgs = [_make_message("user", "History message")]
    chunk = _make_chunk("Context excerpt.")
    prompt = builder.build(
        question="Current question?",
        retrieved_chunks=[chunk],
        history=msgs,
    )

    sys_pos = prompt.index("Answer ONLY")
    hist_pos = prompt.index("History message")
    ctx_pos = prompt.index("Context excerpt.")
    q_pos = prompt.index("Current question?")

    assert sys_pos < hist_pos < ctx_pos < q_pos


def test_build_no_history_order_is_sys_ctx_question(
    builder: PromptBuilder,
) -> None:
    """Without history, order is: system → context → question."""
    chunk = _make_chunk("Context excerpt.")
    prompt = builder.build(
        question="My question?",
        retrieved_chunks=[chunk],
    )

    sys_pos = prompt.index("Answer ONLY")
    ctx_pos = prompt.index("Context excerpt.")
    q_pos = prompt.index("My question?")

    assert sys_pos < ctx_pos < q_pos


# ---------------------------------------------------------------------------
# Content stripping
# ---------------------------------------------------------------------------


def test_build_history_message_content_is_stripped(
    builder: PromptBuilder,
) -> None:
    """Whitespace around history message content is stripped."""
    msgs = [_make_message("user", "  Padded message  ")]
    prompt = builder.build(question="Q?", retrieved_chunks=[], history=msgs)
    assert "Padded message" in prompt
    # The padded version should not appear verbatim
    assert "  Padded message  " not in prompt


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_build_is_deterministic(builder: PromptBuilder) -> None:
    """Calling build() twice with same inputs produces identical output."""
    msgs = [_make_message("user", "Q1"), _make_message("assistant", "A1")]
    chunk = _make_chunk("Some text.")
    args = dict(question="Q2?", retrieved_chunks=[chunk], history=msgs)

    p1 = builder.build(**args)
    p2 = builder.build(**args)
    assert p1 == p2
