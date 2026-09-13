"""Unit tests for RAGService."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from cortex.core.exceptions import BadRequestError, ServiceUnavailableError
from cortex.db.models.user import User, UserRole
from cortex.llm.base import LLMProvider
from cortex.retrieval.base import Retriever
from cortex.retrieval.models import RetrievalResult
from cortex.services.prompt_builder import PromptBuilder
from cortex.services.rag import RAGResult, RAGService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user() -> User:
    now = datetime.now(UTC)
    return User(
        id=str(uuid4()),
        email="test@example.com",
        username="testuser",
        full_name="Test User",
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
    text: str = "Relevant text.",
    score: float = 0.9,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id or str(uuid4()),
        document_id=document_id or str(uuid4()),
        chunk_index=chunk_index,
        text=text,
        score=score,
    )


def _make_service(
    *,
    retriever: Retriever | None = None,
    llm_provider: LLMProvider | None = None,
    prompt_builder: PromptBuilder | None = None,
) -> RAGService:
    if retriever is None:
        retriever = MagicMock(spec=Retriever)
        retriever.retrieve = AsyncMock(return_value=[])
    if llm_provider is None:
        llm_provider = MagicMock(spec=LLMProvider)
        llm_provider.generate = AsyncMock(return_value="Default answer.")
    if prompt_builder is None:
        prompt_builder = PromptBuilder()
    return RAGService(
        retriever=retriever,
        llm_provider=llm_provider,
        prompt_builder=prompt_builder,
    )


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_answer_returns_rag_result() -> None:
    """answer() returns a RAGResult instance."""
    service = _make_service()
    result = await service.answer(question="What is AI?", user=_make_user())
    assert isinstance(result, RAGResult)


@pytest.mark.asyncio
async def test_answer_returns_llm_text() -> None:
    """answer() returns the text produced by the LLM provider."""
    llm = MagicMock(spec=LLMProvider)
    llm.generate = AsyncMock(return_value="AI is artificial intelligence.")
    service = _make_service(llm_provider=llm)

    result = await service.answer(question="What is AI?", user=_make_user())
    assert result.answer == "AI is artificial intelligence."


@pytest.mark.asyncio
async def test_answer_calls_retriever_with_correct_args() -> None:
    """answer() forwards question, user, top_k, and document_id to the retriever."""
    retriever = MagicMock(spec=Retriever)
    retriever.retrieve = AsyncMock(return_value=[])
    user = _make_user()
    doc_id = str(uuid4())

    service = _make_service(retriever=retriever)
    await service.answer(
        question="What is AI?",
        user=user,
        top_k=7,
        document_id=doc_id,
    )

    retriever.retrieve.assert_called_once_with(
        query="What is AI?",
        user=user,
        top_k=7,
        document_id=doc_id,
    )


@pytest.mark.asyncio
async def test_answer_with_chunks_populates_retrieved_chunks() -> None:
    """retrieved_chunks in the result mirrors what the retriever returned."""
    chunks = [_make_chunk(text=f"Chunk {i}") for i in range(3)]
    retriever = MagicMock(spec=Retriever)
    retriever.retrieve = AsyncMock(return_value=chunks)

    service = _make_service(retriever=retriever)
    result = await service.answer(question="Q?", user=_make_user())

    assert result.retrieved_chunks == chunks


@pytest.mark.asyncio
async def test_answer_empty_retrieval_still_calls_llm() -> None:
    """When no chunks are retrieved, the LLM is still called
    (with a no-context prompt).
    """

    llm = MagicMock(spec=LLMProvider)
    llm.generate = AsyncMock(return_value="I cannot answer.")
    retriever = MagicMock(spec=Retriever)
    retriever.retrieve = AsyncMock(return_value=[])

    service = _make_service(retriever=retriever, llm_provider=llm)
    result = await service.answer(question="Q?", user=_make_user())

    llm.generate.assert_called_once()
    assert result.answer == "I cannot answer."


@pytest.mark.asyncio
async def test_answer_empty_retrieval_returns_empty_citations() -> None:
    """An empty retrieval results in an empty retrieved_chunks list."""
    service = _make_service()
    result = await service.answer(question="Q?", user=_make_user())
    assert result.retrieved_chunks == []


@pytest.mark.asyncio
async def test_answer_strips_whitespace_from_question() -> None:
    """Whitespace is stripped from the question before it reaches the retriever."""
    retriever = MagicMock(spec=Retriever)
    retriever.retrieve = AsyncMock(return_value=[])

    service = _make_service(retriever=retriever)
    await service.answer(question="  What is ML?  ", user=_make_user())

    retriever.retrieve.assert_called_once()
    call_query = retriever.retrieve.call_args.kwargs["query"]
    assert call_query == "What is ML?"


@pytest.mark.asyncio
async def test_answer_passes_prompt_to_llm() -> None:
    """The assembled prompt is forwarded to llm_provider.generate()."""
    llm = MagicMock(spec=LLMProvider)
    llm.generate = AsyncMock(return_value="Answer.")
    chunk = _make_chunk(text="Context text here.")

    retriever = MagicMock(spec=Retriever)
    retriever.retrieve = AsyncMock(return_value=[chunk])

    service = _make_service(retriever=retriever, llm_provider=llm)
    await service.answer(question="My question?", user=_make_user())

    prompt_arg = llm.generate.call_args.args[0]
    assert "My question?" in prompt_arg
    assert "Context text here." in prompt_arg


@pytest.mark.asyncio
async def test_answer_uses_default_top_k_5() -> None:
    """top_k defaults to 5 when not supplied."""
    retriever = MagicMock(spec=Retriever)
    retriever.retrieve = AsyncMock(return_value=[])

    service = _make_service(retriever=retriever)
    await service.answer(question="Q?", user=_make_user())

    assert retriever.retrieve.call_args.kwargs["top_k"] == 5


# ---------------------------------------------------------------------------
# Citation generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_answer_citation_chunk_ids_match_retrieved_chunks() -> None:
    """The chunk IDs in retrieved_chunks match the chunks from the retriever."""
    chunk_id_a = str(uuid4())
    chunk_id_b = str(uuid4())
    doc_id = str(uuid4())

    chunks = [
        _make_chunk(chunk_id=chunk_id_a, document_id=doc_id, chunk_index=0),
        _make_chunk(chunk_id=chunk_id_b, document_id=doc_id, chunk_index=1),
    ]
    retriever = MagicMock(spec=Retriever)
    retriever.retrieve = AsyncMock(return_value=chunks)

    service = _make_service(retriever=retriever)
    result = await service.answer(question="Q?", user=_make_user())

    ids = {c.chunk_id for c in result.retrieved_chunks}
    assert chunk_id_a in ids
    assert chunk_id_b in ids


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_answer_raises_bad_request_for_empty_question() -> None:
    """An empty question raises BadRequestError immediately."""
    service = _make_service()
    with pytest.raises(BadRequestError):
        await service.answer(question="   ", user=_make_user())


@pytest.mark.asyncio
async def test_answer_propagates_service_unavailable_from_llm() -> None:
    """ServiceUnavailableError from the LLM is propagated unchanged."""
    llm = MagicMock(spec=LLMProvider)
    llm.generate = AsyncMock(side_effect=ServiceUnavailableError("LLM down"))

    service = _make_service(llm_provider=llm)
    with pytest.raises(ServiceUnavailableError, match="LLM down"):
        await service.answer(question="Q?", user=_make_user())


@pytest.mark.asyncio
async def test_answer_wraps_unexpected_llm_exception() -> None:
    """Unexpected exceptions from the LLM are wrapped as ServiceUnavailableError."""
    llm = MagicMock(spec=LLMProvider)
    llm.generate = AsyncMock(side_effect=RuntimeError("unexpected"))

    service = _make_service(llm_provider=llm)
    with pytest.raises(ServiceUnavailableError):
        await service.answer(question="Q?", user=_make_user())


@pytest.mark.asyncio
async def test_answer_propagates_retriever_not_found_error() -> None:
    """NotFoundError from the retriever propagates without wrapping."""
    from cortex.core.exceptions import NotFoundError

    retriever = MagicMock(spec=Retriever)
    retriever.retrieve = AsyncMock(side_effect=NotFoundError("Document not found"))

    service = _make_service(retriever=retriever)
    with pytest.raises(NotFoundError):
        await service.answer(question="Q?", user=_make_user(), document_id="bad-id")
