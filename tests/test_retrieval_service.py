"""Unit tests for SemanticRetriever."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from cortex.core.exceptions import BadRequestError, NotFoundError
from cortex.db.models.document import Document, DocumentStatus
from cortex.db.models.document_chunk import DocumentChunk
from cortex.db.models.user import User, UserRole
from cortex.retrieval.models import RetrievalResult
from cortex.retrieval.semantic import SemanticRetriever
from cortex.vectorstore.models import VectorSearchResult

EMBEDDING_DIMENSION = 384


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user(user_id: str | None = None) -> User:
    now = datetime.now(UTC)
    return User(
        id=user_id or str(uuid4()),
        email="tester@example.com",
        username="tester",
        full_name="Tester User",
        hashed_password="hashed",
        role=UserRole.USER,
        is_active=True,
        is_verified=False,
        created_at=now,
        updated_at=now,
    )


def _make_document(
    *,
    document_id: str | None = None,
    user_id: str,
    status: DocumentStatus = DocumentStatus.READY,
) -> Document:
    now = datetime.now(UTC)
    doc_id = document_id or str(uuid4())
    return Document(
        id=doc_id,
        user_id=user_id,
        title="Test Doc",
        original_filename="test.pdf",
        storage_filename=f"{doc_id}.pdf",
        storage_path=f"storage/documents/{doc_id}.pdf",
        mime_type="application/pdf",
        file_size=1024,
        status=status,
        created_at=now,
        updated_at=now,
    )


def _make_chunk(
    *,
    chunk_id: str | None = None,
    document_id: str,
    chunk_index: int = 0,
    text: str = "Hello world chunk text.",
) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id or str(uuid4()),
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        token_count=5,
        created_at=datetime.now(UTC),
    )


def _make_retriever(
    session: object,
    embedding_provider: object,
    vector_store: object,
) -> SemanticRetriever:
    return SemanticRetriever(
        session=session,  # type: ignore[arg-type]
        embedding_provider=embedding_provider,  # type: ignore[arg-type]
        vector_store=vector_store,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Session mock helpers
# ---------------------------------------------------------------------------


def _mock_session_with_chunks(chunks: list[DocumentChunk]) -> MagicMock:
    """Return a mock async session whose execute() yields the given chunks."""
    scalars_mock = MagicMock()
    scalars_mock.all.return_value = chunks

    result_mock = MagicMock()
    result_mock.scalars.return_value = scalars_mock
    result_mock.scalar_one_or_none.return_value = None  # default

    session = MagicMock()
    session.execute = AsyncMock(return_value=result_mock)
    return session


def _mock_session_with_document(document: Document | None) -> MagicMock:
    """Return a mock async session whose scalar_one_or_none returns document."""
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = document

    scalars_mock = MagicMock()
    scalars_mock.all.return_value = []
    result_mock.scalars.return_value = scalars_mock

    session = MagicMock()
    session.execute = AsyncMock(return_value=result_mock)
    return session


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_returns_ranked_results() -> None:
    """SemanticRetriever returns results ordered by vector score."""
    user = _make_user()
    doc_id = str(uuid4())
    chunk_id_a = str(uuid4())
    chunk_id_b = str(uuid4())

    chunk_a = _make_chunk(
        chunk_id=chunk_id_a, document_id=doc_id, chunk_index=0, text="Alpha"
    )
    chunk_b = _make_chunk(
        chunk_id=chunk_id_b, document_id=doc_id, chunk_index=1, text="Beta"
    )

    vector_hits = [
        VectorSearchResult(
            chunk_id=chunk_id_a, document_id=doc_id, chunk_index=0, score=0.95
        ),
        VectorSearchResult(
            chunk_id=chunk_id_b, document_id=doc_id, chunk_index=1, score=0.80
        ),
    ]

    embedding_provider = MagicMock()
    embedding_provider.embed = AsyncMock(return_value=[0.1] * EMBEDDING_DIMENSION)

    vector_store = MagicMock()
    vector_store.similarity_search = AsyncMock(return_value=vector_hits)

    # Session returns both chunks in the scalars result
    scalars_mock = MagicMock()
    scalars_mock.all.return_value = [chunk_a, chunk_b]
    result_mock = MagicMock()
    result_mock.scalars.return_value = scalars_mock
    session = MagicMock()
    session.execute = AsyncMock(return_value=result_mock)

    retriever = _make_retriever(session, embedding_provider, vector_store)
    results = await retriever.retrieve(query="test query", user=user, top_k=5)

    assert len(results) == 2
    assert results[0].chunk_id == chunk_id_a
    assert results[0].text == "Alpha"
    assert results[0].score == pytest.approx(0.95)
    assert results[1].chunk_id == chunk_id_b
    assert results[1].text == "Beta"
    assert results[1].score == pytest.approx(0.80)


@pytest.mark.asyncio
async def test_retrieve_with_document_id_filter_passes_to_vector_store() -> None:
    """When document_id is given, it is forwarded to similarity_search."""
    user = _make_user()
    doc_id = str(uuid4())
    document = _make_document(document_id=doc_id, user_id=user.id)
    chunk = _make_chunk(chunk_id=str(uuid4()), document_id=doc_id)

    embedding_provider = MagicMock()
    embedding_provider.embed = AsyncMock(return_value=[0.1] * EMBEDDING_DIMENSION)

    vector_store = MagicMock()
    vector_store.similarity_search = AsyncMock(
        return_value=[
            VectorSearchResult(
                chunk_id=chunk.id,
                document_id=doc_id,
                chunk_index=0,
                score=0.88,
            )
        ]
    )

    # First execute call → document lookup; second → chunk hydration
    doc_result = MagicMock()
    doc_result.scalar_one_or_none.return_value = document

    chunk_scalars = MagicMock()
    chunk_scalars.all.return_value = [chunk]
    chunk_result = MagicMock()
    chunk_result.scalars.return_value = chunk_scalars

    session = MagicMock()
    session.execute = AsyncMock(side_effect=[doc_result, chunk_result])

    retriever = _make_retriever(session, embedding_provider, vector_store)
    results = await retriever.retrieve(
        query="filtered query",
        user=user,
        top_k=3,
        document_id=doc_id,
    )

    assert len(results) == 1
    assert results[0].document_id == doc_id

    vector_store.similarity_search.assert_called_once_with(
        query_embedding=[0.1] * EMBEDDING_DIMENSION,
        limit=3,
        document_id=doc_id,
    )


@pytest.mark.asyncio
async def test_retrieve_empty_vector_results_returns_empty_list() -> None:
    """When the vector store returns no hits, retrieve returns []."""
    user = _make_user()

    embedding_provider = MagicMock()
    embedding_provider.embed = AsyncMock(return_value=[0.1] * EMBEDDING_DIMENSION)

    vector_store = MagicMock()
    vector_store.similarity_search = AsyncMock(return_value=[])

    session = MagicMock()

    retriever = _make_retriever(session, embedding_provider, vector_store)
    results = await retriever.retrieve(query="nothing matches", user=user, top_k=5)

    assert results == []
    # Session should never be queried for chunks when there are no vector hits
    session.execute.assert_not_called()


@pytest.mark.asyncio
async def test_retrieve_drops_cross_user_vector_hits() -> None:
    """Chunks absent from the owned-chunks query (cross-user leak) are
    silently dropped.
    """
    user = _make_user()
    doc_id = str(uuid4())
    stale_chunk_id = str(uuid4())

    embedding_provider = MagicMock()
    embedding_provider.embed = AsyncMock(return_value=[0.1] * EMBEDDING_DIMENSION)

    # Vector store returns a hit, but the SQL query returns zero owned chunks.
    vector_store = MagicMock()
    vector_store.similarity_search = AsyncMock(
        return_value=[
            VectorSearchResult(
                chunk_id=stale_chunk_id,
                document_id=doc_id,
                chunk_index=0,
                score=0.99,
            )
        ]
    )

    scalars_mock = MagicMock()
    scalars_mock.all.return_value = []  # no owned chunks found
    result_mock = MagicMock()
    result_mock.scalars.return_value = scalars_mock
    session = MagicMock()
    session.execute = AsyncMock(return_value=result_mock)

    retriever = _make_retriever(session, embedding_provider, vector_store)
    results = await retriever.retrieve(query="cross-user test", user=user, top_k=5)

    assert results == []


@pytest.mark.asyncio
async def test_retrieve_returns_correct_result_type() -> None:
    """Each returned item is a RetrievalResult dataclass instance."""
    user = _make_user()
    doc_id = str(uuid4())
    chunk = _make_chunk(document_id=doc_id, text="Some content.")

    embedding_provider = MagicMock()
    embedding_provider.embed = AsyncMock(return_value=[0.2] * EMBEDDING_DIMENSION)

    vector_store = MagicMock()
    vector_store.similarity_search = AsyncMock(
        return_value=[
            VectorSearchResult(
                chunk_id=chunk.id,
                document_id=doc_id,
                chunk_index=0,
                score=0.75,
            )
        ]
    )

    scalars_mock = MagicMock()
    scalars_mock.all.return_value = [chunk]
    result_mock = MagicMock()
    result_mock.scalars.return_value = scalars_mock
    session = MagicMock()
    session.execute = AsyncMock(return_value=result_mock)

    retriever = _make_retriever(session, embedding_provider, vector_store)
    results = await retriever.retrieve(query="type check", user=user, top_k=1)

    assert len(results) == 1
    assert isinstance(results[0], RetrievalResult)
    assert results[0].text == "Some content."
    assert results[0].score == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Validation / error tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_raises_on_empty_query() -> None:
    """An empty (or whitespace-only) query raises BadRequestError."""
    user = _make_user()
    session = MagicMock()
    embedding_provider = MagicMock()
    vector_store = MagicMock()

    retriever = _make_retriever(session, embedding_provider, vector_store)

    with pytest.raises(BadRequestError, match="Query must not be empty"):
        await retriever.retrieve(query="   ", user=user, top_k=5)


@pytest.mark.asyncio
async def test_retrieve_raises_not_found_for_unknown_document_id() -> None:
    """Passing an unowned or nonexistent document_id raises NotFoundError."""
    user = _make_user()

    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = None  # document not found

    session = MagicMock()
    session.execute = AsyncMock(return_value=result_mock)

    embedding_provider = MagicMock()
    vector_store = MagicMock()

    retriever = _make_retriever(session, embedding_provider, vector_store)

    with pytest.raises(NotFoundError, match="Document not found"):
        await retriever.retrieve(
            query="valid query",
            user=user,
            top_k=5,
            document_id=str(uuid4()),
        )


@pytest.mark.asyncio
async def test_retrieve_raises_bad_request_for_non_ready_document() -> None:
    """A document not yet in READY status raises BadRequestError."""
    user = _make_user()
    doc_id = str(uuid4())
    document = _make_document(
        document_id=doc_id,
        user_id=user.id,
        status=DocumentStatus.UPLOADED,
    )

    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = document

    session = MagicMock()
    session.execute = AsyncMock(return_value=result_mock)

    embedding_provider = MagicMock()
    vector_store = MagicMock()

    retriever = _make_retriever(session, embedding_provider, vector_store)

    with pytest.raises(BadRequestError, match="Document has not been processed yet"):
        await retriever.retrieve(
            query="valid query",
            user=user,
            top_k=5,
            document_id=doc_id,
        )


@pytest.mark.asyncio
async def test_retrieve_raises_bad_request_for_processing_document() -> None:
    """A document in PROCESSING status also raises BadRequestError."""
    user = _make_user()
    doc_id = str(uuid4())
    document = _make_document(
        document_id=doc_id,
        user_id=user.id,
        status=DocumentStatus.PROCESSING,
    )

    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = document

    session = MagicMock()
    session.execute = AsyncMock(return_value=result_mock)

    retriever = _make_retriever(session, MagicMock(), MagicMock())

    with pytest.raises(BadRequestError):
        await retriever.retrieve(
            query="valid query",
            user=user,
            top_k=5,
            document_id=doc_id,
        )


@pytest.mark.asyncio
async def test_embedding_called_with_stripped_query() -> None:
    """The query passed to the embedding provider is stripped of whitespace."""
    user = _make_user()

    embedding_provider = MagicMock()
    embedding_provider.embed = AsyncMock(return_value=[0.1] * EMBEDDING_DIMENSION)

    vector_store = MagicMock()
    vector_store.similarity_search = AsyncMock(return_value=[])

    session = MagicMock()

    retriever = _make_retriever(session, embedding_provider, vector_store)
    await retriever.retrieve(query="  hello world  ", user=user, top_k=5)

    embedding_provider.embed.assert_called_once_with("hello world")
