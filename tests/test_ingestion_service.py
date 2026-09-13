"""Unit tests for IngestionService."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from cortex.core.config import Settings
from cortex.core.exceptions import DocumentProcessingError, NotFoundError
from cortex.db.models.document import Document, DocumentStatus
from cortex.db.models.user import User
from cortex.services.chunking import ChunkingService, TextChunk
from cortex.services.cleaning import CleaningService
from cortex.services.ingestion import IngestionService
from cortex.services.parser import ParserService
from cortex.services.storage import StorageService
from tests.pdf_helpers import build_pdf_with_text


def _make_ingestion_service(
    *,
    test_settings: Settings,
    session: AsyncMock | None = None,
    storage: AsyncMock | None = None,
    parser: ParserService | AsyncMock | None = None,
) -> tuple[IngestionService, AsyncMock, AsyncMock]:
    session = session or AsyncMock()
    storage = storage or AsyncMock(spec=StorageService)
    storage.read = AsyncMock(return_value=build_pdf_with_text("Ingestion test content"))
    parser = parser or ParserService()
    cleaning = CleaningService()
    chunking = ChunkingService(
        test_settings.model_copy(
            update={
                "ingestion_chunk_size_words": 50,
                "ingestion_chunk_overlap_words": 10,
            }
        )
    )
    service = IngestionService(
        session=session,
        storage_service=storage,
        parser_service=parser,
        cleaning_service=cleaning,
        chunking_service=chunking,
    )
    return service, session, storage


@pytest.mark.asyncio
async def test_process_persists_chunks_and_sets_ready(
    test_settings: Settings,
    sample_user: User,
    sample_document: Document,
) -> None:
    """Successful ingestion stores chunks and marks the document READY."""
    session = AsyncMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.commit = AsyncMock()

    service, _, storage = _make_ingestion_service(
        test_settings=test_settings,
        session=session,
    )
    service._get_owned_document = AsyncMock(return_value=sample_document)  # type: ignore[method-assign]
    service._delete_existing_chunks = AsyncMock()  # type: ignore[method-assign]

    async def _persist(document_id: str, text_chunks: list[TextChunk]) -> list:
        now = datetime.now(UTC)
        return [
            MagicMock(
                id=str(uuid4()),
                document_id=document_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                token_count=chunk.token_count,
                created_at=now,
            )
            for chunk in text_chunks
        ]

    service._persist_chunks = AsyncMock(side_effect=_persist)  # type: ignore[method-assign]

    result = await service.process(document_id=sample_document.id, user=sample_user)

    storage.read.assert_awaited_once_with(sample_document.storage_path)
    service._delete_existing_chunks.assert_awaited_once_with(sample_document.id)  # type: ignore[attr-defined]
    service._persist_chunks.assert_awaited_once()  # type: ignore[attr-defined]
    assert sample_document.status == DocumentStatus.READY
    assert result.status == DocumentStatus.READY
    assert result.chunk_count >= 1


@pytest.mark.asyncio
async def test_process_deletes_existing_chunks_before_insert(
    test_settings: Settings,
    sample_user: User,
    sample_document: Document,
) -> None:
    """Reprocessing deletes existing chunks before inserting new ones."""
    session = AsyncMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()

    service, _, _ = _make_ingestion_service(
        test_settings=test_settings,
        session=session,
    )
    service._get_owned_document = AsyncMock(return_value=sample_document)  # type: ignore[method-assign]
    service._delete_existing_chunks = AsyncMock()  # type: ignore[method-assign]
    service._persist_chunks = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            MagicMock(
                id=str(uuid4()),
                document_id=sample_document.id,
                chunk_index=0,
                text="chunk",
                token_count=1,
                created_at=datetime.now(UTC),
            )
        ]
    )

    await service.process(document_id=sample_document.id, user=sample_user)

    service._delete_existing_chunks.assert_awaited_once_with(sample_document.id)  # type: ignore[attr-defined]
    service._persist_chunks.assert_awaited_once()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_process_marks_document_failed_on_parser_error(
    test_settings: Settings,
    sample_user: User,
    sample_document: Document,
) -> None:
    """Parser failures mark the document FAILED and raise a structured error."""
    parser = AsyncMock(spec=ParserService)
    parser.parse_pdf = AsyncMock(
        side_effect=DocumentProcessingError("PDF document is unreadable or corrupted")
    )
    session = AsyncMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    service, _, _ = _make_ingestion_service(
        test_settings=test_settings,
        session=session,
        parser=parser,
    )
    service._get_owned_document = AsyncMock(return_value=sample_document)  # type: ignore[method-assign]
    service._delete_existing_chunks = AsyncMock()  # type: ignore[method-assign]

    with pytest.raises(DocumentProcessingError):
        await service.process(document_id=sample_document.id, user=sample_user)

    assert sample_document.status == DocumentStatus.FAILED
    session.commit.assert_awaited()


@pytest.mark.asyncio
async def test_process_raises_not_found_for_foreign_document(
    test_settings: Settings,
    sample_user: User,
) -> None:
    """Foreign or missing documents raise NotFoundError."""
    service, _, _ = _make_ingestion_service(test_settings=test_settings)
    service._get_owned_document = AsyncMock(  # type: ignore[method-assign]
        side_effect=NotFoundError("Document not found", details={"document_id": "x"}),
    )

    with pytest.raises(NotFoundError):
        await service.process(document_id="missing-id", user=sample_user)
