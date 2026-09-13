"""Unit tests for DocumentService."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from cortex.core.config import Settings
from cortex.core.exceptions import BadRequestError, NotFoundError
from cortex.db.models.document import Document, DocumentStatus
from cortex.db.models.user import User
from cortex.schemas.document import DocumentRead
from cortex.services.document import DocumentService
from cortex.services.storage import StorageService


def _make_document_service(
    *,
    test_settings: Settings,
    session: AsyncMock | None = None,
    storage: AsyncMock | None = None,
) -> tuple[DocumentService, AsyncMock, AsyncMock]:
    session = session or AsyncMock()
    storage = storage or AsyncMock(spec=StorageService)
    storage.generate_storage_filename = MagicMock(return_value=f"{uuid4()}.pdf")
    storage.save = AsyncMock(return_value="storage/documents/test.pdf")
    storage.delete = AsyncMock()
    service = DocumentService(
        session=session,
        settings=test_settings,
        storage_service=storage,
    )
    return service, session, storage


@pytest.mark.asyncio
async def test_upload_persists_document(
    test_settings: Settings,
    sample_user: User,
    minimal_pdf_bytes: bytes,
) -> None:
    """Valid PDF uploads are stored and persisted with UPLOADED status."""
    service, session, storage = _make_document_service(test_settings=test_settings)
    session.add = MagicMock()
    session.refresh = AsyncMock()

    async def _assign_id_on_flush() -> None:
        for call in session.add.call_args_list:
            document = call.args[0]
            document.id = str(uuid4())
            document.created_at = datetime.now(UTC)
            document.updated_at = datetime.now(UTC)

    session.flush = AsyncMock(side_effect=_assign_id_on_flush)

    result = await service.upload(
        user=sample_user,
        content=minimal_pdf_bytes,
        original_filename="report.pdf",
        mime_type="application/pdf",
        title="Quarterly Report",
    )

    storage.save.assert_awaited_once()
    session.add.assert_called_once()
    added: Document = session.add.call_args.args[0]
    assert added.user_id == sample_user.id
    assert added.title == "Quarterly Report"
    assert added.status == DocumentStatus.UPLOADED
    assert isinstance(result, DocumentRead)


@pytest.mark.asyncio
async def test_upload_rejects_non_pdf_mime(
    test_settings: Settings,
    sample_user: User,
    minimal_pdf_bytes: bytes,
) -> None:
    """Non-PDF MIME types are rejected."""
    service, _, _ = _make_document_service(test_settings=test_settings)

    with pytest.raises(BadRequestError) as exc_info:
        await service.upload(
            user=sample_user,
            content=minimal_pdf_bytes,
            original_filename="report.pdf",
            mime_type="text/plain",
        )
    assert exc_info.value.code == "bad_request"


@pytest.mark.asyncio
async def test_upload_rejects_oversized_file(
    test_settings: Settings,
    sample_user: User,
    minimal_pdf_bytes: bytes,
) -> None:
    """Files exceeding the configured size limit are rejected."""
    small_settings = test_settings.model_copy(
        update={"document_max_file_size_bytes": 10},
    )
    service, _, _ = _make_document_service(test_settings=small_settings)

    with pytest.raises(BadRequestError) as exc_info:
        await service.upload(
            user=sample_user,
            content=minimal_pdf_bytes,
            original_filename="report.pdf",
            mime_type="application/pdf",
        )
    assert "maximum allowed size" in exc_info.value.message


@pytest.mark.asyncio
async def test_upload_rejects_invalid_pdf_magic(
    test_settings: Settings,
    sample_user: User,
) -> None:
    """Files without PDF magic bytes are rejected."""
    service, _, _ = _make_document_service(test_settings=test_settings)

    with pytest.raises(BadRequestError) as exc_info:
        await service.upload(
            user=sample_user,
            content=b"not-a-pdf",
            original_filename="report.pdf",
            mime_type="application/pdf",
        )
    assert "not a valid PDF" in exc_info.value.message


@pytest.mark.asyncio
async def test_get_returns_owned_document(
    test_settings: Settings,
    sample_user: User,
    sample_document: Document,
) -> None:
    """Owners can retrieve their documents."""
    service, session, _ = _make_document_service(test_settings=test_settings)
    session.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=sample_document))
    )

    result = await service.get(document_id=sample_document.id, user=sample_user)
    assert result.id == sample_document.id


@pytest.mark.asyncio
async def test_get_raises_for_missing_or_foreign_document(
    test_settings: Settings,
    sample_user: User,
) -> None:
    """Missing or foreign documents raise NotFoundError."""
    service, session, _ = _make_document_service(test_settings=test_settings)
    session.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    )

    with pytest.raises(NotFoundError):
        await service.get(document_id=str(uuid4()), user=sample_user)


@pytest.mark.asyncio
async def test_list_returns_paginated_documents(
    test_settings: Settings,
    sample_user: User,
    sample_document: Document,
) -> None:
    """List returns owned documents with pagination metadata."""
    service, session, _ = _make_document_service(test_settings=test_settings)
    list_result = MagicMock(
        scalars=MagicMock(return_value=MagicMock(all=lambda: [sample_document])),
    )
    session.execute = AsyncMock(
        side_effect=[
            MagicMock(scalar_one=MagicMock(return_value=1)),
            list_result,
        ]
    )

    result = await service.list(user=sample_user, skip=0, limit=10)
    assert result.total == 1
    assert len(result.items) == 1
    assert result.items[0].id == sample_document.id


@pytest.mark.asyncio
async def test_delete_removes_record_and_file(
    test_settings: Settings,
    sample_user: User,
    sample_document: Document,
) -> None:
    """Delete removes the database row and stored file."""
    service, session, storage = _make_document_service(test_settings=test_settings)
    session.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=sample_document))
    )
    session.delete = AsyncMock()
    session.flush = AsyncMock()

    await service.delete(document_id=sample_document.id, user=sample_user)

    session.delete.assert_awaited_once_with(sample_document)
    storage.delete.assert_awaited_once_with(sample_document.storage_path)
