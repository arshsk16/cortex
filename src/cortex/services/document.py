"""Document management service — upload, retrieval, listing, and deletion."""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.core.config import Settings
from cortex.core.exceptions import BadRequestError, NotFoundError
from cortex.db.models.document import Document, DocumentStatus
from cortex.db.models.user import User
from cortex.schemas.document import DocumentList, DocumentRead
from cortex.services.storage import PDF_EXTENSION, StorageService

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"


class DocumentService:
    """Coordinates document validation, persistence, and ownership checks."""

    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        storage_service: StorageService,
    ) -> None:
        self._session = session
        self._settings = settings
        self._storage = storage_service

    async def upload(
        self,
        *,
        user: User,
        content: bytes,
        original_filename: str,
        mime_type: str | None,
        title: str | None = None,
    ) -> DocumentRead:
        """Validate, store, and persist metadata for a PDF upload."""
        self._validate_upload(
            content=content,
            original_filename=original_filename,
            mime_type=mime_type,
        )

        resolved_title = self._resolve_title(
            title=title,
            original_filename=original_filename,
        )
        storage_filename = self._storage.generate_storage_filename()
        storage_path: str | None = None

        try:
            storage_path = await self._storage.save(content, storage_filename)
            document = Document(
                user_id=user.id,
                title=resolved_title,
                original_filename=original_filename,
                storage_filename=storage_filename,
                storage_path=storage_path,
                mime_type=self._settings.document_allowed_mime_type,
                file_size=len(content),
                status=DocumentStatus.UPLOADED,
            )
            self._session.add(document)
            await self._session.flush()
            await self._session.refresh(document)
        except Exception:
            if storage_path is not None:
                await self._storage.delete(storage_path)
            raise

        logger.info(
            "Uploaded document id=%s user_id=%s filename=%s size=%d",
            document.id,
            user.id,
            original_filename,
            len(content),
        )
        return DocumentRead.model_validate(document)

    async def get(self, *, document_id: str, user: User) -> DocumentRead:
        """Return a document owned by ``user`` or raise NotFoundError."""
        document = await self._get_owned_document(document_id=document_id, user=user)
        return DocumentRead.model_validate(document)

    async def list(
        self,
        *,
        user: User,
        skip: int = 0,
        limit: int = 50,
    ) -> DocumentList:
        """Return a paginated list of documents owned by ``user``."""
        total_statement = (
            select(func.count())
            .select_from(Document)
            .where(Document.user_id == user.id)
        )
        total_result = await self._session.execute(total_statement)
        total = int(total_result.scalar_one())

        statement: Select[tuple[Document]] = (
            select(Document)
            .where(Document.user_id == user.id)
            .order_by(Document.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self._session.execute(statement)
        documents = list(result.scalars().all())

        return DocumentList(
            items=[DocumentRead.model_validate(doc) for doc in documents],
            total=total,
            skip=skip,
            limit=limit,
        )

    async def delete(self, *, document_id: str, user: User) -> None:
        """Delete a user-owned document and its stored file."""
        document = await self._get_owned_document(document_id=document_id, user=user)
        storage_path = document.storage_path

        await self._session.delete(document)
        await self._session.flush()
        await self._storage.delete(storage_path)

        logger.info(
            "Deleted document id=%s user_id=%s",
            document_id,
            user.id,
        )

    async def _get_owned_document(self, *, document_id: str, user: User) -> Document:
        """Fetch a document ensuring it belongs to ``user``."""
        statement: Select[tuple[Document]] = select(Document).where(
            Document.id == document_id,
            Document.user_id == user.id,
        )
        result = await self._session.execute(statement)
        document = result.scalar_one_or_none()
        if document is None:
            raise NotFoundError(
                "Document not found",
                details={"document_id": document_id},
            )
        return document

    def _validate_upload(
        self,
        *,
        content: bytes,
        original_filename: str,
        mime_type: str | None,
    ) -> None:
        """Validate PDF uploads against size, MIME type, extension, and magic bytes."""
        if not content:
            raise BadRequestError(
                "Uploaded file is empty",
                details={"field": "file"},
            )

        max_size = self._settings.document_max_file_size_bytes
        if len(content) > max_size:
            raise BadRequestError(
                "Uploaded file exceeds the maximum allowed size",
                details={
                    "max_size_bytes": max_size,
                    "received_size_bytes": len(content),
                },
            )

        normalized_mime = (mime_type or "").split(";", maxsplit=1)[0].strip().lower()
        allowed_mime = self._settings.document_allowed_mime_type.lower()
        if normalized_mime != allowed_mime:
            raise BadRequestError(
                "Only PDF files are allowed",
                details={
                    "allowed_mime_type": allowed_mime,
                    "received_mime_type": normalized_mime or None,
                },
            )

        suffix = Path(original_filename).suffix.lower()
        if suffix != PDF_EXTENSION:
            raise BadRequestError(
                "Only PDF files are allowed",
                details={
                    "allowed_extension": PDF_EXTENSION,
                    "received_extension": suffix,
                },
            )

        if not content.startswith(PDF_MAGIC):
            raise BadRequestError(
                "Uploaded file is not a valid PDF",
                details={"field": "file"},
            )

    @staticmethod
    def _resolve_title(*, title: str | None, original_filename: str) -> str:
        """Use an explicit title or derive one from the original filename."""
        if title is not None:
            normalized = title.strip()
            if normalized:
                return normalized[:255]

        stem = Path(original_filename).stem.strip()
        if stem:
            return stem[:255]
        return "Untitled Document"
