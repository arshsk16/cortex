"""Document ingestion orchestration service."""

from __future__ import annotations

import logging

from sqlalchemy import Select, delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.core.exceptions import DocumentProcessingError, NotFoundError
from cortex.db.models.document import Document, DocumentStatus
from cortex.db.models.document_chunk import DocumentChunk
from cortex.db.models.user import User
from cortex.embeddings.base import EmbeddingProvider
from cortex.schemas.chunk import DocumentChunkRead
from cortex.schemas.document import DocumentRead
from cortex.schemas.ingestion import DocumentProcessResponse
from cortex.services.chunking import ChunkingService, TextChunk
from cortex.services.cleaning import CleaningService
from cortex.services.parser import ParserService
from cortex.services.storage import StorageService
from cortex.vectorstore.base import VectorStore
from cortex.vectorstore.models import ChunkVectorRecord

logger = logging.getLogger(__name__)


class IngestionService:
    """Coordinates PDF parsing, cleaning, chunking, embedding, and persistence."""

    def __init__(
        self,
        session: AsyncSession,
        storage_service: StorageService,
        parser_service: ParserService,
        cleaning_service: CleaningService,
        chunking_service: ChunkingService,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._session = session
        self._storage = storage_service
        self._parser = parser_service
        self._cleaner = cleaning_service
        self._chunker = chunking_service
        self._embedder = embedding_provider
        self._vector_store = vector_store

    async def process(self, *, document_id: str, user: User) -> DocumentProcessResponse:
        """Run the ingestion pipeline for an owned document."""
        document = await self._get_owned_document(document_id=document_id, user=user)
        document.status = DocumentStatus.PROCESSING
        await self._session.flush()

        try:
            pdf_bytes = await self._storage.read(document.storage_path)
            raw_text = await self._parser.parse_pdf(pdf_bytes)
            cleaned_text = self._cleaner.clean(raw_text)
            text_chunks = self._chunker.chunk(cleaned_text)

            if not text_chunks:
                raise DocumentProcessingError(
                    "Document contains no processable text after cleaning",
                    details={"document_id": document.id},
                )

            embeddings = await self._generate_embeddings(text_chunks)

            await self._delete_existing_chunks(document.id)
            await self._vector_store.delete_document(document.id)
            persisted_chunks = await self._persist_chunks(document.id, text_chunks)
            await self._store_vectors(document.id, persisted_chunks, embeddings)

            document.status = DocumentStatus.READY
            await self._session.flush()
            await self._session.refresh(document)

            logger.info(
                "Processed document id=%s user_id=%s chunks=%d",
                document.id,
                user.id,
                len(persisted_chunks),
            )
            return DocumentProcessResponse(
                document=DocumentRead.model_validate(document),
                status=document.status,
                chunk_count=len(persisted_chunks),
                chunks=[
                    DocumentChunkRead.model_validate(chunk)
                    for chunk in persisted_chunks
                ],
            )
        except DocumentProcessingError:
            await self._mark_failed(document)
            raise
        except Exception as exc:
            logger.exception(
                "Unexpected ingestion failure for document id=%s",
                document.id,
            )
            await self._mark_failed(document)
            raise DocumentProcessingError(
                "Document processing failed",
                details={"document_id": document.id, "reason": str(exc)},
            ) from exc

    async def _generate_embeddings(
        self,
        text_chunks: list[TextChunk],
    ) -> list[list[float]]:
        """Generate embeddings before relational or vector mutations."""
        texts = [chunk.text for chunk in text_chunks]
        embeddings = await self._embedder.embed_batch(texts)
        if len(embeddings) != len(text_chunks):
            raise DocumentProcessingError(
                "Embedding generation returned an unexpected number of vectors",
                details={
                    "expected": len(text_chunks),
                    "received": len(embeddings),
                },
            )
        return embeddings

    async def _store_vectors(
        self,
        document_id: str,
        persisted_chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Persist chunk vectors after PostgreSQL rows are available."""
        records = [
            ChunkVectorRecord(
                chunk_id=chunk.id,
                document_id=document_id,
                chunk_index=chunk.chunk_index,
                embedding=embedding,
            )
            for chunk, embedding in zip(persisted_chunks, embeddings, strict=True)
        ]
        await self._vector_store.add_chunks(records)

    async def _mark_failed(self, document: Document) -> None:
        """Persist ``FAILED`` status so operators can inspect failed ingestions."""
        document.status = DocumentStatus.FAILED
        await self._session.flush()
        await self._session.commit()

    async def _delete_existing_chunks(self, document_id: str) -> None:
        statement = delete(DocumentChunk).where(
            DocumentChunk.document_id == document_id,
        )
        await self._session.execute(statement)
        await self._session.flush()

    async def _persist_chunks(
        self,
        document_id: str,
        text_chunks: list[TextChunk],
    ) -> list[DocumentChunk]:
        records = [
            DocumentChunk(
                document_id=document_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                token_count=chunk.token_count,
            )
            for chunk in text_chunks
        ]
        self._session.add_all(records)
        await self._session.flush()
        for record in records:
            await self._session.refresh(record)
        return records

    async def _get_owned_document(self, *, document_id: str, user: User) -> Document:
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
