"""Semantic retriever — embedding-based vector search over owned documents."""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.core.exceptions import BadRequestError, NotFoundError
from cortex.db.models.document import Document, DocumentStatus
from cortex.db.models.document_chunk import DocumentChunk
from cortex.db.models.user import User
from cortex.embeddings.base import EmbeddingProvider
from cortex.retrieval.base import Retriever
from cortex.retrieval.models import RetrievalResult
from cortex.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


class SemanticRetriever(Retriever):
    """Retrieve chunks by embedding the query and searching ChromaDB.

    Ownership is enforced in two places:

    1. When ``document_id`` is supplied, the document is fetched from
       PostgreSQL and verified to belong to ``user`` before vector search.
    2. After vector search, chunk texts are fetched from PostgreSQL using
       a join on ``document.user_id``.  Any stale or cross-user vector hits
       are silently dropped — the user only ever sees their own content.
    """

    def __init__(
        self,
        session: AsyncSession,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._session = session
        self._embedder = embedding_provider
        self._vector_store = vector_store

    async def retrieve(
        self,
        *,
        query: str,
        user: User,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[RetrievalResult]:
        """Return up to *top_k* chunks most semantically similar to *query*.

        Parameters
        ----------
        query:
            Natural-language search string; must be non-empty after strip.
        user:
            Authenticated owner — chunks from other users are never returned.
        top_k:
            Maximum result count (1–100).
        document_id:
            When set, search is restricted to this single document.  The
            document must be owned by ``user`` *and* have status ``READY``.
        """
        query = query.strip()
        if not query:
            raise BadRequestError(
                "Query must not be empty",
                details={"field": "query"},
            )

        if document_id is not None:
            await self._assert_document_owned_and_ready(
                document_id=document_id,
                user=user,
            )

        query_embedding = await self._embedder.embed(query)

        vector_hits = await self._vector_store.similarity_search(
            query_embedding=query_embedding,
            limit=top_k,
            document_id=document_id,
        )

        if not vector_hits:
            return []

        # Build a set of chunk IDs from the vector hits for the SQL IN-clause.
        hit_chunk_ids = [hit.chunk_id for hit in vector_hits]

        # Fetch chunk texts from PostgreSQL, enforcing user ownership via the
        # documents join.  Any hit whose document belongs to another user is
        # simply absent from the result set and will be dropped below.
        statement = (
            select(DocumentChunk)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(
                DocumentChunk.id.in_(hit_chunk_ids),
                Document.user_id == user.id,
            )
        )
        result = await self._session.execute(statement)
        chunks_by_id = {chunk.id: chunk for chunk in result.scalars().all()}

        # Re-rank: preserve the vector-store order (highest score first),
        # dropping any hit that was not found in the owned-chunks query.
        retrieval_results: list[RetrievalResult] = []
        for hit in vector_hits:
            chunk = chunks_by_id.get(hit.chunk_id)
            if chunk is None:
                logger.warning(
                    "Vector hit chunk_id=%s not found in owned chunks "
                    "(document_id=%s user_id=%s) — skipping",
                    hit.chunk_id,
                    hit.document_id,
                    user.id,
                )
                continue
            retrieval_results.append(
                RetrievalResult(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    score=hit.score,
                )
            )

        logger.info(
            "Retrieval query=%r user_id=%s document_id=%s top_k=%d hits=%d",
            query[:80],
            user.id,
            document_id,
            top_k,
            len(retrieval_results),
        )
        return retrieval_results

    async def _assert_document_owned_and_ready(
        self,
        *,
        document_id: str,
        user: User,
    ) -> None:
        """Raise if the document does not exist, is not owned, or is not READY."""
        stmt = select(Document).where(
            Document.id == document_id,
            Document.user_id == user.id,
        )
        result = await self._session.execute(stmt)
        document = result.scalar_one_or_none()

        if document is None:
            raise NotFoundError(
                "Document not found",
                details={"document_id": document_id},
            )

        if document.status != DocumentStatus.READY:
            raise BadRequestError(
                "Document has not been processed yet and cannot be searched",
                details={
                    "document_id": document_id,
                    "status": document.status,
                },
            )
