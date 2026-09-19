"""MemoryService -- CRUD and semantic search for long-term user memories.

Design
------
* PostgreSQL is the durable source of truth for every memory record.
* Chroma (via ``MemoryVectorStore``) stores only the embedding vector plus
  a ``user_id`` metadata field used for per-user filtering.
* All mutations touch PostgreSQL first; Chroma is updated after a successful
  PG write.  A Chroma failure on create/delete is surfaced to the caller
  (it is not silently swallowed) so the UI can retry.  Future phases may
  add a reconciliation job for eventual consistency.
* Ownership is enforced at every operation: a user can only read, delete,
  or search their own memories.
"""

from __future__ import annotations

import logging

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.core.exceptions import NotFoundError
from cortex.db.models.memory import Memory
from cortex.db.models.user import User
from cortex.embeddings.base import EmbeddingProvider
from cortex.schemas.memory import (
    MemoryCreate,
    MemoryList,
    MemoryRead,
    MemorySearchResult,
    MemoryUpdate,
)
from cortex.vectorstore.memory_store import MemoryVectorStore

logger = logging.getLogger(__name__)


class MemoryService:
    """Coordinates memory persistence, embedding, and retrieval."""

    def __init__(
        self,
        *,
        session: AsyncSession,
        embedding_provider: EmbeddingProvider,
        memory_vector_store: MemoryVectorStore,
    ) -> None:
        self._session = session
        self._embedding_provider = embedding_provider
        self._vector_store = memory_vector_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def create(self, *, user: User, payload: MemoryCreate) -> MemoryRead:
        """Persist a new memory and index its embedding in Chroma."""
        # 1. Persist to PostgreSQL first.
        memory = Memory(
            user_id=user.id,
            content=payload.content,
            memory_metadata=payload.memory_metadata,
        )
        self._session.add(memory)
        await self._session.flush()
        await self._session.refresh(memory)

        # 2. Embed and upsert to Chroma.
        embedding = await self._embedding_provider.embed(payload.content)
        await self._vector_store.upsert(
            memory_id=memory.id,
            user_id=user.id,
            embedding=embedding,
        )

        logger.info(
            "Created memory id=%s user_id=%s content_len=%d",
            memory.id,
            user.id,
            len(payload.content),
        )
        return MemoryRead.model_validate(memory)

    async def get(self, *, memory_id: str, user: User) -> MemoryRead:
        """Return a memory owned by ``user`` or raise ``NotFoundError``."""
        memory = await self._get_owned(memory_id=memory_id, user=user)
        return MemoryRead.model_validate(memory)

    async def list(
        self,
        *,
        user: User,
        skip: int = 0,
        limit: int = 50,
    ) -> MemoryList:
        """Return a paginated list of memories owned by ``user``."""
        total_stmt = (
            select(func.count())
            .select_from(Memory)
            .where(Memory.user_id == user.id)
        )
        total = int((await self._session.execute(total_stmt)).scalar_one())

        stmt: Select[tuple[Memory]] = (
            select(Memory)
            .where(Memory.user_id == user.id)
            .order_by(Memory.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        rows = list((await self._session.execute(stmt)).scalars().all())

        return MemoryList(
            items=[MemoryRead.model_validate(m) for m in rows],
            total=total,
            skip=skip,
            limit=limit,
        )

    async def update(
        self,
        *,
        memory_id: str,
        user: User,
        payload: MemoryUpdate,
    ) -> MemoryRead:
        """Update a memory owned by ``user`` and re-index its embedding in Chroma."""
        memory = await self._get_owned(memory_id=memory_id, user=user)

        # 1. Update PostgreSQL first
        memory.content = payload.content
        await self._session.flush()
        await self._session.refresh(memory)

        # 2. Embed and upsert to Chroma
        embedding = await self._embedding_provider.embed(payload.content)
        await self._vector_store.upsert(
            memory_id=memory.id,
            user_id=user.id,
            embedding=embedding,
        )

        logger.info(
            "Updated memory id=%s user_id=%s content_len=%d",
            memory.id,
            user.id,
            len(payload.content),
        )
        return MemoryRead.model_validate(memory)

    async def delete(self, *, memory_id: str, user: User) -> None:
        """Delete a memory owned by ``user`` from both PostgreSQL and Chroma."""
        memory = await self._get_owned(memory_id=memory_id, user=user)
        # Delete from Chroma first (reversible if PG fails).
        await self._vector_store.delete(memory.id)
        await self._session.delete(memory)
        await self._session.flush()
        logger.info("Deleted memory id=%s user_id=%s", memory_id, user.id)

    async def search(
        self,
        *,
        user: User,
        query: str,
        limit: int = 5,
    ) -> list[MemorySearchResult]:
        """Semantic search over the calling user's memories.

        Returns up to ``limit`` results ordered by descending cosine
        similarity, each with a ``score`` in [0, 1].
        """
        query_embedding = await self._embedding_provider.embed(query)
        hits = await self._vector_store.search(
            user_id=user.id,
            query_embedding=query_embedding,
            limit=limit,
        )
        if not hits:
            return []

        # Fetch the matching PG records in hit order.
        hit_ids = [h[0] for h in hits]
        score_map: dict[str, float] = {mid: score for mid, score in hits}

        stmt: Select[tuple[Memory]] = select(Memory).where(
            Memory.id.in_(hit_ids),
            Memory.user_id == user.id,  # redundant safety check
        )
        rows = list((await self._session.execute(stmt)).scalars().all())
        row_map: dict[str, Memory] = {m.id: m for m in rows}

        results: list[MemorySearchResult] = []
        for mid in hit_ids:
            mem = row_map.get(mid)
            if mem is None:
                continue  # stale Chroma entry; skip gracefully
            results.append(
                MemorySearchResult(
                    memory=MemoryRead.model_validate(mem),
                    score=score_map[mid],
                )
            )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_owned(self, *, memory_id: str, user: User) -> Memory:
        """Fetch a memory ensuring it belongs to ``user``."""
        stmt: Select[tuple[Memory]] = select(Memory).where(
            Memory.id == memory_id,
            Memory.user_id == user.id,
        )
        memory = (await self._session.execute(stmt)).scalar_one_or_none()
        if memory is None:
            raise NotFoundError(
                "Memory not found",
                details={"memory_id": memory_id},
            )
        return memory
