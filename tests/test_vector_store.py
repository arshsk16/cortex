"""Unit tests for ChromaVectorStore."""

from __future__ import annotations

from uuid import uuid4

import pytest

from cortex.core.config import Settings
from cortex.vectorstore.chroma import ChromaVectorStore
from cortex.vectorstore.models import ChunkVectorRecord


@pytest.fixture
def vector_store(test_settings: Settings) -> ChromaVectorStore:
    """Chroma vector store backed by a temporary directory."""
    return ChromaVectorStore(test_settings)


def _record(
    *,
    chunk_id: str | None = None,
    document_id: str | None = None,
    chunk_index: int = 0,
) -> ChunkVectorRecord:
    return ChunkVectorRecord(
        chunk_id=chunk_id or str(uuid4()),
        document_id=document_id or str(uuid4()),
        chunk_index=chunk_index,
        embedding=[0.1, 0.2, 0.3, 0.4],
    )


@pytest.mark.asyncio
async def test_add_and_search_chunks(vector_store: ChromaVectorStore) -> None:
    """Vectors can be added and retrieved via similarity search."""
    document_id = str(uuid4())
    chunk_id = str(uuid4())
    record = _record(chunk_id=chunk_id, document_id=document_id, chunk_index=0)

    await vector_store.add_chunks([record])
    results = await vector_store.similarity_search(
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        limit=1,
        document_id=document_id,
    )

    assert len(results) == 1
    assert results[0].chunk_id == chunk_id
    assert results[0].document_id == document_id
    assert results[0].chunk_index == 0
    assert results[0].score > 0.9


@pytest.mark.asyncio
async def test_delete_document_removes_vectors(vector_store: ChromaVectorStore) -> None:
    """Deleting a document removes all associated vectors."""
    document_id = str(uuid4())
    await vector_store.add_chunks(
        [
            _record(document_id=document_id, chunk_index=0),
            _record(document_id=document_id, chunk_index=1),
        ]
    )

    await vector_store.delete_document(document_id)
    results = await vector_store.similarity_search(
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        limit=5,
        document_id=document_id,
    )

    assert results == []


@pytest.mark.asyncio
async def test_add_chunks_noop_for_empty_list(vector_store: ChromaVectorStore) -> None:
    """Adding an empty chunk list is a no-op."""
    await vector_store.add_chunks([])
