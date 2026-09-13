"""Pydantic schemas for document ingestion responses."""

from __future__ import annotations

from pydantic import BaseModel, Field

from cortex.db.models.document import DocumentStatus
from cortex.schemas.chunk import DocumentChunkRead
from cortex.schemas.document import DocumentRead


class DocumentProcessResponse(BaseModel):
    """Response returned after a document ingestion run completes."""

    document: DocumentRead
    status: DocumentStatus
    chunk_count: int = Field(..., ge=0)
    chunks: list[DocumentChunkRead]
