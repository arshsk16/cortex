"""Pydantic schemas for document chunks."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentChunkRead(BaseModel):
    """Public representation of a stored document chunk."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    chunk_index: int = Field(..., ge=0)
    text: str
    token_count: int = Field(..., ge=0)
    created_at: datetime
