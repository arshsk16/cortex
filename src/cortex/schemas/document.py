"""Pydantic schemas for document resources."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from cortex.db.models.document import DocumentStatus


class DocumentCreate(BaseModel):
    """Internal payload for persisting a new document record."""

    title: str = Field(..., min_length=1, max_length=255)
    original_filename: str = Field(..., min_length=1, max_length=512)
    mime_type: str = Field(..., min_length=1, max_length=127)
    file_size: int = Field(..., ge=1)


class DocumentRead(BaseModel):
    """Public document metadata returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    title: str
    original_filename: str
    storage_filename: str
    storage_path: str
    mime_type: str
    file_size: int
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime


class DocumentResponse(BaseModel):
    """Single-document API response envelope."""

    document: DocumentRead


class DocumentList(BaseModel):
    """Paginated list of documents owned by the authenticated user."""

    items: list[DocumentRead]
    total: int = Field(..., ge=0)
    skip: int = Field(..., ge=0)
    limit: int = Field(..., ge=1)
