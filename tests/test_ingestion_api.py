"""API tests for document ingestion endpoints."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from cortex.api.deps import get_current_active_user, get_ingestion_service
from cortex.core.exceptions import DocumentProcessingError, NotFoundError
from cortex.db.models.document import DocumentStatus
from cortex.db.models.user import User
from cortex.schemas.chunk import DocumentChunkRead
from cortex.schemas.document import DocumentRead
from cortex.schemas.ingestion import DocumentProcessResponse
from cortex.services.ingestion import IngestionService


@pytest.mark.asyncio
async def test_process_document_success(
    app: FastAPI,
    client: AsyncClient,
    sample_user: User,
    sample_document: Any,
) -> None:
    """POST /api/v1/documents/{id}/process returns processed chunks."""

    async def _override_user() -> User:
        return sample_user

    async def _override_service() -> IngestionService:
        service = AsyncMock(spec=IngestionService)
        document = DocumentRead.model_validate(sample_document)
        service.process = AsyncMock(
            return_value=DocumentProcessResponse(
                document=document.model_copy(update={"status": DocumentStatus.READY}),
                status=DocumentStatus.READY,
                chunk_count=1,
                chunks=[
                    DocumentChunkRead(
                        id="chunk-id",
                        document_id=sample_document.id,
                        chunk_index=0,
                        text="Processed chunk text",
                        token_count=3,
                        created_at=sample_document.created_at,
                    )
                ],
            )
        )
        return service

    app.dependency_overrides[get_current_active_user] = _override_user
    app.dependency_overrides[get_ingestion_service] = _override_service
    try:
        response = await client.post(f"/api/v1/documents/{sample_document.id}/process")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == DocumentStatus.READY.value
        assert body["chunk_count"] == 1
        assert body["chunks"][0]["text"] == "Processed chunk text"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_process_document_not_found(
    app: FastAPI,
    client: AsyncClient,
    sample_user: User,
    sample_document: Any,
) -> None:
    """Missing documents return HTTP 404."""

    async def _override_user() -> User:
        return sample_user

    async def _override_service() -> IngestionService:
        service = AsyncMock(spec=IngestionService)
        service.process = AsyncMock(
            side_effect=NotFoundError(
                "Document not found",
                details={"document_id": sample_document.id},
            )
        )
        return service

    app.dependency_overrides[get_current_active_user] = _override_user
    app.dependency_overrides[get_ingestion_service] = _override_service
    try:
        response = await client.post(f"/api/v1/documents/{sample_document.id}/process")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_process_document_processing_error(
    app: FastAPI,
    client: AsyncClient,
    sample_user: User,
    sample_document: Any,
) -> None:
    """Processing failures return HTTP 422."""

    async def _override_user() -> User:
        return sample_user

    async def _override_service() -> IngestionService:
        service = AsyncMock(spec=IngestionService)
        service.process = AsyncMock(
            side_effect=DocumentProcessingError(
                "PDF document contains no extractable text",
                details={"page_count": 1},
            )
        )
        return service

    app.dependency_overrides[get_current_active_user] = _override_user
    app.dependency_overrides[get_ingestion_service] = _override_service
    try:
        response = await client.post(f"/api/v1/documents/{sample_document.id}/process")
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "document_processing_error"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_process_document_requires_authentication(
    client: AsyncClient,
    sample_document: Any,
) -> None:
    """Processing requires a bearer token."""
    response = await client.post(f"/api/v1/documents/{sample_document.id}/process")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_openapi_includes_process_route(client: AsyncClient) -> None:
    """OpenAPI schema documents the process endpoint."""
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    assert "/api/v1/documents/{document_id}/process" in response.json()["paths"]
