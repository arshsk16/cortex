"""API tests for document management endpoints."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from cortex.api.deps import get_current_active_user, get_document_service
from cortex.core.exceptions import BadRequestError, NotFoundError
from cortex.db.models.document import DocumentStatus
from cortex.db.models.user import User
from cortex.schemas.document import DocumentList, DocumentRead
from cortex.services.document import DocumentService
from tests.conftest import MINIMAL_PDF_BYTES


def _document_read(sample_document: Any) -> DocumentRead:
    return DocumentRead.model_validate(sample_document)


@pytest.mark.asyncio
async def test_upload_document_success(
    app: FastAPI,
    client: AsyncClient,
    sample_user: User,
    sample_document: Any,
) -> None:
    """POST /api/v1/documents/upload returns 201 for valid PDF uploads."""

    async def _override_user() -> User:
        return sample_user

    async def _override_service() -> DocumentService:
        service = AsyncMock(spec=DocumentService)
        service.upload = AsyncMock(return_value=_document_read(sample_document))
        return service

    app.dependency_overrides[get_current_active_user] = _override_user
    app.dependency_overrides[get_document_service] = _override_service
    try:
        response = await client.post(
            "/api/v1/documents/upload",
            data={"title": "Sample Report"},
            files={
                "file": (
                    "report.pdf",
                    MINIMAL_PDF_BYTES,
                    "application/pdf",
                )
            },
        )
        assert response.status_code == 201
        body = response.json()
        assert body["document"]["title"] == sample_document.title
        assert body["document"]["status"] == DocumentStatus.UPLOADED.value
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_upload_document_invalid_file(
    app: FastAPI,
    client: AsyncClient,
    sample_user: User,
) -> None:
    """Invalid uploads map to HTTP 400."""

    async def _override_user() -> User:
        return sample_user

    async def _override_service() -> DocumentService:
        service = AsyncMock(spec=DocumentService)
        service.upload = AsyncMock(
            side_effect=BadRequestError(
                "Only PDF files are allowed",
                details={"allowed_mime_type": "application/pdf"},
            )
        )
        return service

    app.dependency_overrides[get_current_active_user] = _override_user
    app.dependency_overrides[get_document_service] = _override_service
    try:
        response = await client.post(
            "/api/v1/documents/upload",
            files={
                "file": (
                    "notes.txt",
                    b"plain text",
                    "text/plain",
                )
            },
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "bad_request"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_list_documents_success(
    app: FastAPI,
    client: AsyncClient,
    sample_user: User,
    sample_document: Any,
) -> None:
    """GET /api/v1/documents returns owned documents."""

    async def _override_user() -> User:
        return sample_user

    async def _override_service() -> DocumentService:
        service = AsyncMock(spec=DocumentService)
        service.list = AsyncMock(
            return_value=DocumentList(
                items=[_document_read(sample_document)],
                total=1,
                skip=0,
                limit=50,
            )
        )
        return service

    app.dependency_overrides[get_current_active_user] = _override_user
    app.dependency_overrides[get_document_service] = _override_service
    try:
        response = await client.get("/api/v1/documents")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert len(body["items"]) == 1
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_document_success(
    app: FastAPI,
    client: AsyncClient,
    sample_user: User,
    sample_document: Any,
) -> None:
    """GET /api/v1/documents/{id} returns document metadata."""

    async def _override_user() -> User:
        return sample_user

    async def _override_service() -> DocumentService:
        service = AsyncMock(spec=DocumentService)
        service.get = AsyncMock(return_value=_document_read(sample_document))
        return service

    app.dependency_overrides[get_current_active_user] = _override_user
    app.dependency_overrides[get_document_service] = _override_service
    try:
        response = await client.get(f"/api/v1/documents/{sample_document.id}")
        assert response.status_code == 200
        assert response.json()["document"]["id"] == sample_document.id
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_document_not_found(
    app: FastAPI,
    client: AsyncClient,
    sample_user: User,
    sample_document: Any,
) -> None:
    """Missing or foreign documents return HTTP 404."""

    async def _override_user() -> User:
        return sample_user

    async def _override_service() -> DocumentService:
        service = AsyncMock(spec=DocumentService)
        service.get = AsyncMock(
            side_effect=NotFoundError(
                "Document not found",
                details={"document_id": sample_document.id},
            )
        )
        return service

    app.dependency_overrides[get_current_active_user] = _override_user
    app.dependency_overrides[get_document_service] = _override_service
    try:
        response = await client.get(f"/api/v1/documents/{sample_document.id}")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_delete_document_success(
    app: FastAPI,
    client: AsyncClient,
    sample_user: User,
    sample_document: Any,
) -> None:
    """DELETE /api/v1/documents/{id} returns 204."""

    async def _override_user() -> User:
        return sample_user

    async def _override_service() -> DocumentService:
        service = AsyncMock(spec=DocumentService)
        service.delete = AsyncMock(return_value=None)
        return service

    app.dependency_overrides[get_current_active_user] = _override_user
    app.dependency_overrides[get_document_service] = _override_service
    try:
        response = await client.delete(f"/api/v1/documents/{sample_document.id}")
        assert response.status_code == 204
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_documents_require_authentication(client: AsyncClient) -> None:
    """Document routes reject unauthenticated requests."""
    response = await client.get("/api/v1/documents")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_openapi_includes_document_routes(client: AsyncClient) -> None:
    """OpenAPI schema documents the document endpoints."""
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/v1/documents/upload" in paths
    assert "/api/v1/documents" in paths
    assert "/api/v1/documents/{document_id}" in paths
