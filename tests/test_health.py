"""Health endpoint and service tests."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from cortex.api.deps import get_health_service
from cortex.core.config import Settings
from cortex.schemas.health import ComponentHealth, HealthResponse
from cortex.services.health import HealthService


@pytest.mark.asyncio
async def test_root_redirects_to_service_info(
    client: AsyncClient,
    test_settings: Settings,
) -> None:
    """The root path returns basic service metadata."""
    response = await client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == test_settings.app_name
    assert body["version"] == test_settings.app_version
    assert body["docs"] == "/docs"


@pytest.mark.asyncio
async def test_health_endpoint_healthy(app: FastAPI, client: AsyncClient) -> None:
    """Health endpoint returns 200 when all components are healthy."""

    async def _override() -> Any:
        service = AsyncMock(spec=HealthService)
        service.check = AsyncMock(
            return_value=HealthResponse(
                status="healthy",
                service="Cortex",
                version="0.1.0",
                environment="development",
                timestamp=datetime.now(UTC),
                components=[
                    ComponentHealth(
                        name="database",
                        status="healthy",
                        latency_ms=1.23,
                    )
                ],
            )
        )
        return service

    app.dependency_overrides[get_health_service] = _override
    try:
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "healthy"
        assert payload["components"][0]["name"] == "database"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_health_endpoint_unhealthy(app: FastAPI, client: AsyncClient) -> None:
    """Health endpoint returns 503 when a critical component is down."""

    async def _override() -> Any:
        service = AsyncMock(spec=HealthService)
        service.check = AsyncMock(
            return_value=HealthResponse(
                status="unhealthy",
                service="Cortex",
                version="0.1.0",
                environment="development",
                timestamp=datetime.now(UTC),
                components=[
                    ComponentHealth(
                        name="database",
                        status="unhealthy",
                        detail="Connectivity probe failed",
                    )
                ],
            )
        )
        return service

    app.dependency_overrides[get_health_service] = _override
    try:
        response = await client.get("/api/v1/health")
        assert response.status_code == 503
        assert response.json()["status"] == "unhealthy"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_openapi_docs_available(client: AsyncClient) -> None:
    """Swagger OpenAPI schema is published at /openapi.json."""
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "Cortex"
    assert "/api/v1/health" in schema["paths"]
