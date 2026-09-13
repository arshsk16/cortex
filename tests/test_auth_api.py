"""API tests for authentication endpoints."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from cortex.api.deps import get_auth_service, get_current_active_user
from cortex.core.exceptions import ConflictError, UnauthorizedError
from cortex.core.security import create_access_token
from cortex.db.models.user import User
from cortex.schemas.auth import AuthResponse
from cortex.schemas.user import UserRead
from cortex.services.auth import AuthService


@pytest.mark.asyncio
async def test_register_endpoint_success(
    app: FastAPI,
    client: AsyncClient,
    sample_user: User,
    test_settings: Any,
) -> None:
    """POST /api/v1/register returns 201 with token and user."""

    async def _override() -> AuthService:
        service = AsyncMock(spec=AuthService)
        service.register = AsyncMock(
            return_value=AuthResponse(
                access_token=create_access_token(
                    subject=sample_user.id,
                    settings=test_settings,
                ),
                token_type="bearer",
                expires_in=3600,
                user=UserRead.model_validate(sample_user),
            )
        )
        return service

    app.dependency_overrides[get_auth_service] = _override
    try:
        response = await client.post(
            "/api/v1/register",
            json={
                "email": "alice@example.com",
                "username": "alice",
                "full_name": "Alice Example",
                "password": "Password1",
            },
        )
        assert response.status_code == 201
        body = response.json()
        assert "access_token" in body
        assert body["user"]["email"] == "alice@example.com"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_register_endpoint_conflict(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    """POST /api/v1/register maps ConflictError to HTTP 409."""

    async def _override() -> AuthService:
        service = AsyncMock(spec=AuthService)
        service.register = AsyncMock(
            side_effect=ConflictError(
                "A user with this email or username already exists",
                details={"email": "alice@example.com"},
            )
        )
        return service

    app.dependency_overrides[get_auth_service] = _override
    try:
        response = await client.post(
            "/api/v1/register",
            json={
                "email": "alice@example.com",
                "username": "alice",
                "full_name": "Alice Example",
                "password": "Password1",
            },
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "conflict"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_register_validation_error(client: AsyncClient) -> None:
    """Weak passwords are rejected with HTTP 422."""
    response = await client.post(
        "/api/v1/register",
        json={
            "email": "alice@example.com",
            "username": "alice",
            "full_name": "Alice Example",
            "password": "short",
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


@pytest.mark.asyncio
async def test_login_endpoint_success(
    app: FastAPI,
    client: AsyncClient,
    sample_user: User,
    test_settings: Any,
) -> None:
    """POST /api/v1/login returns 200 with a token on success."""

    async def _override() -> AuthService:
        service = AsyncMock(spec=AuthService)
        service.login = AsyncMock(
            return_value=AuthResponse(
                access_token=create_access_token(
                    subject=sample_user.id,
                    settings=test_settings,
                ),
                token_type="bearer",
                expires_in=3600,
                user=UserRead.model_validate(sample_user),
            )
        )
        return service

    app.dependency_overrides[get_auth_service] = _override
    try:
        response = await client.post(
            "/api/v1/login",
            json={"email": "alice@example.com", "password": "Password1"},
        )
        assert response.status_code == 200
        assert response.json()["token_type"] == "bearer"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_login_endpoint_unauthorized(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    """POST /api/v1/login maps UnauthorizedError to HTTP 401."""

    async def _override() -> AuthService:
        service = AsyncMock(spec=AuthService)
        service.login = AsyncMock(
            side_effect=UnauthorizedError("Incorrect email or password")
        )
        return service

    app.dependency_overrides[get_auth_service] = _override
    try:
        response = await client.post(
            "/api/v1/login",
            json={"email": "alice@example.com", "password": "WrongPass1"},
        )
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthorized"
        assert response.headers.get("www-authenticate") == "Bearer"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_me_endpoint_requires_auth(client: AsyncClient) -> None:
    """GET /api/v1/me without a bearer token returns 401."""
    response = await client.get("/api/v1/me")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


@pytest.mark.asyncio
async def test_me_endpoint_success(
    app: FastAPI,
    client: AsyncClient,
    sample_user: User,
) -> None:
    """GET /api/v1/me returns the authenticated user profile."""

    async def _override() -> User:
        return sample_user

    app.dependency_overrides[get_current_active_user] = _override
    try:
        response = await client.get("/api/v1/me")
        assert response.status_code == 200
        body = response.json()
        assert body["email"] == sample_user.email
        assert body["username"] == sample_user.username
        assert "hashed_password" not in body
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_openapi_includes_auth_routes(client: AsyncClient) -> None:
    """OpenAPI schema documents the auth endpoints."""
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/v1/register" in paths
    assert "/api/v1/login" in paths
    assert "/api/v1/me" in paths
