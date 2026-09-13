"""Shared pytest fixtures for Cortex tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from cortex.core.config import Settings
from cortex.core.security import hash_password
from cortex.db.models.user import User, UserRole
from cortex.db.session import Database
from cortex.main import create_app


@pytest.fixture
def test_settings() -> Settings:
    """Settings suitable for unit tests (no live database required)."""
    return Settings(
        app_name="Cortex",
        app_version="0.1.0",
        app_env="development",
        debug=True,
        database_url="postgresql+asyncpg://cortex:cortex@localhost:5432/cortex_test",
        cors_origins=["http://localhost:3000"],
        log_level="WARNING",
        log_json=False,
        jwt_secret_key="test-secret-key-that-is-at-least-32-chars",
        jwt_algorithm="HS256",
        jwt_access_token_expire_minutes=60,
    )


@pytest.fixture
def app(test_settings: Settings) -> FastAPI:
    """Application instance with settings and a Database attached for DI.

    Lifespan is not relied upon here because the installed httpx ASGITransport
    does not expose a lifespan switch; attaching Database mirrors startup wiring.
    """
    application = create_app(settings=test_settings)
    application.state.database = Database(test_settings)
    return application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """HTTPX async client bound to the ASGI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_user() -> User:
    """In-memory User instance for service and auth dependency tests."""
    now = datetime.now(UTC)
    return User(
        id=str(uuid4()),
        email="alice@example.com",
        username="alice",
        full_name="Alice Example",
        hashed_password=hash_password("Password1"),
        role=UserRole.USER,
        is_active=True,
        is_verified=False,
        created_at=now,
        updated_at=now,
    )
