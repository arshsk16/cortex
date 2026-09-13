"""Shared pytest fixtures for Cortex tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from cortex.core.config import Settings
from cortex.core.security import hash_password
from cortex.db.models.document import Document, DocumentStatus
from cortex.db.models.user import User, UserRole
from cortex.db.session import Database
from cortex.embeddings.base import EmbeddingProvider
from cortex.llm.base import LLMProvider
from cortex.main import create_app
from cortex.vectorstore.base import VectorStore

MINIMAL_PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"
EMBEDDING_DIMENSION = 384


@pytest.fixture
def test_settings(tmp_path) -> Settings:
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
        document_storage_path=str(tmp_path / "documents"),
        document_max_file_size_bytes=26_214_400,
        document_allowed_mime_type="application/pdf",
        chroma_persist_directory=str(tmp_path / "chroma"),
        chroma_collection_name="test_document_chunks",
        gemini_api_key="test-key",
        gemini_model_name="gemini-2.0-flash",
    )


@pytest.fixture
def mock_embedding_provider() -> EmbeddingProvider:
    """Fast mock embedding provider for API and service tests."""
    provider = MagicMock(spec=EmbeddingProvider)
    provider.embed = AsyncMock(
        return_value=[0.1] * EMBEDDING_DIMENSION,
    )
    provider.embed_batch = AsyncMock(
        side_effect=lambda texts: [[0.1] * EMBEDDING_DIMENSION for _ in texts],
    )
    return provider


@pytest.fixture
def mock_vector_store() -> VectorStore:
    """Fast mock vector store for API and service tests."""
    store = MagicMock(spec=VectorStore)
    store.add_chunks = AsyncMock()
    store.delete_document = AsyncMock()
    store.similarity_search = AsyncMock(return_value=[])
    return store


@pytest.fixture
def mock_llm_provider() -> LLMProvider:
    """Fast mock LLM provider — returns a canned answer without network calls."""
    provider = MagicMock(spec=LLMProvider)
    provider.generate = AsyncMock(return_value="This is a mocked LLM answer.")
    provider.generate_stream = AsyncMock(return_value=iter([]))
    return provider


@pytest.fixture
def app(
    test_settings: Settings,
    mock_embedding_provider: EmbeddingProvider,
    mock_vector_store: VectorStore,
    mock_llm_provider: LLMProvider,
) -> FastAPI:
    """Application instance with settings and DI state attached for tests."""
    application = create_app(settings=test_settings)
    application.state.database = Database(test_settings)
    application.state.embedding_provider = mock_embedding_provider
    application.state.vector_store = mock_vector_store
    application.state.llm_provider = mock_llm_provider
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


@pytest.fixture
def other_user() -> User:
    """Second user for ownership isolation tests."""
    now = datetime.now(UTC)
    return User(
        id=str(uuid4()),
        email="bob@example.com",
        username="bob",
        full_name="Bob Example",
        hashed_password=hash_password("Password1"),
        role=UserRole.USER,
        is_active=True,
        is_verified=False,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def sample_document(sample_user: User) -> Document:
    """In-memory Document owned by ``sample_user``."""
    now = datetime.now(UTC)
    return Document(
        id=str(uuid4()),
        user_id=sample_user.id,
        title="Sample Report",
        original_filename="report.pdf",
        storage_filename=f"{uuid4()}.pdf",
        storage_path="storage/documents/sample.pdf",
        mime_type="application/pdf",
        file_size=len(MINIMAL_PDF_BYTES),
        status=DocumentStatus.UPLOADED,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def minimal_pdf_bytes() -> bytes:
    """Minimal bytes that satisfy PDF magic-byte validation."""
    return MINIMAL_PDF_BYTES
