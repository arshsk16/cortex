"""FastAPI dependency injection wiring."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.core.config import Settings, get_settings
from cortex.core.exceptions import ForbiddenError, UnauthorizedError
from cortex.core.security import decode_access_token
from cortex.db.models.user import User
from cortex.db.session import Database, get_session
from cortex.embeddings.base import EmbeddingProvider
from cortex.services.auth import AuthService
from cortex.services.chunking import ChunkingService
from cortex.services.cleaning import CleaningService
from cortex.services.document import DocumentService
from cortex.services.health import HealthService
from cortex.services.ingestion import IngestionService
from cortex.services.parser import ParserService
from cortex.services.storage import StorageService
from cortex.services.user import UserService
from cortex.vectorstore.base import VectorStore

_bearer_scheme = HTTPBearer(auto_error=False)


def get_database(request: Request) -> Database:
    """Resolve the application-scoped Database from app state."""
    return request.app.state.database  # type: ignore[no-any-return]


def get_embedding_provider(request: Request) -> EmbeddingProvider:
    """Resolve the application-scoped embedding provider from app state."""
    return request.app.state.embedding_provider  # type: ignore[no-any-return]


def get_vector_store(request: Request) -> VectorStore:
    """Resolve the application-scoped vector store from app state."""
    return request.app.state.vector_store  # type: ignore[no-any-return]


async def get_db_session(
    database: Annotated[Database, Depends(get_database)],
) -> AsyncGenerator[AsyncSession, None]:
    """Yield a request-scoped SQLAlchemy async session."""
    async for session in get_session(database.session_factory):
        yield session


def get_health_service(
    settings: Annotated[Settings, Depends(get_settings)],
    database: Annotated[Database, Depends(get_database)],
) -> HealthService:
    """Construct a HealthService with its required collaborators."""
    return HealthService(settings=settings, database=database)


def get_user_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> UserService:
    """Construct a request-scoped UserService."""
    return UserService(session)


def get_auth_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    user_service: Annotated[UserService, Depends(get_user_service)],
) -> AuthService:
    """Construct a request-scoped AuthService."""
    return AuthService(session=session, settings=settings, user_service=user_service)


def get_storage_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> StorageService:
    """Construct a StorageService bound to configured storage settings."""
    return StorageService(settings)


def get_document_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    storage_service: Annotated[StorageService, Depends(get_storage_service)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
) -> DocumentService:
    """Construct a request-scoped DocumentService."""
    return DocumentService(
        session=session,
        settings=settings,
        storage_service=storage_service,
        vector_store=vector_store,
    )


def get_parser_service() -> ParserService:
    """Construct a ParserService."""
    return ParserService()


def get_cleaning_service() -> CleaningService:
    """Construct a CleaningService."""
    return CleaningService()


def get_chunking_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChunkingService:
    """Construct a ChunkingService bound to ingestion settings."""
    return ChunkingService(settings)


def get_ingestion_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    storage_service: Annotated[StorageService, Depends(get_storage_service)],
    parser_service: Annotated[ParserService, Depends(get_parser_service)],
    cleaning_service: Annotated[CleaningService, Depends(get_cleaning_service)],
    chunking_service: Annotated[ChunkingService, Depends(get_chunking_service)],
    embedding_provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
) -> IngestionService:
    """Construct a request-scoped IngestionService."""
    return IngestionService(
        session=session,
        storage_service=storage_service,
        parser_service=parser_service,
        cleaning_service=cleaning_service,
        chunking_service=chunking_service,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )


async def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(_bearer_scheme),
    ],
    settings: Annotated[Settings, Depends(get_settings)],
    user_service: Annotated[UserService, Depends(get_user_service)],
) -> User:
    """Resolve the authenticated user from a Bearer JWT access token."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise UnauthorizedError("Not authenticated")

    payload = decode_access_token(credentials.credentials, settings)
    user_id = payload["sub"]
    user = await user_service.get_by_id(user_id)
    if user is None:
        raise UnauthorizedError("Could not validate credentials")
    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Ensure the authenticated user account is active."""
    if not current_user.is_active:
        raise ForbiddenError("User account is inactive")
    return current_user


SettingsDep = Annotated[Settings, Depends(get_settings)]
DatabaseDep = Annotated[Database, Depends(get_database)]
SessionDep = Annotated[AsyncSession, Depends(get_db_session)]
HealthServiceDep = Annotated[HealthService, Depends(get_health_service)]
UserServiceDep = Annotated[UserService, Depends(get_user_service)]
AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
StorageServiceDep = Annotated[StorageService, Depends(get_storage_service)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]
CurrentUserDep = Annotated[User, Depends(get_current_user)]
CurrentActiveUserDep = Annotated[User, Depends(get_current_active_user)]
