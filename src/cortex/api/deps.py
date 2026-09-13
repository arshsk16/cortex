"""FastAPI dependency injection wiring."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.agent.registry import ToolRegistry
from cortex.agent.service import AgentService
from cortex.agent.tools.calculator import CalculatorTool
from cortex.agent.tools.document_list import DocumentListTool
from cortex.agent.tools.rag_search import RAGSearchTool
from cortex.core.config import Settings, get_settings
from cortex.core.exceptions import ForbiddenError, UnauthorizedError
from cortex.core.security import decode_access_token
from cortex.db.models.user import User
from cortex.db.session import Database, get_session
from cortex.embeddings.base import EmbeddingProvider
from cortex.llm.base import LLMProvider
from cortex.retrieval.base import Retriever
from cortex.retrieval.semantic import SemanticRetriever
from cortex.services.auth import AuthService
from cortex.services.chunking import ChunkingService
from cortex.services.cleaning import CleaningService
from cortex.services.conversation import ConversationService
from cortex.services.document import DocumentService
from cortex.services.health import HealthService
from cortex.services.ingestion import IngestionService
from cortex.services.memory import MemoryService
from cortex.services.parser import ParserService
from cortex.services.prompt_builder import PromptBuilder
from cortex.services.rag import RAGService
from cortex.services.storage import StorageService
from cortex.services.user import UserService
from cortex.state_store.base import StateStore
from cortex.vectorstore.base import VectorStore
from cortex.vectorstore.memory_store import MemoryVectorStore

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


def get_llm_provider(request: Request) -> LLMProvider:
    """Resolve the application-scoped LLM provider from app state."""
    return request.app.state.llm_provider  # type: ignore[no-any-return]


def get_state_store(request: Request) -> StateStore:
    """Resolve the application-scoped StateStore from app state."""
    return request.app.state.state_store  # type: ignore[no-any-return]


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


def get_retriever(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    embedding_provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
) -> Retriever:
    """Construct a request-scoped SemanticRetriever."""
    return SemanticRetriever(
        session=session,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )


def get_prompt_builder() -> PromptBuilder:
    """Construct a stateless PromptBuilder."""
    return PromptBuilder()


def get_rag_service(
    retriever: Annotated[Retriever, Depends(get_retriever)],
    llm_provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    prompt_builder: Annotated[PromptBuilder, Depends(get_prompt_builder)],
) -> RAGService:
    """Stateless RAGService — used by the Phase 6 /rag/query endpoint."""
    return RAGService(
        retriever=retriever,
        llm_provider=llm_provider,
        prompt_builder=prompt_builder,
    )


def get_conversation_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> ConversationService:
    """Construct a request-scoped ConversationService."""
    return ConversationService(session=session)


def get_conversation_rag_service(
    retriever: Annotated[Retriever, Depends(get_retriever)],
    llm_provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    prompt_builder: Annotated[PromptBuilder, Depends(get_prompt_builder)],
    conv_service: Annotated[ConversationService, Depends(get_conversation_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RAGService:
    """Conversation-aware RAGService — used by the chat endpoint."""
    return RAGService(
        retriever=retriever,
        llm_provider=llm_provider,
        prompt_builder=prompt_builder,
        conversation_service=conv_service,
        conversation_history_limit=settings.conversation_history_limit,
    )


def get_agent_service(
    retriever: Annotated[Retriever, Depends(get_retriever)],
    llm_provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    prompt_builder: Annotated[PromptBuilder, Depends(get_prompt_builder)],
    document_service: Annotated[DocumentService, Depends(get_document_service)],
    conv_service: Annotated[ConversationService, Depends(get_conversation_service)],
    state_store: Annotated[StateStore, Depends(get_state_store)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AgentService:
    """Construct a request-scoped AgentService with all three tools registered."""
    registry = ToolRegistry()
    registry.register(RAGSearchTool(retriever))
    registry.register(DocumentListTool(document_service))
    registry.register(CalculatorTool())
    return AgentService(
        llm_provider=llm_provider,
        tool_registry=registry,
        prompt_builder=prompt_builder,
        conversation_service=conv_service,
        state_store=state_store,
        max_tool_calls=settings.agent_max_tool_calls,
        conversation_history_limit=settings.conversation_history_limit,
        state_ttl_seconds=settings.agent_state_ttl_seconds,
    )




def get_memory_vector_store(request: Request) -> MemoryVectorStore:
    """Resolve the application-scoped MemoryVectorStore from app state."""
    return request.app.state.memory_vector_store  # type: ignore[no-any-return]


def get_memory_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    embedding_provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    memory_vector_store: Annotated[MemoryVectorStore, Depends(get_memory_vector_store)],
) -> MemoryService:
    """Construct a request-scoped MemoryService."""
    return MemoryService(
        session=session,
        embedding_provider=embedding_provider,
        memory_vector_store=memory_vector_store,
    )

SettingsDep = Annotated[Settings, Depends(get_settings)]
DatabaseDep = Annotated[Database, Depends(get_database)]
SessionDep = Annotated[AsyncSession, Depends(get_db_session)]
HealthServiceDep = Annotated[HealthService, Depends(get_health_service)]
UserServiceDep = Annotated[UserService, Depends(get_user_service)]
AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
StorageServiceDep = Annotated[StorageService, Depends(get_storage_service)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]
RetrieverDep = Annotated[Retriever, Depends(get_retriever)]
LLMProviderDep = Annotated[LLMProvider, Depends(get_llm_provider)]
RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
ConversationServiceDep = Annotated[
    ConversationService, Depends(get_conversation_service)
]
ConversationRAGServiceDep = Annotated[
    RAGService, Depends(get_conversation_rag_service)
]
AgentServiceDep = Annotated[AgentService, Depends(get_agent_service)]
StateStoreDep = Annotated[StateStore, Depends(get_state_store)]
MemoryServiceDep = Annotated[MemoryService, Depends(get_memory_service)]
MemoryVectorStoreDep = Annotated[MemoryVectorStore, Depends(get_memory_vector_store)]
CurrentUserDep = Annotated[User, Depends(get_current_user)]
CurrentActiveUserDep = Annotated[User, Depends(get_current_active_user)]






