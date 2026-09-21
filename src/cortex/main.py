"""FastAPI application factory and ASGI entrypoint."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cortex import __version__
from cortex.api.router import api_router
from cortex.core.config import Settings, get_settings
from cortex.core.exceptions import register_exception_handlers
from cortex.core.logging import configure_logging
from cortex.core.middleware import RequestIDMiddleware
from cortex.core.rate_limit import RateLimitMiddleware
from cortex.db.session import Database
from cortex.embeddings.factory import create_embedding_provider
from cortex.llm.factory import create_llm_provider
from cortex.state_store.null import NullStateStore
from cortex.vectorstore.factory import create_vector_store
from cortex.vectorstore.memory_store import MemoryVectorStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown resources."""
    settings: Settings = app.state.settings
    database = Database(settings)
    embedding_provider = create_embedding_provider(settings)
    vector_store = create_vector_store(settings)
    llm_provider = create_llm_provider(settings)
    app.state.database = database
    app.state.embedding_provider = embedding_provider
    app.state.vector_store = vector_store
    app.state.llm_provider = llm_provider

    # Memory vector store -- dedicated Chroma collection for memory embeddings
    import chromadb as _chromadb
    _chroma_client = _chromadb.PersistentClient(path=settings.chroma_persist_directory)
    _memory_collection = _chroma_client.get_or_create_collection(
        name=settings.chroma_memory_collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    app.state.memory_vector_store = MemoryVectorStore(_memory_collection)

    # Redis state store — optional; gracefully falls back to NullStateStore
    redis_client = None
    if settings.redis_url:
        try:
            from redis.asyncio import Redis

            from cortex.state_store.redis_store import RedisStateStore

            redis_client = Redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            app.state.state_store = RedisStateStore(redis_client)
            logger.info("StateStore: Redis connected at %s", settings.redis_url)
        except Exception:
            logger.warning(
                "StateStore: failed to connect to Redis — using NullStateStore",
                exc_info=True,
            )
            app.state.state_store = NullStateStore()
    else:
        app.state.state_store = NullStateStore()

    logger.info(
        "Cortex started (env=%s, version=%s, embedding_model=%s)",
        settings.app_env,
        settings.app_version,
        settings.embedding_model_name,
    )
    try:
        yield
    finally:
        await database.dispose()
        if redis_client is not None:
            try:
                await redis_client.aclose()
                logger.info("StateStore: Redis connection closed")
            except Exception:
                logger.warning(
                    "StateStore: error closing Redis connection", exc_info=True
                )
        logger.info("Cortex shut down cleanly")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application.

    Parameters
    ----------
    settings:
        Optional settings override (useful in tests). When omitted, settings
        are loaded from the environment / ``.env`` file.
    """
    resolved = settings or get_settings()
    configure_logging(resolved)

    application = FastAPI(
        title=resolved.app_name,
        version=resolved.app_version or __version__,
        description=(
            "Cortex is an enterprise-grade AI platform. "
            "This API exposes health, management, and (future) AI capabilities."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        debug=resolved.debug,
    )
    application.state.settings = resolved

    application.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.cors_origins,
        allow_credentials=resolved.cors_allow_credentials,
        allow_methods=resolved.cors_allow_methods,
        allow_headers=resolved.cors_allow_headers,
    )

    # Request-ID middleware — must be added before rate limiting so the ID is
    # available in rate-limit log warnings.
    application.add_middleware(RequestIDMiddleware)

    # Auth-scoped tighter rate limit (brute-force protection on login/register).
    auth_prefix = f"{resolved.api_v1_prefix}/auth"
    application.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=resolved.rate_limit_auth_requests_per_minute,
        path_prefix=auth_prefix,
    )

    # Global rate limit for all remaining routes.
    application.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=resolved.rate_limit_requests_per_minute,
    )

    register_exception_handlers(application)
    application.include_router(api_router, prefix=resolved.api_v1_prefix)

    @application.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "service": resolved.app_name,
            "version": resolved.app_version,
            "docs": "/docs",
        }

    return application


def run() -> None:
    """CLI entrypoint used by the ``cortex`` console script."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "cortex.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug and not settings.is_production,
        log_level=settings.log_level.lower(),
    )


# Lazily constructed module-level app for ASGI servers (uvicorn/gunicorn).
# Tests should prefer ``create_app(settings=...)`` to avoid env coupling.
app = create_app()


if __name__ == "__main__":
    run()


