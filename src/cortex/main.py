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
from cortex.db.session import Database
from cortex.embeddings.factory import create_embedding_provider
from cortex.llm.factory import create_llm_provider
from cortex.vectorstore.factory import create_vector_store

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
