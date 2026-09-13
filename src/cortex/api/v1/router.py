"""API v1 router — aggregates all v1 endpoint routers."""

from fastapi import APIRouter

from cortex.api.v1.endpoints import (
    auth,
    conversations,
    documents,
    health,
    rag,
    retrieval,
)

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(documents.router)
api_router.include_router(retrieval.router)
api_router.include_router(rag.router)
api_router.include_router(conversations.router)
