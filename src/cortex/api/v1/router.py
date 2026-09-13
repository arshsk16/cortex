"""API v1 router — aggregates all v1 endpoint routers."""

from fastapi import APIRouter

from cortex.api.v1.endpoints import auth, health

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(auth.router)
