"""Top-level API router — mounts versioned sub-routers."""

from fastapi import APIRouter

from cortex.api.v1.router import api_router as v1_router

# Version prefix is applied in ``create_app`` from Settings.api_v1_prefix so
# configuration stays centralized and import-time env loading is avoided.
api_router = APIRouter()
api_router.include_router(v1_router)
