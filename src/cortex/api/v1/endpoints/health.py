"""Health-check HTTP endpoints."""

from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from cortex.api.deps import HealthServiceDep
from cortex.schemas.health import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description=(
        "Returns the overall health of the Cortex API and its critical "
        "dependencies (currently PostgreSQL). Use this endpoint for "
        "load-balancer readiness and liveness probes."
    ),
    responses={
        200: {"description": "Service is healthy or degraded but reachable"},
        503: {"description": "Service or a critical dependency is unhealthy"},
    },
)
async def health_check(health_service: HealthServiceDep) -> JSONResponse:
    """Probe application and dependency health."""
    result = await health_service.check()
    status_code = (
        status.HTTP_200_OK
        if result.status != "unhealthy"
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return JSONResponse(
        status_code=status_code,
        content=result.model_dump(mode="json"),
    )
