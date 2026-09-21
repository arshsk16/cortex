"""Health-check HTTP endpoints.

Three routes
------------
* ``GET /health``       — full check (DB probe); backward-compatible.
* ``GET /health/live``  — liveness: always 200 if the process is running.
* ``GET /health/ready`` — readiness: 200 only if all dependencies are healthy.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from cortex.api.deps import HealthServiceDep, SettingsDep
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


@router.get(
    "/health/live",
    summary="Liveness probe",
    description=(
        "Lightweight liveness probe: returns 200 whenever the process is "
        "alive and the event loop is responsive. Does NOT check dependencies. "
        "Suitable for Kubernetes/Docker liveness probes."
    ),
    responses={200: {"description": "Process is alive"}},
    include_in_schema=True,
)
async def liveness(settings: SettingsDep) -> JSONResponse:
    """Return 200 immediately — process is alive."""
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ok",
            "service": settings.app_name,
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    summary="Readiness probe",
    description=(
        "Readiness probe: runs all dependency health checks and returns "
        "200 only when every dependency is healthy. Returns 503 when any "
        "critical dependency is unhealthy. "
        "Suitable for Kubernetes/Docker readiness probes."
    ),
    responses={
        200: {"description": "All dependencies healthy"},
        503: {"description": "One or more dependencies are unhealthy"},
    },
)
async def readiness(health_service: HealthServiceDep) -> JSONResponse:
    """Return 200 only when all dependencies are healthy."""
    result = await health_service.check()
    status_code = (
        status.HTTP_200_OK
        if result.status == "healthy"
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return JSONResponse(
        status_code=status_code,
        content=result.model_dump(mode="json"),
    )
