"""Health-check service — probes application and infrastructure readiness."""

from __future__ import annotations

import logging
import time
from typing import Literal

from cortex.core.config import Settings
from cortex.db.session import Database
from cortex.schemas.health import ComponentHealth, HealthResponse
from cortex.utils.datetime import utc_now

logger = logging.getLogger(__name__)

OverallStatus = Literal["healthy", "degraded", "unhealthy"]


class HealthService:
    """Evaluates the readiness of Cortex and its critical dependencies.

    Constructed once per application lifetime and injected into route handlers
    via FastAPI's dependency injection system.
    """

    def __init__(self, settings: Settings, database: Database) -> None:
        self._settings = settings
        self._database = database

    async def check(self) -> HealthResponse:
        """Run all health probes and return an aggregate status payload."""
        components = [await self._check_database()]

        if all(c.status == "healthy" for c in components):
            overall: OverallStatus = "healthy"
        elif any(c.status == "healthy" for c in components):
            overall = "degraded"
        else:
            overall = "unhealthy"

        return HealthResponse(
            status=overall,
            service=self._settings.app_name,
            version=self._settings.app_version,
            environment=self._settings.app_env,
            timestamp=utc_now(),
            components=components,
        )

    async def _check_database(self) -> ComponentHealth:
        """Probe PostgreSQL connectivity and measure round-trip latency."""
        started = time.perf_counter()
        try:
            healthy = await self._database.health_check()
            latency_ms = (time.perf_counter() - started) * 1000
            if healthy:
                return ComponentHealth(
                    name="database",
                    status="healthy",
                    latency_ms=round(latency_ms, 2),
                )
            logger.warning("Database health probe returned unhealthy")
            return ComponentHealth(
                name="database",
                status="unhealthy",
                latency_ms=round(latency_ms, 2),
                detail="Connectivity probe failed",
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            logger.exception("Database health probe raised an exception")
            return ComponentHealth(
                name="database",
                status="unhealthy",
                latency_ms=round(latency_ms, 2),
                detail=str(exc),
            )
