"""Pydantic schemas for health-check responses."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ComponentHealth(BaseModel):
    """Health status of a single infrastructure dependency."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Component identifier")
    status: Literal["healthy", "unhealthy"] = Field(
        ...,
        description="Component health state",
    )
    latency_ms: float | None = Field(
        default=None,
        description="Probe latency in milliseconds, when measured",
    )
    detail: str | None = Field(
        default=None,
        description="Optional human-readable detail",
    )


class HealthResponse(BaseModel):
    """Aggregate health payload returned by the health-check endpoint."""

    model_config = ConfigDict(frozen=True)

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="Overall service health",
    )
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    environment: str = Field(..., description="Deployment environment")
    timestamp: datetime = Field(..., description="UTC timestamp of the check")
    components: list[ComponentHealth] = Field(
        default_factory=list,
        description="Per-dependency health results",
    )
