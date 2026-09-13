"""Core package — configuration, logging, and shared exceptions."""

from cortex.core.config import Settings, get_settings
from cortex.core.exceptions import (
    BadRequestError,
    ConflictError,
    CortexError,
    DocumentProcessingError,
    ForbiddenError,
    NotFoundError,
    ServiceUnavailableError,
    UnauthorizedError,
)

__all__ = [
    "BadRequestError",
    "ConflictError",
    "CortexError",
    "DocumentProcessingError",
    "ForbiddenError",
    "NotFoundError",
    "ServiceUnavailableError",
    "Settings",
    "UnauthorizedError",
    "get_settings",
]
