"""Domain and HTTP exception hierarchy for Cortex."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class CortexError(Exception):
    """Base exception for all Cortex domain errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "cortex_error",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class NotFoundError(CortexError):
    """Raised when a requested resource does not exist."""

    def __init__(
        self,
        message: str = "Resource not found",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="not_found", details=details)


class ConflictError(CortexError):
    """Raised when a request conflicts with current resource state."""

    def __init__(
        self,
        message: str = "Resource conflict",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="conflict", details=details)


class ServiceUnavailableError(CortexError):
    """Raised when a required dependency is unavailable."""

    def __init__(
        self,
        message: str = "Service unavailable",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="service_unavailable", details=details)


class UnauthorizedError(CortexError):
    """Raised when authentication fails or credentials are invalid."""

    def __init__(
        self,
        message: str = "Could not validate credentials",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="unauthorized", details=details)


class ForbiddenError(CortexError):
    """Raised when an authenticated user lacks permission for an action."""

    def __init__(
        self,
        message: str = "Not enough permissions",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="forbidden", details=details)


def _error_body(
    *,
    code: str,
    message: str,
    details: dict[str, Any] | list[Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return body


def register_exception_handlers(app: FastAPI) -> None:
    """Attach global exception handlers that produce a uniform error envelope."""

    @app.exception_handler(CortexError)
    async def cortex_error_handler(
        _request: Request,
        exc: CortexError,
    ) -> JSONResponse:
        status_map: dict[str, int] = {
            "not_found": status.HTTP_404_NOT_FOUND,
            "conflict": status.HTTP_409_CONFLICT,
            "service_unavailable": status.HTTP_503_SERVICE_UNAVAILABLE,
            "unauthorized": status.HTTP_401_UNAUTHORIZED,
            "forbidden": status.HTTP_403_FORBIDDEN,
        }
        headers: dict[str, str] | None = None
        if exc.code == "unauthorized":
            headers = {"WWW-Authenticate": "Bearer"}
        return JSONResponse(
            status_code=status_map.get(exc.code, status.HTTP_400_BAD_REQUEST),
            content=_error_body(
                code=exc.code,
                message=exc.message,
                details=exc.details or None,
            ),
            headers=headers,
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        _request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(
                code="http_error",
                message=str(exc.detail),
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=_error_body(
                code="validation_error",
                message="Request validation failed",
                details=exc.errors(),
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        _request: Request,
        exc: Exception,
    ) -> JSONResponse:
        import logging

        logging.getLogger("cortex").exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_body(
                code="internal_error",
                message="An unexpected error occurred",
            ),
        )
