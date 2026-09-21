"""HTTP middleware utilities for Cortex.

RequestIDMiddleware
-------------------
Generates a unique ``X-Request-ID`` (UUID4) for every incoming request,
attaches it to the response header, and injects it into the Python logging
system via a thread-local ``LogRecordFactory`` so all log lines emitted
during the request carry ``request_id``.

The ``JsonFormatter`` in ``core/logging.py`` already reads and serialises
the ``request_id`` attribute when present.
"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

logger = logging.getLogger(__name__)

# Context variable that holds the current request ID within an async task.
_request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    """Return the request-ID for the current async context (empty string if unset)."""
    return _request_id_var.get()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique X-Request-ID to every request/response.

    The ID is:
    * Generated as a UUID4 string if the client does not provide one.
    * Accepted from ``X-Request-ID`` header if the client supplies one
      (useful for distributed tracing correlation).
    * Stored in a ``ContextVar`` so downstream log records can access it.
    * Returned in the ``X-Request-ID`` response header.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        # Install a custom LogRecord factory that injects request_id.
        _original_factory = logging.getLogRecordFactory()

        def _record_factory(*args: object, **kwargs: object) -> logging.LogRecord:
            record = _original_factory(*args, **kwargs)
            record.request_id = get_request_id()  # type: ignore[attr-defined]
            return record

        logging.setLogRecordFactory(_record_factory)

    async def dispatch(self, request: Request, call_next: object) -> Response:
        """Generate / propagate request ID and store in context."""
        # Accept client-supplied ID (e.g. from upstream proxy) or generate one.
        request_id = (
            request.headers.get("X-Request-ID") or uuid.uuid4().hex
        )
        token = _request_id_var.set(request_id)
        try:
            response: Response = await call_next(request)  # type: ignore[operator]
        finally:
            _request_id_var.reset(token)

        response.headers["X-Request-ID"] = request_id
        return response
