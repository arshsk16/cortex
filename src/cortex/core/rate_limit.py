"""In-process sliding-window rate limiter middleware for Cortex.

Design
------
* Per remote-IP sliding-window counter stored in an asyncio-safe dict.
* No external dependencies — purely in-process.
* Middleware is instantiated with a ``requests_per_minute`` limit and an
  optional ``path_prefix`` so callers can mount two instances with
  different budgets (e.g. tighter on auth endpoints).
* Returns ``429 Too Many Requests`` with a ``Retry-After`` header when the
  limit is exceeded.

Limitations
-----------
This implementation is single-process.  For multi-process / multi-node
deployments a Redis-backed rate limiter (Phase 16+) is required.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

logger = logging.getLogger(__name__)

_WINDOW_SECONDS = 60  # sliding window width


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window per-IP rate limiter.

    Parameters
    ----------
    app:
        The ASGI application to wrap.
    requests_per_minute:
        Maximum number of requests allowed per IP per minute.
    path_prefix:
        Optional URL path prefix this limiter applies to.  When set,
        requests whose path does NOT start with ``path_prefix`` are
        passed through unchanged.  When ``None`` (default) all paths
        are rate-limited.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        requests_per_minute: int = 60,
        path_prefix: str | None = None,
    ) -> None:
        super().__init__(app)
        self._rpm = requests_per_minute
        self._prefix = path_prefix
        # Per-IP deque of request timestamps (float, seconds since epoch).
        # Access is single-threaded within an asyncio event loop, so no
        # explicit lock is needed.
        self._buckets: dict[str, deque[float]] = {}

    def _get_client_ip(self, request: Request) -> str:
        """Return the best-effort client IP address."""
        # Respect X-Forwarded-For when behind a proxy.
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        if request.client is not None:
            return request.client.host
        return "unknown"

    def _is_limited(self, ip: str) -> tuple[bool, int]:
        """Check and record the current request; return (limited, retry_after).

        The caller is responsible for ensuring this method is only called
        from a single asyncio task at a time (i.e. from dispatch()).
        """
        now = time.monotonic()
        window_start = now - _WINDOW_SECONDS

        if ip not in self._buckets:
            self._buckets[ip] = deque()
        bucket = self._buckets[ip]

        # Evict timestamps outside the sliding window.
        while bucket and bucket[0] < window_start:
            bucket.popleft()

        if len(bucket) >= self._rpm:
            # Retry-After = seconds until the oldest request ages out.
            retry_after = max(1, int(bucket[0] - window_start) + 1)
            return True, retry_after

        bucket.append(now)
        return False, 0

    async def dispatch(self, request: Request, call_next: object) -> Response:
        """Rate-check the request; pass through or return 429."""
        # Skip if this middleware is scoped to a prefix and path doesn't match.
        if self._prefix and not request.url.path.startswith(self._prefix):
            return await call_next(request)  # type: ignore[operator]

        ip = self._get_client_ip(request)
        limited, retry_after = self._is_limited(ip)

        if limited:
            logger.warning(
                "Rate limit exceeded: ip=%s path=%s rpm_limit=%d",
                ip,
                request.url.path,
                self._rpm,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "rate_limit_exceeded",
                        "message": (
                            f"Too many requests. "
                            f"Retry after {retry_after} second(s)."
                        ),
                    }
                },
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)  # type: ignore[operator]
