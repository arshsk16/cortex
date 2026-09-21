"""Phase 15 — Production Hardening tests.

Covers:
- RateLimitMiddleware (sliding window, per-IP isolation, Retry-After header)
- RequestIDMiddleware (X-Request-ID header, UUID format)
- Generic unhandled-exception catch-all -> 500 internal_error
- GeminiProvider timeout -> ServiceUnavailableError
- Input sanitisation: null bytes / control chars stripped from agent/RAG/memory schemas
- Auth email redaction helper
- Health endpoints: /health/live (always 200), /health/ready (DB-dependent)
- Settings: new Phase 15 fields with defaults
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ===========================================================================
# 1. RateLimitMiddleware
# ===========================================================================


class TestRateLimitMiddleware:
    """Tests for the in-process sliding-window rate limiter."""

    def _app(self, rpm: int, prefix: str | None = None) -> FastAPI:
        from cortex.core.rate_limit import RateLimitMiddleware

        app = FastAPI()
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=rpm,
            path_prefix=prefix,
        )

        @app.get("/limited")
        async def ok() -> dict[str, str]:
            return {"ok": "true"}

        @app.get("/other")
        async def other() -> dict[str, str]:
            return {"ok": "true"}

        return app

    def test_requests_below_limit_pass(self) -> None:
        """Requests under the limit should return 200."""
        client = TestClient(self._app(rpm=5, prefix="/limited"))
        for _ in range(5):
            r = client.get("/limited")
            assert r.status_code == 200

    def test_request_over_limit_returns_429(self) -> None:
        """The (N+1)-th request should be rate-limited."""
        client = TestClient(self._app(rpm=3, prefix="/limited"))
        for _ in range(3):
            client.get("/limited")
        r = client.get("/limited")
        assert r.status_code == 429
        body = r.json()
        assert body["error"]["code"] == "rate_limit_exceeded"

    def test_429_has_retry_after_header(self) -> None:
        """Retry-After header must be present and numeric."""
        client = TestClient(self._app(rpm=2, prefix="/limited"))
        client.get("/limited")
        client.get("/limited")
        r = client.get("/limited")
        assert r.status_code == 429
        assert "Retry-After" in r.headers
        retry = int(r.headers["Retry-After"])
        assert retry >= 1

    def test_unscoped_path_not_rate_limited(self) -> None:
        """Paths outside the prefix must pass through regardless of count."""
        client = TestClient(self._app(rpm=2, prefix="/limited"))
        for _ in range(10):
            r = client.get("/other")
            assert r.status_code == 200

    def test_per_ip_isolation(self) -> None:
        """Different IPs get independent buckets."""
        from cortex.core.rate_limit import RateLimitMiddleware

        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, requests_per_minute=2)

        @app.get("/check")
        async def check() -> dict[str, str]:
            return {"ok": "true"}

        client = TestClient(app, raise_server_exceptions=False)
        # Exhaust the budget for IP 1.1.1.1
        for _ in range(2):
            client.get("/check", headers={"X-Forwarded-For": "1.1.1.1"})
        r_ip1 = client.get("/check", headers={"X-Forwarded-For": "1.1.1.1"})
        assert r_ip1.status_code == 429

        # IP 2.2.2.2 has its own independent bucket -- should pass.
        r_ip2 = client.get("/check", headers={"X-Forwarded-For": "2.2.2.2"})
        assert r_ip2.status_code == 200

    def test_no_prefix_applies_globally(self) -> None:
        """Without path_prefix every route shares the same budget."""
        client = TestClient(self._app(rpm=2, prefix=None))
        client.get("/limited")
        client.get("/other")  # second request, same IP bucket
        r = client.get("/limited")  # third request -- should be blocked
        assert r.status_code == 429


# ===========================================================================
# 2. RequestIDMiddleware
# ===========================================================================


class TestRequestIDMiddleware:
    """Tests for request-ID injection middleware."""

    def _app(self) -> FastAPI:
        from cortex.core.middleware import RequestIDMiddleware

        app = FastAPI()
        app.add_middleware(RequestIDMiddleware)

        @app.get("/ping")
        async def ping() -> dict[str, str]:
            return {"pong": "true"}

        return app

    def test_response_has_x_request_id_header(self) -> None:
        client = TestClient(self._app())
        r = client.get("/ping")
        assert "X-Request-ID" in r.headers

    def test_request_id_is_hex_string(self) -> None:
        """Generated IDs should be 32-char hex strings (UUID4 without dashes)."""
        client = TestClient(self._app())
        r = client.get("/ping")
        rid = r.headers["X-Request-ID"]
        assert re.fullmatch(r"[0-9a-f]{32}", rid), f"Bad request_id: {rid}"

    def test_client_supplied_id_is_echoed(self) -> None:
        """If the client supplies X-Request-ID, the same value is returned."""
        client = TestClient(self._app())
        my_id = "test-correlation-abc123"
        r = client.get("/ping", headers={"X-Request-ID": my_id})
        assert r.headers["X-Request-ID"] == my_id

    def test_different_requests_get_unique_ids(self) -> None:
        client = TestClient(self._app())
        ids = {client.get("/ping").headers["X-Request-ID"] for _ in range(5)}
        assert len(ids) == 5, "Expected 5 unique request IDs"


# ===========================================================================
# 3. Generic exception catch-all
# ===========================================================================


class TestUnhandledExceptionCatchAll:
    """Unhandled exceptions should produce a structured 500 response."""

    def _app(self) -> FastAPI:
        from cortex.core.exceptions import register_exception_handlers

        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/boom")
        async def boom() -> dict[str, str]:
            raise RuntimeError("chaos")

        return app

    def test_runtime_error_returns_500(self) -> None:
        client = TestClient(self._app(), raise_server_exceptions=False)
        r = client.get("/boom")
        assert r.status_code == 500

    def test_500_body_has_internal_error_code(self) -> None:
        client = TestClient(self._app(), raise_server_exceptions=False)
        r = client.get("/boom")
        body = r.json()
        assert body["error"]["code"] == "internal_error"

    def test_500_body_does_not_expose_traceback(self) -> None:
        """Stack traces must not leak to the client."""
        client = TestClient(self._app(), raise_server_exceptions=False)
        r = client.get("/boom")
        text = r.text
        assert "Traceback" not in text
        assert "RuntimeError" not in text
        assert "chaos" not in text


# ===========================================================================
# 4. GeminiProvider timeout
# ===========================================================================


class TestGeminiTimeout:
    """GeminiProvider should raise ServiceUnavailableError on timeout."""

    def _provider(self, timeout: int = 1) -> object:
        from cortex.llm.gemini import GeminiProvider

        settings = MagicMock()
        settings.gemini_api_key = "fake-key"
        settings.gemini_model_name = "gemini-test"
        settings.gemini_temperature = 0.2
        settings.gemini_max_output_tokens = 256
        settings.gemini_request_timeout_seconds = timeout
        return GeminiProvider(settings)

    @pytest.mark.asyncio
    async def test_generate_timeout_raises_service_unavailable(self) -> None:
        from cortex.core.exceptions import ServiceUnavailableError
        from cortex.llm.gemini import GeminiProvider

        provider: GeminiProvider = self._provider(timeout=1)  # type: ignore[assignment]

        async def _slow(*args: object, **kwargs: object) -> str:
            await asyncio.sleep(10)
            return "answer"

        with (
            patch("asyncio.to_thread", side_effect=_slow),
            pytest.raises(ServiceUnavailableError) as exc_info,
        ):
            await provider.generate("test prompt")
        assert "timed out" in exc_info.value.message.lower()

    @pytest.mark.asyncio
    async def test_generate_with_tools_timeout_raises_service_unavailable(
        self,
    ) -> None:
        from cortex.core.exceptions import ServiceUnavailableError
        from cortex.llm.gemini import GeminiProvider

        provider: GeminiProvider = self._provider(timeout=1)  # type: ignore[assignment]

        async def _slow(*args: object, **kwargs: object) -> object:
            await asyncio.sleep(10)
            return MagicMock()

        with (
            patch("asyncio.to_thread", side_effect=_slow),
            pytest.raises(ServiceUnavailableError) as exc_info,
        ):
            await provider.generate_with_tools([], [])
        assert "timed out" in exc_info.value.message.lower()

    def test_timeout_stored_from_settings(self) -> None:
        from cortex.llm.gemini import GeminiProvider

        provider: GeminiProvider = self._provider(timeout=45)  # type: ignore[assignment]
        assert provider._timeout == 45


# ===========================================================================
# 5. Input sanitisation
# ===========================================================================


class TestInputSanitisation:
    """Control chars and null bytes must be stripped from user inputs."""

    # --- AgentRunRequest ---

    def test_agent_null_byte_stripped(self) -> None:
        from cortex.schemas.agent import AgentRunRequest

        req = AgentRunRequest(message="hello\x00world")
        assert "\x00" not in req.message
        assert req.message == "helloworld"

    def test_agent_control_chars_stripped(self) -> None:
        from cortex.schemas.agent import AgentRunRequest

        # \x01 (SOH), \x1f (US) stripped; \n and \t survive
        req = AgentRunRequest(message="a\x01b\x1fc\nd\te")
        assert "\x01" not in req.message
        assert "\x1f" not in req.message
        assert "\n" in req.message
        assert "\t" in req.message

    def test_agent_clean_message_unchanged(self) -> None:
        from cortex.schemas.agent import AgentRunRequest

        msg = "What is the capital of France?"
        req = AgentRunRequest(message=msg)
        assert req.message == msg

    def test_agent_all_control_chars_becomes_empty_fails_validation(self) -> None:
        """A message that is all control chars becomes empty -> validation fails."""
        import pydantic

        from cortex.schemas.agent import AgentRunRequest

        with pytest.raises(pydantic.ValidationError):
            AgentRunRequest(message="\x00\x01\x02")

    # --- RAGQueryRequest ---

    def test_rag_null_byte_stripped(self) -> None:
        from cortex.schemas.rag import RAGQueryRequest

        req = RAGQueryRequest(question="q\x00uery")
        assert "\x00" not in req.question
        assert req.question == "query"

    def test_rag_control_chars_stripped(self) -> None:
        from cortex.schemas.rag import RAGQueryRequest

        req = RAGQueryRequest(question="a\x07b")
        assert req.question == "ab"

    # --- MemoryCreate ---

    def test_memory_create_null_byte_stripped(self) -> None:
        from cortex.schemas.memory import MemoryCreate

        mc = MemoryCreate(content="mem\x00ory")
        assert "\x00" not in mc.content

    def test_memory_update_control_chars_stripped(self) -> None:
        from cortex.schemas.memory import MemoryUpdate

        mu = MemoryUpdate(content="val\x01ue")
        assert "\x01" not in mu.content

    def test_memory_search_null_byte_stripped(self) -> None:
        from cortex.schemas.memory import MemorySearchRequest

        msr = MemorySearchRequest(query="find\x00this")
        assert "\x00" not in msr.query


# ===========================================================================
# 6. Auth email redaction
# ===========================================================================


class TestEmailRedaction:
    """_redact_email must safely redact addresses for log output."""

    def _redact(self, email: str) -> str:
        from cortex.services.auth import _redact_email

        return _redact_email(email)

    def test_standard_email_redacted(self) -> None:
        result = self._redact("alice@example.com")
        assert result == "a***@example.com"

    def test_only_first_char_visible(self) -> None:
        result = self._redact("bob@domain.org")
        assert result.startswith("b***@")

    def test_no_at_sign_returns_stars(self) -> None:
        result = self._redact("notanemail")
        assert result == "***"

    def test_domain_preserved(self) -> None:
        result = self._redact("user@corp.internal")
        assert result.endswith("@corp.internal")

    def test_short_local_part(self) -> None:
        result = self._redact("x@y.com")
        assert result == "x***@y.com"


# ===========================================================================
# 7. Health endpoints
# ===========================================================================


class TestHealthEndpoints:
    """Tests for /health/live and /health/ready."""

    def _app_with_db(self, *, healthy: bool) -> FastAPI:
        from cortex.api.deps import get_health_service, get_settings
        from cortex.api.v1.endpoints.health import router as health_router
        from cortex.schemas.health import ComponentHealth, HealthResponse
        from cortex.services.health import HealthService

        app = FastAPI()
        app.include_router(health_router)

        if healthy:
            db_status = "healthy"
            overall = "healthy"
        else:
            db_status = "unhealthy"
            overall = "unhealthy"

        mock_hs = AsyncMock(spec=HealthService)
        mock_hs.check.return_value = HealthResponse(
            status=overall,
            service="Cortex",
            version="test",
            environment="test",
            timestamp=datetime.now(UTC),
            components=[
                ComponentHealth(name="database", status=db_status, latency_ms=1.0)
            ],
        )

        mock_settings = MagicMock()
        mock_settings.app_name = "Cortex"

        app.dependency_overrides[get_health_service] = lambda: mock_hs
        app.dependency_overrides[get_settings] = lambda: mock_settings
        return app

    def test_liveness_always_200_when_healthy(self) -> None:
        client = TestClient(self._app_with_db(healthy=True))
        r = client.get("/health/live")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_liveness_200_even_with_unhealthy_db(self) -> None:
        """Liveness must never reflect DB state."""
        client = TestClient(self._app_with_db(healthy=False))
        r = client.get("/health/live")
        assert r.status_code == 200

    def test_readiness_200_when_healthy(self) -> None:
        client = TestClient(self._app_with_db(healthy=True))
        r = client.get("/health/ready")
        assert r.status_code == 200

    def test_readiness_503_when_unhealthy(self) -> None:
        client = TestClient(self._app_with_db(healthy=False))
        r = client.get("/health/ready")
        assert r.status_code == 503

    def test_legacy_health_endpoint_still_works(self) -> None:
        """The original /health endpoint must remain backward-compatible."""
        client = TestClient(self._app_with_db(healthy=True))
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"


# ===========================================================================
# 8. Settings -- new Phase 15 fields
# ===========================================================================


class TestNewSettings:
    """New Phase 15 settings must have correct defaults and validation."""

    def _s(self, **overrides: object) -> object:
        from cortex.core.config import Settings

        base = {
            "database_url": "postgresql+asyncpg://u:p@localhost/db",
            "jwt_secret_key": "a" * 32,
        }
        base.update(overrides)
        return Settings(**base)

    def test_default_gemini_timeout(self) -> None:
        assert self._s().gemini_request_timeout_seconds == 30

    def test_default_rate_limit_rpm(self) -> None:
        assert self._s().rate_limit_requests_per_minute == 60

    def test_default_auth_rate_limit_rpm(self) -> None:
        assert self._s().rate_limit_auth_requests_per_minute == 10

    def test_custom_timeout(self) -> None:
        s = self._s(gemini_request_timeout_seconds=60)
        assert s.gemini_request_timeout_seconds == 60

    def test_timeout_minimum_enforced(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            self._s(gemini_request_timeout_seconds=1)  # below ge=5
