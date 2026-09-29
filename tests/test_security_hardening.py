"""Tests for Phase 19A — Security & API Hardening.

Covers:
- Rate limiter: spoofed X-Forwarded-For rejected, trusted proxy behaviour
- JWT: jti in tokens, logout endpoint, revoked-token rejection, TTL correctness
- Document upload: size limit enforced via streaming (not full read)
- API docs: disabled in production, available in development
"""

from __future__ import annotations

import ipaddress
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from cortex.core.config import Settings
from cortex.core.rate_limit import (
    RateLimitMiddleware,
    _ip_in_networks,
    _parse_networks,
)
from cortex.core.security import create_access_token, decode_access_token
from cortex.core.token_blocklist import TokenBlocklist

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = dict(
        database_url="postgresql+asyncpg://u:p@localhost/db",
        jwt_secret_key="test-secret-key-that-is-at-least-32-chars",
        app_env="development",
    )
    base.update(overrides)
    return Settings(**base)


# ---------------------------------------------------------------------------
# 1. Rate Limiter helpers — _parse_networks / _ip_in_networks
# ---------------------------------------------------------------------------


class TestParseNetworks:
    def test_valid_ipv4_cidr(self) -> None:
        nets = _parse_networks(["10.0.0.0/8"])
        assert len(nets) == 1
        assert ipaddress.ip_address("10.1.2.3") in nets[0]

    def test_valid_ipv6_cidr(self) -> None:
        nets = _parse_networks(["::1/128"])
        assert len(nets) == 1

    def test_invalid_cidr_skipped(self) -> None:
        nets = _parse_networks(["not-a-cidr", "10.0.0.0/8"])
        assert len(nets) == 1

    def test_empty_list(self) -> None:
        assert _parse_networks([]) == []

    def test_empty_string_skipped(self) -> None:
        assert _parse_networks([""]) == []


class TestIpInNetworks:
    def test_ip_in_single_network(self) -> None:
        nets = _parse_networks(["10.0.0.0/8"])
        assert _ip_in_networks("10.5.6.7", nets) is True

    def test_ip_not_in_network(self) -> None:
        nets = _parse_networks(["10.0.0.0/8"])
        assert _ip_in_networks("192.168.1.1", nets) is False

    def test_empty_networks_always_false(self) -> None:
        assert _ip_in_networks("10.0.0.0", []) is False

    def test_invalid_ip_returns_false(self) -> None:
        nets = _parse_networks(["10.0.0.0/8"])
        assert _ip_in_networks("not-an-ip", nets) is False


# ---------------------------------------------------------------------------
# 2. Rate Limiter — XFF spoof / trusted proxy
# ---------------------------------------------------------------------------


class TestRateLimiterXFFSpoof:
    def _make_middleware(
        self, trusted: list[str] | None = None
    ) -> RateLimitMiddleware:
        from starlette.applications import Starlette

        app = Starlette()
        return RateLimitMiddleware(app, requests_per_minute=60, trusted_proxies=trusted)

    def _mock_request(
        self,
        client_host: str,
        xff_header: str | None = None,
    ) -> MagicMock:
        req = MagicMock()
        req.client = MagicMock()
        req.client.host = client_host
        headers: dict[str, str] = {}
        if xff_header:
            headers["X-Forwarded-For"] = xff_header
        req.headers = headers
        return req

    def test_no_trusted_proxies_xff_ignored(self) -> None:
        mw = self._make_middleware(trusted=None)
        req = self._mock_request("1.2.3.4", xff_header="9.9.9.9")
        assert mw._get_client_ip(req) == "1.2.3.4"

    def test_no_trusted_proxies_attacker_uses_direct_ip(self) -> None:
        mw = self._make_middleware(trusted=None)
        req = self._mock_request("attacker.host", xff_header="legitimate.ip")
        assert mw._get_client_ip(req) == "attacker.host"

    def test_trusted_proxy_xff_honoured(self) -> None:
        mw = self._make_middleware(trusted=["10.0.0.0/8"])
        req = self._mock_request("10.1.2.3", xff_header="203.0.113.5, 10.1.2.3")
        assert mw._get_client_ip(req) == "203.0.113.5"

    def test_untrusted_proxy_xff_ignored(self) -> None:
        mw = self._make_middleware(trusted=["10.0.0.0/8"])
        req = self._mock_request("172.16.5.5", xff_header="9.9.9.9")
        assert mw._get_client_ip(req) == "172.16.5.5"

    def test_multiple_trusted_cidrs(self) -> None:
        mw = self._make_middleware(trusted=["10.0.0.0/8", "172.16.0.0/12"])
        req = self._mock_request("172.16.1.1", xff_header="8.8.8.8")
        assert mw._get_client_ip(req) == "8.8.8.8"
        req2 = self._mock_request("192.168.1.1", xff_header="8.8.8.8")
        assert mw._get_client_ip(req2) == "192.168.1.1"

    def test_spoofed_xff_does_not_affect_rate_bucket(self) -> None:
        mw = self._make_middleware(trusted=None)
        mw._rpm = 2
        req = self._mock_request("attacker", xff_header="victim")
        limited1, _ = mw._is_limited(mw._get_client_ip(req))
        limited2, _ = mw._is_limited(mw._get_client_ip(req))
        limited3, _ = mw._is_limited(mw._get_client_ip(req))
        assert not limited1
        assert not limited2
        assert limited3
        assert "victim" not in mw._buckets


# ---------------------------------------------------------------------------
# 3. JWT — jti in tokens
# ---------------------------------------------------------------------------


class TestJWTJti:
    @pytest.fixture()
    def settings(self) -> Settings:
        return _make_settings()

    def test_token_contains_jti(self, settings: Settings) -> None:
        token, jti = create_access_token(subject="user-1", settings=settings)
        payload = decode_access_token(token, settings)
        assert payload.get("jti") == jti

    def test_jti_unique_per_call(self, settings: Settings) -> None:
        _, jti1 = create_access_token(subject="user-1", settings=settings)
        _, jti2 = create_access_token(subject="user-1", settings=settings)
        assert jti1 != jti2

    def test_jti_is_32_char_hex(self, settings: Settings) -> None:
        _, jti = create_access_token(subject="user-1", settings=settings)
        assert isinstance(jti, str) and len(jti) == 32
        int(jti, 16)  # must be valid hex

    def test_returns_tuple(self, settings: Settings) -> None:
        result = create_access_token(subject="user-1", settings=settings)
        assert isinstance(result, tuple) and len(result) == 2

    def test_extra_claims_preserved(self, settings: Settings) -> None:
        token, _ = create_access_token(
            subject="user-1",
            settings=settings,
            extra_claims={"role": "user"},
        )
        payload = decode_access_token(token, settings)
        assert payload["role"] == "user" and payload["sub"] == "user-1"


# ---------------------------------------------------------------------------
# 4. TokenBlocklist unit tests
# ---------------------------------------------------------------------------


class TestTokenBlocklist:
    @pytest.fixture()
    def redis_mock(self) -> MagicMock:
        client = MagicMock()
        client.setex = AsyncMock(return_value=True)
        client.exists = AsyncMock(return_value=0)
        return client

    @pytest.fixture()
    def blocklist(self, redis_mock: MagicMock) -> TokenBlocklist:
        return TokenBlocklist(redis_mock)

    @pytest.mark.asyncio
    async def test_revoke_calls_setex(
        self, blocklist: TokenBlocklist, redis_mock: MagicMock
    ) -> None:
        await blocklist.revoke("abc123", ttl_seconds=3600)
        redis_mock.setex.assert_awaited_once_with(
            "cortex:blocklist:jti:abc123", 3600, "1"
        )

    @pytest.mark.asyncio
    async def test_revoke_zero_ttl_noop(
        self, blocklist: TokenBlocklist, redis_mock: MagicMock
    ) -> None:
        await blocklist.revoke("abc123", ttl_seconds=0)
        redis_mock.setex.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_revoke_negative_ttl_noop(
        self, blocklist: TokenBlocklist, redis_mock: MagicMock
    ) -> None:
        await blocklist.revoke("abc123", ttl_seconds=-10)
        redis_mock.setex.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_is_revoked_true(
        self, blocklist: TokenBlocklist, redis_mock: MagicMock
    ) -> None:
        redis_mock.exists.return_value = 1
        assert await blocklist.is_revoked("abc123") is True

    @pytest.mark.asyncio
    async def test_is_revoked_false(
        self, blocklist: TokenBlocklist, redis_mock: MagicMock
    ) -> None:
        redis_mock.exists.return_value = 0
        assert await blocklist.is_revoked("abc123") is False

    @pytest.mark.asyncio
    async def test_is_revoked_fail_open(
        self, blocklist: TokenBlocklist, redis_mock: MagicMock
    ) -> None:
        redis_mock.exists.side_effect = ConnectionError("Redis down")
        assert await blocklist.is_revoked("abc123") is False

    @pytest.mark.asyncio
    async def test_revoke_silently_ignores_redis_error(
        self, blocklist: TokenBlocklist, redis_mock: MagicMock
    ) -> None:
        redis_mock.setex.side_effect = ConnectionError("Redis down")
        await blocklist.revoke("abc123", ttl_seconds=60)  # must not raise


# ---------------------------------------------------------------------------
# 5. Logout endpoint
# ---------------------------------------------------------------------------


class TestLogoutEndpoint:
    @pytest.fixture()
    def active_user(self) -> MagicMock:
        user = MagicMock()
        user.id = "user-uuid-1"
        user.email = "test@example.com"
        user.is_active = True
        user.role = MagicMock()
        user.role.value = "user"
        return user

    @pytest.fixture()
    def settings(self) -> Settings:
        return _make_settings()

    @pytest.fixture()
    def token_and_jti(self, settings: Settings) -> tuple[str, str]:
        return create_access_token(subject="user-uuid-1", settings=settings)

    @pytest.fixture()
    def blocklist_mock(self) -> MagicMock:
        bl = MagicMock()
        bl.revoke = AsyncMock()
        bl.is_revoked = AsyncMock(return_value=False)
        return bl

    @pytest.fixture()
    def client(
        self,
        settings: Settings,
        active_user: MagicMock,
        blocklist_mock: MagicMock,
    ) -> TestClient:
        from cortex.api.deps import (
            get_current_active_user,
            get_settings,
            get_token_blocklist,
        )
        from cortex.main import create_app

        app = create_app(settings=settings)
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides[get_current_active_user] = lambda: active_user
        app.dependency_overrides[get_token_blocklist] = lambda: blocklist_mock
        return TestClient(app, raise_server_exceptions=True)

    def test_logout_returns_204(
        self,
        client: TestClient,
        token_and_jti: tuple[str, str],
    ) -> None:
        token, _ = token_and_jti
        resp = client.post(
            "/api/v1/logout",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 204

    def test_logout_calls_blocklist_revoke(
        self,
        client: TestClient,
        token_and_jti: tuple[str, str],
        blocklist_mock: MagicMock,
    ) -> None:
        token, jti = token_and_jti
        client.post(
            "/api/v1/logout",
            headers={"Authorization": f"Bearer {token}"},
        )
        blocklist_mock.revoke.assert_awaited_once()
        call_args = blocklist_mock.revoke.call_args
        revoked_jti = call_args.kwargs.get("jti") or (
            call_args.args[0] if call_args.args else None
        )
        assert revoked_jti == jti

    def test_logout_ttl_positive(
        self,
        client: TestClient,
        token_and_jti: tuple[str, str],
        blocklist_mock: MagicMock,
    ) -> None:
        token, _ = token_and_jti
        client.post(
            "/api/v1/logout",
            headers={"Authorization": f"Bearer {token}"},
        )
        call_args = blocklist_mock.revoke.call_args
        ttl = call_args.kwargs.get("ttl_seconds") or (
            call_args.args[1] if len(call_args.args) > 1 else None
        )
        assert ttl is not None and ttl > 0

    def test_logout_no_auth_returns_4xx(self, client: TestClient) -> None:
        resp = client.post("/api/v1/logout")
        assert resp.status_code in (401, 403)

    def test_logout_succeeds_with_none_blocklist(
        self,
        settings: Settings,
        active_user: MagicMock,
    ) -> None:
        """When Redis is absent (blocklist=None), logout still returns 204."""
        from cortex.api.deps import (
            get_current_active_user,
            get_settings,
            get_token_blocklist,
        )
        from cortex.main import create_app

        app = create_app(settings=settings)
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides[get_current_active_user] = lambda: active_user
        app.dependency_overrides[get_token_blocklist] = lambda: None
        token, _ = create_access_token(subject="user-uuid-1", settings=settings)
        with TestClient(app) as c:
            resp = c.post(
                "/api/v1/logout",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 204


# ---------------------------------------------------------------------------
# 6. Revoked token rejection in get_current_user
# ---------------------------------------------------------------------------


class TestRevokedTokenRejection:
    @pytest.mark.asyncio
    async def test_revoked_token_raises_401(self) -> None:
        from cortex.api.deps import get_current_user
        from cortex.core.exceptions import UnauthorizedError

        settings = _make_settings()
        token, _ = create_access_token(subject="u1", settings=settings)
        bl = MagicMock()
        bl.is_revoked = AsyncMock(return_value=True)
        creds = MagicMock()
        creds.scheme = "bearer"
        creds.credentials = token
        svc = MagicMock()
        svc.get_by_id = AsyncMock(return_value=MagicMock())

        with pytest.raises(UnauthorizedError, match="revoked"):
            await get_current_user(
                credentials=creds,
                settings=settings,
                user_service=svc,
                blocklist=bl,
            )

    @pytest.mark.asyncio
    async def test_valid_token_not_revoked_passes(self) -> None:
        from cortex.api.deps import get_current_user

        settings = _make_settings()
        token, _ = create_access_token(subject="u1", settings=settings)
        bl = MagicMock()
        bl.is_revoked = AsyncMock(return_value=False)
        creds = MagicMock()
        creds.scheme = "bearer"
        creds.credentials = token
        fake_user = MagicMock()
        svc = MagicMock()
        svc.get_by_id = AsyncMock(return_value=fake_user)

        result = await get_current_user(
            credentials=creds, settings=settings, user_service=svc, blocklist=bl
        )
        assert result is fake_user

    @pytest.mark.asyncio
    async def test_none_blocklist_skips_check(self) -> None:
        from cortex.api.deps import get_current_user

        settings = _make_settings()
        token, _ = create_access_token(subject="u1", settings=settings)
        creds = MagicMock()
        creds.scheme = "bearer"
        creds.credentials = token
        fake_user = MagicMock()
        svc = MagicMock()
        svc.get_by_id = AsyncMock(return_value=fake_user)

        result = await get_current_user(
            credentials=creds, settings=settings, user_service=svc, blocklist=None
        )
        assert result is fake_user

    @pytest.mark.asyncio
    async def test_token_without_jti_accepted_even_when_blocklist_set(self) -> None:
        """Old tokens without jti should be accepted (blocklist check skipped)."""
        from jose import jwt as jose_jwt

        from cortex.api.deps import get_current_user

        settings = _make_settings()
        payload: dict[str, Any] = {
            "sub": "u1",
            "iat": datetime.now(UTC),
            "exp": datetime.now(UTC) + timedelta(minutes=60),
            "type": "access",
        }
        token = jose_jwt.encode(
            payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
        )
        bl = MagicMock()
        bl.is_revoked = AsyncMock(return_value=True)  # would reject if called
        creds = MagicMock()
        creds.scheme = "bearer"
        creds.credentials = token
        fake_user = MagicMock()
        svc = MagicMock()
        svc.get_by_id = AsyncMock(return_value=fake_user)

        result = await get_current_user(
            credentials=creds, settings=settings, user_service=svc, blocklist=bl
        )
        assert result is fake_user
        bl.is_revoked.assert_not_awaited()


# ---------------------------------------------------------------------------
# 7. Document upload streaming size enforcement
# ---------------------------------------------------------------------------


class TestDocumentUploadStreaming:
    @pytest.fixture()
    def settings(self) -> Settings:
        return _make_settings(document_max_file_size_bytes=1024)

    @pytest.fixture()
    def active_user(self) -> MagicMock:
        user = MagicMock()
        user.id = "user-1"
        user.is_active = True
        return user

    def _build_client(
        self,
        settings: Settings,
        active_user: MagicMock,
        mock_svc: MagicMock,
    ) -> TestClient:
        from cortex.api.deps import (
            get_current_active_user,
            get_document_service,
            get_settings,
        )
        from cortex.main import create_app

        app = create_app(settings=settings)
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides[get_current_active_user] = lambda: active_user
        app.dependency_overrides[get_document_service] = lambda: mock_svc
        return TestClient(app, raise_server_exceptions=False)

    def _mock_doc(self) -> MagicMock:
        doc = MagicMock()
        doc.id = "doc-1"
        doc.status = "uploaded"
        doc.title = "test.pdf"
        doc.original_filename = "test.pdf"
        doc.storage_filename = "doc-1.pdf"
        doc.storage_path = "storage/documents/doc-1.pdf"
        doc.file_size = 100
        doc.mime_type = "application/pdf"
        doc.created_at = datetime.now(UTC)
        doc.updated_at = datetime.now(UTC)
        doc.user_id = "user-1"
        return doc

    def test_small_file_accepted(
        self, settings: Settings, active_user: MagicMock
    ) -> None:
        svc = MagicMock()
        svc.upload = AsyncMock(return_value=self._mock_doc())
        with self._build_client(settings, active_user, svc) as c:
            resp = c.post(
                "/api/v1/documents/upload",
                files={"file": ("test.pdf", b"x" * 512, "application/pdf")},
            )
        assert resp.status_code == 201

    def test_oversized_file_rejected_400(
        self, settings: Settings, active_user: MagicMock
    ) -> None:
        svc = MagicMock()
        svc.upload = AsyncMock(return_value=self._mock_doc())
        with self._build_client(settings, active_user, svc) as c:
            resp = c.post(
                "/api/v1/documents/upload",
                files={"file": ("big.pdf", b"x" * 2048, "application/pdf")},
            )
        assert resp.status_code == 400

    def test_oversized_error_body_contains_error_key(
        self, settings: Settings, active_user: MagicMock
    ) -> None:
        svc = MagicMock()
        svc.upload = AsyncMock(return_value=self._mock_doc())
        with self._build_client(settings, active_user, svc) as c:
            resp = c.post(
                "/api/v1/documents/upload",
                files={"file": ("big.pdf", b"x" * 2048, "application/pdf")},
            )
        assert "error" in resp.json()

    def test_exact_limit_accepted(
        self, settings: Settings, active_user: MagicMock
    ) -> None:
        svc = MagicMock()
        svc.upload = AsyncMock(return_value=self._mock_doc())
        with self._build_client(settings, active_user, svc) as c:
            resp = c.post(
                "/api/v1/documents/upload",
                files={"file": ("edge.pdf", b"x" * 1024, "application/pdf")},
            )
        assert resp.status_code == 201

    def test_one_byte_over_rejected(
        self, settings: Settings, active_user: MagicMock
    ) -> None:
        svc = MagicMock()
        svc.upload = AsyncMock(return_value=self._mock_doc())
        with self._build_client(settings, active_user, svc) as c:
            resp = c.post(
                "/api/v1/documents/upload",
                files={"file": ("over.pdf", b"x" * 1025, "application/pdf")},
            )
        assert resp.status_code == 400

    def test_service_not_called_on_oversized(
        self, settings: Settings, active_user: MagicMock
    ) -> None:
        svc = MagicMock()
        svc.upload = AsyncMock()
        with self._build_client(settings, active_user, svc) as c:
            c.post(
                "/api/v1/documents/upload",
                files={"file": ("big.pdf", b"x" * 2048, "application/pdf")},
            )
        svc.upload.assert_not_awaited()


# ---------------------------------------------------------------------------
# 8. Production API docs gating
# ---------------------------------------------------------------------------


class TestProductionApiDocs:
    def _client(self, app_env: str) -> TestClient:
        from cortex.main import create_app

        s = _make_settings(app_env=app_env)
        return TestClient(create_app(settings=s), raise_server_exceptions=False)

    def test_docs_available_in_development(self) -> None:
        with self._client("development") as c:
            assert c.get("/docs").status_code == 200

    def test_redoc_available_in_development(self) -> None:
        with self._client("development") as c:
            assert c.get("/redoc").status_code == 200

    def test_openapi_json_available_in_development(self) -> None:
        with self._client("development") as c:
            assert c.get("/openapi.json").status_code == 200

    def test_docs_disabled_in_production(self) -> None:
        with self._client("production") as c:
            assert c.get("/docs").status_code == 404

    def test_redoc_disabled_in_production(self) -> None:
        with self._client("production") as c:
            assert c.get("/redoc").status_code == 404

    def test_openapi_json_disabled_in_production(self) -> None:
        with self._client("production") as c:
            assert c.get("/openapi.json").status_code == 404

    def test_api_still_works_in_production(self) -> None:
        with self._client("production") as c:
            assert c.get("/api/v1/health/live").status_code == 200


# ---------------------------------------------------------------------------
# 9. Settings — trusted_proxies field
# ---------------------------------------------------------------------------


class TestTrustedProxiesConfig:
    def test_default_is_empty_list(self) -> None:
        s = _make_settings()
        assert s.rate_limit_trusted_proxies == []

    def test_from_comma_string(self) -> None:
        s = _make_settings(
            rate_limit_trusted_proxies="10.0.0.0/8,172.16.0.0/12"
        )
        assert "10.0.0.0/8" in s.rate_limit_trusted_proxies
        assert "172.16.0.0/12" in s.rate_limit_trusted_proxies

    def test_from_list(self) -> None:
        s = _make_settings(rate_limit_trusted_proxies=["10.0.0.0/8"])
        assert s.rate_limit_trusted_proxies == ["10.0.0.0/8"]
