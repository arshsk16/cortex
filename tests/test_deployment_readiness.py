"""Phase 16 — Container & Deployment Readiness tests.

Covers things we can verify without Docker:
- Dockerfile and docker-compose.yml exist and have required content
- entrypoint.sh exists and contains key commands
- .dockerignore excludes sensitive paths
- .env.docker template exists and lists required variables
- /health/live and /health/ready work correctly as container health checks
- Alembic env.py resolves database URL from application Settings
- Settings correctly populate DATABASE_URL in asyncpg format
- docs/deployment.md exists and covers key operational commands
"""

from __future__ import annotations

import pathlib

import pytest

ROOT = pathlib.Path(__file__).parent.parent  # project root


# ===========================================================================
# 1. Dockerfile content checks
# ===========================================================================


class TestDockerfile:
    """Verify Dockerfile structure and key directives."""

    @pytest.fixture(scope="class")
    @classmethod
    def dockerfile(cls) -> str:
        p = ROOT / "Dockerfile"
        assert p.exists(), "Dockerfile must exist"
        return p.read_text(encoding="utf-8")

    def test_uses_python312_base(self, dockerfile: str) -> None:
        assert "python:3.12" in dockerfile

    def test_multi_stage_has_builder_and_runtime(self, dockerfile: str) -> None:
        assert "AS builder" in dockerfile
        assert "AS runtime" in dockerfile

    def test_non_root_user_created(self, dockerfile: str) -> None:
        assert "useradd" in dockerfile or "adduser" in dockerfile

    def test_non_root_user_set(self, dockerfile: str) -> None:
        assert "USER cortex" in dockerfile

    def test_port_8000_exposed(self, dockerfile: str) -> None:
        assert "EXPOSE 8000" in dockerfile

    def test_healthcheck_uses_health_live(self, dockerfile: str) -> None:
        assert "/health/live" in dockerfile

    def test_healthcheck_has_start_period(self, dockerfile: str) -> None:
        assert "start-period" in dockerfile

    def test_entrypoint_set(self, dockerfile: str) -> None:
        assert "ENTRYPOINT" in dockerfile
        assert "entrypoint.sh" in dockerfile

    def test_uv_used_for_install(self, dockerfile: str) -> None:
        assert "uv" in dockerfile

    def test_storage_volume_dirs_created(self, dockerfile: str) -> None:
        assert "storage/documents" in dockerfile
        assert "storage/chroma" in dockerfile

    def test_pythonunbuffered_set(self, dockerfile: str) -> None:
        assert "PYTHONUNBUFFERED=1" in dockerfile


# ===========================================================================
# 2. entrypoint.sh content checks
# ===========================================================================


class TestEntrypoint:
    """Verify the container entrypoint script."""

    @pytest.fixture(scope="class")
    @classmethod
    def entrypoint(cls) -> str:
        p = ROOT / "docker" / "entrypoint.sh"
        assert p.exists(), "docker/entrypoint.sh must exist"
        return p.read_text(encoding="utf-8")

    def test_is_bash_script(self, entrypoint: str) -> None:
        assert entrypoint.startswith("#!/usr/bin/env bash") or entrypoint.startswith(
            "#!/bin/bash"
        )

    def test_runs_alembic_migrations(self, entrypoint: str) -> None:
        assert "alembic" in entrypoint
        assert "upgrade head" in entrypoint

    def test_starts_uvicorn(self, entrypoint: str) -> None:
        assert "uvicorn" in entrypoint
        assert "cortex.main:app" in entrypoint

    def test_waits_for_database(self, entrypoint: str) -> None:
        # Must have some DB wait logic
        assert "wait" in entrypoint.lower() or "pg_isready" in entrypoint

    def test_set_euo_pipefail(self, entrypoint: str) -> None:
        assert "set -euo pipefail" in entrypoint or "set -e" in entrypoint

    def test_uses_exec_for_uvicorn(self, entrypoint: str) -> None:
        """exec replaces the shell process so signals are forwarded correctly."""
        assert "exec uvicorn" in entrypoint


# ===========================================================================
# 3. docker-compose.yml content checks
# ===========================================================================


class TestDockerCompose:
    """Verify docker-compose.yml structure."""

    @pytest.fixture(scope="class")
    @classmethod
    def compose(cls) -> str:
        p = ROOT / "docker-compose.yml"
        assert p.exists(), "docker-compose.yml must exist"
        return p.read_text(encoding="utf-8")

    def test_defines_cortex_service(self, compose: str) -> None:
        assert "cortex:" in compose

    def test_defines_db_service(self, compose: str) -> None:
        assert "db:" in compose

    def test_defines_redis_service(self, compose: str) -> None:
        assert "redis:" in compose

    def test_postgres_volume_defined(self, compose: str) -> None:
        assert "postgres_data" in compose

    def test_redis_volume_defined(self, compose: str) -> None:
        assert "redis_data" in compose

    def test_cortex_documents_volume_defined(self, compose: str) -> None:
        assert "cortex_documents" in compose

    def test_cortex_chroma_volume_defined(self, compose: str) -> None:
        assert "cortex_chroma" in compose

    def test_db_healthcheck_present(self, compose: str) -> None:
        assert "pg_isready" in compose

    def test_cortex_depends_on_db(self, compose: str) -> None:
        assert "depends_on" in compose
        assert "service_healthy" in compose

    def test_jwt_secret_required(self, compose: str) -> None:
        """JWT_SECRET_KEY must use the :? syntax to fail fast if unset."""
        assert "JWT_SECRET_KEY:?" in compose

    def test_postgres_password_required(self, compose: str) -> None:
        assert "POSTGRES_PASSWORD:?" in compose

    def test_cortex_healthcheck_uses_health_live(self, compose: str) -> None:
        assert "/health/live" in compose

    def test_network_defined(self, compose: str) -> None:
        assert "cortex_net" in compose

    def test_restart_policy_set(self, compose: str) -> None:
        assert "unless-stopped" in compose


# ===========================================================================
# 4. .dockerignore content checks
# ===========================================================================


class TestDockerignore:
    """Verify .dockerignore excludes critical paths."""

    @pytest.fixture(scope="class")
    @classmethod
    def dockerignore(cls) -> str:
        p = ROOT / ".dockerignore"
        assert p.exists(), ".dockerignore must exist"
        return p.read_text(encoding="utf-8")

    def test_excludes_venv(self, dockerignore: str) -> None:
        assert ".venv/" in dockerignore

    def test_excludes_dot_env(self, dockerignore: str) -> None:
        assert ".env" in dockerignore

    def test_excludes_git(self, dockerignore: str) -> None:
        assert ".git/" in dockerignore

    def test_excludes_storage(self, dockerignore: str) -> None:
        assert "storage/" in dockerignore

    def test_excludes_pytest_cache(self, dockerignore: str) -> None:
        assert ".pytest_cache/" in dockerignore

    def test_excludes_tests(self, dockerignore: str) -> None:
        assert "tests/" in dockerignore


# ===========================================================================
# 5. .env.docker template checks
# ===========================================================================


class TestEnvDockerTemplate:
    """Verify .env.docker template has all required variables."""

    @pytest.fixture(scope="class")
    @classmethod
    def env_docker(cls) -> str:
        p = ROOT / ".env.docker"
        assert p.exists(), ".env.docker template must exist"
        return p.read_text(encoding="utf-8")

    def test_has_postgres_password(self, env_docker: str) -> None:
        assert "POSTGRES_PASSWORD" in env_docker

    def test_has_jwt_secret_key(self, env_docker: str) -> None:
        assert "JWT_SECRET_KEY" in env_docker

    def test_has_redis_password(self, env_docker: str) -> None:
        assert "REDIS_PASSWORD" in env_docker

    def test_has_gemini_api_key(self, env_docker: str) -> None:
        assert "GEMINI_API_KEY" in env_docker

    def test_has_workers(self, env_docker: str) -> None:
        assert "WORKERS" in env_docker

    def test_has_cors_origins(self, env_docker: str) -> None:
        assert "CORS_ORIGINS" in env_docker

    def test_placeholders_marked(self, env_docker: str) -> None:
        """Secrets should have <CHANGE_ME> markers so operators notice them."""
        assert "<CHANGE_ME>" in env_docker


# ===========================================================================
# 6. Health endpoint behaviour (as container health check targets)
# ===========================================================================


class TestHealthEndpointsAsProbes:
    """Verify /health/live and /health/ready behave correctly as probes."""

    def _make_app(self, *, db_healthy: bool) -> object:
        from datetime import UTC, datetime
        from unittest.mock import AsyncMock, MagicMock

        from fastapi import FastAPI

        from cortex.api.deps import get_health_service, get_settings
        from cortex.api.v1.endpoints.health import router as health_router
        from cortex.schemas.health import ComponentHealth, HealthResponse
        from cortex.services.health import HealthService

        app = FastAPI()
        app.include_router(health_router)

        status = "healthy" if db_healthy else "unhealthy"
        mock_hs = AsyncMock(spec=HealthService)
        mock_hs.check.return_value = HealthResponse(
            status=status,
            service="Cortex",
            version="0.1.0",
            environment="test",
            timestamp=datetime.now(UTC),
            components=[
                ComponentHealth(
                    name="database", status=status, latency_ms=1.0
                )
            ],
        )
        mock_settings = MagicMock()
        mock_settings.app_name = "Cortex"

        app.dependency_overrides[get_health_service] = lambda: mock_hs
        app.dependency_overrides[get_settings] = lambda: mock_settings
        return app

    def test_live_always_200(self) -> None:
        """Docker liveness check: always 200 when process is running."""
        from fastapi.testclient import TestClient

        client = TestClient(self._make_app(db_healthy=True))
        r = client.get("/health/live")
        assert r.status_code == 200

    def test_live_200_when_db_down(self) -> None:
        """Liveness must not reflect database state."""
        from fastapi.testclient import TestClient

        client = TestClient(self._make_app(db_healthy=False))
        r = client.get("/health/live")
        assert r.status_code == 200

    def test_ready_200_when_db_up(self) -> None:
        """Readiness returns 200 when DB is healthy."""
        from fastapi.testclient import TestClient

        client = TestClient(self._make_app(db_healthy=True))
        r = client.get("/health/ready")
        assert r.status_code == 200

    def test_ready_503_when_db_down(self) -> None:
        """Readiness returns 503 when DB is unhealthy."""
        from fastapi.testclient import TestClient

        client = TestClient(self._make_app(db_healthy=False))
        r = client.get("/health/ready")
        assert r.status_code == 503

    def test_live_response_has_status_ok(self) -> None:
        from fastapi.testclient import TestClient

        client = TestClient(self._make_app(db_healthy=True))
        body = client.get("/health/live").json()
        assert body["status"] == "ok"

    def test_ready_response_has_components(self) -> None:
        from fastapi.testclient import TestClient

        client = TestClient(self._make_app(db_healthy=True))
        body = client.get("/health/ready").json()
        assert "components" in body
        assert len(body["components"]) >= 1


# ===========================================================================
# 7. Alembic configuration — resolves database URL from Settings
# ===========================================================================


class TestAlembicConfiguration:
    """Alembic env.py must read DATABASE_URL from application Settings."""

    def test_alembic_env_py_exists(self) -> None:
        assert (ROOT / "alembic" / "env.py").exists()

    def test_alembic_env_py_imports_settings(self) -> None:
        content = (ROOT / "alembic" / "env.py").read_text(encoding="utf-8")
        assert "get_settings" in content

    def test_alembic_ini_exists(self) -> None:
        assert (ROOT / "alembic.ini").exists()

    def test_alembic_versions_exist(self) -> None:
        versions_dir = ROOT / "alembic" / "versions"
        assert versions_dir.exists()
        py_files = list(versions_dir.glob("*.py"))
        assert len(py_files) > 0, "At least one migration must exist"

    def test_database_url_uses_asyncpg_dialect(self) -> None:
        """Settings validator enforces asyncpg dialect."""
        import pydantic
        import pytest

        from cortex.core.config import Settings

        with pytest.raises(pydantic.ValidationError):
            Settings(
                database_url="postgresql://u:p@localhost/db",  # missing +asyncpg
                jwt_secret_key="a" * 32,
            )

    def test_database_url_accepted_with_asyncpg(self) -> None:
        from cortex.core.config import Settings

        s = Settings(
            database_url="postgresql+asyncpg://u:p@localhost/db",
            jwt_secret_key="a" * 32,
        )
        assert s.database_url.startswith("postgresql+asyncpg://")


# ===========================================================================
# 8. Deployment documentation
# ===========================================================================


class TestDeploymentDocs:
    """docs/deployment.md must cover key operational topics."""

    @pytest.fixture(scope="class")
    @classmethod
    def docs(cls) -> str:
        p = ROOT / "docs" / "deployment.md"
        assert p.exists(), "docs/deployment.md must exist"
        return p.read_text(encoding="utf-8")

    def test_has_quick_start_section(self, docs: str) -> None:
        assert "Quick Start" in docs or "quick start" in docs.lower()

    def test_has_migration_commands(self, docs: str) -> None:
        assert "alembic" in docs
        assert "upgrade head" in docs

    def test_has_docker_compose_commands(self, docs: str) -> None:
        assert "docker compose" in docs

    def test_has_health_check_section(self, docs: str) -> None:
        assert "/health/live" in docs
        assert "/health/ready" in docs

    def test_has_volume_reference(self, docs: str) -> None:
        assert "postgres_data" in docs

    def test_has_env_variable_table(self, docs: str) -> None:
        assert "JWT_SECRET_KEY" in docs
        assert "GEMINI_API_KEY" in docs

    def test_has_stop_command(self, docs: str) -> None:
        assert "docker compose down" in docs
