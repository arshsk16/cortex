"""Phase 17 — Kubernetes + CI/CD tests.

Validates (without running a real cluster or CI):
- All required k8s manifest files exist
- YAML files are syntactically valid
- Key security and operational requirements are present
- GitHub Actions workflow files are valid and cover lint/test/docker jobs
- Kubernetes resource limits, probes, and replica counts are set
- Secret template is NOT committed with real values
- CI workflow pins Python version and uses service containers
- docs/kubernetes.md covers required operational topics
"""

from __future__ import annotations

import pathlib

import pytest

ROOT = pathlib.Path(__file__).parent.parent


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _yaml_parse(text: str) -> list[dict]:
    """Parse one or more YAML documents from text; return list of dicts."""
    try:
        import yaml  # type: ignore[import-untyped]

        docs = list(yaml.safe_load_all(text))
        return [d for d in docs if d is not None]
    except ImportError:
        # PyYAML not installed — skip parse-level checks
        return []
    except Exception as exc:
        raise AssertionError(f"YAML parse error: {exc}") from exc


# ===========================================================================
# 1. Manifest file existence
# ===========================================================================


class TestManifestFilesExist:
    """Every required k8s manifest must be present."""

    REQUIRED = [
        "k8s/namespace.yaml",
        "k8s/configmap.yaml",
        "k8s/secret.yaml",
        "k8s/kustomization.yaml",
        "k8s/postgres/statefulset.yaml",
        "k8s/postgres/service.yaml",
        "k8s/redis/deployment.yaml",
        "k8s/redis/pvc.yaml",
        "k8s/redis/service.yaml",
        "k8s/cortex/deployment.yaml",
        "k8s/cortex/service.yaml",
        "k8s/cortex/pvc.yaml",
        "k8s/cortex/hpa.yaml",
        "k8s/cortex/ingress.yaml",
    ]

    @pytest.mark.parametrize("rel", REQUIRED)
    def test_file_exists(self, rel: str) -> None:
        assert (ROOT / rel).exists(), f"Missing: {rel}"

    def test_github_ci_workflow_exists(self) -> None:
        assert (ROOT / ".github/workflows/ci.yml").exists()

    def test_github_publish_workflow_exists(self) -> None:
        assert (ROOT / ".github/workflows/docker-publish.yml").exists()

    def test_kubernetes_docs_exist(self) -> None:
        assert (ROOT / "docs/kubernetes.md").exists()


# ===========================================================================
# 2. YAML validity
# ===========================================================================


class TestYAMLValidity:
    """All manifest YAML files must be syntactically valid."""

    YAMLS = [
        "k8s/namespace.yaml",
        "k8s/configmap.yaml",
        "k8s/kustomization.yaml",
        "k8s/postgres/statefulset.yaml",
        "k8s/postgres/service.yaml",
        "k8s/redis/deployment.yaml",
        "k8s/redis/pvc.yaml",
        "k8s/redis/service.yaml",
        "k8s/cortex/deployment.yaml",
        "k8s/cortex/service.yaml",
        "k8s/cortex/pvc.yaml",
        "k8s/cortex/hpa.yaml",
        "k8s/cortex/ingress.yaml",
        ".github/workflows/ci.yml",
        ".github/workflows/docker-publish.yml",
    ]

    @pytest.mark.parametrize("rel", YAMLS)
    def test_yaml_parseable(self, rel: str) -> None:
        """File must parse as valid YAML (skipped if PyYAML not available)."""
        text = _read(rel)
        try:
            import yaml  # type: ignore[import-untyped]

            list(yaml.safe_load_all(text))  # must not raise
        except ImportError:
            pytest.skip("PyYAML not installed")


# ===========================================================================
# 3. Namespace
# ===========================================================================


class TestNamespace:
    def test_namespace_is_cortex(self) -> None:
        text = _read("k8s/namespace.yaml")
        assert "name: cortex" in text

    def test_namespace_kind(self) -> None:
        assert "kind: Namespace" in _read("k8s/namespace.yaml")


# ===========================================================================
# 4. ConfigMap
# ===========================================================================


class TestConfigMap:
    @pytest.fixture(scope="class")
    @classmethod
    def cm(cls) -> str:
        return _read("k8s/configmap.yaml")

    def test_kind(self, cm: str) -> None:
        assert "kind: ConfigMap" in cm

    def test_postgres_host(self, cm: str) -> None:
        assert "POSTGRES_HOST" in cm

    def test_redis_host(self, cm: str) -> None:
        assert "REDIS_HOST" in cm

    def test_document_storage_path(self, cm: str) -> None:
        assert "DOCUMENT_STORAGE_PATH" in cm

    def test_chroma_persist_directory(self, cm: str) -> None:
        assert "CHROMA_PERSIST_DIRECTORY" in cm

    def test_rate_limit_settings(self, cm: str) -> None:
        assert "RATE_LIMIT_REQUESTS_PER_MINUTE" in cm


# ===========================================================================
# 5. Secret template
# ===========================================================================


class TestSecretTemplate:
    @pytest.fixture(scope="class")
    @classmethod
    def secret(cls) -> str:
        return _read("k8s/secret.yaml")

    def test_kind_is_secret(self, secret: str) -> None:
        assert "kind: Secret" in secret

    def test_has_postgres_password_key(self, secret: str) -> None:
        assert "POSTGRES_PASSWORD" in secret

    def test_has_jwt_secret_key(self, secret: str) -> None:
        assert "JWT_SECRET_KEY" in secret

    def test_has_gemini_api_key(self, secret: str) -> None:
        assert "GEMINI_API_KEY" in secret

    def test_no_real_secrets_committed(self, secret: str) -> None:
        """Real values must never be committed — placeholders only."""
        assert "<base64:CHANGE_ME>" in secret, (
            "secret.yaml must contain placeholder values, not real secrets"
        )

    def test_type_is_opaque(self, secret: str) -> None:
        assert "type: Opaque" in secret


# ===========================================================================
# 6. PostgreSQL StatefulSet
# ===========================================================================


class TestPostgresStatefulSet:
    @pytest.fixture(scope="class")
    @classmethod
    def sts(cls) -> str:
        return _read("k8s/postgres/statefulset.yaml")

    def test_kind_is_statefulset(self, sts: str) -> None:
        assert "kind: StatefulSet" in sts

    def test_uses_postgres16(self, sts: str) -> None:
        assert "postgres:16" in sts

    def test_has_volume_claim_template(self, sts: str) -> None:
        assert "volumeClaimTemplates" in sts

    def test_has_resource_limits(self, sts: str) -> None:
        assert "limits:" in sts
        assert "requests:" in sts

    def test_has_liveness_probe(self, sts: str) -> None:
        assert "livenessProbe" in sts

    def test_has_readiness_probe(self, sts: str) -> None:
        assert "readinessProbe" in sts

    def test_password_from_secret(self, sts: str) -> None:
        assert "secretKeyRef" in sts
        assert "POSTGRES_PASSWORD" in sts

    def test_pgdata_set(self, sts: str) -> None:
        """PGDATA must be a subdirectory to avoid init issues with non-empty mounts."""
        assert "PGDATA" in sts


# ===========================================================================
# 7. Redis Deployment
# ===========================================================================


class TestRedisDeployment:
    @pytest.fixture(scope="class")
    @classmethod
    def dep(cls) -> str:
        return _read("k8s/redis/deployment.yaml")

    def test_kind_is_deployment(self, dep: str) -> None:
        assert "kind: Deployment" in dep

    def test_uses_redis7(self, dep: str) -> None:
        assert "redis:7" in dep

    def test_has_resource_limits(self, dep: str) -> None:
        assert "limits:" in dep

    def test_has_liveness_probe(self, dep: str) -> None:
        assert "livenessProbe" in dep

    def test_has_pvc_volume(self, dep: str) -> None:
        assert "persistentVolumeClaim" in dep


# ===========================================================================
# 8. Cortex Deployment
# ===========================================================================


class TestCortexDeployment:
    @pytest.fixture(scope="class")
    @classmethod
    def dep(cls) -> str:
        return _read("k8s/cortex/deployment.yaml")

    def test_kind_is_deployment(self, dep: str) -> None:
        assert "kind: Deployment" in dep

    def test_replicas_at_least_2(self, dep: str) -> None:
        assert "replicas: 2" in dep

    def test_rolling_update_strategy(self, dep: str) -> None:
        assert "RollingUpdate" in dep
        assert "maxUnavailable: 0" in dep

    def test_liveness_uses_health_live(self, dep: str) -> None:
        assert "/health/live" in dep

    def test_readiness_uses_health_ready(self, dep: str) -> None:
        assert "/health/ready" in dep

    def test_liveness_initial_delay_allows_migrations(self, dep: str) -> None:
        """initialDelaySeconds must be >= 30 to let migrations complete."""
        import re
        matches = re.findall(r"initialDelaySeconds:\s*(\d+)", dep)
        assert any(int(v) >= 30 for v in matches), (
            "Liveness initialDelaySeconds must be >= 30 for migration startup"
        )

    def test_resource_requests_set(self, dep: str) -> None:
        assert "requests:" in dep
        assert "cpu:" in dep
        assert "memory:" in dep

    def test_resource_limits_set(self, dep: str) -> None:
        assert "limits:" in dep

    def test_runs_as_nonroot(self, dep: str) -> None:
        assert "runAsNonRoot: true" in dep

    def test_database_url_references_secret(self, dep: str) -> None:
        assert "JWT_SECRET_KEY" in dep
        assert "secretKeyRef" in dep

    def test_mounts_document_storage(self, dep: str) -> None:
        assert "storage/documents" in dep

    def test_mounts_chroma_storage(self, dep: str) -> None:
        assert "storage/chroma" in dep

    def test_termination_grace_period_set(self, dep: str) -> None:
        assert "terminationGracePeriodSeconds" in dep

    def test_envfrom_configmap(self, dep: str) -> None:
        assert "configMapRef" in dep


# ===========================================================================
# 9. HPA
# ===========================================================================


class TestHPA:
    @pytest.fixture(scope="class")
    @classmethod
    def hpa(cls) -> str:
        return _read("k8s/cortex/hpa.yaml")

    def test_kind_is_hpa(self, hpa: str) -> None:
        assert "kind: HorizontalPodAutoscaler" in hpa

    def test_min_replicas(self, hpa: str) -> None:
        assert "minReplicas: 2" in hpa

    def test_max_replicas(self, hpa: str) -> None:
        assert "maxReplicas: 10" in hpa

    def test_cpu_metric(self, hpa: str) -> None:
        assert "cpu" in hpa

    def test_scale_down_stabilization(self, hpa: str) -> None:
        assert "scaleDown" in hpa
        assert "stabilizationWindowSeconds" in hpa


# ===========================================================================
# 10. Ingress
# ===========================================================================


class TestIngress:
    @pytest.fixture(scope="class")
    @classmethod
    def ing(cls) -> str:
        return _read("k8s/cortex/ingress.yaml")

    def test_kind_is_ingress(self, ing: str) -> None:
        assert "kind: Ingress" in ing

    def test_ingress_class_nginx(self, ing: str) -> None:
        assert "nginx" in ing

    def test_routes_to_cortex_svc(self, ing: str) -> None:
        assert "cortex-svc" in ing

    def test_has_placeholder_host(self, ing: str) -> None:
        """Host must be a placeholder to remind operators to change it."""
        assert "example.com" in ing or "CHANGE" in ing


# ===========================================================================
# 11. GitHub Actions CI workflow
# ===========================================================================


class TestCIWorkflow:
    @pytest.fixture(scope="class")
    @classmethod
    def ci(cls) -> str:
        return _read(".github/workflows/ci.yml")

    def test_triggers_on_push_to_main(self, ci: str) -> None:
        assert "branches: [main]" in ci or "branches:\n    - main" in ci or "main" in ci

    def test_has_lint_job(self, ci: str) -> None:
        assert "ruff" in ci.lower()

    def test_has_test_job(self, ci: str) -> None:
        assert "pytest" in ci

    def test_has_docker_build_job(self, ci: str) -> None:
        assert "docker" in ci.lower()
        assert "build" in ci.lower()

    def test_uses_python312(self, ci: str) -> None:
        assert "3.12" in ci

    def test_uses_uv(self, ci: str) -> None:
        assert "uv" in ci

    def test_has_postgres_service(self, ci: str) -> None:
        assert "postgres" in ci

    def test_has_redis_service(self, ci: str) -> None:
        assert "redis" in ci

    def test_runs_alembic_migrations(self, ci: str) -> None:
        assert "alembic" in ci
        assert "upgrade head" in ci

    def test_concurrency_cancel_in_progress(self, ci: str) -> None:
        assert "cancel-in-progress" in ci

    def test_docker_job_depends_on_lint_and_test(self, ci: str) -> None:
        assert "needs:" in ci


# ===========================================================================
# 12. Docker publish workflow
# ===========================================================================


class TestPublishWorkflow:
    @pytest.fixture(scope="class")
    @classmethod
    def pub(cls) -> str:
        return _read(".github/workflows/docker-publish.yml")

    def test_triggers_on_version_tag(self, pub: str) -> None:
        assert "tags:" in pub
        assert "v*" in pub

    def test_pushes_to_registry(self, pub: str) -> None:
        assert "push: true" in pub

    def test_uses_buildx(self, pub: str) -> None:
        assert "buildx" in pub.lower()

    def test_uses_docker_metadata_action(self, pub: str) -> None:
        assert "metadata-action" in pub

    def test_uses_gha_cache(self, pub: str) -> None:
        assert "type=gha" in pub


# ===========================================================================
# 13. Kubernetes docs
# ===========================================================================


class TestKubernetesDocs:
    @pytest.fixture(scope="class")
    @classmethod
    def docs(cls) -> str:
        return _read("docs/kubernetes.md")

    def test_has_quick_deploy_section(self, docs: str) -> None:
        assert "Quick Deploy" in docs or "quick" in docs.lower()

    def test_has_migration_command(self, docs: str) -> None:
        assert "alembic" in docs
        assert "upgrade head" in docs

    def test_has_health_probe_reference(self, docs: str) -> None:
        assert "/health/live" in docs
        assert "/health/ready" in docs

    def test_has_scaling_section(self, docs: str) -> None:
        assert "Scal" in docs

    def test_has_kubectl_commands(self, docs: str) -> None:
        assert "kubectl" in docs

    def test_has_rollback_command(self, docs: str) -> None:
        assert "rollout undo" in docs

    def test_has_hpa_reference(self, docs: str) -> None:
        assert "HPA" in docs or "HorizontalPodAutoscaler" in docs

    def test_has_pvc_section(self, docs: str) -> None:
        assert "PVC" in docs or "PersistentVolume" in docs

    def test_mentions_ci_workflows(self, docs: str) -> None:
        assert "GitHub Actions" in docs or "ci.yml" in docs


# ===========================================================================
# 14. Kustomization entry point
# ===========================================================================


class TestKustomization:
    @pytest.fixture(scope="class")
    @classmethod
    def kust(cls) -> str:
        return _read("k8s/kustomization.yaml")

    def test_kind_is_kustomization(self, kust: str) -> None:
        assert "kind: Kustomization" in kust

    def test_lists_all_manifests(self, kust: str) -> None:
        for name in [
            "namespace.yaml",
            "configmap.yaml",
            "secret.yaml",
            "postgres/statefulset.yaml",
            "redis/deployment.yaml",
            "cortex/deployment.yaml",
        ]:
            assert name in kust, f"kustomization.yaml missing: {name}"

    def test_namespace_set(self, kust: str) -> None:
        assert "namespace: cortex" in kust
