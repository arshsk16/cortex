"""
HTTP-level benchmarks via Starlette TestClient.

Measures end-to-end request latency for key API endpoints including
middleware overhead (rate limiting, request-ID injection, sanitisation).
No external services required.
"""
from __future__ import annotations

import statistics
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from cortex.core.config import Settings


def _make_test_client() -> tuple[TestClient, str]:
    """Build a minimal TestClient with JWT auth token."""
    from cortex.core.security import create_access_token
    from cortex.main import create_app

    settings = Settings(
        database_url="postgresql+asyncpg://u:p@localhost/bench",
        jwt_secret_key="bench-secret-key-that-is-at-least-32-chars-xx",
        redis_url="",
    )

    app = create_app()

    # Attach required app.state objects
    app.state.settings = settings
    app.state.database = MagicMock()
    app.state.embedding_provider = MagicMock()
    app.state.vector_store = MagicMock()
    app.state.llm_provider = MagicMock()
    app.state.state_store = MagicMock()

    token = create_access_token(subject="bench-user", settings=settings)
    client = TestClient(app, raise_server_exceptions=False)
    return client, token


def _measure_http(
    fn,
    n: int = 500,
    warmup: int = 20,
) -> dict[str, Any]:
    """Run fn(client) n times; return latency stats in milliseconds."""
    client, token = _make_test_client()

    # warm-up
    for _ in range(warmup):
        fn(client, token)

    samples: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(client, token)
        samples.append((time.perf_counter() - t0) * 1_000)

    samples.sort()
    return {
        "n": n,
        "mean_ms": statistics.mean(samples),
        "p50_ms": samples[n // 2],
        "p95_ms": samples[int(n * 0.95)],
        "p99_ms": samples[int(n * 0.99)],
    }


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_health_live() -> dict[str, Any]:
    def _req(client, _token):
        client.get("/health/live")

    result = _measure_http(_req)
    result["name"] = "GET /health/live"
    return result


def bench_health_ready_mocked() -> dict[str, Any]:
    from cortex.services.health import HealthService

    mock_svc = MagicMock(spec=HealthService)
    mock_svc.check_database = AsyncMock(return_value=True)
    mock_svc.check_vector_store = AsyncMock(return_value=True)

    def _req(client, _token):
        path = "cortex.api.v1.endpoints.health.get_health_service"
        with patch(path, return_value=mock_svc):
            client.get("/health/ready")

    result = _measure_http(_req, n=200)
    result["name"] = "GET /health/ready (mocked)"
    return result


def bench_auth_me_unauthenticated() -> dict[str, Any]:
    """Measures 401 fast-path through auth middleware."""
    def _req(client, _token):
        client.get("/api/v1/auth/me")

    result = _measure_http(_req)
    result["name"] = "GET /auth/me (401 path)"
    return result


def bench_concurrent_health_requests() -> dict[str, Any]:
    """Simulate 20 concurrent requests using threading (TestClient is sync)."""
    import concurrent.futures

    client, _ = _make_test_client()
    n_threads = 20
    n_requests = 200

    def _worker(_: int) -> list[float]:
        times = []
        for _ in range(n_requests // n_threads):
            t0 = time.perf_counter()
            client.get("/health/live")
            times.append((time.perf_counter() - t0) * 1_000)
        return times

    t_wall_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = [ex.submit(_worker, i) for i in range(n_threads)]
        all_times = []
        for f in concurrent.futures.as_completed(futures):
            all_times.extend(f.result())
    t_wall = time.perf_counter() - t_wall_start

    all_times.sort()
    n = len(all_times)
    throughput = n / t_wall
    return {
        "name": f"GET /health/live ({n_threads} concurrent threads)",
        "n": n,
        "mean_ms": statistics.mean(all_times),
        "p50_ms": all_times[n // 2],
        "p95_ms": all_times[int(n * 0.95)],
        "p99_ms": all_times[int(n * 0.99)],
        "throughput_rps": round(throughput, 1),
    }


ALL_BENCHMARKS = [
    bench_health_live,
    bench_health_ready_mocked,
    bench_auth_me_unauthenticated,
    bench_concurrent_health_requests,
]


def run_all() -> list[dict]:
    results = []
    for bench_fn in ALL_BENCHMARKS:
        print(f"  Running {bench_fn.__name__}...", flush=True)
        try:
            r = bench_fn()
            results.append(r)
            mean_key = "mean_ms" if "mean_ms" in r else "mean_us"
            p99_key = "p99_ms" if "p99_ms" in r else "p99_us"
            unit = "ms" if mean_key == "mean_ms" else "µs"
            print(
                f"    mean={r[mean_key]:.2f}{unit}  "
                f"p99={r[p99_key]:.2f}{unit}"
                + (
                    f"  throughput={r['throughput_rps']} rps"
                    if "throughput_rps" in r
                    else ""
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    SKIPPED ({exc})")
    return results


if __name__ == "__main__":
    print("=== API benchmarks ===")
    run_all()
