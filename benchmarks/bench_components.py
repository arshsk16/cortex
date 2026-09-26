"""
Component-level micro-benchmarks for Cortex.

Each benchmark function returns a dict:
  {"name": str, "n": int, "mean_us": float, "p50_us": float,
   "p95_us": float, "p99_us": float}

All measurements are in microseconds (µs).
No external services required.
"""
from __future__ import annotations

import statistics
import time
import uuid
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _measure(fn, n: int = 5_000) -> dict[str, Any]:
    """Run fn n times; return latency statistics in microseconds."""
    # warm-up
    for _ in range(min(50, n // 10)):
        fn()

    samples: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1_000_000)

    samples.sort()
    return {
        "n": n,
        "mean_us": statistics.mean(samples),
        "p50_us": samples[n // 2],
        "p95_us": samples[int(n * 0.95)],
        "p99_us": samples[int(n * 0.99)],
    }


# ---------------------------------------------------------------------------
# 1. UUID / request-ID generation
# ---------------------------------------------------------------------------

def bench_request_id_generation() -> dict[str, Any]:
    result = _measure(lambda: str(uuid.uuid4()))
    result["name"] = "request_id_generation"
    return result


# ---------------------------------------------------------------------------
# 2. Rate-limiter sliding-window check
# ---------------------------------------------------------------------------

def bench_rate_limiter_check() -> dict[str, Any]:
    from unittest.mock import MagicMock

    from cortex.core.rate_limit import RateLimitMiddleware

    rl = RateLimitMiddleware(app=MagicMock(), requests_per_minute=1000)
    # Populate a bucket so eviction also fires.
    rl._is_limited("127.0.0.1")

    result = _measure(lambda: rl._is_limited("127.0.0.1"))
    result["name"] = "rate_limiter_check"
    return result


# ---------------------------------------------------------------------------
# 3. Control-character sanitisation
# ---------------------------------------------------------------------------

def bench_sanitise_input() -> dict[str, Any]:
    from cortex.core.middleware import (
        sanitise_input,  # type: ignore[attr-defined]  # noqa: F401
    )

    sample = (
        "Hello \x00world\x01! This is a \x1b[31mcoloured\x1b[0m test "
        "string with some normal text mixed in.\n"
    ) * 3

    result = _measure(lambda: sanitise_input(sample))
    result["name"] = "sanitise_input"
    return result


# ---------------------------------------------------------------------------
# 4. Prompt building (PromptBuilder.build)
# ---------------------------------------------------------------------------

def bench_prompt_builder() -> dict[str, Any]:
    from cortex.retrieval.models import RetrievalResult
    from cortex.services.prompt_builder import PromptBuilder

    pb = PromptBuilder()
    chunks = [
        RetrievalResult(
            chunk_id=f"c{i}",
            document_id="doc1",
            content="Cortex is an enterprise AI platform. " * 20,
            score=0.9 - i * 0.05,
            metadata={},
        )
        for i in range(5)
    ]

    result = _measure(
        lambda: pb.build(question="What is Cortex?", retrieved_chunks=chunks),
        n=2_000,
    )
    result["name"] = "prompt_builder_5_chunks"
    return result


# ---------------------------------------------------------------------------
# 5. Pydantic schema validation
# ---------------------------------------------------------------------------

def bench_schema_validation() -> dict[str, Any]:
    from cortex.schemas.rag import RAGRequest

    payload = {"question": "What is Cortex? " * 10}
    result = _measure(lambda: RAGRequest(**payload), n=5_000)
    result["name"] = "schema_validation_rag_request"
    return result


# ---------------------------------------------------------------------------
# 6. JWT encode + decode round-trip
# ---------------------------------------------------------------------------

def bench_jwt_round_trip() -> dict[str, Any]:
    from cortex.core.config import Settings
    from cortex.core.security import create_access_token, decode_access_token

    settings = Settings(
        jwt_secret_key="bench-secret-key-that-is-at-least-32-chars",
        redis_url="",
    )

    def _round_trip() -> None:
        token = create_access_token(subject="user-123", settings=settings)
        decode_access_token(token, settings)

    result = _measure(_round_trip, n=1_000)
    result["name"] = "jwt_encode_decode"
    return result


# ---------------------------------------------------------------------------
# 7. NullStateStore save/load (baseline for Redis path)
# ---------------------------------------------------------------------------

def bench_null_state_store() -> dict[str, Any]:
    import asyncio

    from cortex.state_store.null import NullStateStore

    store = NullStateStore()

    async def _run() -> None:
        await store.save("key1", {"status": "ok", "data": list(range(100))})
        await store.load("key1")

    result = _measure(
        lambda: asyncio.get_event_loop().run_until_complete(_run()), n=2_000
    )
    result["name"] = "null_state_store_save_load"
    return result


# ---------------------------------------------------------------------------
# 8. Concurrent rate-limiter checks (simulated concurrency)
# ---------------------------------------------------------------------------

def bench_rate_limiter_concurrent() -> dict[str, Any]:
    """Simulate 10 distinct IPs hitting the limiter simultaneously."""
    import asyncio
    from unittest.mock import MagicMock

    from cortex.core.rate_limit import RateLimitMiddleware

    rl = RateLimitMiddleware(app=MagicMock(), requests_per_minute=500)
    ips = [f"10.0.0.{i}" for i in range(10)]

    async def _burst() -> None:
        for ip in ips:
            rl._is_limited(ip)

    result = _measure(
        lambda: asyncio.get_event_loop().run_until_complete(_burst()),
        n=1_000,
    )
    result["name"] = "rate_limiter_10ip_burst"
    return result


# ---------------------------------------------------------------------------
# Run all component benchmarks
# ---------------------------------------------------------------------------

ALL_BENCHMARKS = [
    bench_request_id_generation,
    bench_rate_limiter_check,
    bench_prompt_builder,
    bench_schema_validation,
    bench_jwt_round_trip,
    bench_null_state_store,
    bench_rate_limiter_concurrent,
]

# Only add sanitise_input if the function exists in middleware
try:
    from cortex.core.middleware import (
        sanitise_input,  # type: ignore[attr-defined]  # noqa: F401
    )
    ALL_BENCHMARKS.insert(2, bench_sanitise_input)
except (ImportError, AttributeError):
    pass


def run_all() -> list[dict]:
    results = []
    for bench_fn in ALL_BENCHMARKS:
        print(f"  Running {bench_fn.__name__}...", flush=True)
        try:
            r = bench_fn()
            results.append(r)
            print(
                f"    mean={r['mean_us']:.1f}µs  "
                f"p50={r['p50_us']:.1f}µs  "
                f"p95={r['p95_us']:.1f}µs  "
                f"p99={r['p99_us']:.1f}µs"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    SKIPPED ({exc})")
    return results


if __name__ == "__main__":
    print("=== Component benchmarks ===")
    run_all()
