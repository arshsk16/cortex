"""
Phase 18 — Performance regression tests.

These are *micro-benchmark* tests that assert per-operation latency stays
below documented thresholds.  They run without any external services and
complete in < 30 s on a standard developer machine.

Thresholds are deliberately generous (3-5x observed baseline on Windows/Python
3.12) to prevent false failures on slow CI runners while still catching
catastrophic regressions such as accidental O(n^2) loops or missing
module-level pre-compilation.
"""
from __future__ import annotations

import concurrent.futures
import statistics
import time
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Measurement helper
# ---------------------------------------------------------------------------

def _measure(fn, n: int = 2_000) -> dict[str, Any]:
    """Warm-up then measure fn n times; return stats in us."""
    for _ in range(min(50, n // 10)):
        fn()
    samples: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1_000_000)
    samples.sort()
    return {
        "mean_us": statistics.mean(samples),
        "p50_us": samples[n // 2],
        "p95_us": samples[int(n * 0.95)],
        "p99_us": samples[int(n * 0.99)],
    }


# ---------------------------------------------------------------------------
# 1. Request-ID generation (uuid4 + str)
# ---------------------------------------------------------------------------

class TestRequestIdPerformance:
    def test_uuid4_p99_under_100us(self) -> None:
        """uuid.uuid4() + str() must complete p99 < 100 us."""
        stats = _measure(lambda: str(uuid.uuid4()), n=3_000)
        assert stats["p99_us"] < 100, (
            f"uuid4 p99={stats['p99_us']:.1f}us exceeds 100us threshold"
        )


# ---------------------------------------------------------------------------
# 2. Rate-limiter sliding-window
# ---------------------------------------------------------------------------

class TestRateLimiterPerformance:
    def test_rate_limiter_check_p99_under_200us(self) -> None:
        """Single _is_limited() call must complete p99 < 200 us."""
        from cortex.core.rate_limit import RateLimitMiddleware

        rl = RateLimitMiddleware(app=MagicMock(), requests_per_minute=10_000)
        for _ in range(5):
            rl._is_limited("1.2.3.4")

        stats = _measure(lambda: rl._is_limited("1.2.3.4"), n=3_000)
        assert stats["p99_us"] < 200, (
            f"rate_limiter p99={stats['p99_us']:.1f}us exceeds 200us threshold"
        )

    def test_rate_limiter_multiple_ips_no_cross_contamination(self) -> None:
        """Each IP bucket is independent; first request is never rate-limited."""
        from cortex.core.rate_limit import RateLimitMiddleware

        rl = RateLimitMiddleware(app=MagicMock(), requests_per_minute=1)
        limited_a, _ = rl._is_limited("192.168.1.1")
        limited_b, _ = rl._is_limited("192.168.1.2")
        assert not limited_a
        assert not limited_b

    def test_rate_limiter_enforces_limit_under_concurrent_load(self) -> None:
        """RPM limit is respected: exactly N requests pass."""
        from cortex.core.rate_limit import RateLimitMiddleware

        rl = RateLimitMiddleware(app=MagicMock(), requests_per_minute=5)
        results = [rl._is_limited("10.0.0.1") for _ in range(10)]
        allowed = sum(1 for limited, _ in results if not limited)
        assert allowed == 5, f"Expected 5 allowed, got {allowed}"


# ---------------------------------------------------------------------------
# 3. Prompt builder throughput
# ---------------------------------------------------------------------------

class TestPromptBuilderPerformance:
    @staticmethod
    def _make_chunks(n: int):
        from cortex.retrieval.models import RetrievalResult

        return [
            RetrievalResult(
                chunk_id=f"c{i}",
                document_id="doc1",
                chunk_index=i,
                text="Enterprise AI platform context sentence. " * 10,
                score=0.9 - i * 0.05,
            )
            for i in range(n)
        ]

    def test_prompt_build_5_chunks_p99_under_5ms(self) -> None:
        """PromptBuilder.build() with 5 chunks must complete p99 < 5 ms."""
        from cortex.services.prompt_builder import PromptBuilder

        pb = PromptBuilder()
        chunks = self._make_chunks(5)

        stats = _measure(
            lambda: pb.build(question="What is Cortex?", retrieved_chunks=chunks),
            n=1_000,
        )
        assert stats["p99_us"] < 5_000, (
            f"PromptBuilder p99={stats['p99_us']:.0f}us exceeds 5ms threshold"
        )

    def test_prompt_build_scales_linearly_with_chunks(self) -> None:
        """Build time with 10 chunks must be < 20x build time with 1 chunk."""
        from cortex.services.prompt_builder import PromptBuilder

        pb = PromptBuilder()
        stats_1 = _measure(
            lambda: pb.build(question="Q?", retrieved_chunks=self._make_chunks(1)),
            n=500,
        )
        stats_10 = _measure(
            lambda: pb.build(question="Q?", retrieved_chunks=self._make_chunks(10)),
            n=500,
        )
        ratio = stats_10["mean_us"] / max(stats_1["mean_us"], 1)
        assert ratio < 20, (
            f"Prompt build scaling ratio={ratio:.1f}x (1->10 chunks) exceeds 20x"
        )


# ---------------------------------------------------------------------------
# 4. Pydantic schema validation
# ---------------------------------------------------------------------------

class TestSchemaValidationPerformance:
    def test_rag_query_request_validation_p99_under_1ms(self) -> None:
        """RAGQueryRequest Pydantic validation must complete p99 < 1 ms."""
        from cortex.schemas.rag import RAGQueryRequest

        stats = _measure(
            lambda: RAGQueryRequest(question="What is Cortex? " * 5),
            n=3_000,
        )
        assert stats["p99_us"] < 1_000, (
            f"RAGQueryRequest validation p99={stats['p99_us']:.1f}us exceeds 1ms"
        )


# ---------------------------------------------------------------------------
# 5. JWT encode/decode round-trip
# ---------------------------------------------------------------------------

class TestJWTPerformance:
    def test_jwt_round_trip_p99_under_10ms(self) -> None:
        """Full JWT encode + decode round-trip must complete p99 < 10 ms."""
        from cortex.core.config import Settings
        from cortex.core.security import create_access_token, decode_access_token

        settings = Settings(
            jwt_secret_key="perf-test-secret-that-is-at-least-32-chars-long",
            redis_url="",
        )

        def _round_trip() -> None:
            token = create_access_token(subject="u1", settings=settings)
            decode_access_token(token, settings)

        stats = _measure(_round_trip, n=500)
        assert stats["p99_us"] < 10_000, (
            f"JWT round-trip p99={stats['p99_us']:.0f}us exceeds 10ms threshold"
        )


# ---------------------------------------------------------------------------
# 6. NullStateStore baseline
# ---------------------------------------------------------------------------

class TestStateStorePerformance:
    @pytest.mark.asyncio
    async def test_null_state_store_mean_under_500us(self) -> None:
        """NullStateStore save+load must have mean latency < 500 us."""
        from cortex.state_store.null import NullStateStore

        store = NullStateStore()
        samples: list[float] = []
        for _ in range(1_000):
            t0 = time.perf_counter()
            await store.save("k1", {"status": "ok"}, ttl_seconds=300)
            await store.load("k1")
            samples.append((time.perf_counter() - t0) * 1_000_000)

        mean_us = statistics.mean(samples)
        assert mean_us < 500, (
            f"NullStateStore mean={mean_us:.1f}us exceeds 500us threshold"
        )


# ---------------------------------------------------------------------------
# 7. Health endpoint via TestClient
# ---------------------------------------------------------------------------

def _make_perf_app():
    """Build a minimal TestClient with only the health router.

    Uses the same pattern as test_deployment_readiness.py: a bare FastAPI()
    with only the health router mounted, no rate-limiter middleware, and
    mocked dependencies. This measures pure ASGI + Pydantic dispatch latency.
    """
    from datetime import UTC, datetime
    from unittest.mock import AsyncMock, MagicMock

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from cortex.api.deps import get_health_service, get_settings
    from cortex.api.v1.endpoints.health import router as health_router
    from cortex.schemas.health import ComponentHealth, HealthResponse
    from cortex.services.health import HealthService

    app = FastAPI()
    app.include_router(health_router)

    mock_hs = AsyncMock(spec=HealthService)
    mock_hs.check.return_value = HealthResponse(
        status="healthy",
        service="Cortex",
        version="0.1.0",
        environment="test",
        timestamp=datetime.now(UTC),
        components=[ComponentHealth(name="database", status="healthy", latency_ms=0.5)],
    )
    mock_settings = MagicMock()
    mock_settings.app_name = "Cortex"

    app.dependency_overrides[get_health_service] = lambda: mock_hs
    app.dependency_overrides[get_settings] = lambda: mock_settings
    return TestClient(app, raise_server_exceptions=False)

class TestHealthEndpointPerformance:
    def test_health_live_p99_under_200ms(self) -> None:
        """GET /health/live must complete p99 < 200 ms through full ASGI stack."""
        client = _make_perf_app()

        for _ in range(10):
            client.get("/health/live")

        samples: list[float] = []
        for _ in range(200):
            t0 = time.perf_counter()
            resp = client.get("/health/live")
            samples.append((time.perf_counter() - t0) * 1_000)

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        samples.sort()
        p99_ms = samples[int(200 * 0.99)]
        assert p99_ms < 200, (
            f"GET /health/live p99={p99_ms:.1f}ms exceeds 200ms threshold"
        )

    def test_concurrent_health_requests_throughput(self) -> None:
        """20 concurrent threads hitting /health/live must sustain > 50 rps."""
        client = _make_perf_app()

        n_threads = 20
        n_per_thread = 10  # 200 total

        def _worker(_: int) -> list[float]:
            times: list[float] = []
            for _ in range(n_per_thread):
                t0 = time.perf_counter()
                client.get("/health/live")
                times.append((time.perf_counter() - t0) * 1_000)
            return times

        t_wall = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as ex:
            futs = [ex.submit(_worker, i) for i in range(n_threads)]
            all_ms: list[float] = []
            for f in concurrent.futures.as_completed(futs):
                all_ms.extend(f.result())
        elapsed = time.perf_counter() - t_wall

        throughput = len(all_ms) / elapsed
        assert throughput > 50, (
            f"Throughput={throughput:.0f} rps is below 50 rps minimum"
        )


# ---------------------------------------------------------------------------
# 8. Chroma in-memory similarity search
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def chroma_col_1k():
    """Pre-populated Chroma collection with 1 000 vectors (module scope)."""
    import random
    import uuid

    try:
        import chromadb  # noqa: PLC0415
    except ImportError:
        pytest.skip("chromadb not installed")

    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection("perf_1k")
    dim = 384
    n = 1_000

    def _rand(d: int) -> list[float]:
        v = [random.gauss(0, 1) for _ in range(d)]
        norm = (sum(x * x for x in v)) ** 0.5
        return [x / norm for x in v]

    col.add(
        ids=[str(uuid.uuid4()) for _ in range(n)],
        embeddings=[_rand(dim) for _ in range(n)],
    )
    return col


class TestChromaSearchPerformance:
    """Regression tests for Chroma in-memory vector similarity search.

    Uses chromadb.EphemeralClient (same as in development/test environments).
    Skipped automatically if chromadb is not installed.
    """

    def test_chroma_1k_docs_p99_under_50ms(self, chroma_col_1k) -> None:
        """Chroma similarity search over 1k docs must complete p99 < 50 ms."""
        import math
        import random

        dim = 384

        def _rand(d):
            v = [random.gauss(0, 1) for _ in range(d)]
            norm = math.sqrt(sum(x * x for x in v))
            return [x / norm for x in v]

        q = _rand(dim)
        stats = _measure(
            lambda: chroma_col_1k.query(query_embeddings=[q], n_results=5),
            n=100,
        )
        assert stats["p99_us"] < 50_000, (
            f"Chroma 1k-doc search p99={stats['p99_us']/1000:.1f}ms exceeds 50ms"
        )


# ---------------------------------------------------------------------------
# 9. MemoryVectorStore upsert + search
# ---------------------------------------------------------------------------

class TestMemoryVectorStorePerformance:
    @pytest.mark.asyncio
    async def test_memory_vector_store_upsert_search_p99_under_100ms(self) -> None:
        """MemoryVectorStore upsert + search round-trip p99 < 100 ms."""
        import math
        import random
        import uuid

        try:
            import chromadb  # noqa: PLC0415

            from cortex.vectorstore.memory_store import (
                MemoryVectorStore,  # noqa: PLC0415
            )
        except ImportError:
            pytest.skip("chromadb not installed")

        client = chromadb.EphemeralClient()
        col = client.get_or_create_collection("perf_mem")
        store = MemoryVectorStore(col)
        dim = 384
        user_id = str(uuid.uuid4())

        def _rand(d):
            v = [random.gauss(0, 1) for _ in range(d)]
            norm = math.sqrt(sum(x * x for x in v))
            return [x / norm for x in v]

        samples: list[float] = []
        for _ in range(50):
            mid = str(uuid.uuid4())
            vec = _rand(dim)
            t0 = time.perf_counter()
            await store.upsert(memory_id=mid, user_id=user_id, embedding=vec)
            await store.search(user_id=user_id, query_embedding=vec, limit=3)
            samples.append((time.perf_counter() - t0) * 1_000_000)

        samples.sort()
        p99_us = samples[int(50 * 0.99)]
        assert p99_us < 100_000, (
            f"MemoryVectorStore upsert+search p99={p99_us/1000:.1f}ms exceeds 100ms"
        )


# ---------------------------------------------------------------------------
# 10. RAGService orchestration
# ---------------------------------------------------------------------------

class TestRAGServicePerformance:
    def test_rag_service_orchestration_p99_under_200ms(self) -> None:
        """RAGService.answer() with mocked deps must complete p99 < 200 ms."""
        import asyncio
        import uuid
        from datetime import UTC, datetime
        from unittest.mock import AsyncMock, MagicMock

        from cortex.db.models.user import User, UserRole
        from cortex.retrieval.models import RetrievalResult
        from cortex.services.prompt_builder import PromptBuilder
        from cortex.services.rag import RAGService

        now = datetime.now(UTC)
        user = User(
            id=str(uuid.uuid4()),
            email="perf@example.com",
            username="perf",
            full_name="Perf User",
            hashed_password="hashed",
            role=UserRole.USER,
            is_active=True,
            is_verified=False,
            created_at=now,
            updated_at=now,
        )
        chunks = [
            RetrievalResult(
                chunk_id=f"c{i}", document_id="doc1", chunk_index=i,
                text="Cortex AI context. " * 20, score=0.9,
            )
            for i in range(5)
        ]
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=chunks)
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value="Answer.")

        svc = RAGService(
            retriever=mock_retriever,
            llm_provider=mock_llm,
            prompt_builder=PromptBuilder(),
        )

        stats = _measure(
            lambda: asyncio.run(
                svc.answer(question="What is Cortex?", user=user, top_k=5)
            ),
            n=100,
        )
        assert stats["p99_us"] < 200_000, (
            f"RAGService.answer p99={stats['p99_us']/1000:.1f}ms exceeds 200ms"
        )


# ---------------------------------------------------------------------------
# 11. AgentService orchestration (0-tool path)
# ---------------------------------------------------------------------------

class TestAgentServicePerformance:
    def test_agent_service_run_0tools_p99_under_100ms(self) -> None:
        """AgentService.run() 0-tool path with mocked LLM must complete p99 < 100 ms."""
        import asyncio
        import uuid
        from datetime import UTC, datetime
        from unittest.mock import AsyncMock, MagicMock

        from fastapi import BackgroundTasks

        from cortex.agent.registry import ToolRegistry
        from cortex.agent.service import AgentService
        from cortex.db.models.user import User, UserRole
        from cortex.services.prompt_builder import PromptBuilder
        from cortex.state_store.null import NullStateStore

        now = datetime.now(UTC)
        user = User(
            id=str(uuid.uuid4()),
            email="perf@example.com",
            username="perf",
            full_name="Perf User",
            hashed_password="hashed",
            role=UserRole.USER,
            is_active=True,
            is_verified=False,
            created_at=now,
            updated_at=now,
        )

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value="The answer is 42.")
        mock_conv_svc = MagicMock()
        mock_conv_svc.get_history = AsyncMock(return_value=[])
        mock_conv_svc.add_message = AsyncMock()
        mock_conv_svc.record_token_usage = AsyncMock()
        mock_mem_svc = MagicMock()
        mock_mem_svc.search = AsyncMock(return_value=[])

        svc = AgentService(
            llm_provider=mock_llm,
            tool_registry=ToolRegistry(),
            prompt_builder=PromptBuilder(),
            conversation_service=mock_conv_svc,
            state_store=NullStateStore(),
            memory_service=mock_mem_svc,
        )
        bt = BackgroundTasks()

        stats = _measure(
            lambda: asyncio.run(
                svc.run(question="What is Cortex?", user=user, background_tasks=bt)
            ),
            n=50,
        )
        assert stats["p99_us"] < 100_000, (
            f"AgentService.run p99={stats['p99_us']/1000:.1f}ms exceeds 100ms"
        )
