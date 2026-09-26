"""
Service-layer benchmarks for Cortex.

All benchmarks use in-process fakes/mocks; no external services are required:

*  Chroma:    ephemeral in-memory client (chromadb.EphemeralClient)
*  Retriever: not used by AgentService directly (it lives inside RAGSearchTool)
*  LLM:       AsyncMock returning a canned string
*  PostgreSQL: NOT benchmarked — requires a live server
*  Redis:      NOT benchmarked — requires a live server
*  Real embeddings: NOT benchmarked — require GPU/model download

Provides reproducible baselines for:
  - Chroma in-memory vector similarity search (1 k and 10 k docs)
  - MemoryVectorStore upsert + search round-trip
  - RAGService orchestration overhead (mocked retriever + LLM)
  - AgentService.run orchestration overhead (0-tool path, mocked LLM)
"""
from __future__ import annotations

import asyncio
import math
import random
import statistics
import time
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run a coroutine synchronously using asyncio.run()."""
    return asyncio.run(coro)


def _measure(fn, n: int = 200) -> dict[str, Any]:
    """Warm-up then measure fn n times; return latency stats in ms."""
    for _ in range(min(20, n // 5)):
        fn()
    samples: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1_000)
    samples.sort()
    return {
        "n": n,
        "mean_ms": statistics.mean(samples),
        "p50_ms": samples[n // 2],
        "p95_ms": samples[int(n * 0.95)],
        "p99_ms": samples[int(n * 0.99)],
    }


def _rand_vector(dim: int = 384) -> list[float]:
    """Return a unit-normalised random vector of dimension dim."""
    v = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


# ---------------------------------------------------------------------------
# 1. Chroma in-memory similarity search — 1 000 docs
# ---------------------------------------------------------------------------

def bench_chroma_similarity_search_1k() -> dict[str, Any]:
    """Chroma ephemeral similarity search over 1 000 pre-inserted vectors."""
    try:
        import chromadb  # noqa: PLC0415
    except ImportError:
        return {
            "name": "chroma_similarity_search_1k",
            "skipped": "chromadb not installed",
        }

    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection("bench_1k")
    dim = 384
    n_docs = 1_000
    col.add(
        ids=[str(uuid.uuid4()) for _ in range(n_docs)],
        embeddings=[_rand_vector(dim) for _ in range(n_docs)],
    )
    q = _rand_vector(dim)
    result = _measure(lambda: col.query(query_embeddings=[q], n_results=5), n=200)
    result["name"] = "chroma_similarity_search_1k"
    result["n_docs"] = n_docs
    return result


# ---------------------------------------------------------------------------
# 2. Chroma in-memory similarity search — 10 000 docs
# ---------------------------------------------------------------------------

def bench_chroma_similarity_search_10k() -> dict[str, Any]:
    """Chroma ephemeral similarity search over 10 000 pre-inserted vectors."""
    try:
        import chromadb  # noqa: PLC0415
    except ImportError:
        return {
            "name": "chroma_similarity_search_10k",
            "skipped": "chromadb not installed",
        }

    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection("bench_10k")
    dim = 384
    n_docs = 10_000
    batch = 1_000
    for _start in range(0, n_docs, batch):
        col.add(
            ids=[str(uuid.uuid4()) for _ in range(batch)],
            embeddings=[_rand_vector(dim) for _ in range(batch)],
        )
    q = _rand_vector(dim)
    result = _measure(lambda: col.query(query_embeddings=[q], n_results=5), n=100)
    result["name"] = "chroma_similarity_search_10k"
    result["n_docs"] = n_docs
    return result


# ---------------------------------------------------------------------------
# 3. MemoryVectorStore upsert + search (in-process Chroma)
# ---------------------------------------------------------------------------

def bench_memory_vector_store_upsert_search() -> dict[str, Any]:
    """MemoryVectorStore.upsert() + .search() round-trip using in-process Chroma."""
    try:
        import chromadb  # noqa: PLC0415

        from cortex.vectorstore.memory_store import MemoryVectorStore  # noqa: PLC0415
    except ImportError as exc:
        return {"name": "memory_vector_store_upsert_search", "skipped": str(exc)}

    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection("bench_mem")
    store = MemoryVectorStore(col)
    dim = 384
    user_id = str(uuid.uuid4())

    async def _round_trip() -> None:
        mid = str(uuid.uuid4())
        vec = _rand_vector(dim)
        await store.upsert(memory_id=mid, user_id=user_id, embedding=vec)
        await store.search(user_id=user_id, query_embedding=vec, limit=3)

    result = _measure(lambda: _run(_round_trip()), n=100)
    result["name"] = "memory_vector_store_upsert_search"
    return result


# ---------------------------------------------------------------------------
# 4. RAGService.answer orchestration overhead (mocked retriever + LLM)
# ---------------------------------------------------------------------------

def bench_rag_service_orchestration() -> dict[str, Any]:
    """RAGService.answer() total latency with instant mocked retriever and LLM.

    Isolates: question validation, prompt assembly, timer bookkeeping.
    Excludes: real retrieval, real LLM, DB writes.
    """
    from datetime import UTC, datetime  # noqa: PLC0415

    from cortex.db.models.user import User, UserRole  # noqa: PLC0415
    from cortex.retrieval.models import RetrievalResult  # noqa: PLC0415
    from cortex.services.prompt_builder import PromptBuilder  # noqa: PLC0415
    from cortex.services.rag import RAGService  # noqa: PLC0415

    now = datetime.now(UTC)
    user = User(
        id=str(uuid.uuid4()),
        email="bench@example.com",
        username="bench",
        full_name="Bench User",
        hashed_password="hashed",
        role=UserRole.USER,
        is_active=True,
        is_verified=False,
        created_at=now,
        updated_at=now,
    )
    chunks = [
        RetrievalResult(
            chunk_id=f"c{i}",
            document_id="doc1",
            chunk_index=i,
            text="Enterprise AI platform content. " * 10,
            score=0.9 - i * 0.05,
        )
        for i in range(5)
    ]
    mock_retriever = MagicMock()
    mock_retriever.retrieve = AsyncMock(return_value=chunks)
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value="Mocked answer.")

    svc = RAGService(
        retriever=mock_retriever,
        llm_provider=mock_llm,
        prompt_builder=PromptBuilder(),
    )

    async def _answer() -> None:
        await svc.answer(question="What is Cortex?", user=user, top_k=5)

    result = _measure(lambda: _run(_answer()), n=200)
    result["name"] = "rag_service_answer_mocked"
    return result


# ---------------------------------------------------------------------------
# 5. AgentService.run — 0-tool path, all deps mocked
# ---------------------------------------------------------------------------

def bench_agent_service_run_0_tools() -> dict[str, Any]:
    """AgentService.run() latency: 0-tool path, LLM returns plain text.

    Isolates: state construction, security gate, prompt building, result assembly.
    Excludes: real LLM, DB I/O, tool execution.

    AgentService does not accept a 'retriever' directly — the retriever lives
    inside RAGSearchTool.  An empty ToolRegistry means no tools are registered.
    """
    from datetime import UTC, datetime  # noqa: PLC0415

    from fastapi import BackgroundTasks  # noqa: PLC0415

    from cortex.agent.registry import ToolRegistry  # noqa: PLC0415
    from cortex.agent.service import AgentService  # noqa: PLC0415
    from cortex.db.models.user import User, UserRole  # noqa: PLC0415
    from cortex.services.prompt_builder import PromptBuilder  # noqa: PLC0415
    from cortex.state_store.null import NullStateStore  # noqa: PLC0415

    now = datetime.now(UTC)
    user = User(
        id=str(uuid.uuid4()),
        email="bench@example.com",
        username="bench",
        full_name="Bench User",
        hashed_password="hashed",
        role=UserRole.USER,
        is_active=True,
        is_verified=False,
        created_at=now,
        updated_at=now,
    )

    # LLM returns plain text (not a SupportsToolCalling instance)
    # so the service takes the prompt-based fallback path.
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value="The answer is 42.")
    # Explicitly remove generate_with_tools so isinstance check fails
    if hasattr(mock_llm, "generate_with_tools"):
        del mock_llm.generate_with_tools

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

    async def _run_agent() -> None:
        await svc.run(
            question="What is Cortex?",
            user=user,
            background_tasks=bt,
        )

    result = _measure(lambda: _run(_run_agent()), n=100)
    result["name"] = "agent_service_run_0tools_mocked"
    return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_BENCHMARKS = [
    bench_chroma_similarity_search_1k,
    bench_chroma_similarity_search_10k,
    bench_memory_vector_store_upsert_search,
    bench_rag_service_orchestration,
    bench_agent_service_run_0_tools,
]


def run_all() -> list[dict]:
    results = []
    for bench_fn in ALL_BENCHMARKS:
        print(f"  Running {bench_fn.__name__}...", flush=True)
        try:
            r = bench_fn()
            results.append(r)
            if "skipped" in r:
                print(f"    SKIPPED: {r['skipped']}")
            else:
                extra = f"  n_docs={r['n_docs']}" if "n_docs" in r else ""
                print(
                    f"    mean={r['mean_ms']:.2f}ms  "
                    f"p95={r['p95_ms']:.2f}ms  "
                    f"p99={r['p99_ms']:.2f}ms"
                    + extra
                )
        except Exception as exc:  # noqa: BLE001
            print(f"    ERROR ({bench_fn.__name__}): {exc}")
    return results


if __name__ == "__main__":
    print("=== Service-layer benchmarks ===")
    run_all()
