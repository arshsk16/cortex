#!/usr/bin/env python
"""
Run all Cortex benchmarks and save results to benchmarks/results/baseline.json.

Usage:
    uv run python -m benchmarks.run_all
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

# Ensure src/ is on the path when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmarks.bench_api import run_all as run_api
from benchmarks.bench_components import run_all as run_components
from benchmarks.bench_services import run_all as run_services


def main() -> None:
    print("\n" + "=" * 60)
    print("Cortex Performance Benchmark Suite")
    print("=" * 60)

    print("\n[1/3] Component benchmarks (no external services)")
    component_results = run_components()

    print("\n[2/3] API / HTTP benchmarks (TestClient)")
    api_results = run_api()

    print("\n[3/3] Service-layer benchmarks (Chroma in-process, mocked LLM)")
    service_results = run_services()

    all_results = component_results + api_results + service_results

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "results": all_results,
    }

    out_path = Path(__file__).parent / "results" / "baseline.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\nResults saved to: {out_path}")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in all_results:
        if "skipped" in r:
            continue
        name = r.get("name", "?")
        if "mean_ms" in r:
            extra = f"  n_docs={r['n_docs']}" if "n_docs" in r else ""
            print(
                f"  {name:<50} mean={r['mean_ms']:.2f}ms"
                f"  p99={r['p99_ms']:.2f}ms" + extra
            )
        elif "mean_us" in r:
            print(
                f"  {name:<50} mean={r['mean_us']:.1f}\u00b5s"
                f"  p99={r['p99_us']:.1f}\u00b5s"
            )


if __name__ == "__main__":
    main()
