"""Benchmark: fetching Odoo MCP tools with vs without in-memory caching.

Measures the exact latency difference between:
1. WITHOUT cache: force_refresh=True (makes an HTTP tools/list round-trip to Odoo every turn)
2. WITH cache:    served from in-memory TTL cache (dict lookup)

Run from `agent/`:
    uv run python -m ont_agent.bench_tool_cache
    uv run python -m ont_agent.bench_tool_cache --iterations 5
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

from ont_agent.config import load_settings
from ont_agent.tools import clear_odoo_tools_cache, get_odoo_tools


async def benchmark(iterations: int) -> None:
    settings = load_settings()
    mcp_url = settings.mcp_url
    bearer = settings.mcp_bearer_key

    print("=" * 68)
    print(" MCP Tool Discovery Benchmark: Without Cache vs With In-Memory Cache")
    print("=" * 68)
    print(f"Target URL : {mcp_url}")
    print(f"Iterations : {iterations} runs each\n")

    # 1. Warm-up and count tools
    clear_odoo_tools_cache()
    warmup_tools = await get_odoo_tools(mcp_url, bearer, force_refresh=True)
    tool_count = len(warmup_tools)
    print(f"Discovered {tool_count} tools from Odoo MCP server.\n")

    # 2. Benchmark WITHOUT Cache (Force HTTP round-trip every time)
    without_times_ms: list[float] = []
    print("Running WITHOUT cache (live HTTP 'tools/list' request every run)...")
    for i in range(1, iterations + 1):
        t0 = time.perf_counter()
        tools = await get_odoo_tools(mcp_url, bearer, force_refresh=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        without_times_ms.append(elapsed_ms)
        print(f"  Run {i:2d}: {elapsed_ms:8.2f} ms ({len(tools)} tools)")

    # 3. Benchmark WITH Cache (In-memory lookup)
    with_times_ms: list[float] = []
    print("\nRunning WITH in-memory cache (cached in RAM, 0 HTTP requests)...")
    # Prime cache once
    clear_odoo_tools_cache()
    await get_odoo_tools(mcp_url, bearer)

    for i in range(1, iterations + 1):
        t0 = time.perf_counter()
        tools = await get_odoo_tools(mcp_url, bearer)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        with_times_ms.append(elapsed_ms)
        print(f"  Run {i:2d}: {elapsed_ms:8.4f} ms ({len(tools)} tools)")

    # 4. Summary Statistics
    avg_without = statistics.mean(without_times_ms)
    min_without = min(without_times_ms)
    max_without = max(without_times_ms)

    avg_with = statistics.mean(with_times_ms)
    min_with = min(with_times_ms)
    max_with = max(with_times_ms)

    diff_ms = avg_without - avg_with
    speedup = (avg_without / avg_with) if avg_with > 0 else float("inf")

    print("\n" + "=" * 68)
    print(" SUMMARY RESULTS")
    print("=" * 68)
    print(f"{'Metric':<25} | {'Without Cache (HTTP)':<20} | {'With Cache (RAM)':<18}")
    print("-" * 68)
    print(f"{'Average Latency':<25} | {avg_without:17.2f} ms | {avg_with:15.4f} ms")
    print(f"{'Min Latency':<25} | {min_without:17.2f} ms | {min_with:15.4f} ms")
    print(f"{'Max Latency':<25} | {max_without:17.2f} ms | {max_with:15.4f} ms")
    print("-" * 68)
    print(f"Latency Saved Per Turn : {diff_ms:.2f} ms eliminated on EVERY message")
    print(f"Speedup Factor         : {speedup:,.0f}x faster tool discovery")
    print("=" * 68)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MCP tool discovery caching")
    parser.add_argument("--iterations", "-n", type=int, default=5, help="Number of benchmark iterations")
    args = parser.parse_args()
    asyncio.run(benchmark(args.iterations))


if __name__ == "__main__":
    main()
