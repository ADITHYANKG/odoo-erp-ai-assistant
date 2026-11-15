"""End-to-end benchmark comparison for the query: "list all employees".

Measures the entire turn pipeline (Discovery + LLM + Odoo Tool Execution + Formatting):
1. WITHOUT tool caching (forces HTTP tools/list round-trip to Odoo on discovery)
2. WITH tool caching (in-memory tool definitions lookup)

Run from `agent/`:
    uv run python -m ont_agent.bench_query_comparison
    uv run python -m ont_agent.bench_query_comparison --query "list all employees" --runs 3
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass

from ont_agent.agent import ask, build_agent
from ont_agent.config import load_settings
from ont_agent.llm import build_model
from ont_agent.tools import clear_odoo_tools_cache, filter_read_only, get_odoo_tools


@dataclass
class TurnMeasurement:
    discovery_ms: float
    total_turn_ms: float
    llm_and_exec_ms: float
    answer_preview: str


async def run_single_turn(
    query: str,
    *,
    force_refresh_tools: bool,
) -> TurnMeasurement:
    settings = load_settings()

    # Phase 1: Tool Discovery
    t0_discovery = time.perf_counter()
    raw_tools = await get_odoo_tools(
        settings.mcp_url,
        settings.mcp_bearer_key,
        force_refresh=force_refresh_tools,
    )
    tools = filter_read_only(raw_tools)
    discovery_ms = (time.perf_counter() - t0_discovery) * 1000.0

    # Phase 2: Agent Build & LLM + Tool Execution
    t0_turn = time.perf_counter()
    agent = build_agent(build_model(settings), tools)
    result = await ask(agent, query)
    llm_and_exec_ms = (time.perf_counter() - t0_turn) * 1000.0

    total_turn_ms = discovery_ms + llm_and_exec_ms
    text = result.answer if hasattr(result, "answer") else str(result)
    preview = (text.strip().splitlines()[0] if text else "")[:60]

    return TurnMeasurement(
        discovery_ms=discovery_ms,
        total_turn_ms=total_turn_ms,
        llm_and_exec_ms=llm_and_exec_ms,
        answer_preview=preview,
    )


async def benchmark(query: str, runs: int) -> None:
    settings = load_settings()
    print("=" * 76)
    print(" END-TO-END TURN BENCHMARK: 'Without Tool Cache' vs 'With Tool Cache'")
    print("=" * 76)
    print(f"Target Query : \"{query}\"")
    print(f"Model        : {settings.openrouter_model}")
    print(f"MCP URL      : {settings.mcp_url}")
    print(f"Runs         : {runs} per condition\n")

    # Prime in-memory cache once before benchmark starts
    clear_odoo_tools_cache()
    await get_odoo_tools(settings.mcp_url, settings.mcp_bearer_key)

    without_results: list[TurnMeasurement] = []
    with_results: list[TurnMeasurement] = []

    print(f"--- Condition 1: WITHOUT Tool Cache (Live HTTP tools/list every turn) ---")
    for r in range(1, runs + 1):
        print(f"  Run {r} executing...", end="", flush=True)
        m = await run_single_turn(query, force_refresh_tools=True)
        without_results.append(m)
        print(f" Done! Total: {m.total_turn_ms:,.0f} ms (Discovery: {m.discovery_ms:,.1f} ms, LLM+Exec: {m.llm_and_exec_ms:,.0f} ms)")

    print(f"\n--- Condition 2: WITH Tool Cache (In-Memory Tool Discovery) ---")
    for r in range(1, runs + 1):
        print(f"  Run {r} executing...", end="", flush=True)
        m = await run_single_turn(query, force_refresh_tools=False)
        with_results.append(m)
        print(f" Done! Total: {m.total_turn_ms:,.0f} ms (Discovery: {m.discovery_ms:,.4f} ms, LLM+Exec: {m.llm_and_exec_ms:,.0f} ms)")

    # Aggregates
    avg_discovery_without = statistics.mean(m.discovery_ms for m in without_results)
    avg_discovery_with = statistics.mean(m.discovery_ms for m in with_results)

    avg_total_without = statistics.mean(m.total_turn_ms for m in without_results)
    avg_total_with = statistics.mean(m.total_turn_ms for m in with_results)

    avg_llm_without = statistics.mean(m.llm_and_exec_ms for m in without_results)
    avg_llm_with = statistics.mean(m.llm_and_exec_ms for m in with_results)

    saved_ms = avg_total_without - avg_total_with
    pct_turn_saved = (saved_ms / avg_total_without) * 100.0 if avg_total_without > 0 else 0.0

    print("\n" + "=" * 76)
    print(" SUMMARY FOR QUERY: 'list all employees'")
    print("=" * 76)
    print(f"{'Pipeline Stage':<30} | {'Without Cache':<18} | {'With Cache':<18}")
    print("-" * 76)
    print(f"{'1. Tool Discovery (tools/list)':<30} | {avg_discovery_without:14.2f} ms | {avg_discovery_with:14.4f} ms")
    print(f"{'2. LLM Turn + Odoo Tool Exec':<30} | {avg_llm_without:14.2f} ms | {avg_llm_with:14.2f} ms")
    print("-" * 76)
    print(f"{'TOTAL TURN DURATION':<30} | {avg_total_without:14.2f} ms | {avg_total_with:14.2f} ms")
    print("=" * 76)
    print(f"Latency Saved Per Query  : {saved_ms:,.2f} ms")
    print(f"End-to-End Speedup       : {pct_turn_saved:.1f}% faster overall response")
    print(f"Discovery Speedup        : {avg_discovery_without / (avg_discovery_with or 0.001):,.0f}x faster")
    print(f"Sample Answer Preview    : {with_results[0].answer_preview}")
    print("=" * 76)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end query benchmark with/without tool cache")
    parser.add_argument("--query", "-q", default="list all employees", help="User query to test")
    parser.add_argument("--runs", "-r", type=int, default=2, help="Number of runs per condition")
    args = parser.parse_args()
    asyncio.run(benchmark(args.query, args.runs))


if __name__ == "__main__":
    main()
