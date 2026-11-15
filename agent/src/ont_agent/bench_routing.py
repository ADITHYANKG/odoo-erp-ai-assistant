"""Benchmark: answering with vs without semantic routing.

    uv run python -m ont_agent.bench_routing --router-only
    uv run python -m ont_agent.bench_routing --runs 3

"Without" is the bot's full path for one message (fetch Odoo tools, build the
model and agent, run the LLM turn). "With" is the router decision, plus that
same full path only when the router lets the message through. The full path is
measured once per query and reused for both columns, so the comparison is fair
and each query costs one real LLM turn per run. Read-only tools; nothing is written.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import statistics
import time
from dataclasses import dataclass

from ont_agent.agent import ask, build_agent
from ont_agent.config import Settings, load_settings
from ont_agent.llm import build_model
from ont_agent.log import setup_logging
from ont_agent.router import AgentSemanticRouter
from ont_agent.tools import filter_read_only, get_odoo_tools

QUERIES: tuple[str, ...] = (
    "hello",
    "hey there, good afternoon",
    "good morning",
    "how are you doing",
    "tell me a joke",
    "write a poem about flowers",
    "what is the capital of france",
    "sing a song for me",
    "how many contacts do we have?",
    "list the five most recent sales orders",
    "which invoices are overdue?",
    "show employees in the HR department",
)

#: Router calls are microseconds-to-milliseconds, so one sample is mostly noise.
ROUTER_SAMPLES = 20


@dataclass
class Row:
    query: str
    route: str | None
    reply_is_static: bool
    router_ms: float
    agent_ms: float | None


def _router_ms(router: AgentSemanticRouter, query: str) -> tuple[str | None, float]:
    route: str | None = None
    samples: list[float] = []
    for _ in range(ROUTER_SAMPLES):
        started = time.perf_counter()
        route = router.route(query)
        samples.append((time.perf_counter() - started) * 1000)
    return route, statistics.median(samples)


async def _full_path_ms(settings: Settings, query: str) -> float:
    started = time.perf_counter()
    tools = filter_read_only(await get_odoo_tools(settings.mcp_url, settings.mcp_bearer_key))
    agent = build_agent(build_model(settings), tools)
    await ask(agent, query)
    return (time.perf_counter() - started) * 1000


_HEADER = f"{'query':<40} {'route':<13} {'without':>10} {'with':>10} {'speedup':>9}"


def _print_row(row: Row) -> None:
    route = row.route or "(agent)"
    if row.agent_ms is None:
        print(f"{row.query!r:<40} {route:<13} {'-':>10} {row.router_ms:>8.1f}ms {'-':>9}")
        return
    with_ms = row.router_ms if row.reply_is_static else row.router_ms + row.agent_ms
    print(
        f"{row.query!r:<40} {route:<13} {row.agent_ms:>8.0f}ms {with_ms:>8.0f}ms "
        f"{row.agent_ms / with_ms:>8.1f}x",
        flush=True,
    )


async def _measure(settings: Settings, runs: int, router_only: bool) -> list[Row]:
    router = AgentSemanticRouter(
        model_name=settings.semantic_router_model,
        score_threshold=settings.semantic_router_threshold,
    )
    started = time.perf_counter()
    router.route("warmup")  # builds the index; one-off cost, reported separately
    print(f"Router ready in {time.perf_counter() - started:.2f}s\n")
    print(_HEADER)
    print("-" * len(_HEADER), flush=True)

    rows: list[Row] = []
    for query in QUERIES:
        route, router_ms = _router_ms(router, query)
        agent_ms: float | None = None
        if not router_only:
            agent_ms = statistics.median(
                [await _full_path_ms(settings, query) for _ in range(runs)]
            )
        row = Row(query, route, router.get_static_reply(route) is not None, router_ms, agent_ms)
        rows.append(row)
        _print_row(row)  # stream: a full run takes minutes, so show progress as it goes
    return rows


def _summary(rows: list[Row]) -> None:
    timed = [row for row in rows if row.agent_ms is not None]
    if not timed:
        print("\nRouter-only run: no end-to-end comparison. Re-run without --router-only.")
        return
    short = [row for row in timed if row.reply_is_static]
    through = [row for row in timed if not row.reply_is_static]
    print()
    for label, group in (("short-circuited", short), ("fell through", through)):
        if not group:
            continue
        without = statistics.mean(r.agent_ms for r in group if r.agent_ms is not None)
        with_ = statistics.mean(
            r.router_ms + (0 if r.reply_is_static else (r.agent_ms or 0)) for r in group
        )
        print(f"{label:<16} n={len(group):<3} without={without:>7.0f}ms  with={with_:>7.0f}ms")
    if through:
        overhead = statistics.mean(r.router_ms for r in through)
        print(f"Router overhead on fall-through queries: {overhead:.1f}ms")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1, help="full-path runs per query (median)")
    parser.add_argument("--router-only", action="store_true", help="skip the LLM; time routing")
    parser.add_argument("--verbose", action="store_true", help="keep the per-turn INFO logs")
    args = parser.parse_args()
    if not args.verbose:
        logging.getLogger("ont_agent").setLevel(logging.WARNING)
        logging.getLogger("ont_agent.router").setLevel(logging.INFO)
    settings = load_settings(require_llm=not args.router_only)
    try:
        rows = asyncio.run(_measure(settings, max(1, args.runs), args.router_only))
    except Exception as exc:
        raise SystemExit(f"Benchmark failed: {exc!r}") from exc
    _summary(rows)


if __name__ == "__main__":
    main()
