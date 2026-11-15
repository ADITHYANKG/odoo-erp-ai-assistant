"""Per-call latency: how long each model actually takes, recorded for real.

The configured model's speed is a standing question — ``google/gemini-3.7-flash``
measured 25.9s to first token on 2026-09-01 and 3.3s on another live request,
and nothing in the service was writing either number down. Model choice, the
watcher interval and the fallback list all hinge on latency, so it needs to be
evidence rather than a benchmark somebody remembers to run.

One row per model call, in ``llm_calls``. Three properties hold it up:

* **Time to first token is NOT measurable on this side.** The agent streams
  graph state, not tokens, so a callback pair around a call yields its TOTAL
  duration and nothing finer. The real figure comes from OpenRouter's
  ``GET /generation``, whose ``latency`` field is exactly that. It returns
  metadata, not inference, so it costs no tokens.
* **It is never on the answer's critical path.** Samples are queued and a
  background worker does the fetching and the writing, well after the reply
  has gone out. Measurement that made every answer slower would be
  self-defeating on a bot whose latency is the thing under investigation.
* **Both halves degrade independently.** A failed lookup still writes the row
  with our own wall clock; a full queue drops the sample and says so. Losing
  samples is the correct failure for a measurement table — the alternative is
  growing memory while OpenRouter is unreachable.

Deliberately NOT part of ``tracing.py``: that module is Langfuse-specific and
returns ``None`` when no keys are configured. This has to work regardless, the
same argument ``budget.py`` already makes for spend.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID

import httpx
from langchain_core.callbacks import AsyncCallbackHandler

from ont_agent.config import Settings, load_settings
from ont_agent.crypto import TokenCipher
from ont_agent.log import get_logger, setup_logging
from ont_agent.store import BindingStore, LatencyRow, LlmCall

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult

logger = get_logger(__name__)

#: What kind of call this was. Chat questions and unattended work have very
#: different prompt shapes, and mixing them would blur both medians.
KIND_CHAT = "chat"
KIND_BRIEF = "brief"
KIND_WATCHER = "watcher"


@dataclass(frozen=True)
class GenerationStats:
    """OpenRouter's own timings for one generation, measured at their edge.

    ``ttft_ms`` is their ``latency`` — time to first token. ``generation_ms``
    is their ``generation_time``, **the whole request**, so the decode
    duration is the difference between the two, not ``generation_ms`` itself.

    That was established from live traffic on 2026-09-10 after being got
    wrong first. OpenRouter's OpenAPI spec illustrates the two fields with
    ``latency: 1250`` against ``generation_time: 1200`` — a total smaller
    than its own prefix, which reads as proof that ``generation_time``
    excludes the wait. It is not proof: those are hand-written placeholders,
    not a captured request. In four real calls ``latency + generation_time``
    came to 4376-12138ms while OUR OWN wall clock around the entire HTTP
    call was 3350-6951ms. A provider cannot spend longer than the client
    waited, so the decode reading is impossible. ``generation_time`` alone
    was consistently just under our clock, the remaining 464-1494ms being
    our network hop and client overhead — which is what these two columns
    are for: ``total_ms`` minus ``generation_ms`` is the cost of everything
    that is not the model.
    """

    ttft_ms: float | None = None
    generation_ms: float | None = None
    provider: str = ""


def extract_call(response: LLMResult, kind: str, total_ms: float) -> LlmCall | None:
    """Read one finished call out of the callback's ``LLMResult``.

    **The generation id is only reachable HERE, never from ``ainvoke``'s
    return value.** ``ChatOpenRouter`` does not override
    ``_combine_llm_outputs``, whose base implementation returns ``{}``, so the
    ``LLMResult`` handed back to the caller has an empty ``llm_output``.
    Callbacks are given ``flattened_output`` instead, built per result as
    ``LLMResult(generations=..., llm_output=res.llm_output)`` — which keeps
    it. Reading the id from the wrong one yields nothing, silently.

    Returns ``None`` for a response carrying no model name: without knowing
    WHICH model answered, the row cannot be compared against anything, and a
    latency table of unattributed numbers is worse than a shorter one.
    """
    output = getattr(response, "llm_output", None) or {}
    try:
        generation = response.generations[-1][-1]
    except (AttributeError, IndexError, TypeError):
        return None
    message = getattr(generation, "message", None)
    metadata = getattr(message, "response_metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}

    # Both carry the id and the model name — verified live 2026-09-10, where
    # the callback's llm_output held {"model_name", "id", "created", "object"}
    # and response_metadata held the same plus the cost. Reading either means
    # a change on one side cannot silently empty the table.
    #
    # The model that ANSWERED, not the configured one: with fallbacks these
    # differ exactly when the difference is worth knowing about.
    model = str(output.get("model_name") or metadata.get("model_name") or "")
    if not model:
        return None

    usage = getattr(message, "usage_metadata", None)
    if not isinstance(usage, dict):
        usage = {}
    details = usage.get("input_token_details")
    cached = details.get("cache_read") if isinstance(details, dict) else 0
    cost = metadata.get("cost")
    provider = metadata.get("provider")
    return LlmCall(
        model=model,
        kind=kind,
        total_ms=total_ms,
        generation_id=str(output.get("id") or metadata.get("id") or ""),
        provider=provider if isinstance(provider, str) else "",
        prompt_tokens=int(usage.get("input_tokens") or 0),
        completion_tokens=int(usage.get("output_tokens") or 0),
        cached_tokens=int(cached or 0),
        cost_usd=float(cost) if isinstance(cost, int | float) else 0.0,
    )


def parse_generation_stats(payload: Any) -> GenerationStats:
    """Map ``/generation``'s body onto the two timings we keep.

    Tolerant on purpose: a missing or renamed field leaves that half ``None``
    rather than raising, so the row still lands with the wall clock intact.
    """
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return GenerationStats()

    def number(key: str) -> float | None:
        value = data.get(key)
        return float(value) if isinstance(value, int | float) else None

    provider = data.get("provider_name")
    return GenerationStats(
        ttft_ms=number("latency"),
        generation_ms=number("generation_time"),
        provider=provider if isinstance(provider, str) else "",
    )


async def fetch_generation_stats(
    settings: Settings,
    generation_id: str,
    client: httpx.AsyncClient | None = None,
    sleep: Any = asyncio.sleep,
) -> GenerationStats | None:
    """Ask OpenRouter for its own timings. Never raises; ``None`` on failure.

    Retries because the row is not readable the instant the completion
    returns — OpenRouter writes it a beat later, and asking immediately gets a
    404 for a generation that certainly exists.

    **The wait GROWS with each attempt, and that is measured rather than
    cautious.** A 63-token answer's row was ready inside 1.5s; a 5130-token
    one still 404ed after three flat attempts (4.5s) and read fine a moment
    later (2026-09-10, live). Long answers are exactly the slow ones worth
    measuring, so a flat delay would have lost the readings that matter most.
    Backing off covers ~22s over the default five attempts, and it costs
    nothing: this runs on a background worker, after the reply has gone out.
    """
    if not generation_id or not settings.openrouter_api_key:
        return None
    delay = settings.llm_stats_delay_ms / 1000

    async def call(active: httpx.AsyncClient) -> httpx.Response:
        return await active.get(
            f"{settings.openrouter_base_url}/generation",
            params={"id": generation_id},
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
        )

    for attempt in range(max(1, settings.llm_stats_attempts)):
        await sleep(delay * (attempt + 1))
        try:
            if client is not None:
                response = await call(client)
            else:
                async with httpx.AsyncClient(timeout=30.0) as fresh:
                    response = await call(fresh)
        except httpx.HTTPError:
            logger.debug("Generation stats request failed for %s", generation_id, exc_info=True)
            continue
        if response.status_code == httpx.codes.NOT_FOUND:
            logger.debug("Generation %s not ready yet (attempt %d)", generation_id, attempt + 1)
            continue
        if response.status_code != httpx.codes.OK:
            logger.debug(
                "Generation stats refused for %s (HTTP %s)", generation_id, response.status_code
            )
            return None
        try:
            return parse_generation_stats(response.json())
        except ValueError:
            logger.debug("Generation stats returned a non-JSON body for %s", generation_id)
            return None
    logger.debug("Gave up on generation stats for %s", generation_id)
    return None


class LatencyRecorder:
    """Queues finished calls and writes them from a background worker.

    One per process. The queue is bounded: during an OpenRouter outage the
    worker stalls on retries while calls keep finishing, and an unbounded
    queue would turn a measurement feature into a memory leak.
    """

    def __init__(self, store: BindingStore, settings: Settings) -> None:
        self._store = store
        self._settings = settings
        self._queue: asyncio.Queue[LlmCall] = asyncio.Queue(
            maxsize=max(1, settings.llm_metrics_queue_max)
        )
        self._task: asyncio.Task[None] | None = None
        self.dropped = 0

    @property
    def enabled(self) -> bool:
        return self._settings.llm_metrics

    def handler(self, kind: str) -> LatencyHandler:
        """A per-turn callback handler tagged with what kind of call it is."""
        return LatencyHandler(self, kind)

    def submit(self, call: LlmCall) -> None:
        """Hand one finished call to the worker. Never blocks, never raises."""
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(call)
        except asyncio.QueueFull:
            self.dropped += 1
            # One line per drop would itself become the flood; the running
            # count carries the same information at a readable rate.
            if self.dropped % 50 == 1:
                logger.warning("Latency queue full — dropped %d sample(s) so far", self.dropped)

    def start(self) -> None:
        """Start the worker; a no-op when metrics are off or already running."""
        if not self.enabled or self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="latency-recorder")
        logger.info(
            "Latency recording enabled (OpenRouter stats %s)",
            "on" if self._settings.llm_stats_fetch else "off",
        )

    async def aclose(self) -> None:
        """Stop the worker. Queued samples are dropped rather than waited on.

        Shutdown must not block on a retry loop against a service that may be
        the very reason the bot is being restarted.
        """
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug("Latency worker ended with an error", exc_info=True)
        self._task = None

    async def _run(self) -> None:
        while True:
            call = await self._queue.get()
            try:
                await self._record(call)
            except asyncio.CancelledError:
                raise
            except Exception:
                # A measurement failing must never take the worker down with
                # it, or one bad sample silences every later one.
                logger.warning("Recording a latency sample failed", exc_info=True)
            finally:
                self._queue.task_done()

    async def _record(self, call: LlmCall) -> None:
        """Enrich one sample with OpenRouter's timings, then store it."""
        if self._settings.llm_stats_fetch and call.generation_id:
            stats = await fetch_generation_stats(self._settings, call.generation_id)
            if stats is not None:
                call = replace(
                    call,
                    ttft_ms=stats.ttft_ms,
                    generation_ms=stats.generation_ms,
                    provider=stats.provider or call.provider,
                )
        await self._store.record_llm_call(call)


class LatencyHandler(AsyncCallbackHandler):
    """Times each model call and hands the result to the recorder.

    **Async rather than the sync ``BaseCallbackHandler`` that
    ``SpendAccumulator`` uses, and that is load-bearing.** LangChain runs a
    sync handler from an async call through ``run_in_executor`` — on a worker
    thread, where touching an ``asyncio.Queue`` is undefined behaviour. Spend
    gets away with it because it only adds a float; enqueuing does not.

    Chat models fire ``on_chat_model_start``, never ``on_llm_start`` — the end
    and error events are shared with plain LLMs, the start event is not.
    """

    def __init__(self, recorder: LatencyRecorder, kind: str) -> None:
        self._recorder = recorder
        self._kind = kind
        self._started: dict[UUID, float] = {}

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._started[run_id] = time.monotonic()

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        started = self._started.pop(run_id, None)
        if started is None:
            return
        call = extract_call(response, self._kind, (time.monotonic() - started) * 1000)
        if call is not None:
            # Stamped now, not when the worker gets to it: the lookup and its
            # backoff put up to ~22s between the two.
            self._recorder.submit(replace(call, ended_at=datetime.now(UTC)))

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        # Popped, or the map leaks an entry on every failed call — and this
        # handler outlives a turn's retries.
        self._started.pop(run_id, None)


# --- the report -------------------------------------------------------------

#: Column layout for the report, so the header and the rows cannot drift.
_HEADER = (
    f"{'model':<40} {'calls':>6} {'ttft p50':>10} {'ttft p90':>10} {'total p50':>10} {'cost':>10}"
)


def format_report(rows: list[LatencyRow], days: int) -> str:
    """Render the latency table for the terminal."""
    if not rows:
        return f"No model calls recorded in the last {days} day(s)."
    lines = [_HEADER, "-" * len(_HEADER)]
    for row in rows:
        # "n/a" rather than 0: an unmeasured call and an instant one are not
        # the same thing, and a zero here would read as the fastest model.
        ttft = "n/a" if row.median_ttft_ms is None else f"{row.median_ttft_ms / 1000:.2f}s"
        p90 = "n/a" if row.p90_ttft_ms is None else f"{row.p90_ttft_ms / 1000:.2f}s"
        total = f"{row.median_total_ms / 1000:.2f}s"
        cost = f"${row.cost_usd:.4f}"
        lines.append(f"{row.model:<40} {row.calls:>6} {ttft:>10} {p90:>10} {total:>10} {cost:>10}")
        if row.ttft_samples < row.calls:
            # Without this a median over three samples out of two hundred
            # calls reads exactly like one over all of them.
            lines.append(f"{'':<40} {'':>6} (time to first token known for {row.ttft_samples})")
    return "\n".join(lines)


async def report(days: int, kind: str) -> str:
    settings = load_settings()
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    # The cipher is passed even though this reads no tokens: opening the
    # store without one makes its startup check report "AGENT_TOKEN_KEY was
    # removed", which is a false alarm about people's logins on every run of
    # a read-only report.
    async with BindingStore(settings.agent_db_url, TokenCipher.from_key(settings.token_key)) as (
        store
    ):
        rows = await store.llm_latency_report(since, kind)
    return format_report(rows, days)


def main() -> None:
    """``ont-agent-latency`` — model latency over a window."""
    setup_logging()
    if sys.platform == "win32":
        # psycopg's async mode refuses the default ProactorEventLoop; the same
        # policy bot.py and conftest.py set, for the same reason.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    parser = argparse.ArgumentParser(description="Model latency from recorded calls.")
    parser.add_argument("--days", type=int, default=7, help="window to report on (default 7)")
    parser.add_argument(
        "--kind", default="", help=f"only one kind ({KIND_CHAT}, {KIND_BRIEF}, {KIND_WATCHER})"
    )
    args = parser.parse_args()
    print(asyncio.run(report(args.days, args.kind)))


if __name__ == "__main__":
    main()
