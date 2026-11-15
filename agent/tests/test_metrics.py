"""Unit tests for ont_agent.metrics (no network, no model calls)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, LLMResult

from ont_agent.config import Settings
from ont_agent.metrics import (
    KIND_CHAT,
    KIND_WATCHER,
    GenerationStats,
    LatencyRecorder,
    extract_call,
    fetch_generation_stats,
    format_report,
    parse_generation_stats,
)
from ont_agent.store import BindingStore, LatencyRow, LlmCall

SETTINGS = Settings(
    mcp_url="http://odoo:8160/mcp",
    mcp_bearer_key="k",
    openrouter_api_key="sk-or-v1-test",
    llm_stats_delay_ms=0,
)


def _result(
    *,
    llm_output: dict[str, Any] | None = None,
    cost: float | None = 0.0021,
    provider: str = "Google",
    usage: dict[str, Any] | None = None,
) -> LLMResult:
    """An LLMResult shaped the way a callback actually receives one.

    That shape is the point: LangChain hands callbacks ``flattened_output``,
    which keeps the chat model's own ``llm_output``.
    """
    metadata: dict[str, Any] = {"provider": provider}
    if cost is not None:
        metadata["cost"] = cost
    message = AIMessage(content="hello", response_metadata=metadata)
    if usage is not None:
        message.usage_metadata = cast("UsageMetadata", usage)
    return LLMResult(
        generations=[[ChatGeneration(message=message)]],
        llm_output=llm_output if llm_output is not None else {},
    )


# --- reading a finished call ------------------------------------------------


def test_the_generation_id_and_answering_model_are_read_from_llm_output() -> None:
    """The id lives ONLY on the callback's LLMResult.

    ``ChatOpenRouter`` does not override ``_combine_llm_outputs``, whose base
    returns ``{}``, so the result handed back to the caller has none of this.
    Reading it from there would silently yield nothing for ever.
    """
    result = _result(
        llm_output={"id": "gen-abc123", "model_name": "google/gemini-3.7-flash"},
        usage={
            "input_tokens": 12804,
            "output_tokens": 373,
            "input_token_details": {"cache_read": 0},
        },
    )

    call = extract_call(result, KIND_CHAT, 4742.0)

    assert call is not None
    assert call.generation_id == "gen-abc123"
    assert call.model == "google/gemini-3.7-flash"
    assert call.provider == "Google"
    assert call.kind == KIND_CHAT
    assert call.total_ms == 4742.0
    assert (call.prompt_tokens, call.completion_tokens) == (12804, 373)
    assert call.cost_usd == pytest.approx(0.0021)
    # Not yet known — the /generation lookup fills these in later.
    assert call.ttft_ms is None
    assert call.generation_ms is None


def test_the_model_that_answered_wins_over_the_one_configured() -> None:
    # With fallbacks configured the two differ exactly when it matters, so
    # the row must name the model that actually served the request.
    call = extract_call(
        _result(llm_output={"id": "gen-1", "model_name": "moonshotai/kimi-k2.6"}),
        KIND_CHAT,
        100.0,
    )

    assert call is not None
    assert call.model == "moonshotai/kimi-k2.6"


def test_a_response_with_no_model_name_is_not_recorded() -> None:
    # An unattributed latency number cannot be compared with anything, which
    # makes it worse than a shorter table.
    assert extract_call(_result(llm_output={"id": "gen-1"}), KIND_CHAT, 100.0) is None


def test_missing_usage_and_cost_still_yield_a_row() -> None:
    call = extract_call(_result(llm_output={"model_name": "m"}, cost=None), KIND_CHAT, 50.0)

    assert call is not None
    assert (call.prompt_tokens, call.completion_tokens, call.cost_usd) == (0, 0, 0.0)
    assert call.generation_id == ""  # nothing to look up later, and that is fine


def test_a_malformed_result_is_dropped_rather_than_raised() -> None:
    assert extract_call(LLMResult(generations=[]), KIND_CHAT, 1.0) is None


# --- OpenRouter's own numbers -----------------------------------------------


def test_latency_maps_to_time_to_first_token() -> None:
    stats = parse_generation_stats(
        {
            "data": {
                "id": "gen-abc",
                "latency": 3316,
                "generation_time": 4742,
                "provider_name": "Google",
            }
        }
    )

    assert stats.ttft_ms == 3316
    assert stats.generation_ms == 4742
    assert stats.provider == "Google"


def test_missing_or_renamed_fields_leave_the_half_unknown() -> None:
    # Fails toward "we do not know" rather than toward a wrong number: the
    # wall clock is still recorded either way.
    assert parse_generation_stats({"data": {"id": "gen-abc"}}) == GenerationStats()
    assert parse_generation_stats({"data": {"latency": None}}).ttft_ms is None
    assert parse_generation_stats({}) == GenerationStats()
    assert parse_generation_stats("nonsense") == GenerationStats()


async def test_the_lookup_retries_while_the_row_is_not_written_yet() -> None:
    """A 404 straight after a completion is normal, not an error.

    OpenRouter writes the generation row a beat later; giving up on the first
    404 would lose the reading on every fast answer.
    """
    attempts: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        attempts.append(str(request.url))
        if len(attempts) < 3:
            return httpx.Response(404)
        return httpx.Response(200, json={"data": {"latency": 1250, "generation_time": 1200}})

    transport = httpx.MockTransport(handle)
    async with httpx.AsyncClient(transport=transport) as client:
        stats = await fetch_generation_stats(SETTINGS, "gen-abc", client)

    assert len(attempts) == 3
    assert "gen-abc" in attempts[0]
    assert stats is not None
    assert stats.ttft_ms == 1250


async def test_the_lookup_gives_up_after_the_configured_attempts() -> None:
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        stats = await fetch_generation_stats(
            SETTINGS.model_copy(update={"llm_stats_attempts": 2}), "g", client
        )

    assert calls == 2
    assert stats is None


async def test_the_wait_grows_with_each_attempt() -> None:
    """A flat delay lost exactly the readings worth having.

    Measured live 2026-09-10: a 63-token answer's row was ready inside 1.5s,
    a 5130-token one still 404ed after three flat attempts — and the long,
    slow answers are the ones a latency table exists to catch.
    """
    waits: list[float] = []

    async def record(seconds: float) -> None:
        waits.append(seconds)

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        await fetch_generation_stats(
            SETTINGS.model_copy(update={"llm_stats_delay_ms": 1000, "llm_stats_attempts": 4}),
            "gen-slow",
            client,
            sleep=record,
        )

    assert waits == [1.0, 2.0, 3.0, 4.0]


async def test_the_id_is_taken_from_response_metadata_when_llm_output_is_bare() -> None:
    # Both carriers hold it live; reading either means a change on one side
    # cannot silently empty the whole table.
    message = AIMessage(
        content="x",
        response_metadata={"id": "gen-from-metadata", "model_name": "google/gemini-3.7-flash"},
    )
    result = LLMResult(generations=[[ChatGeneration(message=message)]], llm_output={})

    call = extract_call(result, KIND_CHAT, 10.0)

    assert call is not None
    assert call.generation_id == "gen-from-metadata"
    assert call.model == "google/gemini-3.7-flash"


async def test_a_transport_failure_is_swallowed() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("openrouter is down")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        assert await fetch_generation_stats(SETTINGS, "gen-abc", client) is None


async def test_no_generation_id_means_no_request_at_all() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should never be called")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        assert await fetch_generation_stats(SETTINGS, "", client) is None


# --- the handler ------------------------------------------------------------


class _Recorder:
    """Collects submitted samples in place of the real recorder."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.calls: list[LlmCall] = []

    def submit(self, call: LlmCall) -> None:
        self.calls.append(call)


def _handler(recorder: Any, kind: str = KIND_CHAT) -> Any:
    from ont_agent.metrics import LatencyHandler

    return LatencyHandler(recorder, kind)


async def test_a_call_is_timed_from_start_to_end() -> None:
    recorder = _Recorder()
    handler = _handler(recorder)
    run_id = uuid4()

    await handler.on_chat_model_start({}, [], run_id=run_id)
    await handler.on_llm_end(_result(llm_output={"id": "gen-1", "model_name": "m"}), run_id=run_id)

    assert len(recorder.calls) == 1
    assert recorder.calls[0].total_ms >= 0


async def test_the_row_is_stamped_when_the_call_ended_not_when_it_was_written(
    store_dsn: str,
) -> None:
    """A timing table must not be timestamped by the time we got round to it.

    The row waits on the OpenRouter lookup and its backoff — up to ~22
    seconds — so a write-time stamp would misorder the last ten calls
    whenever one call's stats took longer to appear than a later call's, and
    would push a call near midnight into the wrong day.
    """
    recorder = _Recorder()
    handler = _handler(recorder)
    run_id = uuid4()

    await handler.on_chat_model_start({}, [], run_id=run_id)
    await handler.on_llm_end(_result(llm_output={"model_name": "m"}), run_id=run_id)
    ended = recorder.calls[0].ended_at
    assert ended is not None

    # Written much later, as a slow lookup would: the row keeps the call's
    # own moment rather than the writer's.
    async with BindingStore(store_dsn) as store:
        await store.record_llm_call(recorder.calls[0], datetime(2030, 1, 1, tzinfo=UTC))
        async with (
            store.connection() as db,
            await db.execute("SELECT created_at FROM llm_calls") as cursor,
        ):
            row = await cursor.fetchone()

    assert row is not None
    assert row[0] == ended.isoformat()


async def test_an_end_without_a_start_records_nothing() -> None:
    # Handlers are per turn but LangChain reuses them across nested runs;
    # an unmatched end is a run this handler never saw begin.
    recorder = _Recorder()

    await _handler(recorder).on_llm_end(
        _result(llm_output={"id": "g", "model_name": "m"}), run_id=uuid4()
    )

    assert recorder.calls == []


async def test_a_failed_call_leaves_no_entry_behind() -> None:
    """The leak this prevents: a handler outlives a turn's retries, so an
    unpopped start would accumulate one dead entry per failed call."""
    handler = _handler(_Recorder())
    run_id = uuid4()

    await handler.on_chat_model_start({}, [], run_id=run_id)
    await handler.on_llm_error(RuntimeError("provider down"), run_id=run_id)

    assert handler._started == {}


async def test_the_kind_travels_onto_the_row() -> None:
    recorder = _Recorder()
    handler = _handler(recorder, KIND_WATCHER)
    run_id = uuid4()

    await handler.on_chat_model_start({}, [], run_id=run_id)
    await handler.on_llm_end(_result(llm_output={"model_name": "m"}), run_id=run_id)

    assert recorder.calls[0].kind == KIND_WATCHER


# --- the recorder -----------------------------------------------------------


class _FakeStore:
    def __init__(self) -> None:
        self.written: list[LlmCall] = []

    async def record_llm_call(self, call: LlmCall) -> None:
        self.written.append(call)


def _recorder(store: Any, **overrides: Any) -> LatencyRecorder:
    return LatencyRecorder(store, SETTINGS.model_copy(update=overrides))


async def test_metrics_off_records_absolutely_nothing() -> None:
    store = _FakeStore()
    recorder = _recorder(store, llm_metrics=False)

    recorder.start()
    recorder.submit(LlmCall(model="m", kind=KIND_CHAT, total_ms=1.0))
    await asyncio.sleep(0)

    assert store.written == []
    await recorder.aclose()


async def test_a_full_queue_drops_the_sample_instead_of_growing() -> None:
    """Backpressure, not a memory leak.

    While OpenRouter is unreachable the worker stalls on retries and calls
    keep finishing; an unbounded queue would make a measurement feature the
    reason the process runs out of memory.
    """
    recorder = _recorder(_FakeStore(), llm_metrics_queue_max=2)

    for _ in range(5):
        recorder.submit(LlmCall(model="m", kind=KIND_CHAT, total_ms=1.0))

    assert recorder.dropped == 3  # queued 2, dropped the rest, raised nothing


async def test_openrouter_timings_are_merged_onto_the_stored_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _FakeStore()
    recorder = _recorder(store)

    async def fake_fetch(settings: Settings, generation_id: str) -> GenerationStats:
        assert generation_id == "gen-abc"
        return GenerationStats(ttft_ms=3316, generation_ms=4742, provider="Google")

    monkeypatch.setattr("ont_agent.metrics.fetch_generation_stats", fake_fetch)
    recorder.start()
    recorder.submit(LlmCall(model="m", kind=KIND_CHAT, total_ms=8058.0, generation_id="gen-abc"))
    await asyncio.sleep(0.05)
    await recorder.aclose()

    assert len(store.written) == 1
    row = store.written[0]
    assert (row.ttft_ms, row.generation_ms) == (3316, 4742)
    assert row.total_ms == 8058.0  # our own clock is never overwritten
    assert row.provider == "Google"


async def test_a_failed_lookup_still_stores_the_wall_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The half that always works: without this the whole row would be lost
    # whenever OpenRouter's stats were unavailable.
    store = _FakeStore()
    recorder = _recorder(store)

    async def fake_fetch(settings: Settings, generation_id: str) -> GenerationStats | None:
        return None

    monkeypatch.setattr("ont_agent.metrics.fetch_generation_stats", fake_fetch)
    recorder.start()
    recorder.submit(LlmCall(model="m", kind=KIND_CHAT, total_ms=900.0, generation_id="g"))
    await asyncio.sleep(0.05)
    await recorder.aclose()

    assert len(store.written) == 1
    assert store.written[0].ttft_ms is None
    assert store.written[0].total_ms == 900.0


async def test_stats_fetching_off_skips_the_lookup_entirely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _FakeStore()
    recorder = _recorder(store, llm_stats_fetch=False)

    async def explode(settings: Settings, generation_id: str) -> GenerationStats:
        raise AssertionError("should not be called")

    monkeypatch.setattr("ont_agent.metrics.fetch_generation_stats", explode)
    recorder.start()
    recorder.submit(LlmCall(model="m", kind=KIND_CHAT, total_ms=42.0, generation_id="g"))
    await asyncio.sleep(0.05)
    await recorder.aclose()

    assert len(store.written) == 1
    assert store.written[0].ttft_ms is None


async def test_one_bad_sample_does_not_silence_the_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker is a single long-lived loop, so an escaping exception would
    stop every later sample, not just this one."""

    class _AngryStore(_FakeStore):
        async def record_llm_call(self, call: LlmCall) -> None:
            if call.model == "bad":
                raise RuntimeError("write failed")
            await super().record_llm_call(call)

    store = _AngryStore()
    recorder = _recorder(store, llm_stats_fetch=False)
    recorder.start()
    recorder.submit(LlmCall(model="bad", kind=KIND_CHAT, total_ms=1.0))
    recorder.submit(LlmCall(model="good", kind=KIND_CHAT, total_ms=2.0))
    await asyncio.sleep(0.05)
    await recorder.aclose()

    assert [c.model for c in store.written] == ["good"]


# --- the report -------------------------------------------------------------


def test_the_report_says_when_a_median_rests_on_few_samples() -> None:
    """A median over 3 readings out of 200 calls is a different claim from
    one over all 200, and the table has to be able to say so."""
    text = format_report(
        [
            LatencyRow("google/gemini-3.7-flash", 200, 3, 25900.0, 27000.0, 26100.0, 1.25),
            LatencyRow("anthropic/claude-haiku-4.5", 50, 50, 1100.0, 1400.0, 2910.0, 0.31),
        ],
        7,
    )

    assert "25.90s" in text
    assert "1.10s" in text
    assert "time to first token known for 3" in text
    assert "known for 50" not in text  # complete data needs no caveat


def test_an_unmeasured_model_reads_as_unknown_not_as_instant() -> None:
    # A zero here would sort and read as the fastest model on the list.
    text = format_report([LatencyRow("m", 4, 0, None, None, 3000.0, 0.0)], 1)

    assert "n/a" in text
    assert "3.00s" in text


def test_an_empty_window_says_so() -> None:
    assert "No model calls" in format_report([], 3)


# --- the round trip through Postgres ----------------------------------------


async def test_a_recorded_call_survives_the_database(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        await store.record_llm_call(
            LlmCall(
                model="google/gemini-3.7-flash",
                kind=KIND_CHAT,
                total_ms=8058.0,
                generation_id="gen-abc",
                provider="Google",
                ttft_ms=3316.0,
                generation_ms=4742.0,
                prompt_tokens=12804,
                completion_tokens=373,
                cost_usd=0.0021,
            )
        )
        rows = await store.llm_latency_report("2000-01-01T00:00:00+00:00")

    assert len(rows) == 1
    assert rows[0].model == "google/gemini-3.7-flash"
    assert rows[0].median_ttft_ms == 3316.0
    assert rows[0].cost_usd == pytest.approx(0.0021)
