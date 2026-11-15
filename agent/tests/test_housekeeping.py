"""Unit tests for ont_agent.housekeeping — the daily retention pass."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row

from ont_agent.config import Settings
from ont_agent.housekeeping import LAST_RUN_KEY, run_if_due
from ont_agent.store import BindingStore

SETTINGS = Settings(mcp_url="http://odoo:8160/mcp", mcp_bearer_key="k")

NOW = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)


class _FakeMemory:
    """Records deletions, and can be told to fail on one thread.

    A fake rather than the real saver for most cases: what is under test is
    which threads housekeeping CHOOSES, not LangGraph's delete — that is
    exercised end to end in the last test.
    """

    def __init__(self, fail_on: str | None = None) -> None:
        self.deleted: list[str] = []
        self._fail_on = fail_on

    async def adelete_thread(self, thread_id: str) -> None:
        if thread_id == self._fail_on:
            raise RuntimeError("thread is wedged")
        self.deleted.append(thread_id)


async def _add_checkpoint(dsn: str, thread_id: str, ts: datetime) -> None:
    """Insert one checkpoint row directly, so a thread can be given an age."""
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        await conn.execute(
            "INSERT INTO checkpoints"
            " (thread_id, checkpoint_ns, checkpoint_id, checkpoint, metadata)"
            " VALUES (%s, '', %s, %s, '{}')",
            (
                thread_id,
                f"cp-{ts.isoformat()}",
                json.dumps({"v": 4, "id": "x", "ts": ts.isoformat(), "channel_values": {}}),
            ),
        )


# --- which threads get deleted ---------------------------------------------


async def _run(store: BindingStore, memory: Any, settings: Settings, now: datetime) -> bool:
    """Cross the typed boundary once.

    _FakeMemory is a deliberately minimal stand-in for the checkpointer;
    funnelling it through one Any-typed helper keeps the tests readable
    instead of repeating an ignore comment at every call.
    """
    return await run_if_due(store, memory, settings, now)


async def test_an_idle_thread_is_deleted_and_a_live_one_is_not(checkpoint_dsn: str) -> None:
    await _add_checkpoint(checkpoint_dsn, "idle-chat", NOW - timedelta(days=40))
    await _add_checkpoint(checkpoint_dsn, "busy-chat", NOW - timedelta(hours=2))

    memory = _FakeMemory()
    async with BindingStore(checkpoint_dsn) as store:
        assert await _run(store, memory, SETTINGS, NOW) is True

    assert memory.deleted == ["idle-chat"]


async def test_a_thread_at_the_cutoff_is_kept(checkpoint_dsn: str) -> None:
    """An off-by-one here deletes a conversation someone is still using, so
    the boundary is pinned rather than left to chance."""
    cutoff = NOW - timedelta(days=SETTINGS.conversation_retention_days)
    await _add_checkpoint(checkpoint_dsn, "just-inside", cutoff + timedelta(seconds=1))
    await _add_checkpoint(checkpoint_dsn, "just-outside", cutoff - timedelta(seconds=1))

    memory = _FakeMemory()
    async with BindingStore(checkpoint_dsn) as store:
        await _run(store, memory, SETTINGS, NOW)

    assert memory.deleted == ["just-outside"]


async def test_a_threads_newest_checkpoint_decides_its_age(checkpoint_dsn: str) -> None:
    # A long-running conversation has old checkpoints in it. It is idle only
    # if the LATEST one is old — otherwise every active chat looks stale.
    await _add_checkpoint(checkpoint_dsn, "long-chat", NOW - timedelta(days=200))
    await _add_checkpoint(checkpoint_dsn, "long-chat", NOW - timedelta(minutes=5))

    memory = _FakeMemory()
    async with BindingStore(checkpoint_dsn) as store:
        await _run(store, memory, SETTINGS, NOW)

    assert memory.deleted == []


async def test_one_wedged_thread_does_not_stop_the_others(checkpoint_dsn: str) -> None:
    for name in ("a-chat", "b-chat", "c-chat"):
        await _add_checkpoint(checkpoint_dsn, name, NOW - timedelta(days=40))

    memory = _FakeMemory(fail_on="b-chat")
    async with BindingStore(checkpoint_dsn) as store:
        await _run(store, memory, SETTINGS, NOW)

    assert memory.deleted == ["a-chat", "c-chat"]


# --- usage counters ---------------------------------------------------------


async def test_old_usage_rows_go_and_recent_ones_stay(checkpoint_dsn: str) -> None:
    async with BindingStore(checkpoint_dsn) as store:
        old_day = (NOW - timedelta(days=200)).date().isoformat()
        today = NOW.date().isoformat()
        await store.record_usage("user:a@acme", old_day, 1.0)
        await store.record_usage("user:a@acme", today, 2.0)

        await _run(store, _FakeMemory(), SETTINGS, NOW)

        assert (await store.get_usage("user:a@acme", old_day)).message_count == 0
        assert (await store.get_usage("user:a@acme", today)).message_count == 1


# --- latency samples --------------------------------------------------------


async def test_old_latency_rows_go_and_recent_ones_stay(checkpoint_dsn: str) -> None:
    """The largest counter table by a wide margin — a row per model CALL,
    several per question, where usage_daily holds a handful a day."""
    from ont_agent.store import LlmCall

    async with BindingStore(checkpoint_dsn) as store:
        await store.record_llm_call(
            LlmCall(model="m", kind="chat", total_ms=1.0), NOW - timedelta(days=200)
        )
        await store.record_llm_call(LlmCall(model="m", kind="chat", total_ms=2.0), NOW)

        await _run(store, _FakeMemory(), SETTINGS, NOW)

        rows = await store.llm_latency_report("2000-01-01T00:00:00+00:00")

    assert len(rows) == 1
    assert rows[0].calls == 1  # only the recent one survived


async def test_zero_disables_latency_retention(checkpoint_dsn: str) -> None:
    from ont_agent.store import LlmCall

    off = SETTINGS.model_copy(update={"llm_metrics_retention_days": 0})

    async with BindingStore(checkpoint_dsn) as store:
        await store.record_llm_call(
            LlmCall(model="m", kind="chat", total_ms=1.0), NOW - timedelta(days=999)
        )

        await _run(store, _FakeMemory(), off, NOW)

        rows = await store.llm_latency_report("2000-01-01T00:00:00+00:00")

    assert rows[0].calls == 1


# --- scheduling: state-driven, once a day ----------------------------------


async def test_it_runs_once_a_day(checkpoint_dsn: str) -> None:
    await _add_checkpoint(checkpoint_dsn, "idle-chat", NOW - timedelta(days=40))

    async with BindingStore(checkpoint_dsn) as store:
        first = _FakeMemory()
        assert await _run(store, first, SETTINGS, NOW) is True

        # The tick fires every few minutes; without the stamp this would
        # re-run the whole pass on each one.
        second = _FakeMemory()
        assert await _run(store, second, SETTINGS, NOW + timedelta(hours=6)) is False
        assert second.deleted == []


async def test_a_stamp_from_yesterday_runs_today(checkpoint_dsn: str) -> None:
    """Catch-up falls out of storing a date rather than firing on a clock: a
    machine that was off simply runs one pass when it wakes."""
    await _add_checkpoint(checkpoint_dsn, "idle-chat", NOW - timedelta(days=40))

    async with BindingStore(checkpoint_dsn) as store:
        await store.set_meta(LAST_RUN_KEY, (NOW - timedelta(days=1)).date().isoformat())
        memory = _FakeMemory()

        assert await _run(store, memory, SETTINGS, NOW) is True
        assert memory.deleted == ["idle-chat"]


async def test_the_stamp_is_written_before_the_work(checkpoint_dsn: str) -> None:
    # A pass that dies halfway must not retry on every tick for the rest of
    # the day. Missing one day of retention costs nothing; hammering a
    # failing delete every five minutes does.
    await _add_checkpoint(checkpoint_dsn, "idle-chat", NOW - timedelta(days=40))

    async with BindingStore(checkpoint_dsn) as store:
        await _run(store, _FakeMemory(fail_on="idle-chat"), SETTINGS, NOW)
        assert await store.get_meta(LAST_RUN_KEY) == NOW.date().isoformat()


# --- switches ---------------------------------------------------------------


async def test_zero_disables_conversation_retention(checkpoint_dsn: str) -> None:
    await _add_checkpoint(checkpoint_dsn, "idle-chat", NOW - timedelta(days=999))
    off = SETTINGS.model_copy(update={"conversation_retention_days": 0})

    memory = _FakeMemory()
    async with BindingStore(checkpoint_dsn) as store:
        await _run(store, memory, off, NOW)

    assert memory.deleted == []


async def test_zero_disables_usage_retention(checkpoint_dsn: str) -> None:
    off = SETTINGS.model_copy(update={"usage_retention_days": 0})

    async with BindingStore(checkpoint_dsn) as store:
        old_day = (NOW - timedelta(days=999)).date().isoformat()
        await store.record_usage("user:a@acme", old_day, 1.0)

        await _run(store, _FakeMemory(), off, NOW)

        assert (await store.get_usage("user:a@acme", old_day)).message_count == 1


async def test_no_checkpointer_is_survivable(checkpoint_dsn: str) -> None:
    # AGENT_LONG_TERM_MEMORY off, or a startup where the saver never built.
    async with BindingStore(checkpoint_dsn) as store:
        assert await run_if_due(store, None, SETTINGS, NOW) is True


# --- the real checkpointer, end to end -------------------------------------


async def test_the_real_saver_clears_checkpoints_blobs_and_writes(checkpoint_dsn: str) -> None:
    """A partial delete would leave orphaned rows behind, which is why this
    goes through LangGraph's own adelete_thread rather than a DELETE of
    ours — and why the test counts all three tables."""
    from langchain_core.messages import HumanMessage

    async with await psycopg.AsyncConnection.connect(
        checkpoint_dsn, autocommit=True, row_factory=dict_row
    ) as conn:
        saver = AsyncPostgresSaver(conn)
        config: Any = {"configurable": {"thread_id": "real-chat", "checkpoint_ns": ""}}
        await saver.aput(
            config,
            {
                "v": 4,
                "id": "cp-1",
                "ts": (NOW - timedelta(days=40)).isoformat(),
                "channel_values": {"messages": [HumanMessage(content="hi")]},
                "channel_versions": {},
                "versions_seen": {},
                "updated_channels": None,
            },
            {"source": "input", "step": 1},
            {},
        )

        async with BindingStore(checkpoint_dsn) as store:
            await _run(store, saver, SETTINGS, NOW)

        for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
            cursor = await conn.execute(f"SELECT COUNT(*) AS n FROM {table}")
            row = await cursor.fetchone()
            assert row is not None
            assert row["n"] == 0, f"{table} still has rows"
