"""Unit tests for ont_agent.rotation — bounding an active user's thread.

The safety property under test is the two-phase split: until ``commit``
runs, the epoch is unchanged and the old thread is intact, so nothing a
crash can interrupt loses a conversation.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row

from ont_agent.agent import ask, build_agent, thread_messages
from ont_agent.config import Settings
from ont_agent.rotation import Rotation, current_thread, plan_rotation, thread_name
from ont_agent.store import BindingStore

SETTINGS = Settings(mcp_url="http://odoo:8160/mcp", mcp_bearer_key="k")
#: Small enough that a handful of inserted rows crosses it.
TIGHT = SETTINGS.model_copy(update={"thread_max_checkpoints": 3})

CHAT = 42


# --- fakes -----------------------------------------------------------------


class _Task:
    interrupts = ("waiting for approval",)


class _Snapshot:
    def __init__(self, messages: list[Any], pending: bool) -> None:
        self.values = {"messages": messages}
        self.tasks = [_Task()] if pending else []


class _FakeAgent:
    """Answers the two questions rotation asks: is this thread paused, and
    what state does it hold. A real graph is used end to end below."""

    def __init__(self, messages: list[Any] | None = None, pending: bool = False) -> None:
        self._messages = messages or []
        self._pending = pending

    async def aget_state(self, config: Any) -> _Snapshot:
        return _Snapshot(self._messages, self._pending)


class _FakeMemory:
    def __init__(self, fail: bool = False) -> None:
        self.deleted: list[str] = []
        self._fail = fail

    async def adelete_thread(self, thread_id: str) -> None:
        if self._fail:
            raise RuntimeError("thread is wedged")
        self.deleted.append(thread_id)


async def _plan(
    store: BindingStore, agent: Any, settings: Settings, chat_id: int, memory: Any
) -> Rotation:
    """Cross the typed boundary once, the pattern test_housekeeping uses."""
    return await plan_rotation(store, agent, settings, chat_id, memory)


async def _commit(rotation: Rotation, store: BindingStore, memory: Any, chat_id: int) -> None:
    """Same funnel for the fake checkpointer."""
    await rotation.commit(store, memory, chat_id)


async def _add_checkpoints(dsn: str, thread_id: str, count: int) -> None:
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        for index in range(count):
            await conn.execute(
                "INSERT INTO checkpoints"
                " (thread_id, checkpoint_ns, checkpoint_id, checkpoint, metadata)"
                " VALUES (%s, '', %s, %s, '{}')",
                (
                    thread_id,
                    f"cp-{index}",
                    json.dumps(
                        {
                            "v": 4,
                            "id": f"cp-{index}",
                            "ts": datetime.now(UTC).isoformat(),
                            "channel_values": {},
                        }
                    ),
                ),
            )


# --- naming and epoch reading ----------------------------------------------


def test_epoch_zero_keeps_the_bare_chat_id() -> None:
    # Threads that exist today are on the chat id. If epoch 0 renamed them,
    # every live conversation would be orphaned by the deploy.
    assert thread_name(CHAT, 0) == "42"
    assert thread_name(CHAT, 1) == "42-1"


async def test_current_thread_follows_the_stored_epoch(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        assert await current_thread(store, CHAT) == "42"
        await store.set_meta("thread_epoch:42", "7")
        assert await current_thread(store, CHAT) == "42-7"


async def test_an_unreadable_epoch_falls_back_to_zero(store_dsn: str) -> None:
    # Benign by design: epoch 0's thread was deleted at the first rotation,
    # so the chat starts fresh rather than reading somebody else's state.
    async with BindingStore(store_dsn) as store:
        await store.set_meta("thread_epoch:42", "not-a-number")
        assert await current_thread(store, CHAT) == "42"


# --- when rotation fires ----------------------------------------------------


async def test_a_short_thread_is_left_alone(checkpoint_dsn: str) -> None:
    await _add_checkpoints(checkpoint_dsn, "42", 2)
    async with BindingStore(checkpoint_dsn) as store:
        rotation = await _plan(store, _FakeAgent(), TIGHT, CHAT, _FakeMemory())
    assert rotation.rotated is False
    assert rotation.thread_id == "42"
    assert rotation.history == []


async def test_a_full_thread_rotates_and_carries_its_messages(checkpoint_dsn: str) -> None:
    await _add_checkpoints(checkpoint_dsn, "42", 3)
    carried = [HumanMessage(content="codename Bluebird"), AIMessage(content="noted")]
    async with BindingStore(checkpoint_dsn) as store:
        rotation = await _plan(store, _FakeAgent(carried), TIGHT, CHAT, _FakeMemory())
    assert rotation.rotated is True
    assert rotation.thread_id == "42-1"
    assert rotation.previous_thread == "42"
    assert rotation.history == carried


async def test_an_open_approval_defers_rotation(checkpoint_dsn: str) -> None:
    # The interrupt lives in that thread's checkpoint and cannot be carried
    # across; rotating around it would strand the pending write.
    await _add_checkpoints(checkpoint_dsn, "42", 10)
    async with BindingStore(checkpoint_dsn) as store:
        rotation = await _plan(store, _FakeAgent(pending=True), TIGHT, CHAT, _FakeMemory())
    assert rotation.rotated is False
    assert rotation.thread_id == "42"


async def test_zero_disables_rotation(checkpoint_dsn: str) -> None:
    await _add_checkpoints(checkpoint_dsn, "42", 50)
    off = SETTINGS.model_copy(update={"thread_max_checkpoints": 0})
    async with BindingStore(checkpoint_dsn) as store:
        assert (await _plan(store, _FakeAgent(), off, CHAT, _FakeMemory())).rotated is False


async def test_no_checkpointer_never_rotates(checkpoint_dsn: str) -> None:
    await _add_checkpoints(checkpoint_dsn, "42", 50)
    async with BindingStore(checkpoint_dsn) as store:
        assert (await _plan(store, _FakeAgent(), TIGHT, CHAT, None)).rotated is False


async def test_a_broken_snapshot_does_not_break_the_turn(checkpoint_dsn: str) -> None:
    """Rotation is maintenance. Failing it costs disk; raising would break
    the question the user actually asked."""

    class _Exploding(_FakeAgent):
        async def aget_state(self, config: Any) -> _Snapshot:
            raise RuntimeError("state unreadable")

    await _add_checkpoints(checkpoint_dsn, "42", 10)
    async with BindingStore(checkpoint_dsn) as store:
        rotation = await _plan(store, _Exploding(), TIGHT, CHAT, _FakeMemory())
    assert rotation.rotated is False
    assert rotation.thread_id == "42"


async def test_counting_fails_safe_without_the_checkpoint_tables(store_dsn: str) -> None:
    # No checkpoint tables in this schema at all. The count returns 0 rather
    # than raising, so rotation quietly never fires.
    async with BindingStore(store_dsn) as store:
        assert await store.count_checkpoints("42") == 0


# --- the two-phase guarantee ------------------------------------------------


async def test_nothing_is_written_until_commit(checkpoint_dsn: str) -> None:
    # This is the crash-safety property: a turn that dies after planning
    # leaves the epoch and the old thread exactly as they were, so the next
    # turn simply rotates again.
    await _add_checkpoints(checkpoint_dsn, "42", 5)
    memory = _FakeMemory()
    async with BindingStore(checkpoint_dsn) as store:
        rotation = await _plan(store, _FakeAgent([HumanMessage(content="x")]), TIGHT, CHAT, memory)
        assert rotation.rotated is True

        assert await store.get_meta("thread_epoch:42") is None
        assert memory.deleted == []
        assert await store.count_checkpoints("42") == 5
        assert await current_thread(store, CHAT) == "42"


async def test_commit_advances_the_epoch_and_retires_the_old_thread(
    checkpoint_dsn: str,
) -> None:
    await _add_checkpoints(checkpoint_dsn, "42", 5)
    memory = _FakeMemory()
    async with BindingStore(checkpoint_dsn) as store:
        rotation = await _plan(store, _FakeAgent([HumanMessage(content="x")]), TIGHT, CHAT, memory)
        await _commit(rotation, store, memory, CHAT)

        assert await current_thread(store, CHAT) == "42-1"
        assert memory.deleted == ["42"]


async def test_a_failed_delete_leaves_the_rotation_standing(checkpoint_dsn: str) -> None:
    # The old thread becomes an orphan, which the 30-day retention pass
    # collects. Undoing the rotation instead would lose the carried state.
    await _add_checkpoints(checkpoint_dsn, "42", 5)
    async with BindingStore(checkpoint_dsn) as store:
        rotation = await _plan(
            store, _FakeAgent([HumanMessage(content="x")]), TIGHT, CHAT, _FakeMemory()
        )
        await _commit(rotation, store, _FakeMemory(fail=True), CHAT)
        assert await current_thread(store, CHAT) == "42-1"


async def test_committing_a_non_rotation_does_nothing(checkpoint_dsn: str) -> None:
    memory = _FakeMemory()
    async with BindingStore(checkpoint_dsn) as store:
        rotation = await _plan(store, _FakeAgent(), TIGHT, CHAT, memory)
        await _commit(rotation, store, memory, CHAT)
        assert await store.get_meta("thread_epoch:42") is None
        assert memory.deleted == []


async def test_epochs_are_monotonic(checkpoint_dsn: str) -> None:
    # Never reset, not even by /logout: a reused id could otherwise collide
    # with a thread whose delete had failed.
    memory = _FakeMemory()
    async with BindingStore(checkpoint_dsn) as store:
        for expected in ("42-1", "42-2", "42-3"):
            thread = await current_thread(store, CHAT)
            await _add_checkpoints(checkpoint_dsn, thread, 5)
            rotation = await _plan(store, _FakeAgent(), TIGHT, CHAT, memory)
            await _commit(rotation, store, memory, CHAT)
            assert await current_thread(store, CHAT) == expected


# --- end to end, with the real saver and a real graph -----------------------


class _ScriptedModel(BaseChatModel):
    """Replays fixed replies; the graph around it is entirely real."""

    script: list[AIMessage]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


async def test_a_rotated_conversation_still_remembers(checkpoint_dsn: str) -> None:
    """The whole point: the user must notice nothing.

    Seeding the new thread through a normal invocation, rather than
    ``aupdate_state``, is deliberate — on a thread with no checkpoint yet
    that call raises InvalidUpdateError("Ambiguous update, specify as_node")
    because a compiled agent's input channel is not itself a node.
    """
    async with await psycopg.AsyncConnection.connect(
        checkpoint_dsn, autocommit=True, row_factory=dict_row
    ) as connection:
        saver = AsyncPostgresSaver(connection)
        model = _ScriptedModel(script=[AIMessage(content="noted"), AIMessage(content="Bluebird")])
        agent = build_agent(model, [], checkpointer=saver)

        await ask(agent, "my project codename is Bluebird", thread_id="42")

        async with BindingStore(checkpoint_dsn) as store:
            rotation = await _plan(store, agent, TIGHT, CHAT, saver)
            assert rotation.rotated is True

            answer = await ask(
                agent,
                "what is my project codename?",
                thread_id=rotation.thread_id,
                history=rotation.history,
            )
            assert answer.answer == "Bluebird"

            # The carried question is on the NEW thread, verbatim.
            carried = await thread_messages(agent, rotation.thread_id)
            assert any("Bluebird" in str(message.content) for message in carried)

            await _commit(rotation, store, saver, CHAT)
            assert await store.count_checkpoints("42") == 0
            assert await store.count_checkpoints("42-1") > 0


@pytest.mark.parametrize("table", ["checkpoints", "checkpoint_blobs", "checkpoint_writes"])
async def test_the_retired_thread_leaves_no_orphans(checkpoint_dsn: str, table: str) -> None:
    async with await psycopg.AsyncConnection.connect(
        checkpoint_dsn, autocommit=True, row_factory=dict_row
    ) as connection:
        saver = AsyncPostgresSaver(connection)
        agent = build_agent(
            _ScriptedModel(script=[AIMessage(content="ok")]), [], checkpointer=saver
        )
        await ask(agent, "hello", thread_id="42")

        async with BindingStore(checkpoint_dsn) as store:
            rotation = await _plan(store, agent, TIGHT, CHAT, saver)
            await _commit(rotation, store, saver, CHAT)

        cursor = await connection.execute(
            f"SELECT COUNT(*) AS n FROM {table} WHERE thread_id = '42'"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row["n"] == 0
