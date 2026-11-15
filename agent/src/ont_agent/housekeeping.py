"""Retention: delete what nothing will ever read again.

Two tables grow without bound. Conversation checkpoints are the real
problem — each one is a full snapshot of a chat's state, and a single
chat's history reached 98 MB before this existed. Usage counters are
housekeeping by comparison, a handful of rows a day, but equally unbounded.

Kept out of ``proactive.py`` on purpose: that module is about the bot
SPEAKING unprompted. This is maintenance, and it needs the checkpointer,
which the proactive tick has no access to.

**Scheduled by state, not by cron** — the same pattern as the morning brief
and the watchers, for the same reasons: asking "is this due?" from a stored
stamp is idempotent, survives restarts, and gives catch-up after downtime
for free. A machine that was off for a week runs one pass when it wakes,
not seven.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

from ont_agent.config import Settings
from ont_agent.log import get_logger
from ont_agent.store import BindingStore

logger = get_logger(__name__)

#: Where the last run is stamped, in the existing ``meta`` table.
LAST_RUN_KEY = "last_housekeeping_date"


async def run_if_due(
    store: BindingStore,
    memory: BaseCheckpointSaver[Any] | None,
    settings: Settings,
    now: datetime | None = None,
) -> bool:
    """Run one retention pass if today's has not happened yet.

    :return: whether a pass actually ran, so the caller can log it and a
        test can pin the once-a-day behaviour.
    """
    moment = now or datetime.now(UTC)
    today = moment.date().isoformat()
    if await store.get_meta(LAST_RUN_KEY) == today:
        return False

    # Stamped BEFORE the work, not after: a pass that dies halfway must not
    # retry on every tick for the rest of the day. Retention missing a day
    # costs nothing; hammering a failing delete every five minutes does.
    await store.set_meta(LAST_RUN_KEY, today)

    await _prune_conversations(store, memory, settings, moment)
    await _prune_usage(store, settings, moment)
    await _prune_llm_calls(store, settings, moment)
    return True


async def _prune_conversations(
    store: BindingStore,
    memory: BaseCheckpointSaver[Any] | None,
    settings: Settings,
    moment: datetime,
) -> None:
    """Delete threads nobody has spoken to in ``conversation_retention_days``.

    Whole threads only. LangGraph implements ``adelete_thread`` for Postgres
    and nothing finer — ``aprune`` raises NotImplementedError, and its own
    docstring warns that dropping ancestor checkpoints can make delta
    channels reconstruct as EMPTY, which would corrupt the surviving state
    rather than merely lose history.
    """
    days = settings.conversation_retention_days
    if days <= 0 or memory is None:
        return
    cutoff = (moment - timedelta(days=days)).isoformat()
    threads = await store.idle_threads(cutoff)
    if not threads:
        return

    deleted = 0
    for thread_id in threads:
        try:
            # Clears checkpoints, blobs AND writes — a partial delete would
            # leave orphans behind, which is why this goes through their API
            # rather than a DELETE of our own.
            await memory.adelete_thread(thread_id)
            deleted += 1
        except Exception:
            # One unreadable thread must not stop the pass, the same
            # isolation the proactive tick uses per user.
            logger.error("Could not delete idle thread %s", thread_id, exc_info=True)
    logger.info("Retention: deleted %d conversation(s) idle over %d days", deleted, days)


async def _prune_usage(store: BindingStore, settings: Settings, moment: datetime) -> None:
    """Drop usage counters older than ``usage_retention_days``."""
    days = settings.usage_retention_days
    if days <= 0:
        return
    cutoff_day = (moment - timedelta(days=days)).date().isoformat()
    removed = await store.prune_usage(cutoff_day)
    if removed:
        logger.info("Retention: deleted %d usage row(s) before %s", removed, cutoff_day)


async def _prune_llm_calls(store: BindingStore, settings: Settings, moment: datetime) -> None:
    """Drop latency samples older than ``llm_metrics_retention_days``.

    The larger of the two counter tables by a wide margin: ``usage_daily``
    holds a handful of rows a day, this one holds a row per model CALL —
    several per question. Nothing reads a sample older than the report
    window, so age is the whole criterion.
    """
    days = settings.llm_metrics_retention_days
    if days <= 0:
        return
    cutoff = (moment - timedelta(days=days)).isoformat()
    removed = await store.prune_llm_calls(cutoff)
    if removed:
        logger.info("Retention: deleted %d latency row(s) before %s", removed, cutoff)
