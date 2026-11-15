"""Abuse and spend limits.

Three guards, each keyed to the identity it is actually about:

* **Burst** — how fast one Telegram SENDER may talk. In-memory, seconds.
* **Daily message + spend** — what one ODOO USER may consume in a day.
  Keyed by login, so the same person in two chats shares one budget.
* **Global daily spend** — what the whole bot may cost in a day. The
  per-user cap bounds a person; only this bounds the bill.

Spend is OpenRouter's real billed cost, not an estimate, and none of this
depends on Langfuse: spend tracking is a safety feature, not observability,
so it must work whether or not tracing is configured.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from ont_agent.config import Settings
from ont_agent.log import get_logger
from ont_agent.store import Binding, UsageRow, chat_subject, user_subject
from ont_agent.tracing import extract_cost_details

logger = get_logger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.outputs import LLMResult


@dataclass(frozen=True)
class TurnUsage:
    """Complete OpenRouter usage for one model-backed Telegram turn."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    call_count: int = 0
    tokens_available: bool = False
    cost_available: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class SpendAccumulator(BaseCallbackHandler):
    """Sums OpenRouter cost and tokens across one turn's LLM calls."""

    def __init__(self) -> None:
        self.total_usd = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.call_count = 0
        self._tokens_complete = True
        self._cost_complete = True

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        self.call_count += 1
        cost = extract_cost_details(response)
        if cost is not None:
            self.total_usd += cost["total"]
        else:
            self._cost_complete = False

        try:
            generation = response.generations[-1][-1]
        except (AttributeError, IndexError, TypeError):
            self._tokens_complete = False
            return
        message = getattr(generation, "message", None)
        usage = getattr(message, "usage_metadata", None)
        if not isinstance(usage, dict):
            self._tokens_complete = False
            return
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if not (
            isinstance(input_tokens, int)
            and not isinstance(input_tokens, bool)
            and input_tokens >= 0
            and isinstance(output_tokens, int)
            and not isinstance(output_tokens, bool)
            and output_tokens >= 0
        ):
            self._tokens_complete = False
            return
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def snapshot(self) -> TurnUsage:
        """Freeze the totals for transport with the finished reply."""
        return TurnUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost_usd=self.total_usd,
            call_count=self.call_count,
            tokens_available=self.call_count > 0 and self._tokens_complete,
            cost_available=self.call_count > 0 and self._cost_complete,
        )


def subject_for(binding: Binding | None, chat_id: int) -> str:
    """The budget subject for a chat: its Odoo user, or the chat itself.

    The single place the ``user:`` / ``chat:`` prefixes are produced, so a
    metered turn and its limit check can never disagree about the key.
    """
    return user_subject(binding.odoo_login) if binding is not None else chat_subject(chat_id)


class BurstLimiter:
    """Sliding-window guard on how fast one sender may send.

    The daily caps bound a day; they do nothing about 200 messages in a
    minute, each of which is a real paid model call before the counter
    catches up. This bounds the seconds.

    Measures when messages were SENT, not when they were processed. The bot
    answers serially at several seconds per turn, so timestamping at
    processing time made the window slide faster than the messages could be
    counted and the limit was unreachable — 13 messages in 40s sailed
    through a 10-per-60s limit (found live 2026-08-31). Callers pass the
    platform's own send timestamp; the clock is only a fallback.

    That forces wall-clock (``time.time``) rather than ``time.monotonic``,
    since the two cannot be compared: message timestamps are epoch seconds.
    A clock step from time sync could briefly distort one 60-second window,
    which is an acceptable trade for counting the right thing.

    Deliberately in-memory: a burst window is meaningless across a restart,
    and the bot runs on a single asyncio event loop, so a plain dict needs
    no lock. Stale keys are pruned on every call — otherwise a flood from
    many distinct senders would grow this map without bound.
    """

    def __init__(
        self,
        limit: int,
        window_seconds: float,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._seen: dict[int, deque[float]] = {}
        self._warned: set[int] = set()

    def _prune(self, now: float) -> None:
        cutoff = now - self._window
        for key in list(self._seen):
            hits = self._seen[key]
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if not hits:
                del self._seen[key]
                self._warned.discard(key)

    def allow(self, key: int, when: float | None = None) -> bool:
        """Record an attempt; False when it breaches the window.

        :param when: epoch seconds the message was SENT. Falls back to the
            clock when the update carries no timestamp (button taps).

        A refused attempt is NOT recorded — otherwise a sustained flood
        would keep extending its own window and never recover.
        """
        if self._limit <= 0:  # 0 disables the guard entirely
            return True
        now = self._clock() if when is None else when
        self._prune(now)
        hits = self._seen.setdefault(key, deque())
        if len(hits) >= self._limit:
            return False
        hits.append(now)
        self._warned.discard(key)
        return True

    def should_warn(self, key: int) -> bool:
        """True only on the FIRST refusal in a window.

        This is what makes the bot say "slow down" once and then go quiet:
        answering every message of a flood would make the bot the amplifier
        and trip Telegram's own per-chat send limit.
        """
        if key in self._warned:
            return False
        self._warned.add(key)
        return True


BURST_REPLY = (
    "You're sending messages faster than I can answer them. Give me a moment and try again."
)


def over_limit(usage: UsageRow, global_usage: UsageRow, settings: Settings) -> str | None:
    """User-facing refusal if today's usage has hit a configured limit.

    Checked BEFORE a turn starts, never mid-turn — a single expensive
    question can still exceed the cap in one shot; this stops repeated
    abuse, it does not meter to the cent in real time.
    """
    if usage.message_count >= settings.daily_message_limit:
        return (
            f"You've reached today's message limit ({settings.daily_message_limit}). "
            "Please try again after midnight, or ask your "
            "administrator to raise the limit."
        )
    if (
        settings.daily_spend_limit_usd is not None
        and usage.spend_usd >= settings.daily_spend_limit_usd
    ):
        return (
            f"You've reached today's usage budget "
            f"(${settings.daily_spend_limit_usd:.2f}). Please try again after "
            "midnight, or ask your administrator to raise the limit."
        )
    if (
        settings.global_daily_spend_limit_usd is not None
        and global_usage.spend_usd >= settings.global_daily_spend_limit_usd
    ):
        # Deliberately vague: this ceiling is shared, and naming the figure
        # or the shortfall would leak other people's usage to whoever
        # happens to be the one refused.
        return (
            "The assistant has reached its overall usage budget for today. "
            "Please try again after midnight, or ask your administrator to "
            "raise the limit."
        )
    return None
