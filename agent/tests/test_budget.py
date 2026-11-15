"""Unit tests for ont_agent.budget — spend accumulation and limit checks."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from ont_agent.budget import BurstLimiter, SpendAccumulator, over_limit, subject_for
from ont_agent.config import Settings
from ont_agent.store import Binding, UsageRow

NO_GLOBAL = UsageRow()  # global row empty: only the per-user caps are in play

BASE_SETTINGS = Settings(mcp_url="http://odoo:8160/mcp", mcp_bearer_key="k")


def _llm_result(metadata: dict[str, Any], usage_metadata: dict[str, Any] | None = None) -> Any:
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    if usage_metadata is not None and "total_tokens" not in usage_metadata:
        usage_metadata = {
            **usage_metadata,
            "total_tokens": int(usage_metadata.get("input_tokens") or 0)
            + int(usage_metadata.get("output_tokens") or 0),
        }
    message = AIMessage(content="hi", response_metadata=metadata, usage_metadata=usage_metadata)
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_spend_accumulator_sums_across_multiple_llm_calls() -> None:
    accumulator = SpendAccumulator()
    accumulator.on_llm_end(
        _llm_result({"cost": 0.001}, {"input_tokens": 1_000, "output_tokens": 100}),
        run_id=uuid4(),
    )
    accumulator.on_llm_end(
        _llm_result({"cost": 0.0025}, {"input_tokens": 2_000, "output_tokens": 250}),
        run_id=uuid4(),
    )

    usage = accumulator.snapshot()
    assert accumulator.total_usd == 0.0035
    assert usage.total_tokens == 3_350
    assert usage.input_tokens == 3_000
    assert usage.output_tokens == 350
    assert usage.call_count == 2
    assert usage.tokens_available is True
    assert usage.cost_available is True


def test_spend_accumulator_ignores_missing_or_bad_cost() -> None:
    accumulator = SpendAccumulator()
    accumulator.on_llm_end(_llm_result({}), run_id=uuid4())
    accumulator.on_llm_end(_llm_result({"cost": "free"}), run_id=uuid4())
    assert accumulator.total_usd == 0.0
    assert accumulator.snapshot().cost_available is False


def test_spend_accumulator_marks_incomplete_token_metadata_unavailable() -> None:
    accumulator = SpendAccumulator()
    accumulator.on_llm_end(
        _llm_result(
            {"cost": 0.001},
            {
                "input_tokens": 100,
                "output_tokens": 20,
                "input_token_details": {"cache_read": 40},
            },
        ),
        run_id=uuid4(),
    )
    accumulator.on_llm_end(_llm_result({"cost": 0.001}), run_id=uuid4())

    usage = accumulator.snapshot()
    assert usage.total_tokens == 120  # cache reads are already included in input tokens
    assert usage.tokens_available is False  # one call lacked usage, so 120 is incomplete


def test_over_limit_none_when_under_both_limits() -> None:
    settings = BASE_SETTINGS.model_copy(
        update={"daily_message_limit": 200, "daily_spend_limit_usd": 5.0}
    )
    assert over_limit(UsageRow(message_count=1, spend_usd=0.10), NO_GLOBAL, settings) is None


def test_over_limit_message_count() -> None:
    settings = BASE_SETTINGS.model_copy(update={"daily_message_limit": 5})
    refusal = over_limit(UsageRow(message_count=5, spend_usd=0.0), NO_GLOBAL, settings)
    assert refusal is not None
    assert "5" in refusal


def test_over_limit_spend_cap() -> None:
    settings = BASE_SETTINGS.model_copy(update={"daily_spend_limit_usd": 1.0})
    refusal = over_limit(UsageRow(message_count=1, spend_usd=1.0), NO_GLOBAL, settings)
    assert refusal is not None
    assert "1.00" in refusal


def test_over_limit_disabled_spend_cap_never_trips() -> None:
    # daily_spend_limit_usd defaults to None (disabled).
    refusal = over_limit(UsageRow(message_count=1, spend_usd=999.0), NO_GLOBAL, BASE_SETTINGS)
    assert refusal is None


def test_over_limit_message_count_and_spend_refusals_are_distinguishable() -> None:
    settings = BASE_SETTINGS.model_copy(
        update={"daily_message_limit": 3, "daily_spend_limit_usd": 2.0}
    )
    count_refusal = over_limit(UsageRow(message_count=3, spend_usd=0.0), NO_GLOBAL, settings)
    spend_refusal = over_limit(UsageRow(message_count=0, spend_usd=2.0), NO_GLOBAL, settings)
    assert count_refusal != spend_refusal


# --- global spend cap ------------------------------------------------------


def test_global_cap_refuses_a_user_who_is_personally_under() -> None:
    # The point of the global cap: one person's own budget being fine says
    # nothing about the bill, which every user shares.
    settings = BASE_SETTINGS.model_copy(
        update={"daily_spend_limit_usd": 5.0, "global_daily_spend_limit_usd": 10.0}
    )
    refusal = over_limit(
        UsageRow(message_count=1, spend_usd=0.10),
        UsageRow(message_count=400, spend_usd=10.0),
        settings,
    )
    assert refusal is not None
    assert "overall usage budget" in refusal


def test_global_refusal_leaks_no_other_usage_figures() -> None:
    # Whoever happens to be refused must not learn what everyone else spent.
    settings = BASE_SETTINGS.model_copy(update={"global_daily_spend_limit_usd": 10.0})
    refusal = over_limit(UsageRow(), UsageRow(message_count=987, spend_usd=42.5), settings)
    assert refusal is not None
    assert "42.5" not in refusal
    assert "987" not in refusal


def test_global_cap_disabled_by_default() -> None:
    assert over_limit(UsageRow(), UsageRow(spend_usd=999.0), BASE_SETTINGS) is None


# --- subjects --------------------------------------------------------------


def test_subject_is_the_odoo_user_not_the_chat() -> None:
    from datetime import UTC, datetime, timedelta

    binding = Binding(
        chat_id=5,
        odoo_login="priya@acme",
        odoo_name="Priya",
        access_token="jwt",
        refresh_token="ref",
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    # Same person, two chats -> one budget. This is the whole point of the
    # re-key: per-chat metering handed a second chat a second allowance.
    assert subject_for(binding, 5) == subject_for(binding, 99) == "user:priya@acme"
    # Unbound chats still get metered, just under their own key.
    assert subject_for(None, 5) == "chat:5"


# --- burst limiter ---------------------------------------------------------


class FakeClock:
    """Hand-advanced clock: burst tests must never actually sleep."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_burst_allows_up_to_the_limit_then_refuses() -> None:
    clock = FakeClock()
    limiter = BurstLimiter(limit=3, window_seconds=60, clock=clock)
    assert [limiter.allow(7) for _ in range(3)] == [True, True, True]
    assert limiter.allow(7) is False


def test_burst_window_slides_so_a_sender_recovers() -> None:
    clock = FakeClock()
    limiter = BurstLimiter(limit=2, window_seconds=60, clock=clock)
    assert limiter.allow(7) and limiter.allow(7)
    assert limiter.allow(7) is False

    clock.now += 61  # the window has passed
    assert limiter.allow(7) is True


def test_burst_keys_are_independent() -> None:
    limiter = BurstLimiter(limit=1, window_seconds=60, clock=FakeClock())
    assert limiter.allow(1) is True
    assert limiter.allow(1) is False
    assert limiter.allow(2) is True  # one flooder must not silence everyone


def test_burst_warns_once_then_stays_silent() -> None:
    # Replying to every blocked message would make the bot amplify the flood
    # and trip Telegram's own per-chat send limit.
    limiter = BurstLimiter(limit=1, window_seconds=60, clock=FakeClock())
    limiter.allow(7)
    assert limiter.allow(7) is False
    assert limiter.should_warn(7) is True
    assert limiter.should_warn(7) is False
    assert limiter.should_warn(7) is False


def test_burst_warns_again_after_the_window_clears() -> None:
    clock = FakeClock()
    limiter = BurstLimiter(limit=1, window_seconds=60, clock=clock)
    limiter.allow(7)
    limiter.allow(7)
    assert limiter.should_warn(7) is True

    clock.now += 61
    assert limiter.allow(7) is True  # recovered
    limiter.allow(7)
    assert limiter.should_warn(7) is True  # a NEW flood is worth one warning


def test_burst_prunes_stale_senders() -> None:
    # A flood from many distinct ids must not grow the map without bound.
    clock = FakeClock()
    limiter = BurstLimiter(limit=1, window_seconds=60, clock=clock)
    for sender in range(50):
        limiter.allow(sender)
    assert len(limiter._seen) == 50

    clock.now += 61
    limiter.allow(999)
    assert len(limiter._seen) == 1  # the 50 expired ones are gone


def test_burst_limit_of_zero_disables_the_guard() -> None:
    limiter = BurstLimiter(limit=0, window_seconds=60, clock=FakeClock())
    assert all(limiter.allow(7) for _ in range(100))


def test_slow_processing_must_not_defeat_the_burst_window() -> None:
    """The live failure of 2026-08-31.

    The bot answers serially at ~7s a turn, so ten messages take ~70s to
    work through. Timestamping each one when it was PROCESSED made the
    60s window slide faster than the messages could be counted, and 13
    messages sent in 40 seconds were all answered.
    """
    processing_clock = FakeClock()
    limiter = BurstLimiter(limit=10, window_seconds=60, clock=processing_clock)

    sent = processing_clock.now  # the whole flood arrives at once
    verdicts = []
    for _ in range(13):
        verdicts.append(limiter.allow(7, sent))
        processing_clock.now += 7  # ...but each answer takes 7 seconds

    # Counting by SEND time, the flood is caught from the 11th message.
    assert verdicts[:10] == [True] * 10
    assert verdicts[10:] == [False] * 3


def test_processing_time_is_the_fallback_when_no_timestamp_is_given() -> None:
    # Button taps carry the date of the message the button hangs off, which
    # may be hours old — those fall back to the clock instead.
    clock = FakeClock()
    limiter = BurstLimiter(limit=2, window_seconds=60, clock=clock)
    assert limiter.allow(7) is True
    assert limiter.allow(7) is True
    assert limiter.allow(7) is False
