"""Unit tests for ont_agent.proactive — due logic, verdicts, fan-out.

No sleeping and no network: the schedule is a pure function with an injected
clock, and the model turn is monkeypatched.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from ont_agent import proactive
from ont_agent.budget import subject_for
from ont_agent.config import Settings
from ont_agent.proactive import (
    Verdict,
    brief_due,
    digest_due,
    in_quiet_hours,
    parse_verdict,
    watcher_cutoff,
    zone_for,
)
from ont_agent.store import Binding, BindingStore, Watcher

SETTINGS = Settings(mcp_url="http://odoo:8160/mcp", mcp_bearer_key="k")


def _binding(
    chat_id: int = 5,
    tz: str = "Asia/Kolkata",
    brief_enabled: bool = True,
    last_brief_date: str = "",
) -> Binding:
    return Binding(
        chat_id=chat_id,
        odoo_login="priya@acme",
        odoo_name="CFO Priya",
        access_token="jwt",
        refresh_token="ref",
        expires_at=datetime.now(UTC) + timedelta(minutes=30),
        tz=tz,
        brief_enabled=brief_enabled,
        last_brief_date=last_brief_date,
    )


def _at(hour: int, minute: int = 0, tz: str = "Asia/Kolkata") -> datetime:
    """A UTC instant that reads as hour:minute in ``tz`` on 2026-09-01."""
    from zoneinfo import ZoneInfo

    local = datetime(2026, 9, 1, hour, minute, tzinfo=ZoneInfo(tz))
    return local.astimezone(UTC)


# --- timezone ---------------------------------------------------------------


def test_zone_falls_back_when_the_user_has_none() -> None:
    # Odoo users routinely have no timezone set; the brief must still land at
    # a sane hour rather than the server's.
    assert str(zone_for(_binding(tz=""), SETTINGS)) == SETTINGS.default_timezone
    assert str(zone_for(_binding(tz="Not/AZone"), SETTINGS)) == SETTINGS.default_timezone


def test_zone_prefers_the_users_own() -> None:
    assert str(zone_for(_binding(tz="Europe/London"), SETTINGS)) == "Europe/London"


# --- brief_due --------------------------------------------------------------


def test_brief_not_due_before_the_hour() -> None:
    assert brief_due(_binding(), _at(7, 59), SETTINGS) is False


def test_brief_due_at_the_hour() -> None:
    assert brief_due(_binding(), _at(8, 0), SETTINGS) is True


def test_brief_not_due_twice_in_one_day() -> None:
    already = _binding(last_brief_date="2026-09-01")
    assert brief_due(already, _at(9), SETTINGS) is False


def test_brief_catches_up_after_downtime_but_not_forever() -> None:
    # A laptop asleep at 08:00 and woken at 11:00 should still brief; one
    # woken at 14:00 has missed the moment and should stay quiet.
    assert brief_due(_binding(), _at(11), SETTINGS) is True
    assert brief_due(_binding(), _at(14), SETTINGS) is False


def test_catchup_window_is_configurable() -> None:
    wide = SETTINGS.model_copy(update={"brief_catchup_hours": 12})
    assert brief_due(_binding(), _at(19), wide) is True

    none_at_all = SETTINGS.model_copy(update={"brief_catchup_hours": 0})
    assert brief_due(_binding(), _at(8, 0), none_at_all) is True  # exactly on time
    assert brief_due(_binding(), _at(8, 30), none_at_all) is False


def test_brief_respects_the_opt_out() -> None:
    assert brief_due(_binding(brief_enabled=False), _at(9), SETTINGS) is False


def test_the_hour_is_the_users_local_hour_not_the_servers() -> None:
    # The point of storing tz: 08:00 in London is not 08:00 in Kolkata, and
    # one bot serves both.
    london = _binding(tz="Europe/London")
    kolkata = _binding(tz="Asia/Kolkata")
    moment = _at(8, 0, tz="Europe/London")  # 08:00 London = 12:30 Kolkata

    assert brief_due(london, moment, SETTINGS) is True
    # Kolkata is past its catch-up window by then, so it must NOT fire again.
    assert brief_due(kolkata, moment, SETTINGS) is False


# --- quiet hours ------------------------------------------------------------


def test_quiet_at_night_and_awake_in_the_day() -> None:
    assert in_quiet_hours(_binding(), _at(3), SETTINGS) is True
    assert in_quiet_hours(_binding(), _at(9), SETTINGS) is False


def test_the_quiet_window_wraps_midnight() -> None:
    """The default window is 21:00-08:00, so it spans midnight. A naive
    `start <= hour < end` gets every one of these backwards."""
    assert in_quiet_hours(_binding(), _at(22), SETTINGS) is True  # before midnight
    assert in_quiet_hours(_binding(), _at(2), SETTINGS) is True  # after midnight
    assert in_quiet_hours(_binding(), _at(12), SETTINGS) is False  # the middle of the day
    # The boundaries themselves: start is quiet, end is not.
    assert in_quiet_hours(_binding(), _at(21), SETTINGS) is True
    assert in_quiet_hours(_binding(), _at(8), SETTINGS) is False


def test_a_window_inside_one_day_also_works() -> None:
    siesta = SETTINGS.model_copy(update={"quiet_hours_start": 13, "quiet_hours_end": 16})
    assert in_quiet_hours(_binding(), _at(14), siesta) is True
    assert in_quiet_hours(_binding(), _at(9), siesta) is False
    assert in_quiet_hours(_binding(), _at(22), siesta) is False  # not quiet by default here


def test_equal_values_disable_quiet_hours() -> None:
    # An existing deployment that sets neither must behave exactly as before.
    off = SETTINGS.model_copy(update={"quiet_hours_start": 0, "quiet_hours_end": 0})
    for hour in (0, 3, 9, 21, 23):
        assert in_quiet_hours(_binding(), _at(hour), off) is False


def test_quiet_hours_use_the_users_local_hour_not_the_servers() -> None:
    # One bot, two users: 05:00 in London (BST) is 09:30 in Kolkata, so the
    # same instant must be quiet for one and not the other.
    moment = _at(5, 0, tz="Europe/London")

    assert in_quiet_hours(_binding(tz="Europe/London"), moment, SETTINGS) is True
    assert in_quiet_hours(_binding(tz="Asia/Kolkata"), moment, SETTINGS) is False


# --- watcher scheduling -----------------------------------------------------


def test_watcher_cutoff_follows_the_configured_interval() -> None:
    now = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    # Eight hours by default. Pinned deliberately: this number is the bot's
    # largest recurring cost, so shortening it should require editing a test
    # that says so rather than passing unnoticed.
    assert watcher_cutoff(now, SETTINGS) == now - timedelta(hours=8)

    brisk = SETTINGS.model_copy(update={"watcher_interval_minutes": 15})
    assert watcher_cutoff(now, brisk) == now - timedelta(minutes=15)


# --- verdict parsing --------------------------------------------------------


def test_verdict_parses_a_clean_reply() -> None:
    verdict = parse_verdict('{"triggered": true, "fingerprint": "INV/1,INV/2", "message": "Two."}')
    assert verdict == Verdict(triggered=True, fingerprint="INV/1,INV/2", message="Two.")


def test_verdict_survives_prose_and_code_fences_around_the_json() -> None:
    # Models wrap JSON in fences and commentary; the alert must still work.
    fenced = 'Here you go:\n```json\n{"triggered": false, "fingerprint": ""}\n```'
    assert parse_verdict(fenced).triggered is False


def test_malformed_verdicts_never_raise_and_never_alert() -> None:
    # A confused model must not manufacture an alert, and must not take the
    # ticker down either.
    for junk in ("", "not json at all", "{oh dear", "[1, 2, 3]", "null"):
        assert parse_verdict(junk) == Verdict()


def test_missing_keys_default_safely() -> None:
    assert parse_verdict('{"triggered": true}') == Verdict(triggered=True)


# --- the dedupe rule (the reason watchers are bearable) ---------------------


async def _watcher_run(
    monkeypatch: pytest.MonkeyPatch, store_dsn: str, verdicts: list[str]
) -> tuple[list[bool], list[tuple[int, str]], Watcher]:
    """Run check_watcher repeatedly against scripted model replies."""
    replies = iter(verdicts)
    sent: list[tuple[int, str]] = []

    async def fake_turn(*args: object, **kwargs: object) -> str:
        return next(replies)

    monkeypatch.setattr(proactive, "_run_readonly_turn", fake_turn)

    async def send(chat_id: int, text: str) -> None:
        sent.append((chat_id, text))

    async with BindingStore(store_dsn) as store:
        binding = _binding()
        await store.save_binding(binding)
        watcher_id = await store.add_watcher(5, "priya@acme", "any invoice 60 days overdue")
        fired = []
        for _ in verdicts:
            current = (await store.list_watchers(5))[0]
            fired.append(await proactive.check_watcher(store, SETTINGS, binding, current, send))
        stored = (await store.list_watchers(5))[0]
    assert watcher_id > 0
    return fired, sent, stored


async def test_the_same_finding_alerts_once_not_every_hour(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rule that makes watchers usable.

    A watcher that repeats itself hourly is worse than no watcher — people
    mute it, and then miss the one that mattered.
    """
    same = '{"triggered": true, "fingerprint": "INV/2026/00003", "message": "One overdue."}'
    fired, sent, stored = await _watcher_run(monkeypatch, store_dsn, [same, same, same])

    assert fired == [True, False, False]
    assert len(sent) == 1
    assert "One overdue." in sent[0][1]
    assert stored.fingerprint == "INV/2026/00003"


async def test_a_new_finding_alerts_again(store_dsn: str, monkeypatch: pytest.MonkeyPatch) -> None:
    first = '{"triggered": true, "fingerprint": "INV/1", "message": "One."}'
    second = '{"triggered": true, "fingerprint": "INV/1,INV/2", "message": "Now two."}'
    fired, sent, _ = await _watcher_run(monkeypatch, store_dsn, [first, second])

    assert fired == [True, True]
    assert "Now two." in sent[1][1]


async def test_a_condition_that_clears_can_alert_afresh(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Fires, resolves, then happens again: the third check is news, because
    # a cleared condition wipes the stored fingerprint.
    alert = '{"triggered": true, "fingerprint": "INV/1", "message": "Overdue."}'
    quiet = '{"triggered": false, "fingerprint": "", "message": ""}'
    fired, sent, _ = await _watcher_run(monkeypatch, store_dsn, [alert, quiet, alert])

    assert fired == [True, False, True]
    assert len(sent) == 2


async def test_an_untriggered_check_is_still_stamped(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Otherwise a quiet watcher would be re-evaluated on every single tick,
    # burning a model call each time.
    _, sent, stored = await _watcher_run(monkeypatch, store_dsn, ['{"triggered": false}'])
    assert sent == []
    assert stored.last_checked_at is not None


# --- budget -----------------------------------------------------------------


async def test_proactive_work_stops_at_the_global_spend_ceiling(store_dsn: str) -> None:
    # Background jobs running unattended are exactly how a bill gets away
    # from you, so both spend caps apply even though the message cap does not.
    capped = SETTINGS.model_copy(update={"global_daily_spend_limit_usd": 1.0})
    async with BindingStore(store_dsn) as store:
        binding = _binding()
        await store.save_binding(binding)
        assert await proactive.spend_exhausted(store, capped, binding) is False

        await store.record_usage("user:someone-else", datetime.now(UTC).date().isoformat(), 1.0)
        assert await proactive.spend_exhausted(store, capped, binding) is True


async def test_proactive_work_stops_at_the_per_user_ceiling(store_dsn: str) -> None:
    capped = SETTINGS.model_copy(update={"daily_spend_limit_usd": 0.5})
    async with BindingStore(store_dsn) as store:
        binding = _binding()
        await store.save_binding(binding)
        await store.record_usage("user:priya@acme", datetime.now(UTC).date().isoformat(), 0.5)
        assert await proactive.spend_exhausted(store, capped, binding) is True


async def test_no_caps_configured_never_blocks(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        binding = _binding()
        await store.save_binding(binding)
        await store.record_usage("user:priya@acme", datetime.now(UTC).date().isoformat(), 999.0)
        assert await proactive.spend_exhausted(store, SETTINGS, binding) is False


# --- the tick ---------------------------------------------------------------


async def _quiet_hours_tick(
    store_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
    hours: list[int],
) -> tuple[list[tuple[int, str]], str]:
    """Run tick at each local hour; return what was sent and the final stamp."""
    sent: list[tuple[int, str]] = []

    async def send(chat_id: int, text: str) -> None:
        sent.append((chat_id, text))

    async def fake_turn(*args: object, **kwargs: object) -> str:
        return '{"triggered": true, "fingerprint": "INV/7", "message": "One overdue."}'

    async def fake_refresh(store: Any, settings: Any, chat_id: int) -> Binding | None:
        binding: Binding | None = await store.get_binding(chat_id)
        return binding

    monkeypatch.setattr(proactive, "_run_readonly_turn", fake_turn)
    monkeypatch.setattr(proactive, "refresh_binding", fake_refresh)

    async with BindingStore(store_dsn) as store:
        await store.save_binding(_binding())
        # The brief is opted out through the real API: save_binding does not
        # persist brief_enabled or last_brief_date (mark_brief_sent owns
        # that), so only the watcher path is under test here.
        await store.set_brief_enabled(5, False)
        await store.add_watcher(5, "priya@acme", "any invoice 60 days overdue")
        for hour in hours:
            await proactive.tick(store, SETTINGS, send, now=_at(hour))
        stored = (await store.list_watchers(5))[0]
    return sent, str(stored.last_checked_at or "")


async def test_a_watcher_is_not_checked_during_quiet_hours(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skipped, not suppressed — and the difference is the whole design.

    Nothing is sent AND last_checked_at stays empty, so due_watchers still
    returns it. Had the check run and only the message been withheld, the
    fingerprint would have been stamped and the alert lost for good.
    """
    sent, stamp = await _quiet_hours_tick(store_dsn, monkeypatch, [2, 3, 4])

    assert sent == []
    assert stamp == ""  # still due


async def test_a_deferred_alert_arrives_when_the_day_starts(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The test that would fail under "check anyway, suppress the send": by
    # 09:00 the fingerprint would already match and nothing would fire.
    sent, stamp = await _quiet_hours_tick(store_dsn, monkeypatch, [3, 9])

    assert len(sent) == 1
    assert "One overdue." in sent[0][1]
    assert stamp != ""  # checked at last


async def test_disabling_quiet_hours_restores_the_old_behaviour(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[tuple[int, str]] = []

    async def send(chat_id: int, text: str) -> None:
        sent.append((chat_id, text))

    async def fake_turn(*args: object, **kwargs: object) -> str:
        return '{"triggered": true, "fingerprint": "INV/7", "message": "Overdue at 3am."}'

    async def fake_refresh(store: Any, settings: Any, chat_id: int) -> Binding | None:
        binding: Binding | None = await store.get_binding(chat_id)
        return binding

    monkeypatch.setattr(proactive, "_run_readonly_turn", fake_turn)
    monkeypatch.setattr(proactive, "refresh_binding", fake_refresh)

    off = SETTINGS.model_copy(update={"quiet_hours_start": 0, "quiet_hours_end": 0})
    async with BindingStore(store_dsn) as store:
        await store.save_binding(_binding())
        await store.set_brief_enabled(5, False)
        await store.add_watcher(5, "priya@acme", "any invoice 60 days overdue")
        await proactive.tick(store, off, send, now=_at(3))

    assert len(sent) == 1  # 3am, and that is what was asked for


def test_the_weekly_digest_waits_out_the_quiet_hours() -> None:
    # digest_due's catch-up branch returns True for any later day in a week
    # with no digest yet, regardless of hour — so after downtime it could
    # have landed at 3am on the Tuesday.
    tuesday_night = datetime(2026, 9, 8, 3, 0, tzinfo=ZoneInfo("Asia/Kolkata")).astimezone(UTC)
    tuesday_day = datetime(2026, 9, 8, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata")).astimezone(UTC)

    assert digest_due(_binding(), tuesday_night, SETTINGS) is False
    assert digest_due(_binding(), tuesday_day, SETTINGS) is True


async def test_tick_is_a_no_op_when_proactive_is_off(store_dsn: str) -> None:
    sent: list[tuple[int, str]] = []

    async def send(chat_id: int, text: str) -> None:
        sent.append((chat_id, text))

    off = SETTINGS.model_copy(update={"proactive": False})
    async with BindingStore(store_dsn) as store:
        await store.save_binding(_binding())
        await proactive.tick(store, off, send)
    assert sent == []


async def test_tick_briefs_a_due_user_once_and_stamps_it(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[tuple[int, str]] = []

    async def send(chat_id: int, text: str) -> None:
        sent.append((chat_id, text))

    async def fake_turn(*args: object, **kwargs: object) -> str:
        return "Cash: 12 lakh. Overdue: 3 invoices."

    async def fake_refresh(store: Any, settings: Any, chat_id: int) -> Binding | None:
        binding: Binding | None = await store.get_binding(chat_id)
        return binding

    monkeypatch.setattr(proactive, "_run_readonly_turn", fake_turn)
    monkeypatch.setattr(proactive, "refresh_binding", fake_refresh)

    async with BindingStore(store_dsn) as store:
        await store.save_binding(_binding())
        await proactive.tick(store, SETTINGS, send, now=_at(9))
        # A second pass in the same day must NOT re-brief.
        await proactive.tick(store, SETTINGS, send, now=_at(10))

    assert len(sent) == 1
    assert "Overdue" in sent[0][1]
    # The very first brief explains itself, since the default is opt-out.
    assert "brief off" in sent[0][1]


async def test_logging_out_and_in_again_does_not_send_a_second_brief(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[tuple[int, str]] = []

    async def send(chat_id: int, text: str) -> None:
        sent.append((chat_id, text))

    async def fake_turn(*args: object, **kwargs: object) -> str:
        return "Cash: 12 lakh."

    async def fake_refresh(store: Any, settings: Any, chat_id: int) -> Binding | None:
        binding: Binding | None = await store.get_binding(chat_id)
        return binding

    monkeypatch.setattr(proactive, "_run_readonly_turn", fake_turn)
    monkeypatch.setattr(proactive, "refresh_binding", fake_refresh)
    no_digest = SETTINGS.model_copy(update={"digest": False})

    async with BindingStore(store_dsn) as store:
        await store.save_binding(_binding())
        await proactive.tick(store, no_digest, send, now=_at(9))
        await store.delete_binding(5)  # /logout
        await store.save_binding(_binding())  # and straight back in
        await proactive.tick(store, no_digest, send, now=_at(10))

    assert len(sent) == 1


async def test_one_broken_user_does_not_stop_everyone_elses_brief(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[tuple[int, str]] = []

    async def send(chat_id: int, text: str) -> None:
        sent.append((chat_id, text))

    async def explode_for_one(
        store: Any, settings: Any, binding: Any, *a: object, **k: object
    ) -> str:
        if binding.chat_id == 1:
            raise RuntimeError("Odoo hiccup")
        return "fine"

    async def fake_refresh(store: Any, settings: Any, chat_id: int) -> Binding | None:
        binding: Binding | None = await store.get_binding(chat_id)
        return binding

    monkeypatch.setattr(proactive, "_run_readonly_turn", explode_for_one)
    monkeypatch.setattr(proactive, "refresh_binding", fake_refresh)

    async with BindingStore(store_dsn) as store:
        await store.save_binding(_binding(chat_id=1))
        await store.save_binding(_binding(chat_id=2))
        await proactive.tick(store, SETTINGS, send, now=_at(9))

    assert [chat_id for chat_id, _ in sent] == [2]


async def test_a_dead_binding_is_skipped_quietly(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[tuple[int, str]] = []

    async def send(chat_id: int, text: str) -> None:
        sent.append((chat_id, text))

    async def dead(store: Any, settings: Any, chat_id: int) -> Binding | None:
        return None  # refresh failed; the grant is gone

    monkeypatch.setattr(proactive, "refresh_binding", dead)

    async with BindingStore(store_dsn) as store:
        await store.save_binding(_binding())
        await proactive.tick(store, SETTINGS, send, now=_at(9))

    assert sent == []


# --- presentation -----------------------------------------------------------


def test_proactive_text_is_markdown_never_raw_html() -> None:
    """Proactive messages ride the same pipeline as answers.

    ``prepare_reply_text`` escapes the text and THEN converts Markdown to
    Telegram HTML, so a raw tag written here reaches the user as a literal
    "<b>" (found live 2026-09-01, in the very first brief ever sent).
    """
    from ont_agent.bot import BotReply, prepare_reply_text
    from ont_agent.proactive import FIRST_BRIEF_PREAMBLE

    assert "<" not in FIRST_BRIEF_PREAMBLE
    rendered, parse_mode = prepare_reply_text(BotReply(text=FIRST_BRIEF_PREAMBLE))
    assert parse_mode == "HTML"
    # Bold survives as real formatting, and no escaped tag is shown.
    assert "<b>Your morning brief</b>" in rendered
    assert "&lt;" not in rendered


async def test_a_watcher_alert_is_markdown_too(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ont_agent.bot import BotReply, prepare_reply_text

    alert = '{"triggered": true, "fingerprint": "INV/1", "message": "One is overdue."}'
    _, sent, _ = await _watcher_run(monkeypatch, store_dsn, [alert])

    text = sent[0][1]
    assert "<" not in text
    rendered, _mode = prepare_reply_text(BotReply(text=text))
    assert "&lt;" not in rendered


# --- weekly digest ---------------------------------------------------------


def _monday(hour: int, tz: str = "Asia/Kolkata") -> datetime:
    """A UTC instant reading as Monday 2026-08-31 hour:00 in ``tz``."""
    from zoneinfo import ZoneInfo

    return datetime(2026, 8, 31, hour, 0, tzinfo=ZoneInfo(tz)).astimezone(UTC)


def test_iso_week_stamp_is_a_week_not_a_date() -> None:
    from ont_agent.proactive import iso_week

    monday = _monday(9)
    wednesday = monday + timedelta(days=2)
    assert iso_week(monday) == iso_week(wednesday)
    assert iso_week(monday) != iso_week(monday + timedelta(days=7))


def test_digest_waits_for_mondays_hour() -> None:
    from ont_agent.proactive import digest_due

    assert digest_due(_binding(), _monday(7), SETTINGS) is False
    assert digest_due(_binding(), _monday(8), SETTINGS) is True


def test_digest_goes_out_once_a_week() -> None:
    from ont_agent.proactive import digest_due, iso_week

    sent = _binding()
    sent = Binding(**{**sent.__dict__, "last_digest_week": iso_week(_monday(9))})
    assert digest_due(sent, _monday(9), SETTINGS) is False
    # ...but the next week is due again.
    assert digest_due(sent, _monday(9) + timedelta(days=7), SETTINGS) is True


def test_a_bot_that_was_off_on_monday_still_sends_later() -> None:
    """Catch-up falls out of the state-driven design: "from Monday onward"
    stays true all week, so a machine first switched on midweek still
    delivers rather than silently skipping."""
    from ont_agent.proactive import digest_due

    wednesday = _monday(9) + timedelta(days=2)
    assert digest_due(_binding(), wednesday, SETTINGS) is True


def test_the_digest_can_be_switched_off() -> None:
    from ont_agent.proactive import digest_due

    off = SETTINGS.model_copy(update={"digest": False})
    assert digest_due(_binding(), _monday(9), off) is False


def test_an_unused_week_sends_nothing_at_all() -> None:
    # Someone who did not use the bot does not need telling.
    from ont_agent.proactive import compose_digest

    assert compose_digest([], 0, 0.0, "Priya") == ""


def test_a_week_of_questions_but_no_changes_says_so() -> None:
    """The reassuring case — and it only means something because the busy
    weeks say otherwise."""
    from ont_agent.proactive import compose_digest

    text = compose_digest([], 42, 0.51, "Priya")
    assert "No changes were made" in text
    assert "42 question" in text


def test_actions_are_listed_with_what_became_of_them() -> None:
    from ont_agent.proactive import compose_digest
    from ont_agent.store import LoggedAction

    when = datetime(2026, 8, 27, 10, 0, tzinfo=UTC)
    actions = [
        LoggedAction(1, 5, "priya@acme", "send_email", "send_email\n  to: a@b", "done", when),
        LoggedAction(2, 5, "priya@acme", "send_email", "send_email\n  to: x@y", "blocked", when),
    ]
    text = compose_digest(actions, 10, 0.2, "Priya")

    assert "2 action(s)" in text
    assert "done" in text and "blocked by Odoo" in text
    # One line per action, not the indented argument transcript.
    assert "to: a@b" not in text
    assert "send email: a@b" in text


def test_an_action_headline_names_who_and_what_but_not_the_body() -> None:
    from ont_agent.proactive import action_headline

    summary = (
        "send_email\n"
        "  to: ravi@apex.in\n"
        "  subject: Overdue invoice INV/2026/0042\n"
        "  body: Dear Ravi, a long body that must not appear in a digest line"
    )
    headline = action_headline(summary)

    assert headline == 'send email: ravi@apex.in, "Overdue invoice INV/2026/0042"'
    assert "Dear Ravi" not in headline


def test_an_action_headline_without_arguments_is_just_the_tool() -> None:
    from ont_agent.proactive import action_headline

    assert action_headline("archive_records") == "archive records"
    assert action_headline("") == ""


def test_long_headline_values_are_truncated() -> None:
    from ont_agent.proactive import action_headline

    headline = action_headline("send_email\n  subject: " + "x" * 200)
    assert headline.endswith('…"')
    assert len(headline) < 80


async def test_the_digest_counts_the_whole_week_not_just_today(
    store_dsn: str,
) -> None:
    sent: list[tuple[int, str]] = []

    async def send(chat_id: int, text: str) -> None:
        sent.append((chat_id, text))

    monday = _monday(9)
    async with BindingStore(store_dsn) as store:
        binding = _binding()
        await store.save_binding(binding)
        subject = subject_for(binding, binding.chat_id)
        local = monday.astimezone(ZoneInfo("Asia/Kolkata")).date()
        for days_ago in (0, 2, 6):  # inside the seven days
            await store.record_usage(subject, (local - timedelta(days=days_ago)).isoformat(), 0.5)
        # Eight days ago: last week, must not count.
        await store.record_usage(subject, (local - timedelta(days=8)).isoformat(), 9.0)

        assert await proactive.send_digest(store, SETTINGS, binding, send, monday)

    assert "3 question(s)" in sent[0][1]
    assert "$1.50" in sent[0][1]


async def test_a_brief_with_no_readable_data_sends_a_clear_notice(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[tuple[int, str]] = []

    async def send(chat_id: int, text: str) -> None:
        sent.append((chat_id, text))

    async def fake_turn(*args: object, **kwargs: object) -> str:
        return "BRIEF_UNAVAILABLE"

    monkeypatch.setattr(proactive, "_run_readonly_turn", fake_turn)

    async with BindingStore(store_dsn) as store:
        binding = _binding()
        await store.save_binding(binding)
        assert await proactive.send_brief(store, SETTINGS, binding, send, now=_at(9))
        stored = await store.get_binding(binding.chat_id)

    # The fixed notice, without the first-brief preamble promising more.
    assert sent[0][1] == proactive.BRIEF_UNAVAILABLE_TEXT
    # Still stamped, so the ticker does not retry it all morning.
    assert stored is not None
    assert not proactive.brief_due(stored, _at(10), SETTINGS)
