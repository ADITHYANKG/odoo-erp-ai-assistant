"""What the bot does unprompted: the morning brief and standing watchers.

Everything here runs with **no human present**, which drives three rules the
rest of the module exists to honour:

* **Read-only, always.** Tools go through :func:`filter_read_only`. A write
  would pause for an approval tap nobody is there to give.
* **Stateless.** Proactive turns never touch the conversation checkpointer.
  Sharing the thread would let an 08:00 brief auto-cancel an approval the
  user left open overnight (``answer_question`` cancels a pending action on
  new input), and would push their own conversation out of the
  summarization window.
* **Never fatal.** A scheduled job that raises must not take the bot down,
  and one dead binding must not stop everyone else's brief.

Scheduling is deliberately *state-driven* rather than cron-driven: a ticker
asks "what is due now?" and the answer comes from stored timestamps. That is
idempotent, survives restarts, and gives catch-up after downtime for free —
where a fired-and-forgotten cron job would simply have been missed.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, tzinfo
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.callbacks import BaseCallbackHandler

from ont_agent.agent import ask, build_agent
from ont_agent.auth import ensure_access
from ont_agent.budget import SpendAccumulator, subject_for
from ont_agent.channels import is_whatsapp
from ont_agent.config import Settings
from ont_agent.llm import build_model
from ont_agent.log import get_logger
from ont_agent.memory import recall
from ont_agent.metrics import KIND_BRIEF, KIND_WATCHER, LatencyRecorder
from ont_agent.render import Artifact, discard
from ont_agent.store import (
    GLOBAL_SUBJECT,
    Binding,
    BindingStore,
    LoggedAction,
    Watcher,
    binding_id,
)
from ont_agent.tools import filter_read_only, get_odoo_tools, redact_tool_results

logger = get_logger(__name__)

#: The fixed brief. Sections are spelled out so the output is comparable
#: morning to morning — a brief whose shape changes daily cannot be skimmed,
#: which is the only way a brief ever gets read.
BRIEF_INSTRUCTION = """\
Produce my morning business brief from Odoo. Use these sections, in order,
and omit a section only if its tools return no data at all:

1. *Cash* — total outstanding customer invoices (unpaid, posted).
2. *Overdue* — count and total of customer invoices past their due date.
   Name the three largest customers by overdue amount.
3. *Payables* — count and total of vendor bills past their due date.
4. *Pipeline* — sales orders still in quotation, count and total.
5. *Worth knowing* — at most two short lines on anything unusual: a big
   single balance, a long-overdue account, an empty section you expected
   data in.

Keep the whole thing under 200 words. No preamble, no sign-off, no offer to
help further — this is a briefing, not a conversation. Use a compact
Markdown table only where it genuinely helps; prose otherwise. Every number
must come from a tool call.

If NONE of the sections could be filled because the tools refused access
or the data is not available to me, reply with exactly BRIEF_UNAVAILABLE
and nothing else.\
"""

#: What the model answers when it could fill no section at all.
BRIEF_UNAVAILABLE_SENTINEL = "BRIEF_UNAVAILABLE"

#: Sent in place of a brief the model could not build. A fixed, actionable
#: line rather than the model's own explanation of a permissions error,
#: which read as a broken brief (seen live 2026-10-19).
BRIEF_UNAVAILABLE_TEXT = (
    "📋 **Morning brief unavailable** — your Odoo account can't read invoices, "
    "vendor bills or quotations through the assistant. Ask your Odoo "
    "administrator for access, or use /brief off to stop these."
)

#: Asked of the model for each watcher. A structured verdict rather than
#: prose, so the decision to alert is code's to make, not the model's.
WATCHER_INSTRUCTION = """\
You are evaluating a standing alert condition against current Odoo data.

CONDITION: {condition}

Check it with read-only tools, then reply with ONE JSON object and nothing
else:

{{"triggered": true or false,
  "fingerprint": "stable identifier of exactly what triggered, e.g. sorted
                  record numbers joined by commas; empty string when not
                  triggered",
  "message": "one or two sentences telling the user what happened, with the
              numbers; empty string when not triggered"}}

The fingerprint must be identical if the same records still satisfy the
condition later, and different if different records do — it is what stops
the same alert repeating. Never include commentary outside the JSON.\
"""

#: Prefixed to a user's very first brief. Opt-out defaults mean the first one
#: arrives unannounced; this is what pays for that.
#:
#: MARKDOWN, not HTML: proactive messages go through the same
#: ``prepare_reply_text`` pipeline as answers, which escapes the text before
#: converting Markdown to Telegram HTML. Raw tags here would reach the user
#: as literal "<b>" (found live 2026-09-01).
FIRST_BRIEF_PREAMBLE = (
    "☀️ **Your morning brief** — I'll send this each morning. "
    "Use /brief off to stop, or /brief any time to ask for it early.\n\n"
)


@dataclass(frozen=True)
class Verdict:
    """A watcher evaluation, after defensive parsing."""

    triggered: bool = False
    fingerprint: str = ""
    message: str = ""


def zone_for(binding: Binding, settings: Settings) -> tzinfo:
    """The user's timezone, falling back to the configured default, then UTC.

    Odoo users routinely have no timezone set, and a client deployment can
    sit in a different region from this process — so "morning" is never
    assumed to be the server's morning.

    The last resort is ``datetime.UTC``, not ``ZoneInfo("UTC")``: Windows
    ships no system timezone database, so every ``ZoneInfo`` lookup there
    depends on the ``tzdata`` package. It is a declared dependency, but a
    stdlib fallback means a missing one degrades to UTC instead of taking
    the whole ticker down.
    """
    for name in (binding.tz, settings.default_timezone):
        if not name:
            continue
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            logger.warning("Unknown timezone %r; falling back", name)
    return UTC


def in_quiet_hours(binding: Binding, now: datetime, settings: Settings) -> bool:
    """Whether the bot should stay silent for this user right now.

    Briefs already wait for ``brief_hour`` in the user's own timezone, but
    watcher checks were pure interval arithmetic with no timezone at all: at
    an 8-hour interval one of the three daily checks lands in their night by
    construction, and a 3am alert is the fastest way to get a bot muted.

    Callers must SKIP the work rather than suppress the message. Checking a
    watcher and swallowing its alert would stamp the fingerprint, so by
    morning the verdict matches its own stamp and the alert is lost for
    good — worse than sending it at 3am. Skipped work leaves
    ``last_checked_at`` untouched, so the watcher stays due and fires when
    their day starts.

    Pure: no I/O, injected clock, every branch unit-testable.
    """
    start, end = settings.quiet_hours_start, settings.quiet_hours_end
    if start == end:
        return False  # disabled, and an existing deployment behaves as before
    hour = now.astimezone(zone_for(binding, settings)).hour
    if start < end:
        return start <= hour < end  # a window inside one day
    # Wraps midnight (21 -> 8), which is the normal case and the one a naive
    # `start <= hour < end` gets exactly backwards.
    return hour >= start or hour < end


def brief_due(binding: Binding, now: datetime, settings: Settings) -> bool:
    """Whether this user's brief should go out at ``now``.

    Pure: no I/O, injected clock, every branch unit-testable.
    """
    if not binding.brief_enabled:
        return False
    local = now.astimezone(zone_for(binding, settings))
    if binding.last_brief_date == local.date().isoformat():
        return False  # already briefed today
    slot = local.replace(
        hour=settings.brief_hour, minute=settings.brief_minute, second=0, microsecond=0
    )
    if local < slot:
        return False  # too early
    # Catch-up: a brief missed while the machine slept is still worth having
    # a few hours late, but one missed all day is noise, not news.
    return local - slot <= timedelta(hours=settings.brief_catchup_hours)


#: How a decided action reads in the digest.
_OUTCOME_WORDS: dict[str, str] = {
    "done": "✅ done",
    "blocked": "🚫 blocked by Odoo",
    "cancelled": "❌ cancelled by you",
    "revised": "✏️ revised",
}


def iso_week(moment: datetime) -> str:
    """Year-and-week stamp, e.g. 2026-W36. Weeks, not dates, so a digest
    cannot go out twice however often the ticker runs."""
    year, week, _ = moment.isocalendar()
    return f"{year}-W{week:02d}"


def digest_due(binding: Binding, now: datetime, settings: Settings) -> bool:
    """Whether this user's weekly digest is due.

    Pure, injected clock. Due once per ISO week, from Monday's brief hour
    onward. "From Monday onward" stays true for the rest of the week, so a
    bot that was off on Monday still sends on its first run — catch-up falls
    out of the state-driven design rather than needing a rule of its own.
    """
    if not settings.digest:
        return False
    # The catch-up branch below returns True for any later day in a week
    # with no digest yet, regardless of the hour — so without this a digest
    # missed on Monday could arrive at 3am Tuesday.
    if in_quiet_hours(binding, now, settings):
        return False
    local = now.astimezone(zone_for(binding, settings))
    if binding.last_digest_week == iso_week(local):
        return False
    if local.weekday() == 0:  # Monday: wait for the hour
        slot = local.replace(
            hour=settings.brief_hour, minute=settings.brief_minute, second=0, microsecond=0
        )
        return local >= slot
    return True  # any later day in a week with no digest yet


#: Arguments worth naming in a digest line, in order of preference. Bodies
#: and other long free text stay out — the digest is a list, not a transcript.
_HEADLINE_KEYS = ("to", "subject", "model", "name", "partner", "record", "res_id", "ids")
_HEADLINE_VALUE_MAX = 60


def action_headline(summary: str) -> str:
    """One readable line for a logged action.

    ``summary`` is the approval text: the tool name, then one indented
    ``key: value`` line per argument. The bare tool name alone ("send_email"
    six times) told the reader nothing, so name who and what as well.
    """
    first, *rest = summary.splitlines() or [""]
    args: dict[str, str] = {}
    for line in rest:
        key, sep, value = line.strip().partition(": ")
        if sep and key not in args:
            args[key] = value.strip()
    parts = [first.replace("_", " ").strip()]
    for key in _HEADLINE_KEYS:
        value = args.get(key, "")
        if not value:
            continue
        if len(value) > _HEADLINE_VALUE_MAX:
            value = value[: _HEADLINE_VALUE_MAX - 1] + "…"
        parts.append(f'"{value}"' if key == "subject" else value)
        if len(parts) == 3:
            break
    return parts[0] if len(parts) == 1 else f"{parts[0]}: " + ", ".join(parts[1:])


def compose_digest(actions: list[LoggedAction], messages: int, spend_usd: float, name: str) -> str:
    """The weekly note. Returns "" when there is nothing worth sending.

    Silence for a week with no activity at all — someone who did not use the
    bot does not need telling. But "no changes made" when they DID use it:
    that is the reassuring case, and it only means something because the
    busy weeks say otherwise.
    """
    if not actions and not messages:
        return ""

    lines = [f"📋 **Your week with the assistant**, {name}", ""]
    if actions:
        lines.append(f"**{len(actions)} action(s) on your behalf:**")
        for action in actions:
            outcome = _OUTCOME_WORDS.get(action.outcome, action.outcome)
            headline = action_headline(action.summary)
            lines.append(f"• {action.created_at:%a %d %b} — {headline} — {outcome}")
    else:
        lines.append("**No changes were made to Odoo on your behalf this week.**")
    lines.append("")
    lines.append(f"_You asked {messages} question(s); ${spend_usd:.2f} of model usage._")
    return "\n".join(lines)


def watcher_cutoff(now: datetime, settings: Settings) -> datetime:
    """Watchers not checked since this instant are due."""
    return now - timedelta(minutes=settings.watcher_interval_minutes)


def parse_verdict(answer: str) -> Verdict:
    """Read a watcher verdict out of the model's reply, defensively.

    Anything unparseable means "not triggered": a confused model must never
    manufacture an alert, and it must never crash the ticker either.
    """
    text = answer.strip()
    if text.startswith("```"):  # strip a fenced block if one slipped through
        text = text.strip("`")
        text = text[text.find("{") :] if "{" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return Verdict()
    try:
        data: Any = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, TypeError):
        logger.warning("Unparseable watcher verdict")
        return Verdict()
    if not isinstance(data, dict):
        return Verdict()
    return Verdict(
        triggered=bool(data.get("triggered")),
        fingerprint=str(data.get("fingerprint") or ""),
        message=str(data.get("message") or ""),
    )


async def _run_readonly_turn(
    store: BindingStore,
    settings: Settings,
    binding: Binding,
    prompt: str,
    facts: Any = None,
    latency: LatencyRecorder | None = None,
    kind: str = KIND_BRIEF,
) -> str:
    """One stateless, read-only agent turn on this user's behalf.

    Spend is recorded against the user's own subject (and so the global
    total) exactly as an interactive turn would be — background work that
    did not show up in the ledger would be the easiest way to lose track of
    the bill.
    """
    artifacts: list[Artifact] = []
    tools = redact_tool_results(
        filter_read_only(await get_odoo_tools(settings.mcp_url, binding.access_token)),
        settings.secret_scan,
    )
    memories: list[str] = []
    if facts is not None:
        memories = await recall(facts, binding.odoo_login, prompt, settings.memory_max_recall)
    agent = build_agent(build_model(settings), tools, memories=memories)
    spend = SpendAccumulator()
    callbacks: list[BaseCallbackHandler] = [spend]
    if latency is not None:
        # Tagged apart from chat: an unattended prompt is a different
        # shape and length, so mixing them would blur both medians.
        callbacks.append(latency.handler(kind))
    try:
        result = await ask(
            agent, prompt, recursion_limit=settings.recursion_limit, callbacks=callbacks
        )
    finally:
        discard(artifacts)
        await store.record_usage(
            subject_for(binding, binding.chat_id),
            datetime.now(UTC).date().isoformat(),
            spend.total_usd,
            count_message=False,  # spend yes; nobody asked a question
        )
    return result.answer


async def run_brief(
    store: BindingStore,
    settings: Settings,
    binding: Binding,
    facts: Any = None,
    latency: LatencyRecorder | None = None,
) -> str:
    """Compose one user's brief. Returns the text to send.

    Every channel's on-demand /brief calls this too, so the "could not build
    it" case is translated here, once.
    """
    text = await _run_readonly_turn(
        store, settings, binding, BRIEF_INSTRUCTION, facts, latency, KIND_BRIEF
    )
    if BRIEF_UNAVAILABLE_SENTINEL in text:
        logger.warning("Brief unavailable for %s — no readable data", binding.odoo_login)
        return BRIEF_UNAVAILABLE_TEXT
    return text


async def run_watcher(
    store: BindingStore,
    settings: Settings,
    binding: Binding,
    watcher: Watcher,
    latency: LatencyRecorder | None = None,
) -> Verdict:
    """Evaluate one watcher. Memories are deliberately not recalled — a
    formatting preference must not sway whether an alert fires."""
    answer = await _run_readonly_turn(
        store,
        settings,
        binding,
        WATCHER_INSTRUCTION.format(condition=watcher.condition),
        latency=latency,
        kind=KIND_WATCHER,
    )
    return parse_verdict(answer)


async def refresh_binding(store: BindingStore, settings: Settings, chat_id: int) -> Binding | None:
    """The binding with a live token, or None when the grant is dead.

    Wraps ``ensure_access`` so a dead grant is a skip, never an exception
    that would abort the whole fan-out.
    """
    try:
        return await ensure_access(store, settings, chat_id)
    except Exception:
        logger.warning("Could not refresh chat %s for proactive work", binding_id(chat_id))
        return None


#: How the caller delivers a message: (chat_id, text) -> None. Injected so
#: this module needs no Telegram import and stays testable with a list.
Sender = Callable[[int, str], Awaitable[None]]


async def spend_exhausted(store: BindingStore, settings: Settings, binding: Binding) -> bool:
    """Whether either spend ceiling is reached.

    Proactive work is deliberately NOT subject to the daily *message* cap —
    that exists to stop a person hammering the bot, and nobody is hammering
    anything at 08:00. It IS subject to both *spend* caps: background jobs
    running unattended are exactly how a bill gets away from you.
    """
    day = datetime.now(UTC).date().isoformat()
    subject = subject_for(binding, binding.chat_id)
    if settings.daily_spend_limit_usd is not None:
        usage = await store.get_usage(subject, day)
        if usage.spend_usd >= settings.daily_spend_limit_usd:
            return True
    if settings.global_daily_spend_limit_usd is not None:
        total = await store.get_usage(GLOBAL_SUBJECT, day)
        if total.spend_usd >= settings.global_daily_spend_limit_usd:
            return True
    return False


async def send_brief(
    store: BindingStore,
    settings: Settings,
    binding: Binding,
    send: Sender,
    facts: Any = None,
    now: datetime | None = None,
    latency: LatencyRecorder | None = None,
) -> bool:
    """Compose and deliver one brief; stamp it so it cannot repeat today.

    The stamp uses the moment this tick is FOR, not the wall clock. Reading
    the clock separately meant a tick evaluated for one day could stamp
    another — invisible in production, where they always agree, but it made
    "already briefed today" wrong the moment the two diverged.
    """
    text = await run_brief(store, settings, binding, facts, latency)
    if not text.strip():
        logger.warning("Empty brief for %s — not sending", binding.odoo_login)
        return False
    if not binding.last_brief_date and text != BRIEF_UNAVAILABLE_TEXT:  # their very first one
        text = FIRST_BRIEF_PREAMBLE + text
    await send(binding.chat_id, text)
    moment = now or datetime.now(UTC)
    local_day = moment.astimezone(zone_for(binding, settings)).date().isoformat()
    await store.mark_brief_sent(binding.chat_id, local_day)
    logger.info("Brief sent to %s (%s)", binding_id(binding.chat_id), binding.odoo_login)
    return True


async def check_watcher(
    store: BindingStore,
    settings: Settings,
    binding: Binding,
    watcher: Watcher,
    send: Sender,
    latency: LatencyRecorder | None = None,
) -> bool:
    """Evaluate one watcher and alert only if this is NEWS.

    The check is always stamped, even when nothing fires, so a watcher whose
    condition stays false does not get re-evaluated every tick.
    """
    verdict = await run_watcher(store, settings, binding, watcher, latency)
    fires = verdict.triggered and verdict.fingerprint != watcher.fingerprint
    if fires:
        await send(
            watcher.chat_id,
            f"🔔 **Watcher** — {watcher.condition}\n\n{verdict.message}",
        )
        logger.info("Watcher %d fired for %s", watcher.id, watcher.odoo_login)
    await store.record_watcher_check(watcher.id, verdict.fingerprint if verdict.triggered else "")
    return fires


async def send_digest(
    store: BindingStore,
    settings: Settings,
    binding: Binding,
    send: Sender,
    now: datetime,
) -> bool:
    """Compose and deliver one weekly digest; stamp the week either way.

    Stamped even when nothing is sent, so an inactive user is not re-checked
    on every tick for the rest of the week.
    """
    local = now.astimezone(zone_for(binding, settings))
    since = now - timedelta(days=7)
    actions = await store.actions_for(binding.odoo_login, since)
    # The seven days ending today. Reading only today's row (as this once
    # did) reported Monday morning — mostly the brief's own cost — as "your
    # week".
    today = local.date()
    usage = await store.usage_between(
        subject_for(binding, binding.chat_id),
        (today - timedelta(days=6)).isoformat(),
        today.isoformat(),
    )
    text = compose_digest(actions, usage.message_count, usage.spend_usd, binding.odoo_name)
    await store.mark_digest_sent(binding.chat_id, iso_week(local))
    if not text:
        return False
    await send(binding.chat_id, text)
    logger.info("Weekly digest sent to %s", binding.odoo_login)
    return True


async def tick(
    store: BindingStore,
    settings: Settings,
    send: Sender,
    facts: Any = None,
    latency: LatencyRecorder | None = None,
    now: datetime | None = None,
) -> None:
    """One pass of the scheduler: send due briefs, check due watchers.

    Every user is isolated: one dead binding, one Odoo hiccup or one model
    failure must never stop everybody else's brief, so each unit of work
    gets its own try/except.
    """
    if not settings.proactive:
        return
    moment = now or datetime.now(UTC)

    for stale in await store.list_bindings():
        if is_whatsapp(stale.chat_id) or not brief_due(stale, moment, settings):
            continue
        try:
            binding = await refresh_binding(store, settings, stale.chat_id)
            if binding is None or await spend_exhausted(store, settings, binding):
                continue
            await send_brief(store, settings, binding, send, facts, moment, latency)
        except Exception:
            logger.error("Brief failed for %s", binding_id(stale.chat_id), exc_info=True)

    for stale in await store.list_bindings():
        if is_whatsapp(stale.chat_id) or not digest_due(stale, moment, settings):
            continue
        try:
            await send_digest(store, settings, stale, send, moment)
        except Exception:
            logger.error("Digest failed for %s", binding_id(stale.chat_id), exc_info=True)

    for watcher in await store.due_watchers(watcher_cutoff(moment, settings)):
        if is_whatsapp(watcher.chat_id):
            continue
        try:
            binding = await refresh_binding(store, settings, watcher.chat_id)
            if binding is None or await spend_exhausted(store, settings, binding):
                continue
            if in_quiet_hours(binding, moment, settings):
                # Skipped, not suppressed: check_watcher never runs, so
                # last_checked_at is not stamped, the watcher stays due, and
                # it fires on the first tick after their day starts.
                continue
            await check_watcher(store, settings, binding, watcher, send, latency)
        except Exception:
            logger.error("Watcher %d failed", watcher.id, exc_info=True)
