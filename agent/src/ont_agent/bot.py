"""Phase 3: the Telegram bridge.

Long-polling bot (no inbound endpoint needed for Telegram itself) plus the
one-route OAuth callback server, sharing a single asyncio loop. Each bound
chat talks to Odoo **as its own Odoo user**: per-user JSON Web Tokens, so
Odoo's access rules and audit trail apply to the person, not to the bot.

    uv run ont-agent-bot
"""

from __future__ import annotations

import asyncio
import datetime as dt
import re
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import nullcontext, suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.base import BaseStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ChatType, KeyboardButtonStyle
from telegram.ext import (
    Application,
    ApplicationBuilder,
    ApplicationHandlerStop,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    TypeHandler,
    filters,
)

from ont_agent.agent import (
    PendingAction,
    ToolFailure,
    ask,
    build_agent,
    has_pending,
    replace_message,
    resume,
)
from ont_agent.auth import (
    LINK_CODE_PROMPT,
    LinkCode,
    begin_login,
    confirm_link,
    ensure_access,
    logout,
)
from ont_agent.budget import (
    BURST_REPLY,
    BurstLimiter,
    SpendAccumulator,
    TurnUsage,
    over_limit,
    subject_for,
)
from ont_agent.callback_server import ServerHandle, build_app, start_server
from ont_agent.channels import channel_name, is_google_chat, is_slack, is_whatsapp
from ont_agent.config import Settings, load_settings
from ont_agent.crypto import TokenCipher
from ont_agent.errortracking import init_error_tracking
from ont_agent.housekeeping import run_if_due
from ont_agent.llm import build_model
from ont_agent.log import BLOCKED_LOGGER, add_file_logs, get_logger, setup_logging
from ont_agent.memory import build_memory_tools, build_store_index, forget_all, recall
from ont_agent.metrics import KIND_CHAT, LatencyRecorder
from ont_agent.proactive import Sender, run_brief, tick, zone_for
from ont_agent.promptguard import REFUSAL, leaks_prompt, looks_like_extraction
from ont_agent.redaction import redact_secrets
from ont_agent.render import Artifact, build_local_tools, discard
from ont_agent.rotation import Rotation, current_thread, plan_rotation
from ont_agent.router import get_semantic_router
from ont_agent.store import (
    GLOBAL_SUBJECT,
    ActionOutcome,
    Binding,
    BindingStore,
    LoggedAction,
    binding_id,
)
from ont_agent.telegram_format import plain_text, render_markdown
from ont_agent.tools import (
    CONFIRM_TOOLS,
    DOCUMENT_TOOLS,
    EDITABLE_TOOLS,
    IRREVERSIBLE_TOOLS,
    Evidence,
    bridge_export_tool,
    bridge_record_links,
    filter_for_bot,
    get_odoo_tools,
    redact_tool_results,
)
from ont_agent.tracing import Tracing, Turn, init_tracing
from ont_agent.transcribe import transcribe
from ont_agent.uploads import (
    DEFAULT_QUESTION,
    SUPPORTED,
    Attachment,
    describe,
    message_content,
    thread_placeholder,
)
from ont_agent.uploads import (
    check as check_upload,
)
from ont_agent.whimsy import whimsy

if TYPE_CHECKING:
    from ont_agent.googlechat_bot import GoogleChatChannel
    from ont_agent.slack import SlackSocket
    from ont_agent.slack_bot import SlackChannel
    from ont_agent.whatsapp_bot import WhatsAppChannel

logger = get_logger(__name__)
blocked_logger = get_logger(BLOCKED_LOGGER)

CONNECT_PROMPT = (
    "You are not connected to Odoo yet. Use /start and tap the login button "
    "to connect your account."
)
ERROR_REPLY = "Sorry — something went wrong while answering. Please try again."
TIMEOUT_REPLY = "Sorry — that took too long, so I stopped. Please try again."
#: Langfuse tag on turns the semantic router answered without the model.
ROUTER_TRACE_TAG = "semantic-router"

#: Registered with Telegram at startup: powers the "/" autocomplete menu and
#: the Menu button, and feeds /help. (/reset stays a working alias of /new
#: but is kept out of the menu — no duplicates there.)
BOT_COMMANDS: list[tuple[str, str]] = [
    ("start", "Connect your Odoo account / show status"),
    ("new", "Start a fresh conversation (forget history)"),
    ("forget", "Erase the preferences I remember about you"),
    ("brief", "Send my business brief now (/brief off to stop the daily one)"),
    ("watch", "Alert me when something happens: /watch <condition>"),
    ("watchers", "List the alerts I have set up"),
    ("unwatch", "Remove one of my alerts: /unwatch <number>"),
    ("audit", "Administrators: what the assistant did, across everyone"),
    ("logout", "Disconnect and revoke access"),
    ("help", "What this bot can do"),
]


def bot_commands() -> list[BotCommand]:
    return [BotCommand(command, description) for command, description in BOT_COMMANDS]


def help_text() -> str:
    lines = "\n".join(f"/{command} — {description}" for command, description in BOT_COMMANDS)
    return (
        "I answer questions about your company's Odoo data — just ask in "
        "plain language. Follow-up questions work; any change to data always "
        "asks for your approval first.\n\n"
        f"{lines}"
    )


def clean_reply(text: str) -> str:
    """Normalize a model answer for the reply pipeline (whitespace only —
    Markdown is rendered to Telegram HTML at send time, see
    ``prepare_reply_text``)."""
    return text.strip()


def _without_secrets(reply: str, settings: Settings, chat_id: int) -> str:
    """Second net for credentials the tool-result scan could not reach.

    It earns its place on the upload path: OpenRouter parses a PDF
    server-side, so a key inside one arrives in the model's context without
    passing through any tool.

    Redacts rather than replacing, unlike the prompt guard — a reply with one
    stray key in it still has a legitimate answer around it.
    """
    if not settings.secret_scan:
        return reply
    clean, kinds = redact_secrets(reply)
    if kinds:
        logger.warning(
            "Chat %s: redacted %s from the reply", binding_id(chat_id), ", ".join(kinds)
        )
    return clean


def _without_prompt_leak(reply: str, settings: Settings, chat_id: int) -> str:
    """Replace a reply that reproduces the system prompt.

    The layer that actually guarantees something. The input patterns and the
    prompt rule are both bypassable by rephrasing; this inspects the text on
    its way out, so how the request was worded stops mattering.

    The matched fragment is logged so a false positive can be diagnosed and
    AGENT_PROMPT_LEAK_NGRAM tuned, rather than guessed at.
    """
    if not settings.prompt_guard:
        return reply
    leaked = leaks_prompt(reply, settings.prompt_leak_ngram)
    if leaked is None:
        return reply
    logger.warning(
        "Chat %s: reply reproduced the system prompt (%r) — replaced",
        binding_id(chat_id),
        leaked[:80],
    )
    return REFUSAL


@dataclass
class BotReply:
    """What the chat handler should show: text, approval buttons if a write
    action is paused, and any rendered files (charts, PDFs) to send along."""

    text: str
    pending: PendingAction | None = None
    attachments: list[Artifact] = field(default_factory=list)
    #: Langfuse trace id of the answering run — powers the 👍/👎 buttons.
    trace_id: str | None = None
    #: The approved action was refused by Odoo. Changes the frozen verdict
    #: badge from "Approved" to "Blocked", so the message reports the
    #: OUTCOME rather than merely which button was tapped.
    blocked: bool = False
    #: What the tools saw this turn: record links and per-model row counts.
    #: Powers deep links, the freshness stamp and "show your working".
    evidence: Evidence = field(default_factory=Evidence)
    #: Was this answer read out of a DOCUMENT rather than Odoo records? Still
    #: grounded and still cited — but a policy is prose, not a dataset, so
    #: there is nothing in it to chart or split by customer.
    from_documents: bool = False
    #: Did this answer come from Odoo data, or is it conversational? Follow-up
    #: buttons hang off this: "what can you help me with?" fetched nothing, so
    #: there is nothing to chart, break down or email.
    grounded: bool = False
    #: OpenRouter usage across every model call that produced this reply.
    usage: TurnUsage = field(default_factory=TurnUsage)


#: Telegram rejects messages over 4096 characters; leave headroom for the
#: header/footer so a pathological approval prompt still gets delivered.
_CONFIRMATION_BUDGET = 3800

#: Values longer than this (or with line breaks) get their own paragraph —
#: inline "label: long wrapped text" reads misaligned in Telegram.
_LONG_VALUE = 60


def _format_action_html(name: str, args: dict[str, object]) -> str:
    """One action as Telegram HTML: bold labels, short fields first, long
    fields (like an email body) as their own bold paragraph."""
    import html as html_lib
    import json as json_lib

    short_lines: list[str] = []
    long_blocks: list[str] = []
    for key, value in args.items():
        rendered = value if isinstance(value, str) else json_lib.dumps(value, default=str)
        label = key.replace("_", " ").capitalize()
        if len(rendered) > _LONG_VALUE or "\n" in rendered:
            long_blocks.append(
                f"<b>{html_lib.escape(label)}:</b>\n<b>{html_lib.escape(rendered)}</b>"
            )
        else:
            short_lines.append(f"<b>{html_lib.escape(label)}:</b> {html_lib.escape(rendered)}")
    parts = [f"▶ <b>{html_lib.escape(name)}</b>"]
    if short_lines:
        parts.append("\n".join(short_lines))
    parts.extend(long_blocks)
    return "\n\n".join(parts) if long_blocks else "\n".join(parts)


#: Email preview body cap, so the whole prompt stays under Telegram's limit.
_EMAIL_BODY_BUDGET = 2800


def _format_email_html(args: dict[str, object], assistant: str, user: str) -> str:
    """The email as the recipient will get it (mirrors ont_mcp_mail send_email)."""
    import html as html_lib

    esc = html_lib.escape
    body = str(args.get("body") or "")
    shortened = len(body) > _EMAIL_BODY_BUDGET
    paragraphs = [line for line in body[:_EMAIL_BODY_BUDGET].splitlines() if line.strip()]

    parts = [
        f"<b>From:</b> {esc(assistant)}",
        f"<b>To:</b> {esc(str(args.get('to') or ''))}",
        f"<b>Subject:</b> {esc(str(args.get('subject') or ''))}",
    ]
    export = args.get("export")
    if isinstance(export, dict) and export.get("model"):
        extension = str(export.get("format") or "xlsx")
        filename = f"{str(export['model']).replace('.', '_')}.{extension}"
        parts.append(f"<b>Attachment:</b> 📎 {esc(filename)}")
    elif args.get("record_model") and args.get("record_id"):
        record = f"{args['record_model']} #{args['record_id']}"
        parts.append(f"<b>Attachment:</b> 📎 PDF of {esc(record)}")

    footer = f"— Sent by {assistant} on behalf of {user}." if user else f"— Sent by {assistant}."
    message = ["\n\n".join(esc(line) for line in paragraphs) + ("…" if shortened else "")]
    message.append(f"<i>{esc(footer)}</i>")
    return "📧 <b>Email preview</b>\n" + "\n".join(parts) + "\n\n" + "\n\n".join(message)


def confirmation_message(
    pending: PendingAction, assistant_name: str = "AI Assistant", user_name: str = ""
) -> str:
    """Approval prompt (Telegram HTML). Falls back to the plain prebuilt
    descriptions when no structured arguments are available."""
    import html as html_lib

    if pending.arguments and len(pending.arguments) == len(pending.names):
        blocks = [
            _format_email_html(args, assistant_name, user_name)
            if name == "send_email"
            else _format_action_html(name, args)
            for name, args in zip(pending.names, pending.arguments, strict=True)
        ]
        lines = "\n\n".join(blocks)
    else:
        lines = "\n\n".join(f"▶ {html_lib.escape(action)}" for action in pending.actions)
    if len(lines) > _CONFIRMATION_BUDGET:  # never let Telegram reject the prompt
        # Cut on a closed tag so the clamp can never produce broken HTML.
        cut = lines.rfind("</b>", 0, _CONFIRMATION_BUDGET)
        safe = lines[: cut + 4] if cut != -1 else html_lib.escape(lines[:_CONFIRMATION_BUDGET])
        lines = safe + "\n… (details shortened to fit Telegram)"
    return (
        "⚠️ <b>Approval needed</b> — I want to make this change in Odoo:\n\n"
        f"{lines}\n\n"
        "Nothing happens until you decide."
    )


#: Confirm callback data: verdict plus (optionally) the proposing trace id,
#: so a tap can be scored onto the exact trace that proposed the action.
_CONFIRM_DATA = re.compile(r"^confirm:(approve|edit|reject)(?::([0-9a-f]{32}))?$")


def confirm_markup(
    editable: bool = False, trace_id: str | None = None, irreversible: bool = False
) -> InlineKeyboardMarkup:
    """The approval buttons, coloured so the consequential one is obvious.

    This tap is the last gate before the bot writes to Odoo or sends mail,
    and it is usually made on a phone, quickly. Three identical grey buttons
    make "send this" and "don't" distinguishable only by reading; colour is
    the faster signal.

    A plain traffic light: **green means go, red means stop**, which is the
    fastest reading there is and never changes meaning between prompts.

    The caution for an action that cannot be taken back (see
    ``IRREVERSIBLE_TOOLS``) therefore lives in the LABEL rather than the
    colour. Colouring Approve red as well would put two red buttons side by
    side on exactly the prompts where telling them apart matters most, and a
    warning word is unambiguous where a second red is not.

    ``style`` is additive: a client too old to understand it renders an
    ordinary button, so nothing breaks for anyone.
    """
    suffix = f":{trace_id}" if trace_id and re.fullmatch(r"[0-9a-f]{32}", trace_id) else ""
    row = [
        InlineKeyboardButton(
            "⚠️ Approve" if irreversible else "✅ Approve",
            callback_data=f"confirm:approve{suffix}",
            style=KeyboardButtonStyle.SUCCESS,
        )
    ]
    if editable:
        row.append(
            InlineKeyboardButton(
                "✏️ Edit",
                callback_data=f"confirm:edit{suffix}",
                style=KeyboardButtonStyle.PRIMARY,
            )
        )
    row.append(
        InlineKeyboardButton(
            "❌ Cancel",
            callback_data=f"confirm:reject{suffix}",
            style=KeyboardButtonStyle.DANGER,
        )
    )
    return InlineKeyboardMarkup([row])


def parse_confirm_data(data: str) -> tuple[str, str | None] | None:
    """(verdict, proposing trace id) from a confirm tap; None if malformed.
    Old-format buttons (no trace id, pre-deploy messages) parse fine."""
    match = _CONFIRM_DATA.fullmatch(data)
    if match is None:
        return None
    return match.group(1), match.group(2)


def pending_is_editable(pending: PendingAction | None) -> bool:
    """Edit is offered only when EVERY paused action supports revision
    (mail only — record mutations stay approve-or-cancel)."""
    return pending is not None and bool(pending.names) and set(pending.names) <= EDITABLE_TOOLS


def _pending_markup(
    pending: PendingAction | None, trace_id: str | None = None
) -> InlineKeyboardMarkup | None:
    if pending is None:
        return None
    # The only place that knows WHAT is pending, so the only place that can
    # decide whether approving it is reversible.
    return confirm_markup(
        editable=pending_is_editable(pending),
        trace_id=trace_id,
        irreversible=bool(set(pending.names) & IRREVERSIBLE_TOOLS),
    )


#: Feedback callback data: "fb:<1|0>:<32-hex Langfuse trace id>" (≤64 bytes).
_FEEDBACK_DATA = re.compile(r"^fb:([01]):([0-9a-f]{32})$")


def feedback_markup(trace_id: str) -> InlineKeyboardMarkup | None:
    """👍/👎 buttons scoring the answering trace; None for invalid ids."""
    if not re.fullmatch(r"[0-9a-f]{32}", trace_id):
        return None
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("👍", callback_data=f"fb:1:{trace_id}"),
                InlineKeyboardButton("👎", callback_data=f"fb:0:{trace_id}"),
            ]
        ]
    )


def parse_feedback_data(data: str) -> tuple[bool, str] | None:
    """(positive, trace_id) from a feedback button tap; None if malformed."""
    match = _FEEDBACK_DATA.fullmatch(data)
    if match is None:
        return None
    return match.group(1) == "1", match.group(2)


def usage_line(usage: TurnUsage) -> str | None:
    """Compact per-turn OpenRouter usage, or None when no model completed."""
    if usage.call_count <= 0:
        return None
    if usage.tokens_available:
        if usage.total_tokens >= 1_000:
            compact = f"{usage.total_tokens / 1_000:.1f}".rstrip("0").rstrip(".")
            tokens = f"{compact}K tokens"
        else:
            tokens = f"{usage.total_tokens} tokens"
    else:
        tokens = "Tokens: unavailable"
    if not usage.cost_available:
        cost = "unavailable"
    elif 0 < usage.cost_usd < 0.000001:
        cost = "<$0.000001"
    else:
        cost = f"${usage.cost_usd:.6f}"
    return f"{tokens} · Cost: {cost}"


def _usage_html(usage: TurnUsage) -> str | None:
    line = usage_line(usage)
    if line is None:
        return None
    import html as html_lib

    return f"<i>{html_lib.escape(line)}</i>"


def _append_usage(body: str, footer: str | None, *, clamp: bool = False) -> str:
    if footer is None:
        return body[:4096] if clamp else body
    suffix = f"\n\n{footer}"
    if clamp:
        body = body[: max(0, 4096 - len(suffix))]
    return body + suffix


def prepare_plain_reply_text(reply: BotReply) -> str:
    """Telegram-safe plain reply that always preserves its usage footer."""
    return _append_usage(plain_text(reply.text), usage_line(reply.usage), clamp=True)


def _prepare_plain_reply_html(reply: BotReply) -> str:
    """Escaped oversized reply with a muted usage footer."""
    import html as html_lib

    footer = _usage_html(reply.usage)
    if footer is None:
        return html_lib.escape(plain_text(reply.text))[:4096]
    suffix = f"\n\n{footer}"
    budget = 4096 - len(suffix)
    raw = plain_text(reply.text)
    escaped = html_lib.escape(raw)
    while len(escaped) > budget:
        raw = raw[: max(0, len(raw) - (len(escaped) - budget))]
        escaped = html_lib.escape(raw)
    return escaped + suffix


def prepare_reply_text(reply: BotReply) -> tuple[str, str | None]:
    """Final text + parse_mode for a finished reply.

    Pending approval prompts are prebuilt HTML. Normal answers arrive as
    Markdown from the model and are rendered to Telegram HTML (headers ->
    bold, tables -> aligned monospace); if rendering would exceed Telegram's
    4096-character limit, fall back to clamped escaped text.
    """
    footer = _usage_html(reply.usage)
    if reply.pending is not None:
        return _append_usage(reply.text, footer), "HTML"
    rendered = render_markdown(reply.text, reply.evidence.links or None)
    candidate = _append_usage(rendered, footer)
    if len(candidate) > 4096 and reply.evidence.links:
        # Links inflate the markup. Drop them before giving up on formatting
        # altogether: losing navigation is a smaller loss than losing tables.
        rendered = render_markdown(reply.text)
        candidate = _append_usage(rendered, footer)
    if not rendered or len(candidate) > 4096:
        if footer is not None:
            return _prepare_plain_reply_html(reply), "HTML"
        return prepare_plain_reply_text(reply), None
    return candidate, "HTML"


_FOLLOWUP_DATA = re.compile(r"^fu:(chart|customer|email)$")

#: What each button actually asks. Phrased as a follow-up, not a fresh
#: question: it runs on the SAME conversation thread, so "that" resolves
#: against the answer just given and no data is re-fetched.
FOLLOWUPS: dict[str, tuple[str, str]] = {
    "chart": ("📊 Show as chart", "Show that as a chart."),
    "customer": ("👥 By customer", "Break that down by customer."),
    "email": (
        "📧 Email me this",
        "Email that actual result to me, not just a cover note. For a list or table, "
        "attach the complete result as XLSX and keep the body to a short summary "
        "(what it is and how many rows) - do not repeat the rows in the body. "
        "Otherwise, include a concise result in the body.",
    ),
}

#: Three or more figures reads as data worth plotting.
_FIGURES = re.compile(r"\b\d[\d,.]*\d\b")
_CUSTOMER_WORDS = re.compile(r"\b(customer|client|partner|contact|account)s?\b", re.I)
_LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.M)


def followup_codes(
    text: str, can_write: bool, grounded: bool = True, from_documents: bool = False
) -> list[str]:
    """Which follow-ups suit this answer.

    Chosen in code from what the turn actually did, rather than by asking the
    model: free, instant, and testable. Offering a button that leads nowhere
    trains people to stop looking at the row, so each one must earn its place.

    ``grounded`` — whether the turn called any Odoo tool — is the gate that
    matters most. Text shape alone is not enough: "what can you help me with?"
    lists capabilities including the phrase "customer invoices", which was
    enough to offer *Break down by customer* and *Email me this* under an
    answer containing no data at all (live, 2026-09-01). If nothing was
    fetched, there is nothing to chart, split or send.

    ``from_documents`` is the SECOND gate, and grounding alone does not
    cover it: a policy answer does call a tool, so it is grounded, and its
    section numbers ("Section 29", "Section 30") read as figures to the
    heuristic below — which offered *Show as chart* under the sick-leave
    policy (live, 2026-09-02). Prose from a document is not a dataset.
    Emailing a document is still perfectly sensible, so only chart and
    breakdown are withheld. For ordinary record answers, email is offered
    only when the result has enough structure to be worth sending: a table,
    a multi-item list, or a report with several figures. This keeps the
    button off acknowledgements, preference updates and one-value lookups.
    """
    if not text.strip() or not grounded:
        return []
    codes: list[str] = []
    has_table = any(line.strip().startswith("|") for line in text.splitlines())
    if not from_documents:
        if has_table or len(_FIGURES.findall(text)) >= 3:
            codes.append("chart")
        if _CUSTOMER_WORDS.search(text):
            codes.append("customer")
    # Withheld without write scope: the server would refuse it, and an
    # approval prompt that cannot succeed is the exact failure fixed on
    # 2026-08-31 — do not reintroduce it through a button.
    email_worthy = (
        from_documents
        or has_table
        or len(_LIST_ITEM.findall(text)) >= 2
        or len(_FIGURES.findall(text)) >= 3
    )
    if can_write and email_worthy:
        codes.append("email")
    return codes


def _document_answer(tool_calls: Sequence[str]) -> bool:
    """True when EVERY tool this turn used was a document search.

    A mixed turn — "is this invoice overdue, and what does the policy say?"
    — touched real records too, so it keeps its data buttons.
    """
    names = {call.split("(", 1)[0].strip() for call in tool_calls}
    return bool(names) and names <= DOCUMENT_TOOLS


def parse_followup_data(data: str) -> str | None:
    match = _FOLLOWUP_DATA.fullmatch(data)
    return match.group(1) if match else None


def reply_markup_for(reply: BotReply, can_write: bool = False) -> InlineKeyboardMarkup | None:
    """Buttons for a finished reply: approval buttons for a paused write,
    otherwise a follow-up row plus 👍/👎 feedback."""
    if reply.pending is not None:
        return _pending_markup(reply.pending, reply.trace_id)
    rows: list[list[InlineKeyboardButton]] = []
    codes = (
        followup_codes(reply.text, can_write, reply.grounded, reply.from_documents)
        if not reply.blocked
        else []
    )
    if codes:
        rows.append(
            [
                InlineKeyboardButton(FOLLOWUPS[code][0], callback_data=f"fu:{code}")
                for code in codes
            ]
        )
    if reply.trace_id and re.fullmatch(r"[0-9a-f]{32}", reply.trace_id):
        rows.append(
            [
                InlineKeyboardButton("👍", callback_data=f"fb:1:{reply.trace_id}"),
                InlineKeyboardButton("👎", callback_data=f"fb:0:{reply.trace_id}"),
            ]
        )
    return InlineKeyboardMarkup(rows) if rows else None


CANCELLED_PREFIX = "⚠️ Your earlier pending action was cancelled.\n\n"

EDIT_PROMPT = (
    '✏️ Tell me what to change — for example: "shorten the subject and send it '
    'to x@y.com instead". Your next message revises the proposal above.'
)

#: Frozen verdict when the user approved but Odoo refused. Deliberately not
#: "Cancelled" — that means the USER declined, and the two must stay
#: distinguishable when looking back at a past decision.
BLOCKED_VERDICT = "🚫 <b>Blocked by Odoo</b>"

#: Plain-language phrase per confirm-gated tool, for the refusal sentence.
#: Presentation copy like the reply strings around it, not configuration.
_ACTION_PHRASE: dict[str, str] = {
    "send_email": "send that email",
    "create_records": "create that record",
    "update_records": "make that change",
    "delete_records": "delete that record",
    "post_message": "post that message",
    "call_method": "run that action",
}


def blocked_reply(failures: Sequence[ToolFailure]) -> str:
    """The entire reply when an approved action was refused by the server.

    Only the refusal: the user asked for an ACTION, so answering the
    underlying question instead hands them data they never requested —
    which, on a blocked exfiltration attempt, is precisely the payload the
    server just refused to send (live incident 2026-08-31).
    """
    failure = failures[0]
    phrase = _ACTION_PHRASE.get(failure.name, failure.name or "complete that action")
    return f"I could not {phrase}.\n\nOdoo refused it: {failure.message}"


def status_text(original: str | None, status: str) -> str:
    """Append a status line UNDER the original proposal text — the proposal
    must stay readable after a decision; clamped to Telegram's 4096 limit."""
    if not original:
        return status
    combined = f"{original}\n\n{status}"
    if len(combined) > 4096:
        keep = 4096 - len(status) - len("\n…\n\n")
        combined = f"{original[:keep]}\n…\n\n{status}"
    return combined


def edit_prompt_text(original: str | None) -> str:
    """The message shown after tapping ✏️ Edit: the FULL proposal stays
    visible (the user must see what they are editing) with the instructions
    appended."""
    return status_text(original, EDIT_PROMPT)


def _revision_message(instructions: str) -> str:
    return (
        "The user did NOT approve the proposed action; they request changes "
        f'instead: "{instructions}". Propose a corrected version of the action '
        "incorporating these changes — it will be shown for approval again. If "
        "the message is not a change request, treat the proposal as cancelled "
        "and respond to the message instead."
    )


def _awaiting_edit(context: ContextTypes.DEFAULT_TYPE) -> set[int]:
    result: set[int] = context.application.bot_data.setdefault("awaiting_edit", set())
    return result


#: Action suffix per tool; anything else falls back to "checking <name>".
_PROGRESS_LINES = {
    "render_chart": "drawing your chart",
    "create_pdf_report": "building the PDF",
}


def tool_progress_line(call: str) -> str:
    """Progress line: a random whimsical gerund + what is actually running."""
    name = call.split("(", 1)[0]
    action = _PROGRESS_LINES.get(name, f"checking {name}")
    return f"⏳ {whimsy()} — {action}"


#: Technical model names read as jargon in a freshness line; anything not
#: listed falls back to the neutral "records" rather than leaking one.
_RECORD_WORDS: dict[str, tuple[str, str]] = {
    "account.move": ("invoice", "invoices"),
    "account.move.line": ("line", "lines"),
    "account.payment": ("payment", "payments"),
    "res.partner": ("contact", "contacts"),
    "sale.order": ("order", "orders"),
    "sale.order.line": ("order line", "order lines"),
    "purchase.order": ("purchase order", "purchase orders"),
    "crm.lead": ("opportunity", "opportunities"),
    "stock.picking": ("delivery", "deliveries"),
    "stock.quant": ("stock line", "stock lines"),
    "product.template": ("product", "products"),
    "product.product": ("product", "products"),
    "hr.employee": ("employee", "employees"),
    "res.users": ("user", "users"),
}

#: Models named in the stamp before it stops being a glance.
_FRESHNESS_MODELS = 3


def _plural(model: str, count: int) -> str:
    singular, plural = _RECORD_WORDS.get(model, ("record", "records"))
    return f"{count} {singular if count == 1 else plural}"


def freshness_line(
    evidence: Evidence, binding: Binding | None, settings: Settings, now: dt.datetime
) -> str:
    """ "As of 17:42 · 19 invoices" — when this was true, and from how much.

    Silently answers "is this current?" before anyone thinks to ask, which
    matters most for a number someone scrolls back to next week.

    In the USER's timezone, not the server's: the same binding field the
    morning brief uses. Counts come from rows the tools actually returned,
    so an aggregate-only answer (search_count returns a number, not records)
    gets a timestamp and no count rather than a dishonest "0 records".
    """
    zone = zone_for(binding, settings) if binding is not None else now.tzinfo
    stamp = now.astimezone(zone).strftime("%H:%M")
    if not evidence.counts:
        return f"As of {stamp}"
    ranked = sorted(evidence.counts.items(), key=lambda item: -item[1])
    shown = ", ".join(_plural(model, count) for model, count in ranked[:_FRESHNESS_MODELS])
    return f"As of {stamp} · {shown}"


def initial_status() -> str:
    """The instant placeholder shown before the first tool call."""
    return f"🔎 {whimsy()}"


def fast_route(settings: Settings, text: str) -> tuple[str, str] | None:
    """``(route name, static answer)`` from the semantic router, or None."""
    if not settings.semantic_router:
        return None
    router = get_semantic_router(settings)
    if router is None:
        return None
    route_name = router.route(text)
    reply = router.get_static_reply(route_name)
    if route_name is None or reply is None:
        return None
    return route_name, reply


def fast_reply(settings: Settings, text: str) -> str | None:
    """The semantic router's static answer, or None when the agent must run."""
    routed = fast_route(settings, text)
    return routed[1] if routed is not None else None


def _now() -> dt.datetime:
    """Timezone-aware present. Separate from _today() so tests can freeze it."""
    return dt.datetime.now(dt.UTC)


def _today() -> str:
    """Host-local ISO date — the same "today" concept the system prompt
    uses (see agent.system_prompt_with_date), so usage days and the
    model's own sense of "today" never disagree."""
    return dt.date.today().isoformat()


async def _within_turn_limit(
    settings: Settings, chat_id: int, work: Awaitable[BotReply]
) -> BotReply:
    """Run one turn under ``settings.turn_timeout_seconds``.

    Every channel handles a chat's messages one at a time, so a turn that
    awaits something which never returns (a stalled MCP session, a hung
    socket) would hold that chat's lock until the next restart: the user
    sees "typing…" and nothing else, for every later message too. Past the
    limit the turn is cancelled and the user is told to try again.
    """
    limit = settings.turn_timeout_seconds
    if limit <= 0:
        return await work
    deadline = asyncio.timeout(limit)
    try:
        async with deadline:
            return await work
    except TimeoutError:
        if not deadline.expired():
            raise  # a TimeoutError from inside the turn, not our ceiling
        logger.error("Chat %s: turn exceeded %ss and was stopped", binding_id(chat_id), limit)
        return BotReply(text=TIMEOUT_REPLY)


async def answer_question(
    store: BindingStore,
    settings: Settings,
    chat_id: int,
    text: str,
    on_tool: Callable[[str], Awaitable[None]] | None = None,
    on_text: Callable[[str], Awaitable[None]] | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    tracing: Tracing | None = None,
    facts: BaseStore | None = None,
    attachment: Attachment | None = None,
    latency: LatencyRecorder | None = None,
) -> BotReply:
    """The per-message pipeline, bounded by the turn limit (see ``_answer_question``)."""
    return await _within_turn_limit(
        settings,
        chat_id,
        _answer_question(
            store,
            settings,
            chat_id,
            text,
            on_tool=on_tool,
            on_text=on_text,
            checkpointer=checkpointer,
            tracing=tracing,
            facts=facts,
            attachment=attachment,
            latency=latency,
        ),
    )


async def _answer_question(
    store: BindingStore,
    settings: Settings,
    chat_id: int,
    text: str,
    on_tool: Callable[[str], Awaitable[None]] | None = None,
    on_text: Callable[[str], Awaitable[None]] | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    tracing: Tracing | None = None,
    facts: BaseStore | None = None,
    attachment: Attachment | None = None,
    latency: LatencyRecorder | None = None,
) -> BotReply:
    """The per-message pipeline; returns the reply for the chat.

    :param attachment: an uploaded image or PDF file to answer about. The
        Odoo tools stay available, so "is this bill already in our system?"
        is one turn; ``text`` remains the question, and still drives memory
        recall and tracing, because the bytes are not something to search
        preferences by.

    With a ``checkpointer`` the chat keeps conversation memory: the agent
    graph is rebuilt per message (tools carry the user's own token), but the
    thread state lives in the checkpointer keyed by chat id, so follow-up
    questions resolve against the previous turns. If the agent proposes a
    write, the reply carries ``pending`` and the graph stays paused until
    ``resume_pending`` delivers the user's decision.
    """
    binding = await ensure_access(store, settings, chat_id)
    if binding is None:
        return BotReply(text=CONNECT_PROMPT)

    if settings.prompt_guard and looks_like_extraction(text):
        # Before the tools and the agent are built, so a probe costs nothing
        # -- but the message is still recorded, so repeated probing burns
        # the sender's daily allowance rather than being free.
        logger.warning(
            "Chat %s (%s): refused a prompt-extraction attempt",
            binding_id(chat_id),
            binding.odoo_login,
        )
        await store.record_usage(subject_for(binding, chat_id), _today(), 0.0)
        return BotReply(text=REFUSAL)

    # Fast semantic routing: greetings and out-of-scope chit-chat get a static
    # reply without building tools or the agent. Attachments always need the model.
    if not attachment:
        routing_started = time.monotonic()
        routed = fast_route(settings, text)
        if routed is not None:
            route_name, static_reply = routed
            logger.info("Chat %s: semantic router answered without the LLM", binding_id(chat_id))
            if tracing is not None:
                # Traced like any other turn (same name, tagged), so Langfuse
                # shows every message — and a misrouted business question can
                # be found by filtering on the tag.
                channel = channel_name(chat_id)
                with tracing.turn(
                    trace_name=f"{channel}-ask",
                    user_id=binding.odoo_login,
                    session_id=str(chat_id),
                    input_text=text,
                    tags=[channel, ROUTER_TRACE_TAG],
                    metadata={
                        "route": route_name,
                        "routing_ms": round((time.monotonic() - routing_started) * 1000, 1),
                    },
                ) as turn:
                    turn.set_output(static_reply)
            await store.record_usage(subject_for(binding, chat_id), _today(), 0.0)
            return BotReply(text=static_reply)

    started = time.monotonic()
    artifacts: list[Artifact] = []
    evidence = Evidence()
    tools = redact_tool_results(
        bridge_export_tool(
            bridge_record_links(
                filter_for_bot(
                    await get_odoo_tools(settings.mcp_url, binding.access_token),
                    allow_method_calls=binding.can_call_methods,
                    allow_writes=binding.can_write,
                ),
                settings.odoo_public_url if settings.deep_links else "",
                evidence,
            ),
            artifacts,
        ),
        settings.secret_scan,
    ) + build_local_tools(artifacts)
    memories: list[str] = []
    if facts is not None:
        memories = await recall(facts, binding.odoo_login, text, settings.memory_max_recall)
        tools = tools + build_memory_tools(facts, binding.odoo_login)
    agent = build_agent(
        build_model(settings, for_upload=attachment is not None),
        tools,
        checkpointer=checkpointer,
        memories=memories,
        summary_trigger_tokens=settings.summary_trigger_tokens,
        summary_keep_messages=settings.summary_keep_messages,
    )
    thread_id = await current_thread(store, chat_id) if checkpointer is not None else None

    # New text while a confirmation is still open (user typed instead of
    # tapping a button, possibly across a bot restart): cancel the pending
    # action first — the thread cannot accept fresh input while paused.
    cancelled = False
    if thread_id is not None and await has_pending(agent, thread_id):
        await resume(agent, approved=False, thread_id=thread_id)
        cancelled = True
        logger.info("Chat %s: auto-cancelled pending action on new message", binding_id(chat_id))

    # Rotation is decided AFTER that cancellation, so the cancellation is
    # carried across rather than stranded on the thread being retired — and
    # nothing is written until commit() below, once the turn has returned.
    rotation: Rotation | None = None
    if thread_id is not None:
        rotation = await plan_rotation(store, agent, settings, chat_id, checkpointer)
        thread_id = rotation.thread_id

    # Every channel answers through here, so the trace is named after the
    # chat's channel rather than Telegram.
    channel = channel_name(chat_id)
    turn_context = (
        tracing.turn(
            trace_name=f"{channel}-ask",
            user_id=binding.odoo_login,
            session_id=str(chat_id),
            input_text=text,
            tags=[channel],
        )
        if tracing is not None
        else nullcontext(Turn("", None))
    )
    # Spend tracking runs regardless of Langfuse (safety feature, not
    # observability); the handler list carries both when tracing is live,
    # so test fakes (and the no-tracing path) keep the pre-observability
    # call shape for the tracing.handler-less case.
    spend = SpendAccumulator()
    callbacks: list[BaseCallbackHandler] = [tracing.handler, spend] if tracing else [spend]
    # Latency is recorded per model call, not per turn: a turn is several
    # calls and they are what the model is judged on.
    if latency is not None:
        callbacks.append(latency.handler(KIND_CHAT))
    # An uploaded file gets an id so it can be swapped for a one-line
    # description once answered; see replace_message() for why that matters.
    upload_id = f"upload-{uuid4().hex}" if attachment is not None else None
    with turn_context as turn:
        result = await ask(
            agent,
            message_content(text, attachment) if attachment is not None else text,
            on_tool=on_tool,
            thread_id=thread_id,
            recursion_limit=settings.recursion_limit,
            callbacks=callbacks,
            message_id=upload_id,
            history=rotation.history if rotation is not None else (),
            on_text=on_text if settings.token_streaming else None,
        )
        turn.set_output(result.answer or "<pending approval>")
    if rotation is not None:
        # After the turn, whatever its outcome — a pending approval lives on
        # the new thread, and resume_pending has to find it there.
        await rotation.commit(store, checkpointer, chat_id)
    if attachment is not None and thread_id is not None and upload_id is not None:
        # Whatever the outcome — including a pending approval, whose thread
        # stays alive until the user taps a button.
        await replace_message(agent, thread_id, upload_id, thread_placeholder(text, attachment))
    await store.record_usage(subject_for(binding, chat_id), _today(), spend.total_usd)
    logger.info(
        "Chat %s (%s): answered in %.2fs after %d tool call(s)%s",
        binding_id(chat_id),
        binding.odoo_login,
        time.monotonic() - started,
        len(result.tool_calls),
        " — pending approval" if result.pending else "",
    )
    if result.pending is not None:
        discard(artifacts)  # a paused turn sends buttons, not files
        return BotReply(
            text=confirmation_message(
                result.pending, settings.mail_sender_name, binding.odoo_name
            ),
            pending=result.pending,
            trace_id=turn.trace_id or None,  # decision taps score THIS trace
            usage=spend.snapshot(),
        )
    # An empty answer almost always means the model declined to respond.
    reply = clean_reply(result.answer) or (
        "I can't help with that request. Ask me about your Odoo business "
        "data — sales, invoices, customers, inventory, and more."
    )
    reply = _without_secrets(_without_prompt_leak(reply, settings, chat_id), settings, chat_id)
    if cancelled:
        reply = CANCELLED_PREFIX + reply
    return BotReply(
        text=reply,
        attachments=artifacts,
        trace_id=turn.trace_id or None,
        evidence=evidence,
        grounded=bool(result.tool_calls),
        from_documents=_document_answer(result.tool_calls),
        usage=spend.snapshot(),
    )


async def resume_pending(
    store: BindingStore,
    settings: Settings,
    chat_id: int,
    approved: bool,
    on_tool: Callable[[str], Awaitable[None]] | None = None,
    on_text: Callable[[str], Awaitable[None]] | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    revision: str | None = None,
    tracing: Tracing | None = None,
    facts: BaseStore | None = None,
    latency: LatencyRecorder | None = None,
) -> BotReply:
    """Resume a paused action, bounded by the turn limit (see ``_resume_pending``)."""
    return await _within_turn_limit(
        settings,
        chat_id,
        _resume_pending(
            store,
            settings,
            chat_id,
            approved,
            on_tool=on_tool,
            on_text=on_text,
            checkpointer=checkpointer,
            revision=revision,
            tracing=tracing,
            facts=facts,
            latency=latency,
        ),
    )


async def _resume_pending(
    store: BindingStore,
    settings: Settings,
    chat_id: int,
    approved: bool,
    on_tool: Callable[[str], Awaitable[None]] | None = None,
    on_text: Callable[[str], Awaitable[None]] | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    revision: str | None = None,
    tracing: Tracing | None = None,
    facts: BaseStore | None = None,
    latency: LatencyRecorder | None = None,
) -> BotReply:
    """Deliver an Approve/Cancel/Edit decision for the chat's paused action.

    With ``revision`` (and ``approved=False``) the rejection carries the
    user's plain-language change request; the model then proposes a corrected
    action, which comes back as a fresh pending approval.
    """
    binding = await ensure_access(store, settings, chat_id)
    if binding is None:
        return BotReply(text=CONNECT_PROMPT)

    started = time.monotonic()
    artifacts: list[Artifact] = []
    evidence = Evidence()
    tools = redact_tool_results(
        bridge_export_tool(
            bridge_record_links(
                filter_for_bot(
                    await get_odoo_tools(settings.mcp_url, binding.access_token),
                    allow_method_calls=binding.can_call_methods,
                    allow_writes=binding.can_write,
                ),
                settings.odoo_public_url if settings.deep_links else "",
                evidence,
            ),
            artifacts,
        ),
        settings.secret_scan,
    ) + build_local_tools(artifacts)
    memories: list[str] = []
    if facts is not None:
        memories = await recall(
            facts, binding.odoo_login, revision or "", settings.memory_max_recall
        )
        tools = tools + build_memory_tools(facts, binding.odoo_login)
    agent = build_agent(
        build_model(settings),
        tools,
        checkpointer=checkpointer,
        memories=memories,
        summary_trigger_tokens=settings.summary_trigger_tokens,
        summary_keep_messages=settings.summary_keep_messages,
    )
    channel = channel_name(chat_id)
    turn_context = (
        tracing.turn(
            trace_name=f"{channel}-resume",
            user_id=binding.odoo_login,
            session_id=str(chat_id),
            input_text=revision or ("approve" if approved else "cancel"),
            tags=[channel],
        )
        if tracing is not None
        else nullcontext(Turn("", None))
    )
    spend = SpendAccumulator()
    callbacks: list[BaseCallbackHandler] = [tracing.handler, spend] if tracing else [spend]
    if latency is not None:
        callbacks.append(latency.handler(KIND_CHAT))
    with turn_context as turn:
        result = await resume(
            agent,
            approved,
            await current_thread(store, chat_id),
            on_tool=on_tool,
            recursion_limit=settings.recursion_limit,
            reject_message=_revision_message(revision) if revision else None,
            callbacks=callbacks,
            on_text=on_text if settings.token_streaming else None,
        )
        turn.set_output(result.answer or "<pending approval>")
    await store.record_usage(subject_for(binding, chat_id), _today(), spend.total_usd)
    logger.info(
        "Chat %s (%s): %s resumed in %.2fs",
        binding_id(chat_id),
        binding.odoo_login,
        "approve" if approved else "reject",
        time.monotonic() - started,
    )
    # Record what became of the actions the user just decided on. Written
    # AFTER the outcome is known, so an approved action the server then
    # refused is logged as blocked rather than done.
    refused_names = {failure.name for failure in result.failures}
    for action in result.executed:
        if not approved:
            outcome: ActionOutcome = "revised" if revision else "cancelled"
        else:
            outcome = "blocked" if action.name in refused_names else "done"
        with suppress(Exception):  # the log must never break the reply
            await store.log_action(
                chat_id,
                binding.odoo_login,
                action.name,
                action.summary,
                outcome,
                settings.action_log_days,
            )

    if result.pending is not None:  # the model proposed ANOTHER write
        discard(artifacts)
        return BotReply(
            text=confirmation_message(
                result.pending, settings.mail_sender_name, binding.odoo_name
            ),
            pending=result.pending,
            trace_id=turn.trace_id or None,
            usage=spend.snapshot(),
        )
    # The server refused the action the user just approved. Report exactly
    # that and nothing else, replacing whatever the model said: left to
    # itself it answers the underlying question instead, handing over data
    # the user never asked for (live incident 2026-08-31).
    refused = [failure for failure in result.failures if failure.name in CONFIRM_TOOLS]
    if approved and refused:
        discard(artifacts)  # no charts or exports from a failed write turn
        return BotReply(
            text=blocked_reply(refused),
            blocked=True,
            trace_id=turn.trace_id or None,
            usage=spend.snapshot(),
        )
    return BotReply(
        text=_without_secrets(
            _without_prompt_leak(clean_reply(result.answer) or "Done.", settings, chat_id),
            settings,
            chat_id,
        ),
        attachments=artifacts,
        trace_id=turn.trace_id or None,
        evidence=evidence,
        grounded=bool(result.tool_calls),
        from_documents=_document_answer(result.tool_calls),
        usage=spend.snapshot(),
    )


def _deps(context: ContextTypes.DEFAULT_TYPE) -> tuple[BindingStore, Settings]:
    return context.application.bot_data["store"], context.application.bot_data["settings"]


def _memory(context: ContextTypes.DEFAULT_TYPE) -> BaseCheckpointSaver[Any] | None:
    return context.application.bot_data.get("memory")


def _memory_of(application: Application) -> BaseCheckpointSaver[Any] | None:  # type: ignore[type-arg]
    """Same thing for the ticker, which holds the application directly."""
    memory: BaseCheckpointSaver[Any] | None = application.bot_data.get("memory")
    return memory


def _tracing(context: ContextTypes.DEFAULT_TYPE) -> Tracing | None:
    return context.application.bot_data.get("tracing")


def _latency(context: ContextTypes.DEFAULT_TYPE) -> LatencyRecorder | None:
    """The latency recorder (None before startup, or when metrics are off)."""
    result: LatencyRecorder | None = context.application.bot_data.get("latency")
    return result


def _facts(context: ContextTypes.DEFAULT_TYPE) -> BaseStore | None:
    """The long-term preference store (None when memory is disabled)."""
    result: BaseStore | None = context.application.bot_data.get("facts")
    return result


async def clear_memory(
    store: BindingStore, memory: BaseCheckpointSaver[Any] | None, chat_id: int
) -> None:
    """Forget a chat's conversation thread; failures are logged, not raised.

    Goes through the store because the thread id is no longer the chat id:
    a rotated chat is on ``<chat>-<epoch>``, and deleting ``<chat>`` would
    clear nothing while telling the user their conversation was forgotten.
    The epoch itself is left alone — epochs are monotonic, so a reused id
    can never collide with a thread whose delete failed.
    """
    if memory is None:
        return
    try:
        await memory.adelete_thread(await current_thread(store, chat_id))
        logger.info("Cleared conversation memory for chat %s", binding_id(chat_id))
    except Exception:
        logger.error("Failed clearing memory for chat %s", binding_id(chat_id), exc_info=True)


def login_message(url: str) -> tuple[str, InlineKeyboardMarkup | None]:
    """Compose the login prompt.

    Telegram only accepts public HTTPS URLs in inline keyboard buttons, so a
    local-dev authorize URL (http://localhost:...) is sent as plain text the
    user opens manually; a production HTTPS URL gets a proper button.
    """
    if url.lower().startswith("https://"):
        text = (
            "Let's connect you to Odoo. Tap the button, log in with YOUR Odoo "
            "account, and approve the access request."
        )
        markup = InlineKeyboardMarkup([[InlineKeyboardButton("Log in to Odoo", url=url)]])
        return text, markup
    text = (
        "Let's connect you to Odoo.\n\n"
        "Open this link in your browser, log in with YOUR Odoo account, and "
        "approve the access request:\n\n"
        f"{url}"
    )
    return text, None


def _burst(context: ContextTypes.DEFAULT_TYPE) -> BurstLimiter | None:
    limiter: BurstLimiter | None = context.application.bot_data.get("burst")
    return limiter


def sent_at(update: Update) -> float | None:
    """Epoch seconds the message was SENT, per Telegram, or None.

    The burst window must count arrival, not processing: answers take
    several seconds each and run one at a time, so timestamping on
    processing let a flood outlast its own window (found live 2026-08-31).
    Only genuine new messages carry a usable time — a button tap reports
    the date of the message the button hangs off, which may be hours old.
    """
    message = update.message
    if message is None or message.date is None:
        return None
    return message.date.timestamp()


GROUP_REPLY = "I only work in private chats — message me directly and I'll help there."

#: Chats already told, and when. Replying on every message would make the
#: bot a spammer in someone's group, the same failure the burst limiter's
#: warn-once rule avoids. In memory, so a restart forgets — acceptable,
#: because the bot leaves the chat immediately afterwards.
#:
#: A dict rather than a set because entries are PRUNED (see below); as a
#: set this only ever grew.
_TOLD_ABOUT_PRIVATE: dict[int, float] = {}


def _should_tell_about_private(chat_id: int, ttl: float, now: float) -> bool:
    """True the first time this chat is refused, then not again for ``ttl``.

    Stale keys are pruned on every call — the same rule, for the same
    reason, as ``BurstLimiter._prune``: without it this map only ever grows,
    which is the shape of a leak even though reaching it needs an
    allowlisted sender adding the bot to thousands of groups.

    A time window rather than a size cap because it self-empties: after the
    TTL with no group traffic nothing is retained at all. A group re-added
    later gets one more notice, which is the harmless direction.
    """
    for key, told_at in list(_TOLD_ABOUT_PRIVATE.items()):
        if now - told_at >= ttl:
            del _TOLD_ABOUT_PRIVATE[key]
    if chat_id in _TOLD_ABOUT_PRIVATE:
        return False
    _TOLD_ABOUT_PRIVATE[chat_id] = now
    return True


async def guard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Front door for EVERY update: who may talk, and how fast.

    Registered in handler group -1 and raising ApplicationHandlerStop, so it
    covers every handler — including ones added later — instead of repeating
    the same two checks in seven places.

    Keyed on the Telegram USER id, not the chat id: in a group chat they
    differ, and the question here is who is sending, not where.
    """
    sender = update.effective_user
    if sender is None:
        return
    settings: Settings = context.application.bot_data["settings"]

    # Unknown senders get no reply at all: an answer of any kind confirms
    # the bot is live and tells them their id is simply not on the list.
    if settings.allowed_telegram_ids and sender.id not in settings.allowed_telegram_ids:
        blocked_logger.warning(
            "Blocked message from Telegram id %s (@%s, %s) (not allowlisted)",
            sender.id,
            sender.username or "-",
            sender.full_name,
        )
        raise ApplicationHandlerStop

    # The entire security model is "one chat = one Odoo user": every question
    # runs under that person's token, record rules and audit trail. In a group
    # the binding would be the GROUP's id, so one person's /start would hand
    # every other member their Odoo identity. Deliberately AFTER the allowlist
    # check — a stranger must still get silence, since any reply confirms the
    # bot is live; but someone known who added it to a group needs telling,
    # because they already know it exists.
    chat = update.effective_chat
    if chat is not None and chat.type != ChatType.PRIVATE:
        logger.warning("Refusing %s chat %s — this bot is private-chat only", chat.type, chat.id)
        if _should_tell_about_private(chat.id, settings.group_notice_ttl_seconds, time.time()):
            with suppress(Exception):
                await context.bot.send_message(chat_id=chat.id, text=GROUP_REPLY)
        # Leaving removes the risk rather than declining message by message.
        # Suppressed: the bot may already have been removed, and a failure to
        # leave must not turn into a failure to refuse.
        with suppress(Exception):
            await context.bot.leave_chat(chat.id)
        raise ApplicationHandlerStop

    limiter = _burst(context)
    if limiter is not None and not limiter.allow(sender.id, sent_at(update)):
        logger.warning("Burst limit hit by Telegram id %s", sender.id)
        # Warn once, then go silent: replying to every message of a flood
        # would make the bot the amplifier and trip Telegram's own per-chat
        # send limit.
        if limiter.should_warn(sender.id) and update.effective_chat is not None:
            with suppress(Exception):
                await context.bot.send_message(chat_id=update.effective_chat.id, text=BURST_REPLY)
        raise ApplicationHandlerStop


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.message is None:
        return
    store, settings = _deps(context)
    chat_id = update.effective_chat.id

    binding = await store.get_binding(chat_id)
    if binding is not None:
        await update.message.reply_text(
            f"You are connected as {binding.odoo_name} ({binding.odoo_login}). "
            "Ask me anything about your Odoo data — I remember our conversation, "
            "so follow-up questions work. /new starts a fresh conversation, "
            "/logout disconnects."
        )
        return

    url = await begin_login(store, settings, chat_id)
    text, markup = login_message(url)
    await update.message.reply_text(text, reply_markup=markup)


async def cmd_logout(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.message is None:
        return
    store, settings = _deps(context)
    chat_id = update.effective_chat.id
    removed = await logout(store, settings, chat_id)
    # The next person binding this chat must never inherit this conversation.
    # Long-term preferences are deliberately NOT cleared: they belong to the
    # Odoo user, not to this chat, and a different login simply reads a
    # different namespace. Use /forget to erase them.
    _awaiting_edit(context).discard(chat_id)
    await clear_memory(store, _memory(context), chat_id)
    await update.message.reply_text(
        "Disconnected. Your access tokens were revoked and the conversation was cleared."
        if removed
        else "You were not connected."
    )


async def cmd_new(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/new (alias /reset): forget the conversation, keep the login."""
    if update.effective_chat is None or update.message is None:
        return
    store, _settings = _deps(context)
    _awaiting_edit(context).discard(update.effective_chat.id)
    await clear_memory(store, _memory(context), update.effective_chat.id)
    await update.message.reply_text(
        "Fresh start — I've forgotten our previous conversation. Ask away."
    )


async def cmd_forget(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/forget: erase the long-term preferences stored for THIS Odoo user."""
    if update.effective_chat is None or update.message is None:
        return
    store, _settings = _deps(context)
    facts = _facts(context)
    if facts is None:
        await update.message.reply_text("I'm not storing any preferences right now.")
        return
    binding = await store.get_binding(update.effective_chat.id)
    if binding is None:
        await update.message.reply_text(CONNECT_PROMPT)
        return
    cleared = await forget_all(facts, binding.odoo_login)
    # The conversation must go too: it still contains the assistant repeating
    # the preference, and the model would read its own earlier reply and
    # "remember" what was just deleted (found live 2026-08-24).
    _awaiting_edit(context).discard(update.effective_chat.id)
    await clear_memory(store, _memory(context), update.effective_chat.id)
    await update.message.reply_text(
        f"Forgotten — I cleared {cleared} saved preference(s), "
        "and this conversation too so nothing lingers."
        if cleared
        else "There was nothing saved about you to forget."
    )


async def _proactive_sender(context: ContextTypes.DEFAULT_TYPE) -> Sender:
    """Deliver a proactive message through the normal reply pipeline, so
    Markdown, HTML escaping and the 4096-character clamp behave exactly as
    they do for an answer the user asked for."""

    async def send(chat_id: int, text: str) -> None:
        body, parse_mode = prepare_reply_text(BotReply(text=text))
        try:
            await context.bot.send_message(chat_id=chat_id, text=body, parse_mode=parse_mode)
        except Exception:  # last resort: clamped plain text
            logger.warning("Telegram refused a proactive message; sending plain text")
            await context.bot.send_message(chat_id=chat_id, text=plain_text(text)[:4096])

    return send


#: Actions listed by /audit before it stops being readable in a chat.
_AUDIT_LIMIT = 25

#: How a logged outcome reads in the audit list.
_AUDIT_WORDS: dict[str, str] = {
    "done": "✅",
    "blocked": "🚫",
    "cancelled": "❌",
    "revised": "✏️",
}


def audit_report(actions: list[LoggedAction], total: int, days: int) -> str:
    """The operator view: what the bot did, for whom, and what became of it.

    Truncation is STATED. An audit view that silently drops rows is worse
    than none, because it reads as "that was everything".
    """
    if not actions:
        return f"No actions recorded in the last {days} day(s)."
    lines = [f"🔎 **{total} action(s) in the last {days} day(s)**", ""]
    for action in actions[:_AUDIT_LIMIT]:
        mark = _AUDIT_WORDS.get(action.outcome, "•")
        headline = action.summary.splitlines()[0]
        lines.append(f"{mark} {action.created_at:%d %b %H:%M} · {action.odoo_login} · {headline}")
    if total > _AUDIT_LIMIT:
        lines.append("")
        lines.append(f"… and {total - _AUDIT_LIMIT} more not shown.")
    return "\n".join(lines)


async def cmd_audit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/audit [days]: every user's actions. Administrators only."""
    if update.effective_chat is None or update.message is None:
        return
    store, settings = _deps(context)
    binding = await store.get_binding(update.effective_chat.id)
    if binding is None:
        await update.message.reply_text(CONNECT_PROMPT)
        return
    if not binding.is_admin:
        # Deliberately incurious: no hint at what is behind it.
        await update.message.reply_text("That command is for administrators.")
        return

    days = 7
    if context.args:
        try:
            days = max(1, min(int(context.args[0]), settings.action_log_days))
        except ValueError:
            await update.message.reply_text("Usage: /audit [days], for example /audit 30")
            return

    since = _now() - dt.timedelta(days=days)
    actions = await store.all_actions(since, limit=_AUDIT_LIMIT)
    total = await store.count_actions(since)
    text, parse_mode = prepare_reply_text(BotReply(text=audit_report(actions, total, days)))
    await update.message.reply_text(text, parse_mode=parse_mode)


async def cmd_brief(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/brief - send it now; /brief off|on - opt out of the daily one."""
    if update.effective_chat is None or update.message is None:
        return
    store, settings = _deps(context)
    chat_id = update.effective_chat.id
    argument = context.args[0].lower() if context.args else ""

    if argument in ("off", "on"):
        binding = await store.get_binding(chat_id)
        if binding is None:
            await update.message.reply_text(CONNECT_PROMPT)
            return
        await store.set_brief_enabled(chat_id, argument == "on")
        await update.message.reply_text(
            "Daily brief is back on — it will arrive each morning."
            if argument == "on"
            else "Daily brief off. Ask for one any time with /brief."
        )
        return

    placeholder = await update.message.reply_text("📋 Putting your brief together …")
    try:
        # Inside the try: ensure_access raises on a TRANSIENT refresh failure
        # (Odoo restarting, a 5xx through the tunnel), and outside it that
        # would escape the handler and leave the placeholder sitting there
        # for ever with no explanation.
        binding = await ensure_access(store, settings, chat_id)
        if binding is None:
            await placeholder.edit_text(CONNECT_PROMPT)
            return
        text = await run_brief(store, settings, binding, _facts(context))
        body, parse_mode = prepare_reply_text(BotReply(text=text or "Nothing to report."))
        await placeholder.edit_text(body, parse_mode=parse_mode)
    except Exception:
        logger.error("On-demand brief failed for %s", binding_id(chat_id), exc_info=True)
        await placeholder.edit_text(ERROR_REPLY)


async def cmd_watch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/watch <condition>: store a standing alert."""
    if update.effective_chat is None or update.message is None:
        return
    store, settings = _deps(context)
    chat_id = update.effective_chat.id
    condition = " ".join(context.args or []).strip()
    if not condition:
        await update.message.reply_text(
            "Tell me what to watch for, like:\n"
            "/watch any customer invoice more than 60 days overdue"
        )
        return

    binding = await store.get_binding(chat_id)
    if binding is None:
        await update.message.reply_text(CONNECT_PROMPT)
        return
    if await store.count_watchers(chat_id) >= settings.max_watchers_per_user:
        await update.message.reply_text(
            f"You already have {settings.max_watchers_per_user} alerts, which is the "
            "limit. Use /watchers to see them and /unwatch to free one up."
        )
        return

    await store.add_watcher(chat_id, binding.odoo_login, condition)
    await update.message.reply_text(
        f"👀 Watching: {condition}\n\nI check about every "
        f"{settings.watcher_interval_minutes} minutes and will tell you when it "
        "happens — once per change, not every time I look."
    )


async def cmd_watchers(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/watchers: list this chat's standing alerts."""
    if update.effective_chat is None or update.message is None:
        return
    store, _settings = _deps(context)
    watchers = await store.list_watchers(update.effective_chat.id)
    if not watchers:
        await update.message.reply_text(
            "No alerts set. Add one with /watch, for example:\n"
            "/watch any customer invoice more than 60 days overdue"
        )
        return
    lines = [
        f"{index}. {watcher.condition}"
        + (
            f"\n   last checked {watcher.last_checked_at:%Y-%m-%d %H:%M} UTC"
            if watcher.last_checked_at
            else "\n   not checked yet"
        )
        for index, watcher in enumerate(watchers, start=1)
    ]
    await update.message.reply_text(
        "👀 Your alerts:\n\n" + "\n".join(lines) + "\n\nRemove one with /unwatch <number>."
    )


async def cmd_unwatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/unwatch <n>: remove one alert, numbered as /watchers lists them."""
    if update.effective_chat is None or update.message is None:
        return
    store, _settings = _deps(context)
    chat_id = update.effective_chat.id
    raw = context.args[0] if context.args else ""
    watchers = await store.list_watchers(chat_id)
    try:
        index = int(raw)
    except ValueError:
        await update.message.reply_text("Which one? Use /watchers to see the numbers.")
        return
    if not 1 <= index <= len(watchers):
        await update.message.reply_text(f"There is no alert {index}. You have {len(watchers)}.")
        return
    # Deleting by position, but scoped by chat in the store too: a watcher id
    # is a small integer and must never be addressable across chats.
    removed = await store.deactivate_watcher(watchers[index - 1].id, chat_id)
    await update.message.reply_text(
        f"Removed: {watchers[index - 1].condition}" if removed else "That alert is already gone."
    )


async def _ticker(application: Application) -> None:  # type: ignore[type-arg]
    """Background loop: ask what work is due, do it, sleep.

    Two jobs share the loop — proactive messages, and the daily retention
    pass. Both decide for themselves whether they are due, so the loop only
    has to keep its cadence.

    Deliberately forgiving — a scheduled job must never take the bot down,
    so every pass is wrapped and the loop keeps going regardless.
    """
    settings: Settings = application.bot_data["settings"]
    store: BindingStore = application.bot_data["store"]

    async def send(chat_id: int, text: str) -> None:
        if is_slack(chat_id):
            slack: SlackChannel | None = application.bot_data.get("slack")
            if slack is not None:
                await slack.send_plain(chat_id, text)
            return
        if is_google_chat(chat_id):
            googlechat: GoogleChatChannel | None = application.bot_data.get("googlechat")
            if googlechat is not None:
                await googlechat.send_plain(chat_id, text)
            return
        body, parse_mode = prepare_reply_text(BotReply(text=text))
        try:
            await application.bot.send_message(chat_id=chat_id, text=body, parse_mode=parse_mode)
        except Exception:
            await application.bot.send_message(chat_id=chat_id, text=text[:4096])

    logger.info(
        "Proactive ticker running: brief %02d:%02d local, watchers every %dm, tick %ds",
        settings.brief_hour,
        settings.brief_minute,
        settings.watcher_interval_minutes,
        settings.tick_seconds,
    )
    while True:
        try:
            await asyncio.sleep(settings.tick_seconds)
            await tick(
                store,
                settings,
                send,
                application.bot_data.get("facts"),
                application.bot_data.get("latency"),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("Proactive tick failed", exc_info=True)
        try:
            # Its own try: retention failing must not cost anyone their
            # morning brief, and a brief failing must not skip retention.
            await run_if_due(store, _memory_of(application), settings)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("Retention pass failed", exc_info=True)


async def send_attachments(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int, attachments: list[Artifact]
) -> None:
    """Send rendered files to the chat; always deletes the temp files."""
    for artifact in attachments:
        try:
            with artifact.path.open("rb") as handle:
                if artifact.kind == "photo":
                    await context.bot.send_photo(
                        chat_id=chat_id, photo=handle, caption=artifact.caption
                    )
                else:
                    await context.bot.send_document(
                        chat_id=chat_id,
                        document=handle,
                        filename=artifact.path.name,
                        caption=artifact.caption,
                    )
        except Exception:
            logger.error("Failed sending %s to chat %s", artifact.kind, binding_id(chat_id))
        finally:
            artifact.path.unlink(missing_ok=True)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is not None:
        await update.message.reply_text(help_text())


#: Produces the reply for one turn, given a progress callback. Lets the
#: text, edit-revision and voice paths share one delivery implementation.
Runner = Callable[
    [Callable[[str], Awaitable[None]], Callable[[str], Awaitable[None]]],
    Awaitable["BotReply"],
]


@dataclass
class _DraftUpdater:
    edit: Callable[[str], Awaitable[Any]]
    interval_seconds: float = 1.0
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    last_text: str = ""
    last_edit_at: float | None = None
    pending_text: str = ""
    task: asyncio.Task[None] | None = None
    disabled: bool = False
    closed: bool = False

    async def update(self, text: str, *, immediate: bool = False) -> None:
        if self.disabled or self.closed:
            return
        draft = plain_text(text).strip()[:4096]
        if not draft or draft in (self.last_text, self.pending_text):
            return

        if immediate:
            await self._cancel_pending()
            self.pending_text = draft
            await self._flush()
            return

        self.pending_text = draft
        if self.last_edit_at is None:
            await self._flush()
        elif self.task is None:
            self.task = asyncio.create_task(self._flush_later())

    async def close(self) -> None:
        self.closed = True
        await self._cancel_pending()

    async def _cancel_pending(self) -> None:
        self.pending_text = ""
        task = self.task
        self.task = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def _flush_later(self) -> None:
        try:
            elapsed = self.clock() - (self.last_edit_at or 0.0)
            await self.sleep(max(0.0, self.interval_seconds - elapsed))
            await self._flush()
        except asyncio.CancelledError:
            raise
        finally:
            self.task = None
            if self.pending_text and not self.disabled and not self.closed:
                self.task = asyncio.create_task(self._flush_later())

    async def _flush(self) -> None:
        draft = self.pending_text
        self.pending_text = ""
        if not draft or draft == self.last_text or self.disabled or self.closed:
            return
        try:
            await self.edit(draft)
        except Exception:
            self.disabled = True
            self.pending_text = ""
            logger.debug("Draft edit failed; falling back to final reply", exc_info=True)
            return
        self.last_text = draft
        self.last_edit_at = self.clock()


async def _no_edit(text: str) -> None:
    return None


async def _run_and_deliver(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    run: Runner,
    *,
    instant: bool = False,
    native_draft: bool = True,
) -> None:
    """Stream a live draft, run the turn, deliver the final answer.

    Private chats use Telegram's animated, ephemeral draft API. Other chats,
    clients where native drafts fail, and ``native_draft=False`` use the
    editable placeholder, which BECOMES the reply instead of lingering beside it.
    ``instant`` skips both: the answer is already known to arrive at once,
    so a draft would only flash before it.
    """
    if update.effective_chat is None or update.message is None:
        return
    chat_id = update.effective_chat.id
    placeholder = None
    if instant:
        updater = _DraftUpdater(_no_edit, disabled=True)
    elif native_draft and update.effective_chat.type == ChatType.PRIVATE:
        draft_id = (uuid4().int % 2_147_483_647) or 1
        message_thread_id = getattr(update.message, "message_thread_id", None)

        async def send_draft(text: str) -> None:
            await context.bot.send_message_draft(
                chat_id=chat_id,
                draft_id=draft_id,
                text=text,
                message_thread_id=message_thread_id,
            )

        try:
            await send_draft(initial_status())
            updater = _DraftUpdater(send_draft)
        except Exception:
            logger.debug(
                "Native Telegram draft failed; using editable placeholder",
                exc_info=True,
            )
            placeholder = await update.message.reply_text(initial_status())
            updater = _DraftUpdater(placeholder.edit_text)
    else:
        placeholder = await update.message.reply_text(initial_status())
        updater = _DraftUpdater(placeholder.edit_text)

    async def show_progress(call: str) -> None:
        # A gated write is only proposed here, not running; its approval prompt
        # replaces the draft, and a "checking send_email" line would linger.
        if call.split("(", 1)[0].strip() in CONFIRM_TOOLS:
            return
        await updater.update(tool_progress_line(call), immediate=True)

    async def show_text(text: str) -> None:
        await updater.update(text)

    async def keep_typing() -> None:
        while True:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            await asyncio.sleep(4)

    typing_task = None if instant else asyncio.create_task(keep_typing())
    try:
        reply = await run(show_progress, show_text)
    except Exception:
        logger.error("Failed answering chat %s", binding_id(chat_id), exc_info=True)
        reply = BotReply(text=ERROR_REPLY)
    finally:
        if typing_task is not None:
            typing_task.cancel()
        await updater.close()

    store, settings = _deps(context)
    binding = await store.get_binding(chat_id)
    if reply.grounded and reply.text:
        # Only data-backed answers: a conversational reply has no "as of".
        stamp = freshness_line(reply.evidence, binding, settings, _now())
        reply.text += f"\n\n*{stamp}*"
    markup = reply_markup_for(reply, can_write=bool(binding and binding.can_write))
    text, parse_mode = prepare_reply_text(reply)
    try:
        if placeholder is None:
            await update.message.reply_text(text, reply_markup=markup, parse_mode=parse_mode)
        else:
            await placeholder.edit_text(text, reply_markup=markup, parse_mode=parse_mode)
    except Exception:
        logger.debug("Rendered final reply failed; sending plain text", exc_info=True)
        try:
            await update.message.reply_text(prepare_plain_reply_text(reply), reply_markup=markup)
        except Exception:  # last resort: no buttons
            logger.warning("Telegram refused the rendered reply; sending plain text")
            await update.message.reply_text(prepare_plain_reply_text(reply))
    await send_attachments(context, chat_id, reply.attachments)


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.message is None or not update.message.text:
        return
    store, settings = _deps(context)
    chat_id = update.effective_chat.id

    # Resolving a pending action (✏️ Edit revision) is never rate-limited —
    # a half-finished approval must always be resolvable. Only fresh
    # questions are gated, and gated BEFORE the placeholder: a blocked chat
    # incurs no LLM call and no "thinking" message.
    awaiting = _awaiting_edit(context)
    editing = chat_id in awaiting
    if not editing:
        binding = await store.get_binding(chat_id)
        if binding is None:
            link_reply = await confirm_link(store, settings, chat_id, update.message.text or "")
            if link_reply is not None:
                await update.message.reply_text(link_reply)
                return
        # get_binding, not ensure_access: the subject only needs the stored
        # login, and a token refresh would be wasted work on a refusal.
        subject = subject_for(binding, chat_id)
        today = _today()
        refusal = over_limit(
            await store.get_usage(subject, today),
            await store.get_usage(GLOBAL_SUBJECT, today),
            settings,
        )
        if refusal is not None:
            await update.message.reply_text(refusal)
            return

    awaiting.discard(chat_id)
    question = update.message.text or ""

    async def run(
        on_tool: Callable[[str], Awaitable[None]],
        on_text: Callable[[str], Awaitable[None]],
    ) -> BotReply:
        if editing:
            # The user tapped ✏️ Edit: this message is the change request for
            # the paused action, not a new question.
            return await resume_pending(
                store,
                settings,
                chat_id,
                approved=False,
                on_tool=on_tool,
                on_text=on_text,
                checkpointer=_memory(context),
                revision=question,
                tracing=_tracing(context),
                facts=_facts(context),
                latency=_latency(context),
            )
        return await answer_question(
            store,
            settings,
            chat_id,
            question,
            on_tool=on_tool,
            on_text=on_text,
            checkpointer=_memory(context),
            tracing=_tracing(context),
            facts=_facts(context),
            latency=_latency(context),
        )

    instant = not editing and fast_reply(settings, question) is not None
    await _run_and_deliver(update, context, run, instant=instant)


async def on_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Approve/Cancel button tap: deliver the decision to the paused thread."""
    query = update.callback_query
    if query is None or update.effective_chat is None:
        return
    await query.answer()  # stop the button's loading spinner immediately
    chat_id = update.effective_chat.id
    store, settings = _deps(context)

    # The proposal text is preserved through every state transition — the
    # user must always be able to look back at exactly what they decided on.
    # text_html keeps the bold formatting; a plain-text fallback is escaped
    # so it stays valid under parse_mode HTML.
    import html as html_lib

    plain = getattr(query.message, "text", None)
    original = getattr(query.message, "text_html", None) or (
        html_lib.escape(plain) if plain else None
    )

    async def edit_status(status: str) -> None:
        try:
            await query.edit_message_text(status_text(original, status), parse_mode="HTML")
        except Exception:  # progress/status must never break the decision flow
            logger.debug("Status edit failed", exc_info=True)

    parsed = parse_confirm_data(query.data or "")
    if parsed is None:
        return
    verdict_kind, proposal_trace_id = parsed
    tracing = _tracing(context)

    def score_decision(decision: str) -> None:
        # Action-proposal precision: the tap becomes a Langfuse score on the
        # trace that PROPOSED the action.
        if tracing is not None and proposal_trace_id:
            tracing.record_decision(proposal_trace_id, decision)

    if verdict_kind == "edit":
        # Keep the action paused; the user's NEXT text message revises it.
        score_decision("edit-requested")
        _awaiting_edit(context).add(chat_id)
        await edit_status(EDIT_PROMPT)
        return

    approved = verdict_kind == "approve"
    # Scores the PROPOSAL, not the outcome: "approved" means the human
    # accepted what the model suggested, which stays true even when Odoo
    # then refuses the call. Do not switch this to the outcome.
    score_decision("approved" if approved else "cancelled")
    verdict = "✅ <b>Approved</b>" if approved else "❌ <b>Cancelled</b>"

    async def show_progress(call: str) -> None:
        await edit_status(f"{verdict} — {tool_progress_line(call)}")

    async def edit_draft(text: str) -> None:
        await query.edit_message_text(
            status_text(original, f"{verdict} — {html_lib.escape(text)}"),
            parse_mode="HTML",
        )

    draft_updater = _DraftUpdater(edit_draft)

    async def show_text(text: str) -> None:
        await draft_updater.update(text)

    await edit_status(f"{verdict} — {whimsy().lower()} …")

    try:
        reply = await resume_pending(
            store,
            settings,
            chat_id,
            approved,
            on_tool=show_progress,
            on_text=show_text,
            checkpointer=_memory(context),
            tracing=_tracing(context),
            facts=_facts(context),
            latency=_latency(context),
        )
    except Exception:
        logger.error("Failed resuming chat %s", binding_id(chat_id), exc_info=True)
        reply = BotReply(text=ERROR_REPLY)
    finally:
        await draft_updater.close()

    # Freeze the proposal message with its final verdict (buttons removed).
    # Progress frames above said "Approved" — accurate, the user did approve
    # — but the FROZEN message must report what actually happened.
    await edit_status(BLOCKED_VERDICT if reply.blocked else verdict)
    # ...and deliver the outcome as its OWN message, so neither is lost.
    markup = reply_markup_for(reply)
    text, parse_mode = prepare_reply_text(reply)
    try:
        await context.bot.send_message(
            chat_id=chat_id, text=text, reply_markup=markup, parse_mode=parse_mode
        )
    except Exception:
        try:  # last resort: clamped plain text, no formatting
            await context.bot.send_message(
                chat_id=chat_id,
                text=prepare_plain_reply_text(reply),
                reply_markup=markup,
            )
        except Exception:
            logger.error("Failed sending decision result to chat %s", binding_id(chat_id))
    await send_attachments(context, chat_id, reply.attachments)


async def on_followup(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """A follow-up button tap: ask the canned question on the same thread."""
    query = update.callback_query
    if query is None or query.data is None or update.effective_chat is None:
        return
    code = parse_followup_data(query.data)
    if code is None:
        await query.answer()
        return
    await query.answer()

    # Retire the row so a double tap cannot run it twice; the feedback row
    # goes with it, which is fine — the follow-up answer carries its own.
    with suppress(Exception):
        await query.edit_message_reply_markup(reply_markup=None)

    store, settings = _deps(context)
    chat_id = update.effective_chat.id
    question = FOLLOWUPS[code][1]

    async def run(
        on_tool: Callable[[str], Awaitable[None]],
        on_text: Callable[[str], Awaitable[None]],
    ) -> BotReply:
        return await answer_question(
            store,
            settings,
            chat_id,
            question,
            on_tool=on_tool,
            on_text=on_text,
            checkpointer=_memory(context),
            tracing=_tracing(context),
            facts=_facts(context),
            latency=_latency(context),
        )

    # query.message is the ANSWER the button hung off; replying against it
    # keeps the follow-up visually attached to what it followed up on.
    # No native draft: Telegram keeps showing it for seconds after an
    # approval prompt arrives, and the Bot API cannot clear it.
    await _run_and_deliver(
        _FollowupUpdate(update),  # type: ignore[arg-type]
        context,
        run,
        native_draft=False,
    )


class _FollowupUpdate:
    """Adapts a button tap to the shape _run_and_deliver expects."""

    def __init__(self, update: Update) -> None:
        self.effective_chat = update.effective_chat
        self.effective_user = update.effective_user
        self.message = update.callback_query.message if update.callback_query else None


async def on_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """👍/👎 tap: score the answer's Langfuse trace, then retire the buttons."""
    query = update.callback_query
    if query is None or query.data is None:
        return
    parsed = parse_feedback_data(query.data)
    if parsed is None:
        await query.answer()
        return
    positive, trace_id = parsed
    tracing = _tracing(context)
    if tracing is not None:
        tracing.record_feedback(trace_id, positive)
        await query.answer("Thanks for the feedback!")
    else:
        await query.answer("Feedback recording is currently off.")
    try:
        # Buttons disappear after the tap; a double-tap finds no markup and
        # is answered by the guard above without a second score.
        await query.edit_message_reply_markup(reply_markup=None)
    except Exception:
        logger.debug("Could not clear feedback buttons", exc_info=True)


UPLOADS_OFF = "Reading files is switched off right now — ask me in text instead."

UPLOAD_FAILED = "I couldn't read that file. Try sending it again, or ask me in text."


async def _handle_upload(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    mime: str,
    filename: str,
    size_bytes: int,
    file_id: str,
) -> None:
    """Shared path for an uploaded image or PDF file.

    Follows on_voice's order deliberately: every refusal that can be decided
    from what Telegram DECLARES is decided before the file is fetched, so a
    switched-off, oversized, unsupported or over-budget upload costs no
    download and no model call.
    """
    if update.effective_chat is None or update.message is None:
        return
    store, settings = _deps(context)
    chat_id = update.effective_chat.id

    if not settings.uploads:
        await update.message.reply_text(UPLOADS_OFF)
        return
    refusal = check_upload(mime, size_bytes, settings)
    if refusal is not None:
        await update.message.reply_text(refusal)
        return

    binding = await store.get_binding(chat_id)
    subject = subject_for(binding, chat_id)
    today = _today()
    over = over_limit(
        await store.get_usage(subject, today),
        await store.get_usage(GLOBAL_SUBJECT, today),
        settings,
    )
    if over is not None:
        await update.message.reply_text(over)
        return

    # Whatever they typed with the file is the question. With no caption we
    # ask on their behalf and SAY SO, so a surprising answer traces back to
    # the question rather than looking like the bot inventing a task.
    question = (update.message.caption or "").strip() or DEFAULT_QUESTION
    try:
        handle = await context.bot.get_file(file_id)
        data = bytes(await handle.download_as_bytearray())
    except Exception:
        logger.error("Upload download failed for chat %s", binding_id(chat_id), exc_info=True)
        await update.message.reply_text(UPLOAD_FAILED)
        return

    attachment = Attachment(
        kind=SUPPORTED[mime], mime=mime, filename=filename or "file", data=data
    )
    logger.info("Chat %s: reading upload %s", binding_id(chat_id), describe(attachment))
    if not update.message.caption:
        await update.message.reply_text(f'📎 {describe(attachment)} — asking: "{question}"')

    async def run(
        on_tool: Callable[[str], Awaitable[None]],
        on_text: Callable[[str], Awaitable[None]],
    ) -> BotReply:
        return await answer_question(
            store,
            settings,
            chat_id,
            question,
            on_tool=on_tool,
            on_text=on_text,
            checkpointer=_memory(context),
            tracing=_tracing(context),
            facts=_facts(context),
            latency=_latency(context),
            attachment=attachment,
        )

    await _run_and_deliver(update, context, run)


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """A compressed photo. Telegram sends ascending sizes; take the largest."""
    if update.message is None or not update.message.photo:
        return
    photo = update.message.photo[-1]
    # Compressed photos carry no mime type or name of their own.
    await _handle_upload(
        update, context, "image/jpeg", "photo.jpg", photo.file_size or 0, photo.file_id
    )


async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """A file attachment: a PDF, or an image sent uncompressed.

    Routed by media type, so it does not matter whether the user attached an
    image as a photo or as a file — both reach the same path.
    """
    if update.message is None or update.message.document is None:
        return
    doc = update.message.document
    await _handle_upload(
        update,
        context,
        (doc.mime_type or "").lower(),
        doc.file_name or "file",
        doc.file_size or 0,
        doc.file_id,
    )


VOICE_UNCLEAR = "I couldn't make that out. Try again, or type it instead."

VOICE_OFF = "Voice notes are switched off right now — type your question instead."


async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """A voice note: transcribe it, show what was heard, then answer it.

    The transcript is echoed rather than silently acted on. Transcription is
    never perfect, and seeing the words is how someone catches a misheard
    customer name before they act on the answer built from it.
    """
    if update.effective_chat is None or update.message is None:
        return
    voice = update.message.voice
    if voice is None:
        return
    store, settings = _deps(context)
    chat_id = update.effective_chat.id

    if not settings.voice:
        await update.message.reply_text(VOICE_OFF)
        return
    # python-telegram-bot reports duration as an int or a timedelta
    # depending on its time-type setting; normalise before comparing.
    raw_duration = voice.duration or 0
    seconds = int(
        raw_duration.total_seconds() if isinstance(raw_duration, dt.timedelta) else raw_duration
    )
    if seconds > settings.voice_max_seconds:
        await update.message.reply_text(
            f"That note is {seconds}s — I can handle up to "
            f"{settings.voice_max_seconds}s. Send a shorter one, or type it."
        )
        return

    # Gate on the daily budget BEFORE downloading or transcribing, exactly as
    # the text path does: a blocked chat must incur no cost at all.
    binding = await store.get_binding(chat_id)
    subject = subject_for(binding, chat_id)
    today = _today()
    refusal = over_limit(
        await store.get_usage(subject, today),
        await store.get_usage(GLOBAL_SUBJECT, today),
        settings,
    )
    if refusal is not None:
        await update.message.reply_text(refusal)
        return

    placeholder = await update.message.reply_text("🎤 Listening …")
    try:
        handle = await context.bot.get_file(voice.file_id)
        audio = bytes(await handle.download_as_bytearray())
        result = await transcribe(settings, audio)
    except Exception:
        logger.error("Voice download or transcription failed", exc_info=True)
        await placeholder.edit_text(VOICE_UNCLEAR)
        return

    # Transcription is billed even when it returns nothing, so record it
    # before deciding what to do with the text.
    if result.cost_usd:
        await store.record_usage(subject, today, result.cost_usd)
    if not result.text:
        await placeholder.edit_text(VOICE_UNCLEAR)
        return

    logger.info("Chat %s: voice note transcribed (%ds)", binding_id(chat_id), seconds)
    await placeholder.edit_text(f"🎤 I heard: {result.text}")

    async def run(
        on_tool: Callable[[str], Awaitable[None]],
        on_text: Callable[[str], Awaitable[None]],
    ) -> BotReply:
        return await answer_question(
            store,
            settings,
            chat_id,
            result.text,
            on_tool=on_tool,
            on_text=on_text,
            checkpointer=_memory(context),
            tracing=_tracing(context),
            facts=_facts(context),
            latency=_latency(context),
        )

    await _run_and_deliver(update, context, run)


async def on_other(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Everything not handled above — stickers, video, audio, contacts."""
    if update.message is not None:
        await update.message.reply_text(
            "I can read text, voice notes, photos and PDF files. Send one of those, "
            "or ask me a question."
        )


async def _post_init(application: Application) -> None:  # type: ignore[type-arg]
    settings: Settings = application.bot_data["settings"]
    store = BindingStore(
        settings.agent_db_url,
        TokenCipher.from_key(settings.token_key),
        max_size=settings.store_pool_max_size,
    )
    await store.__aenter__()
    application.bot_data["store"] = store

    # Conversation memory: one shared checkpointer for all chats, keyed by
    # thread (= chat id). Kept apart from credentials so /new and /logout can
    # wipe conversation state without ever touching a token.
    #
    # Long-term memory: preferences keyed by ODOO USER. Survives /new and
    # /logout by design — those wipe the conversation, not who the person is.
    # Identity safety comes from the per-user namespace.
    # ONE pool serves both. They are separate concerns with separate
    # lifetimes, but a single connection pool is what makes that cheap.
    pool: AsyncConnectionPool[AsyncConnection[dict[str, Any]]] = AsyncConnectionPool(
        settings.agent_db_url,
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        open=False,
        # dict_row is required, not a preference: LangGraph's Postgres
        # backends read columns by name. autocommit likewise — they manage
        # their own transactions.
        kwargs={"autocommit": True, "row_factory": dict_row},
    )
    await pool.open(wait=True)
    application.bot_data["db_pool"] = pool

    memory = AsyncPostgresSaver(pool)
    await memory.setup()
    application.bot_data["memory"] = memory
    logger.info("Conversation memory in Postgres")

    if settings.long_term_memory:
        facts = AsyncPostgresStore(pool, index=build_store_index(settings))
        await facts.setup()
        application.bot_data["facts"] = facts
        logger.info("Long-term memory in Postgres")
    else:
        logger.info("Long-term memory disabled (AGENT_LONG_TERM_MEMORY)")

    async def on_bound(link: LinkCode) -> None:
        whatsapp: WhatsAppChannel | None = application.bot_data.get("whatsapp")
        googlechat: GoogleChatChannel | None = application.bot_data.get("googlechat")
        slack: SlackChannel | None = application.bot_data.get("slack")
        if is_slack(link.chat_id):
            if slack is not None:
                await slack.send_plain(link.chat_id, LINK_CODE_PROMPT)
            return
        if is_google_chat(link.chat_id):
            if googlechat is not None:
                await googlechat.send_plain(link.chat_id, LINK_CODE_PROMPT)
            return
        if is_whatsapp(link.chat_id):
            if whatsapp is not None:
                await whatsapp.send_plain(link.chat_id, LINK_CODE_PROMPT)
            return
        await application.bot.send_message(chat_id=link.chat_id, text=LINK_CODE_PROMPT)

    bot_username = application.bot.username  # fetched from Telegram at startup
    logger.info("Bot username: @%s", bot_username)
    try:
        # Powers the "/" autocomplete menu; must never block startup.
        await application.bot.set_my_commands(bot_commands())
        logger.info("Registered %d bot commands with Telegram", len(BOT_COMMANDS))
    except Exception:
        logger.warning("Could not register bot commands", exc_info=True)
    application.bot_data["tracing"] = init_tracing(settings)
    # Built here so it shares the store's pool, started before the ticker
    # so the first proactive call already has somewhere to report.
    recorder = LatencyRecorder(store, settings)
    recorder.start()
    application.bot_data["latency"] = recorder
    application.bot_data["burst"] = BurstLimiter(
        settings.burst_limit, settings.burst_window_seconds
    )
    if settings.whatsapp_enabled:
        # Imported here: whatsapp_bot imports this module, so a top-level
        # import would be circular.
        from ont_agent.whatsapp import WhatsAppClient
        from ont_agent.whatsapp_bot import WhatsAppChannel

        channel = WhatsAppChannel(
            store,
            settings,
            WhatsAppClient(settings),
            memory=memory,
            facts=application.bot_data.get("facts"),
            tracing=application.bot_data["tracing"],
            latency=recorder,
            limiter=application.bot_data["burst"],
        )
        application.bot_data["whatsapp"] = channel
        logger.info(
            "WhatsApp channel enabled (%d allowlisted number(s))",
            len(settings.allowed_whatsapp_numbers),
        )
    else:
        channel = None
        logger.info("WhatsApp channel disabled (WHATSAPP_* not fully set)")
    if settings.googlechat_enabled:
        from ont_agent.googlechat import GoogleChatClient, GoogleChatVerifier
        from ont_agent.googlechat_bot import GoogleChatChannel

        googlechat_channel: GoogleChatChannel | None = GoogleChatChannel(
            store,
            settings,
            GoogleChatClient(settings),
            GoogleChatVerifier(settings.googlechat_audience, settings.googlechat_sender),
            memory=memory,
            facts=application.bot_data.get("facts"),
            tracing=application.bot_data["tracing"],
            latency=recorder,
            limiter=application.bot_data["burst"],
        )
        application.bot_data["googlechat"] = googlechat_channel
        logger.info(
            "Google Chat channel enabled (%d allowlisted email(s)), endpoint %s",
            len(settings.allowed_googlechat_emails),
            settings.googlechat_audience,
        )
    else:
        googlechat_channel = None
        logger.info("Google Chat channel disabled (GOOGLE_CHAT_* not set)")
    slack_channel = await _start_slack(application, store, memory, recorder)
    app = build_app(
        store,
        settings,
        on_bound,
        bot_username=bot_username,
        on_whatsapp=channel.handle_payload if channel is not None else None,
        whatsapp_number=(
            await channel.client.display_number()
            if channel is not None and not settings.whatsapp_business_number
            else ""
        ),
        on_googlechat=googlechat_channel.handle_payload if googlechat_channel else None,
        verify_googlechat=googlechat_channel.verifier.verify if googlechat_channel else None,
        on_slack=(
            slack_channel.handle_envelope
            if slack_channel is not None and settings.slack_mode == "http"
            else None
        ),
        slack_url=application.bot_data.get("slack_url"),
    )
    application.bot_data["http_server"] = await start_server(
        app, settings.callback_host, settings.callback_port
    )
    logger.info(
        "Abuse guards: burst %d/%ds, %d message(s)/user/day, %s allowlisted Telegram id(s)",
        settings.burst_limit,
        settings.burst_window_seconds,
        settings.daily_message_limit,
        len(settings.allowed_telegram_ids) or "no",
    )
    if not settings.proactive:
        logger.info("Proactive work disabled (AGENT_PROACTIVE)")
    # Started either way: the loop also carries the daily retention pass,
    # and an unrelated setting must not quietly switch maintenance off.
    # tick() already returns early on its own when proactive work is off.
    application.bot_data["ticker"] = asyncio.create_task(_ticker(application))
    logger.info("Bot ready — long polling Telegram")


async def _start_slack(
    application: Application,  # type: ignore[type-arg]
    store: BindingStore,
    memory: BaseCheckpointSaver[Any],
    recorder: LatencyRecorder,
) -> SlackChannel | None:
    """Bring the Slack channel up; on any failure it stays off and the rest keeps running."""
    settings: Settings = application.bot_data["settings"]
    if not settings.slack_enabled:
        logger.info("Slack channel OFF (SLACK_BOT_TOKEN not set)")
        return None
    from ont_agent.slack import SlackAuthError, SlackClient, SlackError, SlackSocket
    from ont_agent.slack_bot import SlackChannel

    client = SlackClient(settings)
    try:
        info = await client.auth_test()
    except SlackAuthError:
        logger.error(
            "Slack channel OFF: the bot token was refused (wrong, or the app was uninstalled)"
        )
        await client.aclose()
        return None
    except SlackError:
        logger.error("Slack channel OFF: could not reach Slack", exc_info=True)
        await client.aclose()
        return None

    channel = SlackChannel(
        store,
        settings,
        client,
        workspace_id=info.team_id,
        memory=memory,
        facts=application.bot_data.get("facts"),
        tracing=application.bot_data["tracing"],
        latency=recorder,
        limiter=application.bot_data["burst"],
    )
    if settings.slack_mode == "socket":
        socket = SlackSocket(settings.slack_app_token, client.web, channel.handle_envelope)
        try:
            await socket.start()
        except SlackError as exc:
            logger.error("Slack channel OFF: %s", exc)
            await socket.aclose()
            await client.aclose()
            return None
        application.bot_data["slack_socket"] = socket
        logger.info("Slack channel on (socket mode, workspace %s)", info.team_id)
        logger.info("Slack Socket Mode connected")
    else:
        logger.info(
            "Slack channel on (http mode, workspace %s), endpoint %s/webhook/slack",
            info.team_id,
            settings.agent_public_url,
        )
    application.bot_data["slack"] = channel
    application.bot_data["slack_url"] = f"https://app.slack.com/client/{info.team_id}"
    return channel


async def _post_shutdown(application: Application) -> None:  # type: ignore[type-arg]
    whatsapp: WhatsAppChannel | None = application.bot_data.get("whatsapp")
    if whatsapp is not None:
        await whatsapp.aclose()
    googlechat: GoogleChatChannel | None = application.bot_data.get("googlechat")
    if googlechat is not None:
        await googlechat.aclose()
    slack_socket: SlackSocket | None = application.bot_data.get("slack_socket")
    if slack_socket is not None:
        await slack_socket.aclose()
    slack: SlackChannel | None = application.bot_data.get("slack")
    if slack is not None:
        await slack.aclose()
    recorder: LatencyRecorder | None = application.bot_data.get("latency")
    if recorder is not None:
        await recorder.aclose()
    ticker: asyncio.Task[None] | None = application.bot_data.get("ticker")
    if ticker is not None:
        ticker.cancel()
        with suppress(asyncio.CancelledError):
            await ticker
    tracing: Tracing | None = application.bot_data.get("tracing")
    if tracing is not None:
        tracing.flush()
    http_server: ServerHandle | None = application.bot_data.get("http_server")
    if http_server is not None:
        await http_server.stop()
    pool = application.bot_data.get("db_pool")
    if pool is not None:
        await pool.close()
    store = application.bot_data.get("store")
    if store is not None:
        await store.__aexit__(None, None, None)


def main() -> None:
    setup_logging()
    # Windows defaults to ProactorEventLoop, which psycopg's async mode
    # refuses outright ("Psycopg cannot use the 'ProactorEventLoop'"), so the
    # connection pool never opens and the bot dies at startup. Selector is
    # what psycopg needs; its limits — 512 sockets, no subprocesses — are
    # irrelevant to a bot that polls one API and talks to two databases.
    #
    # Windows-only by construction: the policy does not exist elsewhere, and
    # the Linux container this ships in has no such problem.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    settings = load_settings(require_llm=True, require_telegram=True)
    add_file_logs(settings.log_file, settings.blocked_log_file)
    # First thing after settings, so a crash during startup is reported too.
    init_error_tracking(settings)

    application = (
        ApplicationBuilder()
        .token(settings.telegram_bot_token)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )
    application.bot_data["settings"] = settings

    # Group -1 runs before everything else; the guard stops disallowed or
    # flooding senders here so no later handler ever sees them.
    application.add_handler(TypeHandler(Update, guard), group=-1)
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("logout", cmd_logout))
    application.add_handler(CommandHandler(["new", "reset"], cmd_new))
    application.add_handler(CommandHandler("forget", cmd_forget))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("brief", cmd_brief))
    application.add_handler(CommandHandler("audit", cmd_audit))
    application.add_handler(CommandHandler("watch", cmd_watch))
    application.add_handler(CommandHandler("watchers", cmd_watchers))
    application.add_handler(CommandHandler("unwatch", cmd_unwatch))
    application.add_handler(CallbackQueryHandler(on_confirm, pattern="^confirm:"))
    application.add_handler(CallbackQueryHandler(on_feedback, pattern="^fb:"))
    application.add_handler(CallbackQueryHandler(on_followup, pattern="^fu:"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    # Voice before the catch-all, which otherwise answers "text only".
    application.add_handler(MessageHandler(filters.VOICE, on_voice))
    # Order matters: these must precede the ~TEXT catch-all below, which
    # would otherwise swallow every photo and document.
    application.add_handler(MessageHandler(filters.PHOTO, on_photo))
    application.add_handler(MessageHandler(filters.Document.ALL, on_document))
    application.add_handler(MessageHandler(~filters.TEXT, on_other))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
