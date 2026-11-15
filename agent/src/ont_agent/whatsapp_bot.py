"""The WhatsApp channel: Meta webhook events in, agent answers out.

Mirrors the Telegram handlers in ``bot.py`` and reuses its pipeline
(``answer_question`` / ``resume_pending``), so a WhatsApp chat is one more
chat bound to one Odoo user. A WhatsApp user is keyed by
``channels.whatsapp_chat_id``.

Differences forced by the platform: no message editing (progress is the
typing indicator), at most three reply buttons per message, and
business-initiated messages only inside the 24-hour customer-service window,
which is why briefs and watcher alerts are not sent here yet.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager, suppress
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from ont_agent.auth import begin_login, confirm_link, ensure_access, logout
from ont_agent.bot import (
    _AUDIT_LIMIT,
    EDIT_PROMPT,
    ERROR_REPLY,
    FOLLOWUPS,
    UPLOAD_FAILED,
    UPLOADS_OFF,
    VOICE_OFF,
    VOICE_UNCLEAR,
    BotReply,
    _now,
    _today,
    answer_question,
    audit_report,
    clear_memory,
    followup_codes,
    freshness_line,
    help_text,
    parse_confirm_data,
    parse_feedback_data,
    parse_followup_data,
    pending_is_editable,
    resume_pending,
    usage_line,
)
from ont_agent.budget import BURST_REPLY, BurstLimiter, over_limit, subject_for
from ont_agent.channels import wa_id_of, whatsapp_chat_id
from ont_agent.config import Settings
from ont_agent.log import BLOCKED_LOGGER, get_logger
from ont_agent.memory import forget_all
from ont_agent.metrics import LatencyRecorder
from ont_agent.proactive import run_brief
from ont_agent.store import GLOBAL_SUBJECT, Binding, BindingStore, binding_id
from ont_agent.tools import IRREVERSIBLE_TOOLS
from ont_agent.tracing import Tracing
from ont_agent.transcribe import transcribe
from ont_agent.uploads import DEFAULT_QUESTION, SUPPORTED, Attachment, describe
from ont_agent.uploads import check as check_upload
from ont_agent.whatsapp import (
    KIND_AUDIO,
    KIND_DOCUMENT,
    KIND_IMAGE,
    KIND_REPLY,
    KIND_TEXT,
    MAX_BUTTONS,
    InboundMessage,
    WhatsAppClient,
    parse_events,
)
from ont_agent.whatsapp_format import (
    clamp,
    html_to_whatsapp,
    italic,
    render_whatsapp,
    split_for_body,
)

logger = get_logger(__name__)
blocked_logger = get_logger(BLOCKED_LOGGER)

#: Meta redelivers on any non-200 and after outages; anything older than this
#: is a replay of a conversation that has moved on, so it is dropped.
STALE_AFTER_SECONDS = 600

#: How many recent message ids are remembered to ignore Meta's retries.
_SEEN_MAX = 2048

#: Meta ends the indicator after 25 s or once we send anything; refresh well inside that.
_TYPING_REFRESH_SECONDS = 10

#: A question still running after this long gets an hourglass reaction; instant
#: replies (greetings) never flash one.
_REACT_AFTER_SECONDS = 2.0
_WORKING_REACTION = "⏳"

WA_CONNECT_PROMPT = (
    "You are not connected to Odoo yet. Send */start* and tap the login button "
    "to connect your account."
)
UNSUPPORTED_REPLY = (
    "I can read text, voice notes, photos and PDF files. Send one of those, or ask me a question."
)
NO_WATCH_REPLY = (
    "Alerts need messages I send on my own, which WhatsApp only allows inside "
    "24 hours of your last message. Set alerts up on Telegram for now."
)

_COMMANDS = {
    "start", "logout", "new", "reset", "forget", "help", "brief",
    "watch", "watchers", "unwatch", "audit", "login", "menu",
}  # fmt: skip
#: Typed without a slash, only these count as commands; "watch the Acme
#: invoice" must stay a question.
_BARE_COMMANDS = {"help", "menu"}
_ALIASES = {"login": "start", "menu": "help", "reset": "new"}

_HEX32 = re.compile(r"[0-9a-f]{32}")


def parse_command(text: str) -> tuple[str, list[str]] | None:
    """(command, args) for a command message; ("unknown", []) for an
    unrecognised slash command; None for an ordinary message."""
    words = text.strip().split()
    if not words:
        return None
    head = words[0].lower()
    if head.startswith("/"):
        name = head[1:]
        if name not in _COMMANDS:
            return "unknown", []
        return _ALIASES.get(name, name), words[1:]
    if len(words) == 1 and head in _BARE_COMMANDS:
        return _ALIASES.get(head, head), []
    return None


def confirm_buttons(
    pending_names: list[str], *, editable: bool, trace_id: str | None
) -> list[tuple[str, str]]:
    """(id, title) approval buttons; ids use the same format Telegram's do."""
    suffix = f":{trace_id}" if trace_id and _HEX32.fullmatch(trace_id) else ""
    irreversible = bool(set(pending_names) & IRREVERSIBLE_TOOLS)
    buttons = [(f"confirm:approve{suffix}", "⚠️ Approve" if irreversible else "✅ Approve")]
    if editable:
        buttons.append((f"confirm:edit{suffix}", "✏️ Edit"))
    buttons.append((f"confirm:reject{suffix}", "❌ Cancel"))
    return buttons


class WhatsAppChannel:
    def __init__(
        self,
        store: BindingStore,
        settings: Settings,
        client: WhatsAppClient,
        *,
        memory: BaseCheckpointSaver[Any] | None = None,
        facts: BaseStore | None = None,
        tracing: Tracing | None = None,
        latency: LatencyRecorder | None = None,
        limiter: BurstLimiter | None = None,
    ) -> None:
        self.store = store
        self.settings = settings
        self.client = client
        self.memory = memory
        self.facts = facts
        self.tracing = tracing
        self.latency = latency
        self.limiter = limiter
        self.awaiting_edit: set[int] = set()
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._tasks: set[asyncio.Task[None]] = set()
        self._locks: dict[int, asyncio.Lock] = {}
        self._queued: dict[int, int] = {}

    # ---- webhook entry -------------------------------------------------

    def handle_payload(self, payload: Mapping[str, Any]) -> int:
        """Schedule every new message in a (signature-verified) webhook body.

        Returns at once: Meta wants its 200 fast and retries otherwise, and an
        answer takes seconds.
        """
        scheduled = 0
        for message in parse_events(payload, self.settings.whatsapp_phone_number_id):
            if message.message_id in self._seen:
                continue
            self._seen[message.message_id] = None
            while len(self._seen) > _SEEN_MAX:
                self._seen.popitem(last=False)
            logger.info(
                "WhatsApp %s from %s", message.kind, binding_id(whatsapp_chat_id(message.wa_id))
            )
            task = asyncio.create_task(self._serialized(message))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            scheduled += 1
        return scheduled

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            with suppress(asyncio.CancelledError):
                await task
        await self.client.aclose()

    async def send_plain(self, chat_id: int, text: str) -> None:
        """A message the user did not just ask for (login confirmation)."""
        await self.client.send_text(wa_id_of(chat_id), clamp(render_whatsapp(text)))

    # ---- one message ---------------------------------------------------

    async def _serialized(self, message: InboundMessage) -> None:
        # One chat, one conversation thread: handle its messages in order.
        chat_id = whatsapp_chat_id(message.wa_id)
        lock = self._locks.setdefault(chat_id, asyncio.Lock())
        self._queued[chat_id] = self._queued.get(chat_id, 0) + 1
        try:
            async with lock:
                await self._guarded(message, chat_id)
        finally:
            self._queued[chat_id] -= 1
            if not self._queued[chat_id]:
                del self._queued[chat_id]
                self._locks.pop(chat_id, None)

    async def _guarded(self, message: InboundMessage, chat_id: int) -> None:
        try:
            if not await self._admit(message, chat_id):
                return
            await self._dispatch(message, chat_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("Failed handling WhatsApp chat %s", binding_id(chat_id), exc_info=True)
            with suppress(Exception):
                await self.client.send_text(message.wa_id, ERROR_REPLY)

    async def _admit(self, message: InboundMessage, chat_id: int) -> bool:
        allowed = self.settings.allowed_whatsapp_numbers
        # Unknown senders get silence: any reply confirms the bot is live.
        if allowed and message.wa_id not in allowed:
            blocked_logger.warning(
                "Blocked WhatsApp message from %s (not allowlisted)", binding_id(chat_id)
            )
            return False
        if message.timestamp and time.time() - message.timestamp > STALE_AFTER_SECONDS:
            logger.info("Dropping stale WhatsApp message for %s", binding_id(chat_id))
            return False
        limiter = self.limiter
        if limiter is not None and not limiter.allow(chat_id, message.timestamp or None):
            logger.warning("Burst limit hit by WhatsApp chat %s", binding_id(chat_id))
            if limiter.should_warn(chat_id):
                with suppress(Exception):
                    await self.client.send_text(message.wa_id, BURST_REPLY)
            return False
        return True

    async def _dispatch(self, message: InboundMessage, chat_id: int) -> None:
        if message.kind == KIND_TEXT:
            await self._on_text(message, chat_id)
        elif message.kind == KIND_REPLY:
            await self._on_reply(message, chat_id)
        elif message.kind == KIND_AUDIO:
            await self._on_voice(message, chat_id)
        elif message.kind in (KIND_IMAGE, KIND_DOCUMENT):
            await self._on_upload(message, chat_id)
        else:
            await self.client.send_text(message.wa_id, UNSUPPORTED_REPLY)

    # ---- text and commands ---------------------------------------------

    async def _over_limit(self, chat_id: int) -> str | None:
        subject = subject_for(await self.store.get_binding(chat_id), chat_id)
        today = _today()
        return over_limit(
            await self.store.get_usage(subject, today),
            await self.store.get_usage(GLOBAL_SUBJECT, today),
            self.settings,
        )

    async def _on_text(self, message: InboundMessage, chat_id: int) -> None:
        command = parse_command(message.text)
        if command is not None:
            await self._command(message, chat_id, *command)
            return
        if await self.store.get_binding(chat_id) is None:
            link_reply = await confirm_link(self.store, self.settings, chat_id, message.text)
            await self.client.send_text(message.wa_id, link_reply or WA_CONNECT_PROMPT)
            return

        # Resolving a pending edit is never rate-limited; a fresh question is.
        editing = chat_id in self.awaiting_edit
        if not editing:
            refusal = await self._over_limit(chat_id)
            if refusal is not None:
                await self.client.send_text(message.wa_id, refusal)
                return
        self.awaiting_edit.discard(chat_id)

        if editing:
            await self._run_turn(
                message,
                chat_id,
                lambda: resume_pending(
                    self.store,
                    self.settings,
                    chat_id,
                    approved=False,
                    checkpointer=self.memory,
                    revision=message.text,
                    tracing=self.tracing,
                    facts=self.facts,
                    latency=self.latency,
                ),
            )
            return
        await self._ask(message, chat_id, message.text)

    async def _ask(
        self,
        message: InboundMessage,
        chat_id: int,
        question: str,
        attachment: Attachment | None = None,
    ) -> None:
        await self._run_turn(
            message,
            chat_id,
            lambda: answer_question(
                self.store,
                self.settings,
                chat_id,
                question,
                checkpointer=self.memory,
                tracing=self.tracing,
                facts=self.facts,
                latency=self.latency,
                attachment=attachment,
            ),
        )

    async def _command(
        self, message: InboundMessage, chat_id: int, name: str, args: list[str]
    ) -> None:
        to = message.wa_id
        send = self.client.send_text

        if name == "unknown":
            await send(to, "I don't know that command. Send /help to see what I can do.")
        elif name == "help":
            await send(to, help_text())
        elif name == "start":
            await self._cmd_start(to, chat_id)
        elif name == "logout":
            removed = await logout(self.store, self.settings, chat_id)
            self.awaiting_edit.discard(chat_id)
            await clear_memory(self.store, self.memory, chat_id)
            await send(
                to,
                "Disconnected. Your access tokens were revoked and the conversation was cleared."
                if removed
                else "You were not connected.",
            )
        elif name == "new":
            self.awaiting_edit.discard(chat_id)
            await clear_memory(self.store, self.memory, chat_id)
            await send(to, "Fresh start — I've forgotten our previous conversation. Ask away.")
        else:
            binding = await self.store.get_binding(chat_id)
            if binding is None:
                await send(to, WA_CONNECT_PROMPT)
                return
            await self._bound_command(message, chat_id, binding, name, args)

    async def _cmd_start(self, to: str, chat_id: int) -> None:
        binding = await self.store.get_binding(chat_id)
        if binding is not None:
            await self.client.send_text(
                to,
                f"You are connected as {binding.odoo_name} ({binding.odoo_login}). "
                "Ask me anything about your Odoo data — I remember our conversation, "
                "so follow-up questions work. /new starts a fresh conversation, "
                "/logout disconnects.",
            )
            return
        url = await begin_login(self.store, self.settings, chat_id)
        if url.lower().startswith("https://"):
            await self.client.send_cta_url(
                to,
                "Let's connect you to Odoo. Tap the button, log in with YOUR Odoo "
                "account, and approve the access request.",
                "Log in to Odoo",
                url,
            )
        else:
            await self.client.send_text(
                to,
                "Let's connect you to Odoo.\n\nOpen this link in your browser, log in with "
                f"YOUR Odoo account, and approve the access request:\n\n{url}",
            )

    async def _bound_command(
        self,
        message: InboundMessage,
        chat_id: int,
        binding: Binding,
        name: str,
        args: list[str],
    ) -> None:
        to = message.wa_id
        send = self.client.send_text

        if name == "forget":
            if self.facts is None:
                await send(to, "I'm not storing any preferences right now.")
                return
            cleared = await forget_all(self.facts, binding.odoo_login)
            # The conversation too, or the model reads its own reply and
            # "remembers" what was just deleted.
            self.awaiting_edit.discard(chat_id)
            await clear_memory(self.store, self.memory, chat_id)
            await send(
                to,
                f"Forgotten — I cleared {cleared} saved preference(s), "
                "and this conversation too so nothing lingers."
                if cleared
                else "There was nothing saved about you to forget.",
            )
        elif name == "brief":
            await self._cmd_brief(message, chat_id, args)
        elif name == "watch":
            await send(to, NO_WATCH_REPLY)
        elif name == "watchers":
            watchers = await self.store.list_watchers(chat_id)
            if not watchers:
                await send(to, "No alerts set. Add one with /watch <condition>.")
            else:
                lines = "\n".join(f"{i}. {w.condition}" for i, w in enumerate(watchers, 1))
                await send(to, f"👀 Your alerts:\n\n{lines}\n\nRemove one with /unwatch <number>.")
        elif name == "unwatch":
            watchers = await self.store.list_watchers(chat_id)
            try:
                index = int(args[0]) if args else 0
            except ValueError:
                index = 0
            if not 1 <= index <= len(watchers):
                await send(to, "Which one? Use /watchers to see the numbers.")
                return
            removed = await self.store.deactivate_watcher(watchers[index - 1].id, chat_id)
            await send(
                to,
                f"Removed: {watchers[index - 1].condition}"
                if removed
                else "That alert is already gone.",
            )
        elif name == "audit":
            await self._cmd_audit(message, binding, args)

    async def _cmd_brief(self, message: InboundMessage, chat_id: int, args: list[str]) -> None:
        to = message.wa_id
        argument = args[0].lower() if args else ""
        if argument in ("on", "off"):
            await self.store.set_brief_enabled(chat_id, argument == "on")
            await self.client.send_text(
                to,
                "Daily brief on. WhatsApp can't send it unprompted yet — ask any time with /brief."
                if argument == "on"
                else "Daily brief off. Ask for one any time with /brief.",
            )
            return
        await self.client.send_text(to, "📋 Putting your brief together …")
        async with self._typing(message.message_id):
            binding = await ensure_access(self.store, self.settings, chat_id)
            if binding is None:
                await self.client.send_text(to, WA_CONNECT_PROMPT)
                return
            text = await run_brief(self.store, self.settings, binding, self.facts, self.latency)
        await self.client.send_text(to, clamp(render_whatsapp(text or "Nothing to report.")))

    async def _cmd_audit(self, message: InboundMessage, binding: Binding, args: list[str]) -> None:
        to = message.wa_id
        if not binding.is_admin:
            await self.client.send_text(to, "That command is for administrators.")
            return
        days = 7
        if args:
            try:
                days = max(1, min(int(args[0]), self.settings.action_log_days))
            except ValueError:
                await self.client.send_text(to, "Usage: /audit [days], for example /audit 30")
                return
        since = _now() - dt.timedelta(days=days)
        actions = await self.store.all_actions(since, limit=_AUDIT_LIMIT)
        total = await self.store.count_actions(since)
        await self.client.send_text(to, clamp(render_whatsapp(audit_report(actions, total, days))))

    # ---- button and list taps ------------------------------------------

    async def _on_reply(self, message: InboundMessage, chat_id: int) -> None:
        to = message.wa_id
        tap = message.reply_id

        confirm = parse_confirm_data(tap)
        if confirm is not None:
            await self._on_confirm(message, chat_id, *confirm)
            return

        feedback = parse_feedback_data(tap)
        if feedback is not None:
            positive, trace_id = feedback
            if self.tracing is not None:
                self.tracing.record_feedback(trace_id, positive)
                await self.client.send_text(to, "Thanks for the feedback!")
            return

        code = parse_followup_data(tap)
        if code is not None:
            if await self.store.get_binding(chat_id) is None:
                await self.client.send_text(to, WA_CONNECT_PROMPT)
                return
            refusal = await self._over_limit(chat_id)
            if refusal is not None:
                await self.client.send_text(to, refusal)
                return
            await self._ask(message, chat_id, FOLLOWUPS[code][1])

    async def _on_confirm(
        self, message: InboundMessage, chat_id: int, verdict: str, trace_id: str | None
    ) -> None:
        to = message.wa_id

        def score(decision: str) -> None:
            # The tap scores the trace that PROPOSED the action.
            if self.tracing is not None and trace_id:
                self.tracing.record_decision(trace_id, decision)

        if verdict == "edit":
            score("edit-requested")
            self.awaiting_edit.add(chat_id)
            await self.client.send_text(to, EDIT_PROMPT)
            return

        approved = verdict == "approve"
        score("approved" if approved else "cancelled")
        await self.client.send_text(to, "✅ Approved." if approved else "❌ Cancelled.")
        await self._run_turn(
            message,
            chat_id,
            lambda: resume_pending(
                self.store,
                self.settings,
                chat_id,
                approved,
                checkpointer=self.memory,
                tracing=self.tracing,
                facts=self.facts,
                latency=self.latency,
            ),
        )

    # ---- voice and files -----------------------------------------------

    async def _on_voice(self, message: InboundMessage, chat_id: int) -> None:
        to = message.wa_id
        settings = self.settings
        if await self.store.get_binding(chat_id) is None:
            await self.client.send_text(to, WA_CONNECT_PROMPT)
            return
        if not settings.voice:
            await self.client.send_text(to, VOICE_OFF)
            return
        refusal = await self._over_limit(chat_id)
        if refusal is not None:
            await self.client.send_text(to, refusal)
            return

        try:
            info = await self.client.media_info(message.media_id)
            if info.size > settings.upload_max_mb * 1024 * 1024:
                await self.client.send_text(
                    to, f"That note is too large — keep it under {settings.voice_max_seconds}s."
                )
                return
            audio = await self.client.download(info.url)
            result = await transcribe(settings, audio)
        except Exception:
            logger.error("Voice download or transcription failed", exc_info=True)
            await self.client.send_text(to, VOICE_UNCLEAR)
            return

        # Billed even when empty, so recorded before looking at the text.
        if result.cost_usd:
            subject = subject_for(await self.store.get_binding(chat_id), chat_id)
            await self.store.record_usage(subject, _today(), result.cost_usd)
        if not result.text:
            await self.client.send_text(to, VOICE_UNCLEAR)
            return
        logger.info("Chat %s: voice note transcribed", binding_id(chat_id))
        await self.client.send_text(to, f"🎤 I heard: {result.text}")
        await self._ask(message, chat_id, result.text)

    async def _on_upload(self, message: InboundMessage, chat_id: int) -> None:
        to = message.wa_id
        settings = self.settings
        if await self.store.get_binding(chat_id) is None:
            await self.client.send_text(to, WA_CONNECT_PROMPT)
            return
        if not settings.uploads:
            await self.client.send_text(to, UPLOADS_OFF)
            return
        refusal = await self._over_limit(chat_id)
        if refusal is not None:
            await self.client.send_text(to, refusal)
            return

        try:
            info = await self.client.media_info(message.media_id)
            mime = info.mime or message.mime
            # Decided on what Meta declares, before the download is paid for.
            declined = check_upload(mime, info.size, settings)
            if declined is not None:
                await self.client.send_text(to, declined)
                return
            data = await self.client.download(info.url)
        except Exception:
            logger.error("Upload download failed for %s", binding_id(chat_id), exc_info=True)
            await self.client.send_text(to, UPLOAD_FAILED)
            return

        default_name = "photo.jpg" if message.kind == KIND_IMAGE else "file"
        attachment = Attachment(
            kind=SUPPORTED[mime], mime=mime, filename=message.filename or default_name, data=data
        )
        logger.info("Chat %s: reading upload %s", binding_id(chat_id), describe(attachment))
        question = message.text.strip() or DEFAULT_QUESTION
        if not message.text.strip():
            await self.client.send_text(to, f'📎 {describe(attachment)} — asking: "{question}"')
        await self._ask(message, chat_id, question, attachment)

    # ---- running a turn and delivering it ------------------------------

    @asynccontextmanager
    async def _typing(self, message_id: str):  # type: ignore[no-untyped-def]
        async def keep() -> None:
            while True:
                await self.client.show_typing(message_id)
                await asyncio.sleep(_TYPING_REFRESH_SECONDS)

        task = asyncio.create_task(keep())
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    @asynccontextmanager
    async def _reacting(self, to: str, message_id: str):  # type: ignore[no-untyped-def]
        """Hourglass on the user's message while a slow answer is being worked on.

        WhatsApp cannot edit a message, but a reaction can be set and cleared in
        place, which is the nearest thing to Telegram's changing status line.
        """
        reacted = False

        async def react() -> None:
            nonlocal reacted
            await asyncio.sleep(_REACT_AFTER_SECONDS)
            reacted = True  # before the call: a cancel mid-send must still clear it
            await self.client.send_reaction(to, message_id, _WORKING_REACTION)
            # A message from us ends the typing indicator, and the reaction may count.
            await self.client.show_typing(message_id)

        task = asyncio.create_task(react())
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            if reacted:
                await self.client.send_reaction(to, message_id, "")

    async def _run_turn(
        self,
        message: InboundMessage,
        chat_id: int,
        make_reply: Callable[[], Awaitable[BotReply]],
    ) -> None:
        async with (
            self._typing(message.message_id),
            self._reacting(message.wa_id, message.message_id),
        ):
            try:
                reply = await make_reply()
            except Exception:
                logger.error("Failed answering chat %s", binding_id(chat_id), exc_info=True)
                reply = BotReply(text=ERROR_REPLY)
        await self.deliver(message.wa_id, chat_id, reply)

    async def deliver(self, to: str, chat_id: int, reply: BotReply) -> None:
        """Send a finished reply, with its buttons or options in the same bubble."""
        binding = await self.store.get_binding(chat_id)
        footer = usage_line(reply.usage)
        footer = italic(footer) if footer else None

        if reply.pending is not None:
            # Prebuilt Telegram HTML.
            buttons = confirm_buttons(
                list(reply.pending.names),
                editable=pending_is_editable(reply.pending),
                trace_id=reply.trace_id,
            )
            await self._send_attached(
                to,
                clamp(html_to_whatsapp(reply.text), footer),
                lambda body: self.client.send_buttons(to, body, buttons),
                fallback_body="Your decision?",
                required=True,
            )
            return

        if reply.grounded and reply.text:
            stamp = freshness_line(reply.evidence, binding, self.settings, _now())
            reply.text += f"\n\n*{stamp}*"
        links = reply.evidence.links or None
        body = render_whatsapp(reply.text, links)
        if links and len(body) > 4000:
            body = render_whatsapp(reply.text)  # links inflate it; drop them first
        text = clamp(body, footer)

        buttons, fallback = self._option_buttons(
            reply, can_write=bool(binding and binding.can_write)
        )
        if buttons:
            await self._send_attached(
                to,
                text,
                lambda part: self.client.send_buttons(to, part, buttons),
                fallback_body=fallback,
            )
        else:
            await self.client.send_text(to, text)
        await self._send_files(to, reply)

    async def _send_attached(
        self,
        to: str,
        text: str,
        send_interactive: Callable[[str], Awaitable[None]],
        *,
        fallback_body: str,
        required: bool = False,
    ) -> None:
        """Send ``text`` so an interactive message carries its end as the body.

        WhatsApp cannot attach buttons to an existing message; the closest thing
        is an interactive message whose body is the answer itself. That body holds
        1024 characters, so a longer answer goes out in two parts: the start as
        plain text, the rest under the buttons.
        """
        split = split_for_body(text)
        if split is None:  # no clean cut: separate messages beat a broken one
            await self.client.send_text(to, text)
            body, tail = fallback_body, ""
        else:
            head, body = split
            tail = body
            if head:
                await self.client.send_text(to, head)
        try:
            await send_interactive(body)
        except Exception:
            logger.warning("Could not send WhatsApp buttons", exc_info=True)
            if tail:
                await self.client.send_text(to, tail)  # never lose the answer's end
            if required:
                raise

    def _option_buttons(
        self, reply: BotReply, *, can_write: bool
    ) -> tuple[list[tuple[str, str]], str]:
        """(id, title) reply buttons shown directly under the answer, plus a
        fallback body for when the answer cannot share the message.

        WhatsApp allows three reply buttons. Follow-ups come first because they
        do something; the thumbs take whatever room is left, so both appear only
        when there is at most one follow-up.
        """
        codes = (
            followup_codes(reply.text, can_write, reply.grounded, reply.from_documents)
            if not reply.blocked
            else []
        )
        buttons = [(f"fu:{code}", FOLLOWUPS[code][0]) for code in codes][:MAX_BUTTONS]
        if reply.trace_id and _HEX32.fullmatch(reply.trace_id):
            thumbs = [
                (f"fb:1:{reply.trace_id}", "👍 Helpful"),
                (f"fb:0:{reply.trace_id}", "👎 Not helpful"),
            ]
            room = MAX_BUTTONS - len(buttons)
            if room >= len(thumbs):
                buttons += thumbs
            elif room == 1:
                buttons.append(thumbs[1])  # a lone thumb is only useful as a complaint
        return buttons, "What next?" if codes else "How was this answer?"

    async def _send_files(self, to: str, reply: BotReply) -> None:
        for artifact in reply.attachments:
            try:
                await self.client.send_file(
                    to,
                    artifact.path,
                    as_image=artifact.kind == "photo",
                    caption=artifact.caption or "",
                )
            except Exception:
                logger.error("Failed sending %s to WhatsApp", artifact.kind, exc_info=True)
            finally:
                artifact.path.unlink(missing_ok=True)
