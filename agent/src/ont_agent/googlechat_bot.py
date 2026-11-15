"""The Google Chat channel: Chat events in, agent answers out.

Mirrors ``whatsapp_bot`` and reuses the pipeline in ``bot.py``
(``answer_question`` / ``resume_pending``), so a Google Chat user is one more
chat bound to one Odoo user, keyed by ``channels.googlechat_chat_id``.

Differences forced by the platform: the app answers asynchronously through the
Chat API (the webhook only acknowledges), a message can be edited, so a slow
turn shows a "working" message that the answer then replaces, buttons are a
card under the message, and an app cannot attach files. Direct messages only:
a space would show one person's Odoo data to everybody in it.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
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
from ont_agent.channels import googlechat_chat_id
from ont_agent.config import Settings
from ont_agent.googlechat import (
    KIND_ADDED,
    KIND_TAP,
    KIND_TEXT,
    AttachmentTooLarge,
    ChatEvent,
    GoogleChatClient,
    GoogleChatError,
    GoogleChatVerifier,
    button_cards,
    parse_event,
)
from ont_agent.googlechat_format import html_to_googlechat, render_googlechat
from ont_agent.log import BLOCKED_LOGGER, get_logger
from ont_agent.memory import forget_all
from ont_agent.metrics import LatencyRecorder
from ont_agent.proactive import run_brief
from ont_agent.store import GLOBAL_SUBJECT, Binding, BindingStore, binding_id
from ont_agent.tracing import Tracing
from ont_agent.uploads import DEFAULT_QUESTION, SUPPORTED, Attachment, describe
from ont_agent.uploads import check as check_upload
from ont_agent.whatsapp_bot import confirm_buttons, parse_command
from ont_agent.whatsapp_format import clamp, italic

logger = get_logger(__name__)
blocked_logger = get_logger(BLOCKED_LOGGER)

#: Google retries after outages; anything older than this is a replay of a
#: conversation that has moved on, so it is dropped.
STALE_AFTER_SECONDS = 600

#: How many recent event ids are remembered to ignore Google's retries.
_SEEN_MAX = 2048

#: A question still running after this long gets a "working" message;
#: instant replies (greetings) never flash one.
_WORKING_AFTER_SECONDS = 2.0
_WORKING_TEXT = "⏳ Working on it…"

_HEX32 = re.compile(r"[0-9a-f]{32}")

GC_CONNECT_PROMPT = (
    "You are not connected to Odoo yet. Send /start and click the login button "
    "to connect your account."
)
WELCOME = (
    "Hi, I'm your Odoo assistant. Send /start to connect your Odoo account, "
    "or /help to see what I can do."
)
DIRECT_ONLY_REPLY = (
    "I only work in direct messages, because my answers can contain private Odoo "
    "data. Open a direct message with me to continue."
)
UNSUPPORTED_REPLY = (
    "I can read text, photos and PDF files. Send one of those, or ask me a question."
)
NO_FILES_REPLY = (
    "📎 I produced a file, but Google Chat apps can't attach files yet. "
    "Ask me on Telegram or WhatsApp to receive it."
)


@dataclass
class _Placeholder:
    """The "working" message of one turn, once it has been shown."""

    name: str | None = None


class GoogleChatChannel:
    def __init__(
        self,
        store: BindingStore,
        settings: Settings,
        client: GoogleChatClient,
        verifier: GoogleChatVerifier,
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
        self.verifier = verifier
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
        self._spaces: dict[int, str] = {}
        self._notified: dict[str, float] = {}

    # ---- webhook entry -------------------------------------------------

    def handle_payload(self, payload: Mapping[str, Any]) -> int:
        """Schedule the interaction in a (token-verified) event body.

        Returns at once: Google wants its acknowledgement within 30 seconds,
        and an answer can take longer.
        """
        event = parse_event(payload)
        if event is None or event.event_id in self._seen:
            return 0
        self._seen[event.event_id] = None
        while len(self._seen) > _SEEN_MAX:
            self._seen.popitem(last=False)
        logger.info(
            "Google Chat %s from %s", event.kind, binding_id(googlechat_chat_id(event.user_name))
        )
        task = asyncio.create_task(self._serialized(event))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return 1

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            with suppress(asyncio.CancelledError):
                await task
        await self.client.aclose()
        await self.verifier.aclose()

    async def send_plain(self, chat_id: int, text: str) -> None:
        """A message the user did not just ask for (login confirmation, brief, alert)."""
        space = self._spaces.get(chat_id) or await self.store.get_chat_space(chat_id)
        if space is None:
            logger.warning("No Google Chat space known for %s", binding_id(chat_id))
            return
        await self.client.send_message(space, clamp(render_googlechat(text)))

    # ---- one event -----------------------------------------------------

    async def _serialized(self, event: ChatEvent) -> None:
        # One chat, one conversation thread: handle its events in order.
        chat_id = googlechat_chat_id(event.user_name)
        lock = self._locks.setdefault(chat_id, asyncio.Lock())
        self._queued[chat_id] = self._queued.get(chat_id, 0) + 1
        try:
            async with lock:
                await self._guarded(event, chat_id)
        finally:
            self._queued[chat_id] -= 1
            if not self._queued[chat_id]:
                del self._queued[chat_id]
                self._locks.pop(chat_id, None)

    async def _guarded(self, event: ChatEvent, chat_id: int) -> None:
        try:
            if not await self._admit(event, chat_id):
                return
            await self._dispatch(event, chat_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("Failed handling Google Chat user %s", binding_id(chat_id), exc_info=True)
            with suppress(Exception):
                await self._say(event.space, ERROR_REPLY)

    async def _admit(self, event: ChatEvent, chat_id: int) -> bool:
        allowed = self.settings.allowed_googlechat_emails
        # Unknown senders get silence: any reply confirms the bot is live.
        if allowed and event.email not in allowed:
            blocked_logger.warning(
                "Blocked Google Chat message from %s (not allowlisted)", binding_id(chat_id)
            )
            return False
        if event.timestamp and time.time() - event.timestamp > STALE_AFTER_SECONDS:
            logger.info("Dropping stale Google Chat event for %s", binding_id(chat_id))
            return False
        limiter = self.limiter
        if limiter is not None and not limiter.allow(chat_id, event.timestamp or None):
            logger.warning("Burst limit hit by Google Chat user %s", binding_id(chat_id))
            if limiter.should_warn(chat_id):
                with suppress(Exception):
                    await self._say(event.space, BURST_REPLY)
            return False
        return True

    async def _dispatch(self, event: ChatEvent, chat_id: int) -> None:
        if not event.direct:
            await self._direct_only(event)
            return
        if self._spaces.get(chat_id) != event.space:
            await self.store.set_chat_space(chat_id, event.space)
            self._spaces[chat_id] = event.space
        if event.kind == KIND_ADDED:
            await self._say(event.space, WELCOME)
        elif event.kind == KIND_TAP:
            await self._on_tap(event, chat_id)
        elif event.kind == KIND_TEXT and (event.attachments or event.has_unreadable_file):
            await self._on_upload(event, chat_id)
        elif event.kind == KIND_TEXT and event.text:
            await self._on_text(event, chat_id)
        else:
            await self._say(event.space, UNSUPPORTED_REPLY)

    async def _direct_only(self, event: ChatEvent) -> None:
        # Once a window per space, so a busy room is not answered line by line.
        now = time.monotonic()
        ttl = self.settings.group_notice_ttl_seconds
        last = self._notified.get(event.space)
        if last is not None and now - last < ttl:
            return
        self._notified = {s: t for s, t in self._notified.items() if now - t < ttl}
        self._notified[event.space] = now
        await self._say(event.space, DIRECT_ONLY_REPLY)

    async def _say(self, space: str, text: str) -> None:
        await self.client.send_message(space, clamp(text))

    async def _post(
        self,
        space: str,
        text: str,
        buttons: list[tuple[str, str]] | None = None,
        *,
        replace: str | None = None,
    ) -> None:
        """Send ``text`` with optional buttons, in place of ``replace`` when given."""
        cards = button_cards(buttons or [], self.client.endpoint)
        if replace:
            try:
                await self.client.update_message(replace, text, cards)
                return
            except GoogleChatError:
                logger.warning("Could not edit the working message", exc_info=True)
        await self.client.send_message(space, text, cards)

    # ---- text and commands ---------------------------------------------

    async def _over_limit(self, chat_id: int) -> str | None:
        subject = subject_for(await self.store.get_binding(chat_id), chat_id)
        today = _today()
        return over_limit(
            await self.store.get_usage(subject, today),
            await self.store.get_usage(GLOBAL_SUBJECT, today),
            self.settings,
        )

    async def _on_text(self, event: ChatEvent, chat_id: int) -> None:
        command = parse_command(event.text)
        if command is not None:
            await self._command(event, chat_id, *command)
            return
        if await self.store.get_binding(chat_id) is None:
            link_reply = await confirm_link(self.store, self.settings, chat_id, event.text)
            await self._say(event.space, link_reply or GC_CONNECT_PROMPT)
            return

        # Resolving a pending edit is never rate-limited; a fresh question is.
        editing = chat_id in self.awaiting_edit
        if not editing:
            refusal = await self._over_limit(chat_id)
            if refusal is not None:
                await self._say(event.space, refusal)
                return
        self.awaiting_edit.discard(chat_id)

        if editing:
            await self._run_turn(
                event,
                chat_id,
                lambda: resume_pending(
                    self.store,
                    self.settings,
                    chat_id,
                    approved=False,
                    checkpointer=self.memory,
                    revision=event.text,
                    tracing=self.tracing,
                    facts=self.facts,
                    latency=self.latency,
                ),
            )
            return
        await self._ask(event, chat_id, event.text)

    async def _ask(
        self,
        event: ChatEvent,
        chat_id: int,
        question: str,
        attachment: Attachment | None = None,
    ) -> None:
        await self._run_turn(
            event,
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

    async def _command(self, event: ChatEvent, chat_id: int, name: str, args: list[str]) -> None:
        space = event.space

        if name == "unknown":
            await self._say(space, "I don't know that command. Send /help to see what I can do.")
        elif name == "help":
            await self._say(space, help_text())
        elif name == "start":
            await self._cmd_start(space, chat_id)
        elif name == "logout":
            removed = await logout(self.store, self.settings, chat_id)
            self.awaiting_edit.discard(chat_id)
            await clear_memory(self.store, self.memory, chat_id)
            await self._say(
                space,
                "Disconnected. Your access tokens were revoked and the conversation was cleared."
                if removed
                else "You were not connected.",
            )
        elif name == "new":
            self.awaiting_edit.discard(chat_id)
            await clear_memory(self.store, self.memory, chat_id)
            await self._say(
                space, "Fresh start — I've forgotten our previous conversation. Ask away."
            )
        else:
            binding = await self.store.get_binding(chat_id)
            if binding is None:
                await self._say(space, GC_CONNECT_PROMPT)
                return
            await self._bound_command(event, chat_id, binding, name, args)

    async def _cmd_start(self, space: str, chat_id: int) -> None:
        binding = await self.store.get_binding(chat_id)
        if binding is not None:
            await self._say(
                space,
                f"You are connected as {binding.odoo_name} ({binding.odoo_login}). "
                "Ask me anything about your Odoo data — I remember our conversation, "
                "so follow-up questions work. /new starts a fresh conversation, "
                "/logout disconnects.",
            )
            return
        url = await begin_login(self.store, self.settings, chat_id)
        if url.lower().startswith("https://"):
            await self.client.send_message(
                space,
                "Let's connect you to Odoo. Click the button, log in with YOUR Odoo "
                "account, and approve the access request.",
                button_cards([], self.client.endpoint, links=[(url, "Log in to Odoo")]),
            )
        else:
            await self._say(
                space,
                "Let's connect you to Odoo.\n\nOpen this link in your browser, log in with "
                f"YOUR Odoo account, and approve the access request:\n\n{url}",
            )

    async def _bound_command(
        self,
        event: ChatEvent,
        chat_id: int,
        binding: Binding,
        name: str,
        args: list[str],
    ) -> None:
        space = event.space

        if name == "forget":
            if self.facts is None:
                await self._say(space, "I'm not storing any preferences right now.")
                return
            cleared = await forget_all(self.facts, binding.odoo_login)
            # The conversation too, or the model reads its own reply and
            # "remembers" what was just deleted.
            self.awaiting_edit.discard(chat_id)
            await clear_memory(self.store, self.memory, chat_id)
            await self._say(
                space,
                f"Forgotten — I cleared {cleared} saved preference(s), "
                "and this conversation too so nothing lingers."
                if cleared
                else "There was nothing saved about you to forget.",
            )
        elif name == "brief":
            await self._cmd_brief(space, chat_id, args)
        elif name == "watch":
            await self._cmd_watch(space, chat_id, binding, args)
        elif name == "watchers":
            watchers = await self.store.list_watchers(chat_id)
            if not watchers:
                await self._say(space, "No alerts set. Add one with /watch <condition>.")
            else:
                lines = "\n".join(f"{i}. {w.condition}" for i, w in enumerate(watchers, 1))
                await self._say(
                    space, f"👀 Your alerts:\n\n{lines}\n\nRemove one with /unwatch <number>."
                )
        elif name == "unwatch":
            watchers = await self.store.list_watchers(chat_id)
            try:
                index = int(args[0]) if args else 0
            except ValueError:
                index = 0
            if not 1 <= index <= len(watchers):
                await self._say(space, "Which one? Use /watchers to see the numbers.")
                return
            removed = await self.store.deactivate_watcher(watchers[index - 1].id, chat_id)
            await self._say(
                space,
                f"Removed: {watchers[index - 1].condition}"
                if removed
                else "That alert is already gone.",
            )
        elif name == "audit":
            await self._cmd_audit(space, binding, args)

    async def _cmd_watch(
        self, space: str, chat_id: int, binding: Binding, args: list[str]
    ) -> None:
        condition = " ".join(args).strip()
        if not condition:
            await self._say(
                space,
                "Tell me what to watch for, like:\n"
                "/watch any customer invoice more than 60 days overdue",
            )
        elif await self.store.count_watchers(chat_id) >= self.settings.max_watchers_per_user:
            await self._say(
                space,
                f"You already have {self.settings.max_watchers_per_user} alerts, which is the "
                "limit. Use /watchers to see them and /unwatch to free one up.",
            )
        else:
            await self.store.add_watcher(chat_id, binding.odoo_login, condition)
            await self._say(
                space,
                f"👀 Watching: {condition}\n\nI check about every "
                f"{self.settings.watcher_interval_minutes} minutes and will tell you when it "
                "happens — once per change, not every time I look.",
            )

    async def _cmd_brief(self, space: str, chat_id: int, args: list[str]) -> None:
        argument = args[0].lower() if args else ""
        if argument in ("on", "off"):
            await self.store.set_brief_enabled(chat_id, argument == "on")
            await self._say(
                space,
                "Daily brief is back on — it will arrive each morning."
                if argument == "on"
                else "Daily brief off. Ask for one any time with /brief.",
            )
            return
        placeholder = await self.client.send_message(space, "📋 Putting your brief together …")
        try:
            binding = await ensure_access(self.store, self.settings, chat_id)
            if binding is None:
                await self._post(space, GC_CONNECT_PROMPT, replace=placeholder)
                return
            text = await run_brief(self.store, self.settings, binding, self.facts, self.latency)
        except Exception:
            logger.error("On-demand brief failed for %s", binding_id(chat_id), exc_info=True)
            await self._post(space, ERROR_REPLY, replace=placeholder)
            return
        await self._post(
            space, clamp(render_googlechat(text or "Nothing to report.")), replace=placeholder
        )

    async def _cmd_audit(self, space: str, binding: Binding, args: list[str]) -> None:
        if not binding.is_admin:
            await self._say(space, "That command is for administrators.")
            return
        days = 7
        if args:
            try:
                days = max(1, min(int(args[0]), self.settings.action_log_days))
            except ValueError:
                await self._say(space, "Usage: /audit [days], for example /audit 30")
                return
        since = _now() - dt.timedelta(days=days)
        actions = await self.store.all_actions(since, limit=_AUDIT_LIMIT)
        total = await self.store.count_actions(since)
        await self._say(space, render_googlechat(audit_report(actions, total, days)))

    # ---- button taps ---------------------------------------------------

    async def _on_tap(self, event: ChatEvent, chat_id: int) -> None:
        space = event.space

        confirm = parse_confirm_data(event.tap)
        if confirm is not None:
            await self._on_confirm(event, chat_id, *confirm)
            return

        feedback = parse_feedback_data(event.tap)
        if feedback is not None:
            positive, trace_id = feedback
            if self.tracing is not None:
                self.tracing.record_feedback(trace_id, positive)
                await self._say(space, "Thanks for the feedback!")
            return

        code = parse_followup_data(event.tap)
        if code is not None:
            if await self.store.get_binding(chat_id) is None:
                await self._say(space, GC_CONNECT_PROMPT)
                return
            refusal = await self._over_limit(chat_id)
            if refusal is not None:
                await self._say(space, refusal)
                return
            await self._ask(event, chat_id, FOLLOWUPS[code][1])

    async def _settle(self, event: ChatEvent, note: str) -> None:
        """Strip the buttons from the tapped message and stamp the verdict on it."""
        if event.message_name:
            try:
                text = f"{event.text}\n\n{note}" if event.text else note
                await self.client.update_message(event.message_name, clamp(text))
                return
            except GoogleChatError:
                logger.warning("Could not update the approval message", exc_info=True)
        await self._say(event.space, note)

    async def _on_confirm(
        self, event: ChatEvent, chat_id: int, verdict: str, trace_id: str | None
    ) -> None:
        def score(decision: str) -> None:
            # The tap scores the trace that PROPOSED the action.
            if self.tracing is not None and trace_id:
                self.tracing.record_decision(trace_id, decision)

        if verdict == "edit":
            score("edit-requested")
            self.awaiting_edit.add(chat_id)
            await self._settle(event, "✏️ Editing.")
            await self._say(event.space, EDIT_PROMPT)
            return

        approved = verdict == "approve"
        score("approved" if approved else "cancelled")
        await self._settle(event, "✅ Approved." if approved else "❌ Cancelled.")
        await self._run_turn(
            event,
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

    # ---- files ---------------------------------------------------------

    async def _on_upload(self, event: ChatEvent, chat_id: int) -> None:
        space = event.space
        settings = self.settings
        if await self.store.get_binding(chat_id) is None:
            await self._say(space, GC_CONNECT_PROMPT)
            return
        if not settings.uploads:
            await self._say(space, UPLOADS_OFF)
            return
        if not event.attachments:  # a Drive link: the app cannot download those
            await self._say(space, UNSUPPORTED_REPLY)
            return
        refusal = await self._over_limit(chat_id)
        if refusal is not None:
            await self._say(space, refusal)
            return

        file = event.attachments[0]
        # Decided on what Google declares, before the download is paid for.
        declined = check_upload(file.content_type, 0, settings)
        if declined is not None:
            await self._say(space, declined)
            return
        limit = settings.upload_max_mb * 1024 * 1024
        try:
            data = await self.client.download(file.resource_name, limit)
        except AttachmentTooLarge:
            await self._say(
                space, f"That file is larger than {settings.upload_max_mb} MB. Send a smaller one."
            )
            return
        except GoogleChatError:
            logger.error("Upload download failed for %s", binding_id(chat_id), exc_info=True)
            await self._say(space, UPLOAD_FAILED)
            return

        default_name = "photo.jpg" if file.content_type.startswith("image/") else "file"
        attachment = Attachment(
            kind=SUPPORTED[file.content_type],
            mime=file.content_type,
            filename=file.filename or default_name,
            data=data,
        )
        logger.info("Chat %s: reading upload %s", binding_id(chat_id), describe(attachment))
        question = event.text.strip() or DEFAULT_QUESTION
        if not event.text.strip():
            await self._say(space, f'📎 {describe(attachment)} — asking: "{question}"')
        await self._ask(event, chat_id, question, attachment)

    # ---- running a turn and delivering it ------------------------------

    @asynccontextmanager
    async def _working(self, space: str):  # type: ignore[no-untyped-def]
        """A "working" message that appears only if the turn is slow.

        Chat cannot show a typing indicator, but it can edit a message, so the
        answer later replaces this one in place.
        """
        holder = _Placeholder()

        async def show() -> None:
            await asyncio.sleep(_WORKING_AFTER_SECONDS)
            try:
                holder.name = await self.client.send_message(space, _WORKING_TEXT)
            except GoogleChatError:
                logger.debug("Working message failed", exc_info=True)

        task = asyncio.create_task(show())
        try:
            yield holder
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def _run_turn(
        self,
        event: ChatEvent,
        chat_id: int,
        make_reply: Callable[[], Awaitable[BotReply]],
    ) -> None:
        async with self._working(event.space) as placeholder:
            try:
                reply = await make_reply()
            except Exception:
                logger.error("Failed answering chat %s", binding_id(chat_id), exc_info=True)
                reply = BotReply(text=ERROR_REPLY)
        await self.deliver(event.space, chat_id, reply, placeholder.name)

    async def deliver(
        self, space: str, chat_id: int, reply: BotReply, placeholder: str | None = None
    ) -> None:
        """Send a finished reply, with its buttons in a card under the text."""
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
            await self._post(
                space,
                clamp(html_to_googlechat(reply.text), footer),
                buttons,
                replace=placeholder,
            )
            return

        if reply.grounded and reply.text:
            stamp = freshness_line(reply.evidence, binding, self.settings, _now())
            reply.text += f"\n\n*{stamp}*"
        links = reply.evidence.links or None
        body = render_googlechat(reply.text, links)
        if links and len(body) > 4000:
            body = render_googlechat(reply.text)  # links inflate it; drop them first
        await self._post(
            space,
            clamp(body, footer),
            self._option_buttons(reply, can_write=bool(binding and binding.can_write)),
            replace=placeholder,
        )
        await self._drop_files(space, reply)

    def _option_buttons(self, reply: BotReply, *, can_write: bool) -> list[tuple[str, str]]:
        """(data, title) buttons under the answer: follow-ups, then thumbs."""
        codes = (
            followup_codes(reply.text, can_write, reply.grounded, reply.from_documents)
            if not reply.blocked
            else []
        )
        buttons = [(f"fu:{code}", FOLLOWUPS[code][0]) for code in codes]
        if reply.trace_id and _HEX32.fullmatch(reply.trace_id):
            buttons += [
                (f"fb:1:{reply.trace_id}", "👍 Helpful"),
                (f"fb:0:{reply.trace_id}", "👎 Not helpful"),
            ]
        return buttons

    async def _drop_files(self, space: str, reply: BotReply) -> None:
        """Charts and PDFs cannot be attached by a Chat app (uploads need a user's
        credentials), so say so rather than silently omitting them."""
        if not reply.attachments:
            return
        for artifact in reply.attachments:
            artifact.path.unlink(missing_ok=True)
        with suppress(GoogleChatError):
            await self._say(space, NO_FILES_REPLY)
