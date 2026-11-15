"""The Slack channel: Slack events in, agent answers out.

Mirrors ``googlechat_bot`` and reuses the pipeline in ``bot.py``
(``answer_question`` / ``resume_pending``), so a Slack member is one more chat
bound to one Odoo user, keyed by ``channels.slack_chat_id``.

Differences forced by the platform: Slack has no ``/start``-style commands in a
DM (a leading slash is intercepted), so commands are bare words ("login",
"new") or arguments to the app's slash command; the app answers
asynchronously, a message can be edited, so a slow turn shows a "working"
message that the answer then replaces, and buttons are a block under the text.
Direct messages only: a channel would show one person's Odoo data to everybody
in it. Until a person has logged in, the app stays silent (configurable).
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
    parse_confirm_data,
    parse_feedback_data,
    parse_followup_data,
    pending_is_editable,
    resume_pending,
    usage_line,
)
from ont_agent.budget import BURST_REPLY, BurstLimiter, over_limit, subject_for
from ont_agent.channels import slack_chat_id
from ont_agent.config import Settings
from ont_agent.log import BLOCKED_LOGGER, get_logger
from ont_agent.memory import forget_all
from ont_agent.metrics import LatencyRecorder
from ont_agent.proactive import run_brief
from ont_agent.slack import (
    KIND_COMMAND,
    KIND_MENTION,
    KIND_TAP,
    KIND_TEXT,
    TEXT_LIMIT,
    FileTooLarge,
    SlackClient,
    SlackError,
    SlackEvent,
    SlackFile,
    message_blocks,
    parse_event,
)
from ont_agent.slack_format import html_to_slack, render_slack
from ont_agent.store import GLOBAL_SUBJECT, Binding, BindingStore, binding_id
from ont_agent.tracing import Tracing
from ont_agent.transcribe import transcribe
from ont_agent.uploads import DEFAULT_QUESTION, SUPPORTED, Attachment, describe
from ont_agent.uploads import check as check_upload
from ont_agent.whatsapp_bot import confirm_buttons
from ont_agent.whatsapp_format import clamp, italic

logger = get_logger(__name__)
blocked_logger = get_logger(BLOCKED_LOGGER)

#: Slack redelivers after outages; anything older than this is a replay of a
#: conversation that has moved on, so it is dropped.
STALE_AFTER_SECONDS = 600

#: How many recent event ids are remembered to ignore Slack's retries.
_SEEN_MAX = 2048

#: A question still running after this long gets a "working" message;
#: instant replies (greetings) never flash one.
_WORKING_AFTER_SECONDS = 2.0
_WORKING_TEXT = "⏳ Working on it…"

_HEX32 = re.compile(r"[0-9a-f]{32}")

SLACK_CONNECT_PROMPT = "You are not connected to Odoo yet. Send `login` to connect your account."
DIRECT_ONLY_REPLY = (
    "I only work in direct messages, because my answers can contain private Odoo "
    "data. Open the Messages tab of my app to continue."
)
UNSUPPORTED_REPLY = (
    "I can read text, voice clips, photos and PDF files. Send one of those, or ask me a question."
)

_COMMANDS = {
    "start", "login", "logout", "new", "reset", "forget", "help", "menu", "brief",
    "watch", "watchers", "unwatch", "audit",
}  # fmt: skip
#: Typed alone in the DM, only these count as commands; "watch the Acme
#: invoice" must stay a question, so commands with arguments need the slash command.
_BARE_COMMANDS = {
    "start", "login", "logout", "new", "reset", "forget", "help", "menu", "brief", "watchers",
}  # fmt: skip
_ALIASES = {"login": "start", "menu": "help", "reset": "new"}


def parse_command(text: str, *, slash: bool = False) -> tuple[str, list[str]] | None:
    """(command, args) for a command message, or None for an ordinary question.

    ``slash`` is text typed after the app's slash command, where an empty text
    means help and the first word may carry arguments.
    """
    words = text.strip().split()
    if not words:
        return ("help", []) if slash else None
    head = words[0].lower()
    if slash:
        if head not in _COMMANDS:
            return None
        return _ALIASES.get(head, head), words[1:]
    if len(words) == 1 and head in _BARE_COMMANDS:
        return _ALIASES.get(head, head), []
    if len(words) == 2 and head == "brief" and words[1].lower() in ("on", "off"):
        return "brief", [words[1].lower()]
    return None


def help_text(command: str) -> str:
    return (
        "I answer questions about your company's Odoo data — just ask in plain language. "
        "Follow-up questions work; any change to data always asks for your approval first.\n\n"
        "Type these here, or after "
        f"`{command}`:\n"
        "*login* — connect your Odoo account / show status\n"
        "*new* — start a fresh conversation\n"
        "*forget* — erase the preferences I remember about you\n"
        "*brief* — send my business brief now (*brief off* stops the daily one)\n"
        "*watchers* — list the alerts I have set up\n"
        "*logout* — disconnect and revoke access\n"
        "*help* — this list\n\n"
        f"These take an argument, so use `{command}`:\n"
        f"`{command} watch <condition>` — alert me when something happens\n"
        f"`{command} unwatch <number>` — remove one of my alerts\n"
        f"`{command} audit [days]` — administrators: what the assistant did"
    )


@dataclass
class _Placeholder:
    """The "working" message of one turn, once it has been shown."""

    ts: str | None = None


class SlackChannel:
    def __init__(
        self,
        store: BindingStore,
        settings: Settings,
        client: SlackClient,
        *,
        workspace_id: str = "",
        memory: BaseCheckpointSaver[Any] | None = None,
        facts: BaseStore | None = None,
        tracing: Tracing | None = None,
        latency: LatencyRecorder | None = None,
        limiter: BurstLimiter | None = None,
    ) -> None:
        self.store = store
        self.settings = settings
        self.client = client
        #: The workspace the bot is installed in; its members are always allowed.
        self.workspace_id = workspace_id
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

    # ---- entry ---------------------------------------------------------

    def handle_envelope(self, envelope: str, payload: Mapping[str, Any]) -> int:
        """Schedule the interaction in a (verified) envelope; returns at once.

        Slack wants its acknowledgement within three seconds, and an answer
        takes longer, so the work is a task.
        """
        event = parse_event(envelope, payload)
        if event is None or event.event_id in self._seen:
            return 0
        self._seen[event.event_id] = None
        while len(self._seen) > _SEEN_MAX:
            self._seen.popitem(last=False)
        logger.info("Slack %s from %s", event.kind, binding_id(slack_chat_id(event.user_id)))
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

    async def send_plain(self, chat_id: int, text: str) -> None:
        """A message the user did not just ask for (login confirmation, brief, alert)."""
        channel = self._spaces.get(chat_id) or await self.store.get_chat_space(chat_id)
        if channel is None:
            logger.warning("No Slack DM known for %s", binding_id(chat_id))
            return
        await self._say(channel, render_slack(text))

    # ---- one event -----------------------------------------------------

    async def _serialized(self, event: SlackEvent) -> None:
        # One chat, one conversation thread: handle its events in order.
        chat_id = slack_chat_id(event.user_id)
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

    async def _guarded(self, event: SlackEvent, chat_id: int) -> None:
        try:
            if not await self._admit(event, chat_id):
                return
            await self._dispatch(event, chat_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("Failed handling Slack user %s", binding_id(chat_id), exc_info=True)
            with suppress(Exception):
                await self._say(event.channel, ERROR_REPLY)

    async def _admit(self, event: SlackEvent, chat_id: int) -> bool:
        settings = self.settings
        teams = {self.workspace_id, *settings.allowed_slack_teams} - {""}
        # Unknown senders get silence: any reply confirms the bot is live.
        if (
            settings.allowed_slack_users and event.user_id not in settings.allowed_slack_users
        ) or (teams and event.team_id and event.team_id not in teams):
            blocked_logger.warning(
                "Blocked Slack message from %s (not allowlisted)", binding_id(chat_id)
            )
            return False
        if event.timestamp and time.time() - event.timestamp > STALE_AFTER_SECONDS:
            logger.info("Dropping stale Slack event for %s", binding_id(chat_id))
            return False
        limiter = self.limiter
        if limiter is not None and not limiter.allow(chat_id, event.timestamp or None):
            logger.warning("Burst limit hit by Slack user %s", binding_id(chat_id))
            if limiter.should_warn(chat_id):
                with suppress(Exception):
                    await self._say(event.channel, BURST_REPLY)
            return False
        return True

    async def _dispatch(self, event: SlackEvent, chat_id: int) -> None:
        if not event.direct:
            await self._direct_only(event)
            return
        if self._spaces.get(chat_id) != event.channel:
            await self.store.set_chat_space(chat_id, event.channel)
            self._spaces[chat_id] = event.channel
        if event.kind == KIND_TAP:
            await self._on_tap(event, chat_id)
        elif event.kind in (KIND_TEXT, KIND_COMMAND) and (
            event.files or event.has_unreadable_file
        ):
            await self._on_file(event, chat_id)
        elif event.kind in (KIND_TEXT, KIND_COMMAND):
            await self._on_text(event, chat_id)

    async def _direct_only(self, event: SlackEvent) -> None:
        # Visible to the sender alone, so a busy channel is not answered line by line.
        with suppress(SlackError):
            if event.kind == KIND_COMMAND and event.response_url:
                await self.client.respond(event.response_url, DIRECT_ONLY_REPLY)
            elif event.kind == KIND_MENTION:
                await self.client.send_ephemeral(event.channel, event.user_id, DIRECT_ONLY_REPLY)

    async def _say(self, channel: str, text: str) -> None:
        await self.client.send_message(channel, clamp(text, limit=TEXT_LIMIT))

    async def _post(
        self,
        channel: str,
        text: str,
        buttons: list[tuple[str, str]] | None = None,
        *,
        links: list[tuple[str, str]] | None = None,
        replace: str | None = None,
    ) -> None:
        """Send ``text`` with optional buttons, in place of ``replace`` when given."""
        blocks = message_blocks(text, buttons or [], links or [])
        if replace:
            try:
                await self.client.update_message(channel, replace, text, blocks)
                return
            except SlackError:
                logger.warning("Could not edit the working message", exc_info=True)
        await self.client.send_message(channel, text, blocks)

    # ---- text and commands ---------------------------------------------

    async def _over_limit(self, chat_id: int) -> str | None:
        subject = subject_for(await self.store.get_binding(chat_id), chat_id)
        today = _today()
        return over_limit(
            await self.store.get_usage(subject, today),
            await self.store.get_usage(GLOBAL_SUBJECT, today),
            self.settings,
        )

    async def _unbound(self, channel: str, chat_id: int) -> None:
        """A person who has not logged in: silence, or the login button."""
        if not self.settings.slack_silent_until_login:
            await self._cmd_start(channel, chat_id)

    async def _on_text(self, event: SlackEvent, chat_id: int) -> None:
        command = parse_command(event.text, slash=event.kind == KIND_COMMAND)
        if command is not None:
            await self._command(event, chat_id, *command)
            return
        if not event.text:
            return
        if await self.store.get_binding(chat_id) is None:
            link_reply = await confirm_link(self.store, self.settings, chat_id, event.text)
            if link_reply is not None:
                await self._say(event.channel, link_reply)
            else:
                await self._unbound(event.channel, chat_id)
            return

        # Resolving a pending edit is never rate-limited; a fresh question is.
        editing = chat_id in self.awaiting_edit
        if not editing:
            refusal = await self._over_limit(chat_id)
            if refusal is not None:
                await self._say(event.channel, refusal)
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
        event: SlackEvent,
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

    async def _command(self, event: SlackEvent, chat_id: int, name: str, args: list[str]) -> None:
        channel = event.channel
        if name == "start":
            await self._cmd_start(channel, chat_id)
            return
        binding = await self.store.get_binding(chat_id)
        if binding is None:
            await self._unbound(channel, chat_id)
        elif name == "help":
            await self._say(channel, help_text(self.settings.slack_command))
        elif name == "logout":
            await logout(self.store, self.settings, chat_id)
            self.awaiting_edit.discard(chat_id)
            await clear_memory(self.store, self.memory, chat_id)
            await self._say(
                channel,
                "Disconnected. Your access tokens were revoked and the conversation was cleared.",
            )
        elif name == "new":
            self.awaiting_edit.discard(chat_id)
            await clear_memory(self.store, self.memory, chat_id)
            await self._say(
                channel, "Fresh start — I've forgotten our previous conversation. Ask away."
            )
        else:
            await self._bound_command(event, chat_id, binding, name, args)

    async def _cmd_start(self, channel: str, chat_id: int) -> None:
        binding = await self.store.get_binding(chat_id)
        if binding is not None:
            await self._say(
                channel,
                f"You are connected as {binding.odoo_name} ({binding.odoo_login}). "
                "Ask me anything about your Odoo data — I remember our conversation, "
                "so follow-up questions work. `new` starts a fresh conversation, "
                "`logout` disconnects.",
            )
            return
        url = await begin_login(self.store, self.settings, chat_id)
        if url.lower().startswith("https://"):
            await self._post(
                channel,
                "Let's connect you to Odoo. Click the button, log in with YOUR Odoo "
                "account, and approve the access request.",
                links=[(url, "Log in to Odoo")],
            )
        else:
            await self._say(
                channel,
                "Let's connect you to Odoo.\n\nOpen this link in your browser, log in with "
                f"YOUR Odoo account, and approve the access request:\n\n<{url}>",
            )

    async def _bound_command(
        self,
        event: SlackEvent,
        chat_id: int,
        binding: Binding,
        name: str,
        args: list[str],
    ) -> None:
        channel = event.channel

        if name == "forget":
            if self.facts is None:
                await self._say(channel, "I'm not storing any preferences right now.")
                return
            cleared = await forget_all(self.facts, binding.odoo_login)
            # The conversation too, or the model reads its own reply and
            # "remembers" what was just deleted.
            self.awaiting_edit.discard(chat_id)
            await clear_memory(self.store, self.memory, chat_id)
            await self._say(
                channel,
                f"Forgotten — I cleared {cleared} saved preference(s), "
                "and this conversation too so nothing lingers."
                if cleared
                else "There was nothing saved about you to forget.",
            )
        elif name == "brief":
            await self._cmd_brief(channel, chat_id, args)
        elif name == "watch":
            await self._cmd_watch(channel, chat_id, binding, args)
        elif name == "watchers":
            watchers = await self.store.list_watchers(chat_id)
            if not watchers:
                await self._say(
                    channel, f"No alerts set. Add one with `{self._slash} watch <condition>`."
                )
            else:
                lines = "\n".join(f"{i}. {w.condition}" for i, w in enumerate(watchers, 1))
                await self._say(
                    channel,
                    f"👀 Your alerts:\n\n{lines}\n\nRemove one with "
                    f"`{self._slash} unwatch <number>`.",
                )
        elif name == "unwatch":
            watchers = await self.store.list_watchers(chat_id)
            try:
                index = int(args[0]) if args else 0
            except ValueError:
                index = 0
            if not 1 <= index <= len(watchers):
                await self._say(channel, "Which one? Send `watchers` to see the numbers.")
                return
            removed = await self.store.deactivate_watcher(watchers[index - 1].id, chat_id)
            await self._say(
                channel,
                f"Removed: {watchers[index - 1].condition}"
                if removed
                else "That alert is already gone.",
            )
        elif name == "audit":
            await self._cmd_audit(channel, binding, args)

    @property
    def _slash(self) -> str:
        return self.settings.slack_command

    async def _cmd_watch(
        self, channel: str, chat_id: int, binding: Binding, args: list[str]
    ) -> None:
        condition = " ".join(args).strip()
        if not condition:
            await self._say(
                channel,
                "Tell me what to watch for, like:\n"
                f"`{self._slash} watch any customer invoice more than 60 days overdue`",
            )
        elif await self.store.count_watchers(chat_id) >= self.settings.max_watchers_per_user:
            await self._say(
                channel,
                f"You already have {self.settings.max_watchers_per_user} alerts, which is the "
                "limit. Send `watchers` to see them and "
                f"`{self._slash} unwatch <number>` to free one up.",
            )
        else:
            await self.store.add_watcher(chat_id, binding.odoo_login, condition)
            await self._say(
                channel,
                f"👀 Watching: {condition}\n\nI check about every "
                f"{self.settings.watcher_interval_minutes} minutes and will tell you when it "
                "happens — once per change, not every time I look.",
            )

    async def _cmd_brief(self, channel: str, chat_id: int, args: list[str]) -> None:
        argument = args[0].lower() if args else ""
        if argument in ("on", "off"):
            await self.store.set_brief_enabled(chat_id, argument == "on")
            await self._say(
                channel,
                "Daily brief is back on — it will arrive each morning."
                if argument == "on"
                else "Daily brief off. Ask for one any time with `brief`.",
            )
            return
        placeholder = await self.client.send_message(channel, "📋 Putting your brief together …")
        try:
            binding = await ensure_access(self.store, self.settings, chat_id)
            if binding is None:
                await self._post(channel, SLACK_CONNECT_PROMPT, replace=placeholder)
                return
            text = await run_brief(self.store, self.settings, binding, self.facts, self.latency)
        except Exception:
            logger.error("On-demand brief failed for %s", binding_id(chat_id), exc_info=True)
            await self._post(channel, ERROR_REPLY, replace=placeholder)
            return
        await self._post(
            channel,
            clamp(render_slack(text or "Nothing to report."), limit=TEXT_LIMIT),
            replace=placeholder,
        )

    async def _cmd_audit(self, channel: str, binding: Binding, args: list[str]) -> None:
        if not binding.is_admin:
            await self._say(channel, "That command is for administrators.")
            return
        days = 7
        if args:
            try:
                days = max(1, min(int(args[0]), self.settings.action_log_days))
            except ValueError:
                await self._say(channel, f"Usage: `{self._slash} audit [days]`, e.g. audit 30")
                return
        since = _now() - dt.timedelta(days=days)
        actions = await self.store.all_actions(since, limit=_AUDIT_LIMIT)
        total = await self.store.count_actions(since)
        await self._say(channel, render_slack(audit_report(actions, total, days)))

    # ---- button taps ---------------------------------------------------

    async def _on_tap(self, event: SlackEvent, chat_id: int) -> None:
        channel = event.channel
        if await self.store.get_binding(chat_id) is None:
            await self._unbound(channel, chat_id)
            return

        confirm = parse_confirm_data(event.tap)
        if confirm is not None:
            await self._on_confirm(event, chat_id, *confirm)
            return

        feedback = parse_feedback_data(event.tap)
        if feedback is not None:
            positive, trace_id = feedback
            if self.tracing is not None:
                self.tracing.record_feedback(trace_id, positive)
                await self._say(channel, "Thanks for the feedback!")
            return

        code = parse_followup_data(event.tap)
        if code is not None:
            refusal = await self._over_limit(chat_id)
            if refusal is not None:
                await self._say(channel, refusal)
                return
            await self._ask(event, chat_id, FOLLOWUPS[code][1])

    async def _settle(self, event: SlackEvent, note: str) -> None:
        """Strip the buttons from the tapped message and stamp the verdict on it."""
        if event.message_ts:
            try:
                text = f"{event.text}\n\n{note}" if event.text else note
                await self.client.update_message(
                    event.channel, event.message_ts, clamp(text, limit=TEXT_LIMIT)
                )
                return
            except SlackError:
                logger.warning("Could not update the approval message", exc_info=True)
        await self._say(event.channel, note)

    async def _on_confirm(
        self, event: SlackEvent, chat_id: int, verdict: str, trace_id: str | None
    ) -> None:
        def score(decision: str) -> None:
            # The tap scores the trace that PROPOSED the action.
            if self.tracing is not None and trace_id:
                self.tracing.record_decision(trace_id, decision)

        if verdict == "edit":
            score("edit-requested")
            self.awaiting_edit.add(chat_id)
            await self._settle(event, "✏️ Editing.")
            await self._say(event.channel, EDIT_PROMPT)
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

    # ---- voice and files -----------------------------------------------

    async def _on_file(self, event: SlackEvent, chat_id: int) -> None:
        if await self.store.get_binding(chat_id) is None:
            await self._unbound(event.channel, chat_id)
            return
        if not event.files:  # shared from another workspace: the app cannot fetch it
            await self._say(event.channel, UNSUPPORTED_REPLY)
            return
        file = event.files[0]
        if file.is_audio:
            await self._on_voice(event, chat_id, file)
        else:
            await self._on_upload(event, chat_id, file)

    async def _on_voice(self, event: SlackEvent, chat_id: int, file: SlackFile) -> None:
        channel = event.channel
        settings = self.settings
        if not settings.voice:
            await self._say(channel, VOICE_OFF)
            return
        refusal = await self._over_limit(chat_id)
        if refusal is not None:
            await self._say(channel, refusal)
            return
        limit = settings.slack_voice_max_mb * 1024 * 1024
        if file.size > limit:
            await self._say(
                channel,
                f"That clip is larger than {settings.slack_voice_max_mb} MB. Send a shorter one.",
            )
            return

        try:
            audio = await self.client.download(file.url, limit)
            result = await transcribe(
                settings, audio, filename=file.name or "voice.m4a", mime=file.mime
            )
        except FileTooLarge:
            await self._say(
                channel,
                f"That clip is larger than {settings.slack_voice_max_mb} MB. Send a shorter one.",
            )
            return
        except Exception:
            logger.error("Voice download or transcription failed", exc_info=True)
            await self._say(channel, VOICE_UNCLEAR)
            return

        # Billed even when empty, so recorded before looking at the text.
        if result.cost_usd:
            subject = subject_for(await self.store.get_binding(chat_id), chat_id)
            await self.store.record_usage(subject, _today(), result.cost_usd)
        if not result.text:
            await self._say(channel, VOICE_UNCLEAR)
            return
        logger.info("Chat %s: voice clip transcribed", binding_id(chat_id))
        await self._say(channel, f"🎤 I heard: {result.text}")
        await self._ask(event, chat_id, result.text)

    async def _on_upload(self, event: SlackEvent, chat_id: int, file: SlackFile) -> None:
        channel = event.channel
        settings = self.settings
        if not settings.uploads:
            await self._say(channel, UPLOADS_OFF)
            return
        refusal = await self._over_limit(chat_id)
        if refusal is not None:
            await self._say(channel, refusal)
            return

        # Decided on what Slack declares, before the download is paid for.
        declined = check_upload(file.mime, file.size, settings)
        if declined is not None:
            await self._say(channel, declined)
            return
        try:
            data = await self.client.download(file.url, settings.upload_max_mb * 1024 * 1024)
        except FileTooLarge:
            await self._say(
                channel,
                f"That file is larger than {settings.upload_max_mb} MB. Send a smaller one.",
            )
            return
        except SlackError:
            logger.error("Upload download failed for %s", binding_id(chat_id), exc_info=True)
            await self._say(channel, UPLOAD_FAILED)
            return

        default_name = "photo.jpg" if file.mime.startswith("image/") else "file"
        attachment = Attachment(
            kind=SUPPORTED[file.mime],
            mime=file.mime,
            filename=file.name or default_name,
            data=data,
        )
        logger.info("Chat %s: reading upload %s", binding_id(chat_id), describe(attachment))
        question = event.text.strip() or DEFAULT_QUESTION
        if not event.text.strip():
            await self._say(channel, f'📎 {describe(attachment)} — asking: "{question}"')
        await self._ask(event, chat_id, question, attachment)

    # ---- running a turn and delivering it ------------------------------

    @asynccontextmanager
    async def _working(self, channel: str):  # type: ignore[no-untyped-def]
        """A "working" message that appears only if the turn is slow.

        The answer later replaces it in place, since Slack can edit a message.
        """
        holder = _Placeholder()

        async def show() -> None:
            await asyncio.sleep(_WORKING_AFTER_SECONDS)
            try:
                holder.ts = await self.client.send_message(channel, _WORKING_TEXT)
            except SlackError:
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
        event: SlackEvent,
        chat_id: int,
        make_reply: Callable[[], Awaitable[BotReply]],
    ) -> None:
        async with self._working(event.channel) as placeholder:
            try:
                reply = await make_reply()
            except Exception:
                logger.error("Failed answering chat %s", binding_id(chat_id), exc_info=True)
                reply = BotReply(text=ERROR_REPLY)
        await self.deliver(event.channel, chat_id, reply, placeholder.ts)

    async def deliver(
        self, channel: str, chat_id: int, reply: BotReply, placeholder: str | None = None
    ) -> None:
        """Send a finished reply, with its buttons in a block under the text."""
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
                channel,
                clamp(html_to_slack(reply.text), footer, limit=TEXT_LIMIT),
                buttons,
                replace=placeholder,
            )
            return

        if reply.grounded and reply.text:
            stamp = freshness_line(reply.evidence, binding, self.settings, _now())
            reply.text += f"\n\n*{stamp}*"
        links = reply.evidence.links or None
        body = render_slack(reply.text, links)
        if links and len(body) > TEXT_LIMIT - 100:
            body = render_slack(reply.text)  # links inflate it; drop them first
        await self._post(
            channel,
            clamp(body, footer, limit=TEXT_LIMIT),
            self._option_buttons(reply, can_write=bool(binding and binding.can_write)),
            replace=placeholder,
        )
        await self._send_files(channel, reply)

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

    async def _send_files(self, channel: str, reply: BotReply) -> None:
        for artifact in reply.attachments:
            try:
                await self.client.upload_file(
                    channel, artifact.path, artifact.path.name, artifact.caption or ""
                )
            except SlackError:
                logger.error("Failed sending %s to Slack", artifact.kind, exc_info=True)
            finally:
                artifact.path.unlink(missing_ok=True)
