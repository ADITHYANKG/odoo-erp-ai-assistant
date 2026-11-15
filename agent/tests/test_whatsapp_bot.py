"""Tests for ont_agent.whatsapp_bot: dispatch, guards, buttons, delivery."""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from ont_agent import whatsapp_bot
from ont_agent.agent import PendingAction
from ont_agent.bot import ERROR_REPLY, BotReply
from ont_agent.budget import BURST_REPLY, BurstLimiter
from ont_agent.channels import whatsapp_chat_id
from ont_agent.config import Settings
from ont_agent.render import Artifact
from ont_agent.store import Binding, BindingStore
from ont_agent.tools import Evidence
from ont_agent.whatsapp import (
    KIND_REPLY,
    KIND_TEXT,
    KIND_UNSUPPORTED,
    InboundMessage,
    MediaInfo,
)
from ont_agent.whatsapp_bot import (
    NO_WATCH_REPLY,
    UNSUPPORTED_REPLY,
    WA_CONNECT_PROMPT,
    WhatsAppChannel,
    confirm_buttons,
    parse_command,
)

WA_ID = "919876543210"
CHAT_ID = whatsapp_chat_id(WA_ID)
TRACE = "a" * 32

SETTINGS = Settings(
    mcp_url="http://odoo:8160/mcp",
    mcp_bearer_key="k",
    oauth_client_id="ont_client123",
    whatsapp_access_token="tok",
    whatsapp_phone_number_id="555",
    whatsapp_app_secret="s",
    whatsapp_verify_token="vt",
)


class FakeClient:
    """Records what the channel would have sent."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.media = MediaInfo(url="https://m/x", mime="application/pdf", size=10)
        self.fail_interactive = False

    def _record(self, name: str, *args: Any) -> None:
        self.calls.append((name, args))

    async def send_text(self, to: str, text: str) -> None:
        self._record("text", to, text)

    async def send_buttons(self, to: str, body: str, buttons: Any) -> None:
        if self.fail_interactive:
            raise RuntimeError("Meta refused the interactive message")
        self._record("buttons", to, body, list(buttons))

    async def send_list(self, to: str, body: str, label: str, sections: Any) -> None:
        if self.fail_interactive:
            raise RuntimeError("Meta refused the interactive message")
        self._record("list", to, body, label, list(sections))

    async def send_cta_url(self, to: str, body: str, label: str, url: str) -> None:
        self._record("cta", to, body, label, url)

    async def send_file(self, to: str, path: Path, *, as_image: bool, caption: str = "") -> None:
        self._record("file", to, path.name, as_image, caption)

    async def media_info(self, media_id: str) -> MediaInfo:
        return self.media

    async def download(self, url: str) -> bytes:
        return b"%PDF"

    async def show_typing(self, message_id: str) -> None:
        self._record("typing", message_id)

    async def send_reaction(self, to: str, message_id: str, emoji: str) -> None:
        self._record("reaction", message_id, emoji)

    async def aclose(self) -> None:
        pass

    def sent(self, kind: str) -> list[tuple[Any, ...]]:
        return [args for name, args in self.calls if name == kind]

    def texts(self) -> list[str]:
        return [args[1] for args in self.sent("text")]


def _message(
    text: str = "hello", kind: str = KIND_TEXT, wa_id: str = WA_ID, **extra: Any
) -> InboundMessage:
    return InboundMessage(
        wa_id=wa_id,
        message_id=extra.pop("message_id", f"wamid.{uuid.uuid4().hex}"),
        timestamp=extra.pop("timestamp", time.time()),
        kind=kind,
        text=text,
        **extra,
    )


def _binding() -> Binding:
    return Binding(
        chat_id=CHAT_ID,
        odoo_login="priya@acme",
        odoo_name="CFO Priya",
        access_token="jwt",
        refresh_token="ref",
        expires_at=datetime.now(UTC) + timedelta(minutes=10),
    )


async def _channel(
    store_dsn: str,
    *,
    bound: bool = True,
    settings: Settings = SETTINGS,
    limiter: BurstLimiter | None = None,
) -> tuple[WhatsAppChannel, FakeClient]:
    store = BindingStore(store_dsn)
    await store.__aenter__()
    if bound:
        await store.save_binding(_binding())
    client = FakeClient()
    channel = WhatsAppChannel(store, settings, client, limiter=limiter)  # type: ignore[arg-type]
    return channel, client


@pytest.fixture
def answers(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replaces the agent; each call is recorded and answered from ``replies``."""
    calls: list[dict[str, Any]] = []
    replies: list[BotReply] = [BotReply(text="Total is 42")]

    async def fake_answer(
        store: Any, settings: Any, chat_id: int, text: str, **kw: Any
    ) -> BotReply:
        calls.append({"kind": "answer", "chat_id": chat_id, "text": text, **kw})
        await asyncio.sleep(0.01)  # a real turn takes long enough for typing to start
        return replies[0]

    async def fake_resume(
        store: Any, settings: Any, chat_id: int, approved: bool, **kw: Any
    ) -> BotReply:
        calls.append({"kind": "resume", "chat_id": chat_id, "approved": approved, **kw})
        return replies[0]

    monkeypatch.setattr(whatsapp_bot, "answer_question", fake_answer)
    monkeypatch.setattr(whatsapp_bot, "resume_pending", fake_resume)
    calls.append({"kind": "replies", "set": replies})  # handle to swap the reply
    return calls


def _set_reply(answers: list[dict[str, Any]], reply: BotReply) -> None:
    answers[0]["set"][0] = reply


def _calls(answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [call for call in answers if call["kind"] != "replies"]


# ---- pure helpers --------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/start", ("start", [])),
        ("/Watch any overdue invoice", ("watch", ["any", "overdue", "invoice"])),
        ("/reset", ("new", [])),
        ("start", None),
        ("Login", None),
        ("logout", None),
        ("please login", None),
        ("menu", ("help", [])),
        ("/nonsense", ("unknown", [])),
        ("watch the acme invoice", None),
        ("new customers this week", None),
        ("hello", None),
        ("   ", None),
    ],
)
def test_parse_command(text: str, expected: tuple[str, list[str]] | None) -> None:
    assert parse_command(text) == expected


def test_confirm_buttons_for_email_are_flagged_and_editable() -> None:
    buttons = confirm_buttons(["send_email"], editable=True, trace_id=TRACE)
    assert buttons == [
        (f"confirm:approve:{TRACE}", "⚠️ Approve"),
        (f"confirm:edit:{TRACE}", "✏️ Edit"),
        (f"confirm:reject:{TRACE}", "❌ Cancel"),
    ]


def test_confirm_buttons_for_record_update_have_no_edit() -> None:
    buttons = confirm_buttons(["update_records"], editable=False, trace_id=None)
    assert buttons == [("confirm:approve", "✅ Approve"), ("confirm:reject", "❌ Cancel")]


def test_confirm_buttons_ignore_malformed_trace() -> None:
    assert confirm_buttons(["update_records"], editable=False, trace_id="nothex")[0][0] == (
        "confirm:approve"
    )


# ---- guards --------------------------------------------------------------


async def test_unbound_text_gets_connect_prompt(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, client = await _channel(store_dsn, bound=False)
    await channel._serialized(_message("how much did we sell?"))
    assert client.texts() == [WA_CONNECT_PROMPT]
    assert _calls(answers) == []


async def test_non_allowlisted_sender_gets_silence(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    settings = SETTINGS.model_copy(update={"allowed_whatsapp_numbers": ["14155550123"]})
    channel, client = await _channel(store_dsn, settings=settings)
    await channel._serialized(_message("hi"))
    assert client.calls == []
    assert _calls(answers) == []


async def test_stale_message_is_dropped(store_dsn: str, answers: list[dict[str, Any]]) -> None:
    channel, client = await _channel(store_dsn)
    await channel._serialized(_message("old news", timestamp=time.time() - 3600))
    assert client.calls == []
    assert _calls(answers) == []


async def test_burst_limit_warns_once_then_goes_quiet(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, client = await _channel(store_dsn, limiter=BurstLimiter(1, 60))
    for _ in range(4):
        await channel._serialized(_message("q"))
    assert len(_calls(answers)) == 1
    assert client.texts().count(BURST_REPLY) == 1


async def test_unsupported_kind_gets_hint(store_dsn: str, answers: list[dict[str, Any]]) -> None:
    channel, client = await _channel(store_dsn)
    await channel._serialized(_message("", kind=KIND_UNSUPPORTED))
    assert client.texts() == [UNSUPPORTED_REPLY]


async def test_duplicate_message_ids_are_scheduled_once(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    channel, _client = await _channel(store_dsn)
    handled: list[str] = []

    async def record(message: InboundMessage) -> None:
        handled.append(message.message_id)

    monkeypatch.setattr(channel, "_serialized", record)
    payload = {
        "entry": [{"changes": [{"value": {
            "metadata": {"phone_number_id": "555"},
            "messages": [{"from": WA_ID, "id": "wamid.same", "timestamp": "1",
                          "type": "text", "text": {"body": "hi"}}],
        }}]}]
    }  # fmt: skip
    assert channel.handle_payload(payload) == 1
    assert channel.handle_payload(payload) == 0
    await asyncio.gather(*channel._tasks)
    assert handled == ["wamid.same"]


# ---- answering -----------------------------------------------------------


async def test_quick_answer_never_flashes_a_reaction(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, client = await _channel(store_dsn)
    await channel._serialized(_message("hello"))
    assert client.sent("reaction") == []


async def test_slow_answer_shows_an_hourglass_then_clears_it(
    store_dsn: str, answers: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(whatsapp_bot, "_REACT_AFTER_SECONDS", 0.001)
    message = _message("slow question")
    channel, client = await _channel(store_dsn)
    await channel._serialized(message)
    assert client.sent("reaction") == [(message.message_id, "⏳"), (message.message_id, "")]
    # The hourglass is gone before the answer lands, not after.
    names = [name for name, _args in client.calls]
    assert names.index("reaction") < names.index("text")


async def test_typing_is_reasserted_after_the_hourglass(
    store_dsn: str, answers: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(whatsapp_bot, "_REACT_AFTER_SECONDS", 0.001)
    message = _message("slow question")
    channel, client = await _channel(store_dsn)
    await channel._serialized(message)
    first = client.calls.index(("reaction", (message.message_id, "⏳")))
    # Sending the reaction may dismiss "typing…", so it must be switched back on.
    assert any(name == "typing" for name, _args in client.calls[first + 1 :])


async def test_typing_keeps_refreshing_until_the_answer_arrives(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def slow(*args: Any, **kwargs: Any) -> BotReply:
        await asyncio.sleep(0.12)
        return BotReply(text="done")

    monkeypatch.setattr(whatsapp_bot, "answer_question", slow)
    monkeypatch.setattr(whatsapp_bot, "_TYPING_REFRESH_SECONDS", 0.02)
    channel, client = await _channel(store_dsn)
    await channel._serialized(_message("slow question"))
    typing_calls = [name for name, _args in client.calls if name == "typing"]
    assert len(typing_calls) >= 4
    # None after the answer: typing must stop the moment the reply goes out.
    assert client.calls[-1][0] == "text"


async def test_hourglass_is_cleared_even_when_the_agent_fails(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(*args: Any, **kwargs: Any) -> BotReply:
        await asyncio.sleep(0.02)
        raise RuntimeError("model down")

    monkeypatch.setattr(whatsapp_bot, "answer_question", boom)
    monkeypatch.setattr(whatsapp_bot, "_REACT_AFTER_SECONDS", 0.001)
    message = _message("slow question")
    channel, client = await _channel(store_dsn)
    await channel._serialized(message)
    assert client.sent("reaction")[-1] == (message.message_id, "")
    assert client.texts() == [ERROR_REPLY]


async def test_text_question_runs_the_agent_and_replies(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, client = await _channel(store_dsn)
    await channel._serialized(_message("how much did we sell?"))
    [call] = _calls(answers)
    assert (call["kind"], call["chat_id"], call["text"]) == (
        "answer",
        CHAT_ID,
        "how much did we sell?",
    )
    assert client.texts() == ["Total is 42"]
    assert client.sent("typing")  # progress is the typing indicator


async def test_agent_failure_sends_error_reply(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(*args: Any, **kwargs: Any) -> BotReply:
        raise RuntimeError("model down")

    monkeypatch.setattr(whatsapp_bot, "answer_question", boom)
    channel, client = await _channel(store_dsn)
    await channel._serialized(_message("anything"))
    assert client.texts() == [ERROR_REPLY]


async def test_grounded_answer_gets_freshness_and_visible_followup_buttons(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    _set_reply(
        answers,
        BotReply(
            text="Top customers:\n- Acme 1,200\n- Beta 900\n- Gamma 400",
            grounded=True,
            trace_id=TRACE,
            evidence=Evidence(counts={"res.partner": 3}),
        ),
    )
    channel, client = await _channel(store_dsn)
    await channel._serialized(_message("top customers"))

    # The answer is the buttons' own body: one bubble, buttons visible without a tap.
    assert client.sent("text") == []
    assert client.sent("list") == []
    [(_, body, buttons)] = client.sent("buttons")
    assert "Acme 1,200" in body
    assert "As of" in body
    assert "3 contacts" in body
    ids = [button_id for button_id, _title in buttons]
    assert ids == ["fu:chart", "fu:customer", "fu:email"]  # WhatsApp's three-button cap


async def test_long_answer_puts_its_end_in_the_options_bubble(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    lines = [f"- Customer {i:03d} 1,000" for i in range(120)]
    _set_reply(
        answers,
        BotReply(text="Customers:\n" + "\n".join(lines), grounded=True, trace_id=TRACE),
    )
    channel, client = await _channel(store_dsn)
    await channel._serialized(_message("all customers"))

    [(_, head)] = client.sent("text")
    [(_, tail, _buttons)] = client.sent("buttons")
    assert len(tail) <= 1024
    assert "Customer 000" in head
    assert "Customer 119" in tail  # the last line sits right above the buttons
    assert "Customer 119" not in head


async def test_unsplittable_answer_falls_back_to_a_short_options_message(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    _set_reply(answers, BotReply(text="x" * 3000, trace_id=TRACE))
    channel, client = await _channel(store_dsn)
    await channel._serialized(_message("big blob"))

    assert len(client.sent("text")[0][1]) > 2000
    [(_, body, _buttons)] = client.sent("buttons")
    assert body == "How was this answer?"


async def test_failed_options_still_deliver_the_end_of_the_answer(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    _set_reply(answers, BotReply(text="Total is 42", grounded=True, trace_id=TRACE))
    channel, client = await _channel(store_dsn)
    client.fail_interactive = True
    await channel._serialized(_message("total"))
    assert "Total is 42" in client.texts()[0]


async def test_conversational_answer_has_no_buttons(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, client = await _channel(store_dsn)
    await channel._serialized(_message("thanks"))
    assert client.sent("buttons") == []
    assert client.sent("list") == []


async def test_one_followup_leaves_room_for_both_thumbs(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    _set_reply(
        answers,
        BotReply(text="Leave policy text", grounded=True, from_documents=True, trace_id=TRACE),
    )
    channel, client = await _channel(store_dsn)
    await channel._serialized(_message("leave policy"))
    [(_, _body, buttons)] = client.sent("buttons")
    assert [button_id for button_id, _ in buttons] == [
        "fu:email",
        f"fb:1:{TRACE}",
        f"fb:0:{TRACE}",
    ]


async def test_two_followups_keep_only_the_complaint_thumb(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    _set_reply(
        answers,
        BotReply(text="Revenue 1,200 then 900 and 400", grounded=True, trace_id=TRACE),
    )
    channel, client = await _channel(store_dsn)
    await channel._serialized(_message("revenue"))
    [(_, _body, buttons)] = client.sent("buttons")
    assert [button_id for button_id, _ in buttons] == [
        "fu:chart",
        "fu:email",
        f"fb:0:{TRACE}",
    ]


async def test_answer_with_only_a_trace_gets_both_thumbs(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    _set_reply(answers, BotReply(text="Glad to help", trace_id=TRACE))
    channel, client = await _channel(store_dsn)
    await channel._serialized(_message("thanks"))
    [(_, body, buttons)] = client.sent("buttons")
    assert body == "Glad to help"
    assert [title for _, title in buttons] == ["👍 Helpful", "👎 Not helpful"]


async def test_pending_action_shows_proposal_in_the_buttons_bubble(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    pending = PendingAction(
        actions=["send_email"], names=["send_email"], arguments=[{"to": "a@b.com"}]
    )
    _set_reply(
        answers,
        BotReply(
            text="⚠️ <b>Approval needed</b> — send it?\n\n▶ <b>send_email</b>",
            pending=pending,
            trace_id=TRACE,
        ),
    )
    channel, client = await _channel(store_dsn)
    await channel._serialized(_message("email the report to a@b.com"))

    assert client.sent("text") == []
    [(_, proposal, buttons)] = client.sent("buttons")
    assert "*Approval needed*" in proposal
    assert "<b>" not in proposal
    assert [button_id for button_id, _ in buttons] == [
        f"confirm:approve:{TRACE}",
        f"confirm:edit:{TRACE}",
        f"confirm:reject:{TRACE}",
    ]


async def test_attachments_are_sent_and_deleted(
    store_dsn: str, answers: list[dict[str, Any]], tmp_path: Path
) -> None:
    chart = tmp_path / "sales.png"
    chart.write_bytes(b"png")
    _set_reply(
        answers,
        BotReply(text="Here it is", attachments=[Artifact(chart, "photo", "Sales")]),
    )
    channel, client = await _channel(store_dsn)
    await channel._serialized(_message("chart please"))
    assert client.sent("file") == [(WA_ID, "sales.png", True, "Sales")]
    assert not chart.exists()


# ---- buttons -------------------------------------------------------------


def _tap(reply_id: str) -> InboundMessage:
    return _message("", kind=KIND_REPLY, reply_id=reply_id)


async def test_approve_tap_resumes_the_paused_action(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, client = await _channel(store_dsn)
    await channel._serialized(_tap(f"confirm:approve:{TRACE}"))
    [call] = _calls(answers)
    assert (call["kind"], call["approved"], call["chat_id"]) == ("resume", True, CHAT_ID)
    assert "✅ Approved." in client.texts()


async def test_cancel_tap_rejects(store_dsn: str, answers: list[dict[str, Any]]) -> None:
    channel, client = await _channel(store_dsn)
    await channel._serialized(_tap("confirm:reject"))
    [call] = _calls(answers)
    assert call["approved"] is False
    assert "❌ Cancelled." in client.texts()


async def test_edit_tap_then_next_text_is_the_revision(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, _client = await _channel(store_dsn)
    await channel._serialized(_tap(f"confirm:edit:{TRACE}"))
    assert _calls(answers) == []  # nothing runs until the user says what to change
    assert CHAT_ID in channel.awaiting_edit

    await channel._serialized(_message("send it to x@y.com instead"))
    [call] = _calls(answers)
    assert (call["kind"], call["approved"], call["revision"]) == (
        "resume",
        False,
        "send it to x@y.com instead",
    )
    assert CHAT_ID not in channel.awaiting_edit


async def test_followup_tap_asks_the_canned_question(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, _client = await _channel(store_dsn)
    await channel._serialized(_tap("fu:chart"))
    [call] = _calls(answers)
    assert (call["kind"], call["text"]) == ("answer", "Show that as a chart.")


async def test_feedback_tap_scores_the_trace(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    class FakeTracing:
        def __init__(self) -> None:
            self.scores: list[tuple[str, bool]] = []

        def record_feedback(self, trace_id: str, positive: bool) -> None:
            self.scores.append((trace_id, positive))

    channel, _client = await _channel(store_dsn)
    channel.tracing = tracing = FakeTracing()  # type: ignore[assignment]
    await channel._serialized(_tap(f"fb:0:{TRACE}"))
    assert tracing.scores == [(TRACE, False)]
    assert _calls(answers) == []


# ---- commands ------------------------------------------------------------


async def test_start_when_unbound_sends_login_button(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_begin(store: Any, settings: Any, chat_id: int) -> str:
        assert chat_id == CHAT_ID
        return "https://odoo.example/mcp/oauth/authorize?state=s"

    monkeypatch.setattr(whatsapp_bot, "begin_login", fake_begin)
    channel, client = await _channel(store_dsn, bound=False)
    await channel._serialized(_message("/start"))
    [(_, _, label, url)] = client.sent("cta")
    assert label == "Log in to Odoo"
    assert url.startswith("https://odoo.example/")


async def test_start_when_bound_reports_identity(store_dsn: str) -> None:
    channel, client = await _channel(store_dsn)
    await channel._serialized(_message("/start"))
    assert "CFO Priya" in client.texts()[0]


async def test_watch_is_declined_on_whatsapp(store_dsn: str) -> None:
    channel, client = await _channel(store_dsn)
    await channel._serialized(_message("/watch any overdue invoice"))
    assert client.texts() == [NO_WATCH_REPLY]
    assert await channel.store.list_watchers(CHAT_ID) == []


async def test_logout_removes_binding(store_dsn: str, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_logout(store: BindingStore, settings: Settings, chat_id: int) -> bool:
        await store.delete_binding(chat_id)
        return True

    monkeypatch.setattr(whatsapp_bot, "logout", fake_logout)
    channel, client = await _channel(store_dsn)
    await channel._serialized(_message("/logout"))
    assert await channel.store.get_binding(CHAT_ID) is None
    assert "Disconnected" in client.texts()[0]


async def test_audit_is_admin_only(store_dsn: str) -> None:
    channel, client = await _channel(store_dsn)
    await channel._serialized(_message("/audit"))
    assert client.texts() == ["That command is for administrators."]


# ---- files ---------------------------------------------------------------


async def test_pdf_upload_is_read_and_asked_about(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, _client = await _channel(store_dsn)
    await channel._serialized(
        _message(
            "is this paid?",
            kind="document",
            media_id="m1",
            mime="application/pdf",
            filename="inv.pdf",
        )
    )
    [call] = _calls(answers)
    assert call["text"] == "is this paid?"
    assert call["attachment"].filename == "inv.pdf"
    assert call["attachment"].data == b"%PDF"


async def test_unsupported_file_type_is_refused_before_download(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, client = await _channel(store_dsn)
    client.media = MediaInfo(url="https://m/x", mime="application/zip", size=10)
    await channel._serialized(_message("", kind="document", media_id="m1", mime="application/zip"))
    assert _calls(answers) == []
    assert "archive" in client.texts()[0]
