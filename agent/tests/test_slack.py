"""Tests for the Slack channel: ids, signatures, events, formatting, sending, config."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlencode

import httpx
import pytest
from slack_sdk.errors import SlackApiError

from ont_agent import callback_server, slack_bot
from ont_agent.bot import ERROR_REPLY, BotReply
from ont_agent.channels import (
    GOOGLE_CHAT_OFFSET,
    SLACK_OFFSET,
    googlechat_chat_id,
    is_google_chat,
    is_slack,
    is_whatsapp,
    slack_chat_id,
    whatsapp_chat_id,
)
from ont_agent.config import Settings, load_settings
from ont_agent.render import Artifact
from ont_agent.slack import (
    ENVELOPE_COMMAND,
    ENVELOPE_EVENTS,
    ENVELOPE_INTERACTIVE,
    KIND_COMMAND,
    KIND_MENTION,
    KIND_TAP,
    KIND_TEXT,
    FileTooLarge,
    SlackAuthError,
    SlackClient,
    SlackError,
    message_blocks,
    parse_event,
    split_text,
    verify_signature,
)
from ont_agent.slack_bot import (
    DIRECT_ONLY_REPLY,
    SLACK_CONNECT_PROMPT,
    SlackChannel,
    help_text,
    parse_command,
)
from ont_agent.slack_format import html_to_slack, plain_from_slack, render_slack
from ont_agent.store import Binding, BindingStore

USER = "U0123ABCD"
TEAM = "T0123ABCD"
DM = "D0123ABCD"
CHAT_ID = slack_chat_id(USER)
TRACE = "b" * 32
SECRET = "signing-secret"

SETTINGS = Settings(
    mcp_url="http://odoo:8160/mcp",
    mcp_bearer_key="k",
    oauth_client_id="ont_client123",
    slack_bot_token="xoxb-test",
    slack_app_token="xapp-test",
)


# ---- ids -------------------------------------------------------------------


def test_slack_chat_id_is_stable_and_in_its_own_range() -> None:
    assert slack_chat_id(USER) == slack_chat_id(USER) != slack_chat_id("U0123ABCE")
    assert is_slack(CHAT_ID) and CHAT_ID < 2**63
    assert not is_google_chat(CHAT_ID) and not is_whatsapp(CHAT_ID)


def test_ranges_do_not_overlap() -> None:
    google = googlechat_chat_id("users/104512345678901234567")
    assert is_google_chat(google) and not is_slack(google)
    assert GOOGLE_CHAT_OFFSET + 2**56 < SLACK_OFFSET
    assert not is_slack(whatsapp_chat_id("919876543210"))
    assert not is_slack(123456789)


@pytest.mark.parametrize("bad", ["", "U", "u0123abcd", "T0123ABCD", "C0123ABCD", "U 01"])
def test_slack_chat_id_rejects_bad_ids(bad: str) -> None:
    with pytest.raises(ValueError, match="Slack"):
        slack_chat_id(bad)


# ---- signatures ------------------------------------------------------------


def _sign(body: bytes, timestamp: str | None = None, secret: str = SECRET) -> dict[str, str]:
    stamp = timestamp or str(int(time.time()))
    digest = hmac.new(secret.encode(), b"v0:" + stamp.encode() + b":" + body, hashlib.sha256)
    return {"X-Slack-Request-Timestamp": stamp, "X-Slack-Signature": "v0=" + digest.hexdigest()}


def test_signature_accepts_a_genuine_request() -> None:
    headers = _sign(b"payload")
    assert verify_signature(
        SECRET, b"payload", headers["X-Slack-Request-Timestamp"], headers["X-Slack-Signature"]
    )


@pytest.mark.parametrize(
    ("body", "secret", "age"),
    [(b"other", SECRET, 0), (b"payload", "wrong", 0), (b"payload", SECRET, 3600)],
    ids=["tampered", "wrong-secret", "replayed"],
)
def test_signature_rejects_forgeries(body: bytes, secret: str, age: int) -> None:
    headers = _sign(b"payload", str(int(time.time()) - age), secret)
    assert not verify_signature(
        SECRET, body, headers["X-Slack-Request-Timestamp"], headers["X-Slack-Signature"]
    )


def test_signature_rejects_missing_parts() -> None:
    assert not verify_signature(SECRET, b"x", None, "v0=a")
    assert not verify_signature(SECRET, b"x", "1", None)
    assert not verify_signature(SECRET, b"x", "soon", "v0=a")
    assert not verify_signature("", b"x", str(int(time.time())), "v0=a")


# ---- inbound events --------------------------------------------------------


def _event(
    text: str = "hello", *, event_type: str = "message", channel: str = DM, **extra: Any
) -> tuple[str, dict[str, Any]]:
    event: dict[str, Any] = {
        "type": event_type,
        "user": USER,
        "team": TEAM,
        "channel": channel,
        "channel_type": "im" if channel.startswith("D") else "channel",
        "text": text,
        "ts": f"{time.time():.6f}",
        **extra,
    }
    return ENVELOPE_EVENTS, {
        "team_id": TEAM,
        "event_id": f"Ev{uuid.uuid4().hex[:10]}",
        "event": event,
    }


def _command(text: str, channel: str = DM) -> tuple[str, dict[str, Any]]:
    return ENVELOPE_COMMAND, {
        "command": "/odoo",
        "text": text,
        "user_id": USER,
        "team_id": TEAM,
        "channel_id": channel,
        "trigger_id": uuid.uuid4().hex,
        "response_url": "https://hooks.slack.com/commands/T/1/x",
    }


def _tap(data: str, text: str = "Approval needed") -> tuple[str, dict[str, Any]]:
    return ENVELOPE_INTERACTIVE, {
        "type": "block_actions",
        "trigger_id": uuid.uuid4().hex,
        "user": {"id": USER, "team_id": TEAM},
        "container": {"channel_id": DM, "message_ts": "111.222"},
        "message": {"ts": "111.222", "text": text},
        "actions": [
            {"type": "button", "action_id": "b0", "value": data, "action_ts": f"{time.time():.6f}"}
        ],
    }


def test_parse_text_message() -> None:
    parsed = parse_event(*_event("what is <http://x.io|overdue> &amp; late?"))
    assert parsed is not None
    assert (parsed.kind, parsed.user_id, parsed.team_id, parsed.channel) == (
        KIND_TEXT,
        USER,
        TEAM,
        DM,
    )
    assert parsed.direct and parsed.text == "what is overdue & late?"
    assert parsed.timestamp > 0


def test_parse_team_prefers_the_senders_own_workspace() -> None:
    parsed = parse_event(*_event(user_team="T9999ZZZZ"))
    assert parsed is not None and parsed.team_id == "T9999ZZZZ"


def test_parse_mention_in_a_channel_is_not_direct() -> None:
    parsed = parse_event(*_event("<@UBOT> hi", event_type="app_mention", channel="C0123ABCD"))
    assert parsed is not None
    assert (parsed.kind, parsed.direct, parsed.text) == (KIND_MENTION, False, "@UBOT hi")


def test_parse_slash_command() -> None:
    parsed = parse_event(*_command("login"))
    assert parsed is not None
    assert (parsed.kind, parsed.text, parsed.direct) == (KIND_COMMAND, "login", True)
    assert parsed.response_url.startswith("https://hooks.slack.com/")
    in_channel = parse_event(*_command("login", channel="C0123ABCD"))
    assert in_channel is not None and not in_channel.direct


def test_parse_tap_and_ignore_link_buttons() -> None:
    parsed = parse_event(*_tap("confirm:approve"))
    assert parsed is not None
    assert (parsed.kind, parsed.tap, parsed.message_ts, parsed.text) == (
        KIND_TAP,
        "confirm:approve",
        "111.222",
        "Approval needed",
    )
    envelope, payload = _tap("x")
    del payload["actions"][0]["value"]
    assert parse_event(envelope, payload) is None


def test_parse_files() -> None:
    parsed = parse_event(
        *_event(
            "total?",
            subtype="file_share",
            files=[
                {
                    "url_private_download": "https://files.slack.com/files-pri/T-F/bill.pdf",
                    "mimetype": "application/pdf",
                    "name": "bill.pdf",
                    "size": 1234,
                },
                {
                    "url_private_download": "https://files.slack.com/a.m4a",
                    "subtype": "slack_audio",
                },
                {"file_access": "check_file_info", "id": "F1"},
            ],
        )
    )
    assert parsed is not None
    pdf, clip = parsed.files
    assert (pdf.mime, pdf.name, pdf.size, pdf.is_audio) == (
        "application/pdf",
        "bill.pdf",
        1234,
        False,
    )
    assert clip.is_audio
    assert parsed.has_unreadable_file


@pytest.mark.parametrize(
    "build",
    [
        lambda: _event(bot_id="B1"),
        lambda: _event(subtype="message_changed"),
        lambda: _event(subtype="bot_message"),
        lambda: _event(channel="C0123ABCD"),
        lambda: _event(event_type="reaction_added"),
        lambda: (ENVELOPE_EVENTS, {"event": {"type": "message", "text": "no sender"}}),
        lambda: (ENVELOPE_EVENTS, {}),
        lambda: ("hello", {}),
        lambda: (ENVELOPE_INTERACTIVE, {"type": "view_submission"}),
    ],
    ids=[
        "bot",
        "edit",
        "bot-subtype",
        "channel-chatter",
        "other-event",
        "no-sender",
        "empty",
        "other-envelope",
        "other-interaction",
    ],
)
def test_parse_ignores_what_needs_no_answer(build: Any) -> None:
    assert parse_event(*build()) is None


# ---- formatting ------------------------------------------------------------


def test_format_bold_italic_code_and_links() -> None:
    markup = (
        "<b>Total</b> and <i>note</i> with <code>x</code> and "
        '<a href="https://odoo.example.com/r/1">SO001</a>'
    )
    assert html_to_slack(markup) == (
        "*Total* and _note_ with `x` and <https://odoo.example.com/r/1|SO001>"
    )


def test_format_escapes_what_slack_would_act_on() -> None:
    assert html_to_slack("a &lt;!channel&gt; &amp; b") == "a &lt;!channel&gt; &amp; b"
    assert render_slack("ping <!here> now") == "ping &lt;!here&gt; now"
    assert html_to_slack("<pre>if a &lt; b</pre>") == "```\nif a &lt; b\n```"


def test_link_label_cannot_break_the_syntax() -> None:
    assert (
        html_to_slack('<a href="https://x.io/a">a|b &gt; c</a>') == "<https://x.io/a|a/b &gt; c>"
    )
    assert html_to_slack('<a href="https://x.io/a">https://x.io/a</a>') == "<https://x.io/a>"


def test_render_slack_turns_markdown_bold() -> None:
    assert render_slack("**Hi** there") == "*Hi* there"


def test_plain_from_slack_undoes_markup() -> None:
    assert plain_from_slack("<@U1> see <https://x.io> and <#C1|ops> <!here> 1 &lt; 2") == (
        "@U1 see https://x.io and #ops @here 1 < 2"
    )


def test_split_text_keeps_pieces_under_the_limit_and_fences_balanced() -> None:
    assert split_text("short") == ["short"]
    code = "```\n" + "\n".join(f"row {i}" for i in range(400)) + "\n```"
    pieces = split_text(code, 500)
    assert len(pieces) > 1
    assert all(len(p) <= 500 and p.count("```") % 2 == 0 for p in pieces)
    assert len(split_text("x" * 7000, 3000)) == 3


def test_message_blocks_carry_the_tap_data_and_styles() -> None:
    assert message_blocks("hi", []) == []
    blocks = message_blocks(
        "Approve?",
        [("confirm:approve", "Approve"), ("confirm:reject", "Cancel"), ("fu:chart", "Chart")],
        links=[("https://odoo/login", "Log in")],
    )
    assert blocks[0] == {"type": "section", "text": {"type": "mrkdwn", "text": "Approve?"}}
    elements = blocks[-1]["elements"]
    assert [e.get("value") or e.get("url") for e in elements] == [
        "confirm:approve",
        "confirm:reject",
        "fu:chart",
        "https://odoo/login",
    ]
    assert [e.get("style") for e in elements] == ["primary", "danger", None, "primary"]
    assert len({e["action_id"] for e in elements}) == 4


# ---- commands --------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "slash", "expected"),
    [
        ("login", False, ("start", [])),
        ("Help", False, ("help", [])),
        ("new", False, ("new", [])),
        ("brief off", False, ("brief", ["off"])),
        ("watchers", False, ("watchers", [])),
        ("", True, ("help", [])),
        ("watch big invoices", True, ("watch", ["big", "invoices"])),
        ("unwatch 2", True, ("unwatch", ["2"])),
        ("login", True, ("start", [])),
        ("watch the Acme invoice", False, None),
        ("new customers this week", False, None),
        ("show recent orders", True, None),
        ("hello", False, None),
        ("", False, None),
    ],
)
def test_parse_command(text: str, slash: bool, expected: tuple[str, list[str]] | None) -> None:
    assert parse_command(text, slash=slash) == expected


def test_help_names_the_configured_slash_command() -> None:
    assert "`/ask watch <condition>`" in help_text("/ask")


# ---- client ----------------------------------------------------------------


class FakeWeb:
    """Stands in for slack_sdk's AsyncWebClient."""

    def __init__(self, error: str | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.error = error

    def __getattr__(self, method: str) -> Any:
        async def call(**kwargs: Any) -> SimpleNamespace:
            self.calls.append((method, kwargs))
            if self.error:
                raise SlackApiError("failed", {"error": self.error})
            return SimpleNamespace(
                data={"ts": "123.456", "team_id": TEAM, "user_id": "UBOT", "ok": True}
            )

        return call


def _client(web: FakeWeb, handler: Any = None) -> SlackClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler)) if handler else None
    return SlackClient(SETTINGS, web=web, http=http)  # type: ignore[arg-type]


async def test_client_posts_with_blocks_only_when_given() -> None:
    web = FakeWeb()
    client = _client(web)
    assert await client.send_message(DM, "x" * 9000) == "123.456"
    [(_, plain)] = web.calls
    assert len(plain["text"]) == 4000 and "blocks" not in plain and plain["unfurl_links"] is False
    await client.send_message(DM, "hi", [{"type": "divider"}])
    assert web.calls[1][1]["blocks"] == [{"type": "divider"}]


async def test_client_update_replaces_text_and_clears_buttons() -> None:
    web = FakeWeb()
    await _client(web).update_message(DM, "1.2", "done")
    assert web.calls == [
        ("chat_update", {"channel": DM, "ts": "1.2", "text": "done", "blocks": []})
    ]


async def test_client_reports_auth_and_api_errors_without_secrets() -> None:
    with pytest.raises(SlackAuthError, match="invalid_auth"):
        await _client(FakeWeb("invalid_auth")).auth_test()
    with pytest.raises(SlackError, match="channel_not_found") as raised:
        await _client(FakeWeb("channel_not_found")).send_message(DM, "x")
    assert not isinstance(raised.value, SlackAuthError)
    assert "xoxb" not in str(raised.value)


async def test_client_auth_test_returns_the_workspace() -> None:
    info = await _client(FakeWeb()).auth_test()
    assert (info.team_id, info.user_id) == (TEAM, "UBOT")


async def test_client_download_sends_the_token_and_enforces_the_limit() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"x" * 100, headers={"content-type": "application/pdf"})

    client = _client(FakeWeb(), handler)
    url = "https://files.slack.com/files-pri/T-F/bill.pdf"
    assert await client.download(url, 1000) == b"x" * 100
    assert seen[0].headers["Authorization"] == "Bearer xoxb-test"
    with pytest.raises(FileTooLarge):
        await client.download(url, 50)


@pytest.mark.parametrize(
    "url",
    ["https://evil.example.com/a", "http://files.slack.com/a", "https://slack.com.evil.io/a"],
)
async def test_client_never_sends_the_token_to_other_hosts(url: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not be requested")

    with pytest.raises(SlackError, match="non-Slack"):
        await _client(FakeWeb(), handler).download(url, 1000)


async def test_client_download_detects_a_sign_in_page_and_errors() -> None:
    html_page = httpx.Response(200, text="<html>", headers={"content-type": "text/html"})
    with pytest.raises(SlackError, match="files:read"):
        await _client(FakeWeb(), lambda r: html_page).download("https://files.slack.com/a", 10)
    with pytest.raises(SlackError, match="404"):
        await _client(FakeWeb(), lambda r: httpx.Response(404)).download(
            "https://files.slack.com/a", 10
        )


async def test_client_respond_only_uses_slack_hooks() -> None:
    posted: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posted.append(json.loads(request.content))
        return httpx.Response(200)

    client = _client(FakeWeb(), handler)
    await client.respond("https://hooks.slack.com/commands/T/1/x", "hi")
    assert posted == [{"response_type": "ephemeral", "text": "hi"}]
    with pytest.raises(SlackError):
        await client.respond("https://evil.example.com/x", "hi")


# ---- config ----------------------------------------------------------------

_ENV_KEYS = (
    "SLACK_BOT_TOKEN",
    "SLACK_APP_TOKEN",
    "SLACK_SIGNING_SECRET",
    "SLACK_MODE",
    "SLACK_COMMAND",
    "AGENT_ALLOWED_SLACK_USERS",
    "AGENT_ALLOWED_SLACK_TEAMS",
    "AGENT_SLACK_SILENT_UNTIL_LOGIN",
    "AGENT_SLACK_VOICE_MAX_MB",
)


@pytest.fixture
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _env(tmp_path: Path, extra: str) -> Path:
    path = tmp_path / ".env"
    path.write_text(
        "AGENT_DB_URL=postgresql://u:p@127.0.0.1:5566/db\nMCP_BEARER_KEY=k\n" + extra,
        encoding="utf-8",
    )
    return path


@pytest.mark.usefixtures("_clean_env")
def test_config_enables_slack_in_socket_mode(tmp_path: Path) -> None:
    settings = load_settings(
        _env(
            tmp_path,
            "SLACK_BOT_TOKEN=xoxb-1\nSLACK_APP_TOKEN=xapp-1\nSLACK_COMMAND=ask\n"
            "AGENT_ALLOWED_SLACK_USERS= u0123abcd , W0456EFGH\n"
            "AGENT_ALLOWED_SLACK_TEAMS=T0999ZZZZ\nAGENT_SLACK_SILENT_UNTIL_LOGIN=false\n"
            "AGENT_SLACK_VOICE_MAX_MB=4\n",
        )
    )
    assert settings.slack_enabled and settings.slack_mode == "socket"
    assert settings.slack_command == "/ask"
    assert settings.allowed_slack_users == ["U0123ABCD", "W0456EFGH"]
    assert settings.allowed_slack_teams == ["T0999ZZZZ"]
    assert not settings.slack_silent_until_login and settings.slack_voice_max_mb == 4


@pytest.mark.usefixtures("_clean_env")
def test_config_defaults_and_off_by_default(tmp_path: Path) -> None:
    assert not load_settings(_env(tmp_path, "")).slack_enabled
    settings = load_settings(_env(tmp_path, "SLACK_BOT_TOKEN=xoxb-1\nSLACK_APP_TOKEN=xapp-1\n"))
    assert settings.slack_command == "/odoo" and settings.slack_silent_until_login
    assert settings.slack_voice_max_mb == 2 and settings.allowed_slack_teams == []


@pytest.mark.usefixtures("_clean_env")
def test_config_http_mode_needs_the_signing_secret_not_the_app_token(tmp_path: Path) -> None:
    settings = load_settings(
        _env(tmp_path, "SLACK_BOT_TOKEN=xoxb-1\nSLACK_MODE=http\nSLACK_SIGNING_SECRET=s\n")
    )
    assert settings.slack_mode == "http" and settings.slack_signing_secret == "s"


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.parametrize(
    ("extra", "names"),
    [
        ("SLACK_BOT_TOKEN=xoxb-1\n", "SLACK_APP_TOKEN"),
        ("SLACK_APP_TOKEN=xapp-1\n", "SLACK_BOT_TOKEN"),
        ("SLACK_BOT_TOKEN=xoxp-1\nSLACK_APP_TOKEN=xapp-1\n", "xoxp-"),
        ("SLACK_BOT_TOKEN=nonsense\nSLACK_APP_TOKEN=xapp-1\n", "SLACK_BOT_TOKEN"),
        ("SLACK_BOT_TOKEN=xoxb-1\nSLACK_APP_TOKEN=xoxb-2\n", "SLACK_APP_TOKEN"),
        ("SLACK_BOT_TOKEN=xoxb-1\nSLACK_MODE=http\n", "SLACK_SIGNING_SECRET"),
        ("SLACK_BOT_TOKEN=xoxb-1\nSLACK_APP_TOKEN=xapp-1\nSLACK_MODE=pigeon\n", "SLACK_MODE"),
        (
            "SLACK_BOT_TOKEN=xoxb-1\nSLACK_APP_TOKEN=xapp-1\nAGENT_ALLOWED_SLACK_TEAMS=U123\n",
            "AGENT_ALLOWED_SLACK_TEAMS",
        ),
        (
            "SLACK_BOT_TOKEN=xoxb-1\nSLACK_APP_TOKEN=xapp-1\nAGENT_ALLOWED_SLACK_USERS=bob\n",
            "AGENT_ALLOWED_SLACK_USERS",
        ),
    ],
)
def test_config_rejects_bad_slack_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: str, names: str
) -> None:
    with pytest.raises(SystemExit) as raised:
        load_settings(_env(tmp_path, extra))
    assert names in str(raised.value)


# ---- webhook route (HTTP mode) ---------------------------------------------

HTTP_SETTINGS = SETTINGS.model_copy(update={"slack_mode": "http", "slack_signing_secret": SECRET})


def _webhook(
    settings: Settings = HTTP_SETTINGS, *, on: bool = True
) -> tuple[httpx.AsyncClient, list[tuple[str, dict[str, Any]]]]:
    received: list[tuple[str, dict[str, Any]]] = []
    app = callback_server.build_app(
        None,  # type: ignore[arg-type]
        settings,
        on_slack=(lambda envelope, payload: received.append((envelope, dict(payload))))
        if on
        else None,
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t"), received


async def test_webhook_answers_the_url_verification_challenge() -> None:
    client, received = _webhook()
    body = json.dumps({"type": "url_verification", "challenge": "abc"}).encode()
    async with client:
        response = await client.post(
            "/webhook/slack",
            content=body,
            headers={**_sign(body), "content-type": "application/json"},
        )
    assert (response.status_code, response.json()) == (200, {"challenge": "abc"})
    assert received == []


async def test_webhook_forwards_signed_events() -> None:
    client, received = _webhook()
    body = json.dumps({"type": "event_callback", "event_id": "Ev1", "event": {}}).encode()
    async with client:
        response = await client.post(
            "/webhook/slack",
            content=body,
            headers={**_sign(body), "content-type": "application/json"},
        )
    assert response.status_code == 200
    assert received == [
        (ENVELOPE_EVENTS, {"type": "event_callback", "event_id": "Ev1", "event": {}})
    ]


async def test_webhook_forwards_slash_commands_and_interactions() -> None:
    client, received = _webhook()
    form = {"Content-Type": "application/x-www-form-urlencoded"}
    command = urlencode({"command": "/odoo", "text": "login", "user_id": USER}).encode()
    interaction = urlencode({"payload": json.dumps({"type": "block_actions"})}).encode()
    async with client:
        first = await client.post(
            "/webhook/slack", content=command, headers={**_sign(command), **form}
        )
        second = await client.post(
            "/webhook/slack", content=interaction, headers={**_sign(interaction), **form}
        )
    assert (first.status_code, second.status_code) == (200, 200)
    assert received == [
        (ENVELOPE_COMMAND, {"command": "/odoo", "text": "login", "user_id": USER}),
        (ENVELOPE_INTERACTIVE, {"type": "block_actions"}),
    ]


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"X-Slack-Request-Timestamp": "1", "X-Slack-Signature": "v0=forged"},
        _sign(b"another body"),
        _sign(b'{"a": 1}', str(int(time.time()) - 3600)),
    ],
    ids=["unsigned", "forged", "wrong-body", "replayed"],
)
async def test_webhook_refuses_unverified_posts(headers: dict[str, str]) -> None:
    client, received = _webhook()
    async with client:
        response = await client.post(
            "/webhook/slack",
            content=b'{"a": 1}',
            headers={**headers, "content-type": "application/json"},
        )
    assert response.status_code == 401 and received == []


async def test_webhook_is_closed_in_socket_mode() -> None:
    client, received = _webhook(SETTINGS, on=False)
    body = b"{}"
    async with client:
        response = await client.post(
            "/webhook/slack",
            content=body,
            headers={**_sign(body), "content-type": "application/json"},
        )
    assert response.status_code == 401 and received == []


async def test_webhook_rejects_malformed_bodies() -> None:
    client, received = _webhook()
    async with client:
        for body, kind in [
            (b"{", "application/json"),
            (b"[1]", "application/json"),
            (b"a=b", "text/plain"),
        ]:
            response = await client.post(
                "/webhook/slack", content=body, headers={**_sign(body), "content-type": kind}
            )
            assert response.status_code == 400
    assert received == []


# ---- the channel -----------------------------------------------------------


class FakeSlack:
    """Records what the channel would have sent to Slack."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fail_update = False
        self.downloaded: list[str] = []

    async def send_message(self, channel: str, text: str, blocks: Any = ()) -> str:
        self.calls.append(("send", (channel, text, list(blocks))))
        return f"{len(self.calls)}.000"

    async def update_message(self, channel: str, ts: str, text: str, blocks: Any = ()) -> None:
        if self.fail_update:
            raise SlackError("nope")
        self.calls.append(("update", (channel, ts, text, list(blocks))))

    async def send_ephemeral(self, channel: str, user: str, text: str) -> None:
        self.calls.append(("ephemeral", (channel, user, text)))

    async def respond(self, url: str, text: str) -> None:
        self.calls.append(("respond", (url, text)))

    async def upload_file(self, channel: str, path: Path, title: str, comment: str = "") -> None:
        self.calls.append(("file", (channel, path.name, title, comment)))

    async def download(self, url: str, max_bytes: int) -> bytes:
        self.downloaded.append(url)
        return b"%PDF"

    async def aclose(self) -> None:
        pass

    def of(self, kind: str) -> list[tuple[Any, ...]]:
        return [args for name, args in self.calls if name == kind]

    def texts(self) -> list[str]:
        return [args[1] if name == "send" else args[-1] for name, args in self.calls]


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
    store_dsn: str, *, bound: bool = True, settings: Settings = SETTINGS
) -> tuple[SlackChannel, FakeSlack]:
    store = BindingStore(store_dsn)
    await store.__aenter__()
    if bound:
        await store.save_binding(_binding())
    client = FakeSlack()
    channel = SlackChannel(
        store,
        settings,
        client,  # type: ignore[arg-type]
        workspace_id=TEAM,
    )
    return channel, client


async def _send(channel: SlackChannel, envelope: tuple[str, dict[str, Any]]) -> int:
    scheduled = channel.handle_envelope(*envelope)
    while channel._tasks:
        await asyncio.gather(*list(channel._tasks))
    return scheduled


@pytest.fixture
def answers(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replaces the agent; every call is recorded and answered with ``replies[0]``."""
    calls: list[dict[str, Any]] = []
    replies = [BotReply(text="Total is 42")]

    async def fake_answer(
        store: Any, settings: Any, chat_id: int, text: str, **kw: Any
    ) -> BotReply:
        calls.append({"kind": "answer", "chat_id": chat_id, "text": text, **kw})
        return replies[0]

    async def fake_resume(
        store: Any, settings: Any, chat_id: int, approved: bool, **kw: Any
    ) -> BotReply:
        calls.append({"kind": "resume", "approved": approved, **kw})
        return replies[0]

    monkeypatch.setattr(slack_bot, "answer_question", fake_answer)
    monkeypatch.setattr(slack_bot, "resume_pending", fake_resume)
    calls.append({"kind": "replies", "set": replies})
    return calls


def _asked(answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for c in answers if c["kind"] != "replies"]


async def test_question_from_a_bound_user_is_answered(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, client = await _channel(store_dsn)
    assert await _send(channel, _event("what is overdue?")) == 1
    assert [(c["chat_id"], c["text"]) for c in _asked(answers)] == [(CHAT_ID, "what is overdue?")]
    [sent] = client.of("send")
    assert sent[0] == DM and "Total is 42" in sent[1] and sent[2] == []
    assert await channel.store.get_chat_space(CHAT_ID) == DM


async def test_retried_deliveries_are_answered_once(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, _ = await _channel(store_dsn)
    envelope = _event("hello")
    assert await _send(channel, envelope) == 1
    assert await _send(channel, envelope) == 0
    assert len(_asked(answers)) == 1


async def test_unconnected_people_get_no_reply_by_default(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, client = await _channel(store_dsn, bound=False)
    for envelope in (_event("what is overdue?"), _event("help"), _command(""), _command("new")):
        await _send(channel, envelope)
    assert client.calls == [] and _asked(answers) == []


async def test_unconnected_people_get_the_login_button_when_silence_is_off(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    settings = SETTINGS.model_copy(
        update={"slack_silent_until_login": False, "odoo_public_url": "https://odoo.example.com"}
    )
    channel, client = await _channel(store_dsn, bound=False, settings=settings)
    await _send(channel, _event("what is overdue?"))
    [(_, text, blocks)] = client.of("send")
    assert "connect you to Odoo" in text
    assert blocks[-1]["elements"][0]["url"].startswith("https://odoo.example.com/")
    assert _asked(answers) == []


@pytest.mark.parametrize("envelope", [_event("login"), _command("login"), _event("Login")])
async def test_login_always_answers_with_a_button(
    store_dsn: str, envelope: tuple[str, dict[str, Any]]
) -> None:
    settings = SETTINGS.model_copy(update={"odoo_public_url": "https://odoo.example.com"})
    channel, client = await _channel(store_dsn, bound=False, settings=settings)
    await _send(channel, envelope)
    [(channel_id, text, blocks)] = client.of("send")
    assert channel_id == DM and "connect you to Odoo" in text
    button = blocks[-1]["elements"][0]
    assert button["text"]["text"] == "Log in to Odoo" and button["url"].startswith("https://")


async def test_login_over_plain_http_falls_back_to_a_text_link(store_dsn: str) -> None:
    settings = SETTINGS.model_copy(update={"odoo_public_url": "http://odoo:8160"})
    channel, client = await _channel(store_dsn, bound=False, settings=settings)
    await _send(channel, _event("login"))
    [(_, text, blocks)] = client.of("send")
    assert "http://odoo:8160/mcp/oauth/authorize" in text and blocks == []


async def test_login_when_connected_shows_status(store_dsn: str) -> None:
    channel, client = await _channel(store_dsn)
    await _send(channel, _event("login"))
    assert "connected as CFO Priya" in client.texts()[0]


async def test_slash_command_text_is_a_command_or_a_question(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, client = await _channel(store_dsn)
    await _send(channel, _command(""))
    assert "watch <condition>" in client.texts()[-1]
    await _send(channel, _command("show recent orders"))
    assert [c["text"] for c in _asked(answers)] == ["show recent orders"]


async def test_help_and_new_work_typed_bare(store_dsn: str, answers: list[dict[str, Any]]) -> None:
    channel, client = await _channel(store_dsn)
    await _send(channel, _event("help"))
    await _send(channel, _event("new"))
    assert "login" in client.texts()[0] and "Fresh start" in client.texts()[1]
    assert _asked(answers) == []


async def test_logout_disconnects(store_dsn: str, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_logout(store: BindingStore, settings: Settings, chat_id: int) -> bool:
        await store.delete_binding(chat_id)
        return True

    monkeypatch.setattr(slack_bot, "logout", fake_logout)
    channel, client = await _channel(store_dsn)
    await _send(channel, _event("logout"))
    assert "Disconnected" in client.texts()[-1]
    assert await channel.store.get_binding(CHAT_ID) is None


async def test_watch_alerts_are_managed_through_the_slash_command(store_dsn: str) -> None:
    channel, client = await _channel(store_dsn)
    await _send(channel, _command("watch invoices over 60 days overdue"))
    assert "Watching: invoices over 60 days overdue" in client.texts()[-1]
    await _send(channel, _event("watchers"))
    assert "1. invoices over 60 days overdue" in client.texts()[-1]
    await _send(channel, _command("unwatch 1"))
    assert client.texts()[-1].startswith("Removed")


async def test_channels_get_a_private_only_notice_visible_to_the_sender(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, client = await _channel(store_dsn)
    await _send(channel, _event("<@UBOT> hi", event_type="app_mention", channel="C0123ABCD"))
    await _send(channel, _command("what is overdue?", channel="C0123ABCD"))
    assert client.of("ephemeral") == [("C0123ABCD", USER, DIRECT_ONLY_REPLY)]
    assert client.of("respond") == [("https://hooks.slack.com/commands/T/1/x", DIRECT_ONLY_REPLY)]
    assert client.of("send") == [] and _asked(answers) == []


async def test_user_allowlist_blocks_silently(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    settings = SETTINGS.model_copy(update={"allowed_slack_users": ["U9999ZZZZ"]})
    channel, client = await _channel(store_dsn, settings=settings)
    await _send(channel, _event("hi"))
    assert client.calls == [] and _asked(answers) == []


async def test_other_workspaces_are_ignored_unless_listed(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, client = await _channel(store_dsn)
    await _send(channel, _event("hi", user_team="T9999ZZZZ"))
    assert client.calls == [] and _asked(answers) == []

    settings = SETTINGS.model_copy(update={"allowed_slack_teams": ["T9999ZZZZ"]})
    channel, client = await _channel(store_dsn, settings=settings)
    await _send(channel, _event("hi", user_team="T9999ZZZZ"))
    assert len(_asked(answers)) == 1


async def test_stale_events_are_dropped(store_dsn: str, answers: list[dict[str, Any]]) -> None:
    channel, client = await _channel(store_dsn)
    old = f"{time.time() - 7200:.6f}"
    await _send(channel, _event("hi", ts=old))
    assert client.calls == [] and _asked(answers) == []


async def test_agent_failure_is_reported(store_dsn: str, monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(*args: Any, **kwargs: Any) -> BotReply:
        raise RuntimeError("model down")

    monkeypatch.setattr(slack_bot, "answer_question", boom)
    channel, client = await _channel(store_dsn)
    await _send(channel, _event("hi"))
    assert ERROR_REPLY in client.texts()[-1]


async def test_slow_answer_replaces_the_working_message(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(slack_bot, "_WORKING_AFTER_SECONDS", 0.01)

    async def slow(*args: Any, **kwargs: Any) -> BotReply:
        await asyncio.sleep(0.1)
        return BotReply(text="Done")

    monkeypatch.setattr(slack_bot, "answer_question", slow)
    channel, client = await _channel(store_dsn)
    await _send(channel, _event("hi"))
    [working] = client.of("send")
    [edit] = client.of("update")
    assert "Working" in working[1]
    assert edit[:2] == (DM, "1.000") and "Done" in edit[2]


async def test_answer_falls_back_to_a_new_message_when_the_edit_fails(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(slack_bot, "_WORKING_AFTER_SECONDS", 0.01)

    async def slow(*args: Any, **kwargs: Any) -> BotReply:
        await asyncio.sleep(0.1)
        return BotReply(text="Done")

    monkeypatch.setattr(slack_bot, "answer_question", slow)
    channel, client = await _channel(store_dsn)
    client.fail_update = True
    await _send(channel, _event("hi"))
    assert "Done" in client.texts()[-1]


async def test_answers_are_escaped_so_they_cannot_ping_the_workspace(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    answers[0]["set"][0] = BotReply(text="Notify <!channel> now")
    channel, client = await _channel(store_dsn)
    await _send(channel, _event("hi"))
    assert "<!channel>" not in client.texts()[-1]


async def test_grounded_answer_carries_follow_up_and_feedback_buttons(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    answers[0]["set"][0] = BotReply(
        text="| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |", grounded=True, trace_id=TRACE
    )
    channel, client = await _channel(store_dsn)
    await _send(channel, _event("list them"))
    [(_, _, blocks)] = client.of("send")
    data = [e["value"] for e in blocks[-1]["elements"]]
    assert "fu:chart" in data and f"fb:1:{TRACE}" in data and f"fb:0:{TRACE}" in data


async def test_approval_prompt_has_green_approve_red_cancel_and_edit(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    from ont_agent.agent import PendingAction

    pending = PendingAction(
        actions=["send_email"], names=["send_email"], arguments=[{"to": "a@b.com"}]
    )
    answers[0]["set"][0] = BotReply(
        text="<b>Send this email?</b>", pending=pending, trace_id=TRACE
    )
    channel, client = await _channel(store_dsn)
    await _send(channel, _event("email it"))
    [(_, text, blocks)] = client.of("send")
    assert text == "*Send this email?*"
    elements = blocks[-1]["elements"]
    by_value = {e["value"].split(":")[1]: e for e in elements}
    assert by_value["approve"]["style"] == "primary"
    assert by_value["reject"]["style"] == "danger"
    assert "edit" in by_value


async def test_follow_up_tap_asks_the_follow_up_question(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, _ = await _channel(store_dsn)
    await _send(channel, _tap("fu:chart"))
    assert [c["text"] for c in _asked(answers)] == ["Show that as a chart."]


async def test_approve_tap_removes_the_buttons_and_resumes(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, client = await _channel(store_dsn)
    await _send(channel, _tap(f"confirm:approve:{TRACE}"))
    [edit] = client.of("update")
    assert edit[:2] == (DM, "111.222") and edit[3] == []
    assert edit[2].endswith("✅ Approved.") and edit[2].startswith("Approval needed")
    assert [(c["kind"], c["approved"]) for c in _asked(answers)] == [("resume", True)]


async def test_cancel_tap_resumes_with_rejection(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, _ = await _channel(store_dsn)
    await _send(channel, _tap("confirm:reject"))
    assert [(c["kind"], c["approved"]) for c in _asked(answers)] == [("resume", False)]


async def test_edit_tap_makes_the_next_message_the_revision(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, _ = await _channel(store_dsn)
    await _send(channel, _tap("confirm:edit"))
    assert CHAT_ID in channel.awaiting_edit
    await _send(channel, _event("make it 5 units"))
    [call] = _asked(answers)
    assert (call["kind"], call["revision"]) == ("resume", "make it 5 units")
    assert CHAT_ID not in channel.awaiting_edit


async def test_taps_from_unconnected_people_do_nothing(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, client = await _channel(store_dsn, bound=False)
    await _send(channel, _tap("confirm:approve"))
    assert client.calls == [] and _asked(answers) == []


async def test_charts_are_uploaded_and_the_temp_file_removed(
    store_dsn: str, answers: list[dict[str, Any]], tmp_path: Path
) -> None:
    chart = tmp_path / "chart.png"
    chart.write_bytes(b"png")
    answers[0]["set"][0] = BotReply(
        text="Here", attachments=[Artifact(chart, "photo", "Sales by month")]
    )
    channel, client = await _channel(store_dsn)
    await _send(channel, _event("chart it"))
    assert client.of("file") == [(DM, "chart.png", "chart.png", "Sales by month")]
    assert not chart.exists()


def _upload(name: str, mime: str, size: int = 1000, **extra: Any) -> dict[str, Any]:
    return {
        "url_private_download": f"https://files.slack.com/files-pri/T-F/{name}",
        "mimetype": mime,
        "name": name,
        "size": size,
        **extra,
    }


async def test_pdf_upload_is_read_with_the_caption_as_question(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, client = await _channel(store_dsn)
    await _send(
        channel,
        _event(
            "what is the total?",
            subtype="file_share",
            files=[_upload("bill.pdf", "application/pdf")],
        ),
    )
    [call] = _asked(answers)
    assert call["text"] == "what is the total?"
    assert (call["attachment"].filename, call["attachment"].data) == ("bill.pdf", b"%PDF")
    assert client.downloaded == ["https://files.slack.com/files-pri/T-F/bill.pdf"]


async def test_upload_without_a_caption_asks_the_default_question(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, client = await _channel(store_dsn)
    await _send(
        channel,
        _event("", subtype="file_share", files=[_upload("p.png", "image/png")]),
    )
    [call] = _asked(answers)
    assert call["text"] == "What is in this document?" and "p.png" in client.texts()[0]


async def test_unsupported_and_oversized_uploads_are_declined_before_download(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, client = await _channel(store_dsn)
    await _send(channel, _event("", subtype="file_share", files=[_upload("v.mp4", "video/mp4")]))
    big = _upload("big.pdf", "application/pdf", size=50 * 1024 * 1024)
    await _send(channel, _event("", subtype="file_share", files=[big]))
    assert "video" in client.texts()[0] and "MB" in client.texts()[1]
    assert client.downloaded == [] and _asked(answers) == []


async def test_unreadable_shared_files_get_the_unsupported_reply(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, client = await _channel(store_dsn)
    await _send(
        channel,
        _event("", subtype="file_share", files=[{"file_access": "check_file_info", "id": "F1"}]),
    )
    assert client.texts() == [slack_bot.UNSUPPORTED_REPLY] and _asked(answers) == []


async def test_voice_clip_is_transcribed_and_asked(
    store_dsn: str, answers: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    from ont_agent.transcribe import Transcript

    heard: dict[str, Any] = {}

    async def fake_transcribe(settings: Any, audio: bytes, **kw: Any) -> Transcript:
        heard.update(audio=audio, **kw)
        return Transcript(text="show overdue invoices")

    monkeypatch.setattr(slack_bot, "transcribe", fake_transcribe)
    channel, client = await _channel(store_dsn)
    clip = _upload("clip.m4a", "audio/mp4", size=1000, subtype="slack_audio")
    await _send(channel, _event("", subtype="file_share", files=[clip]))
    assert (heard["filename"], heard["mime"]) == ("clip.m4a", "audio/mp4")
    assert "I heard: show overdue invoices" in client.texts()[0]
    assert [c["text"] for c in _asked(answers)] == ["show overdue invoices"]


async def test_voice_clip_over_the_limit_is_refused_without_download(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, client = await _channel(store_dsn)
    clip = _upload("clip.m4a", "audio/mp4", size=3 * 1024 * 1024)
    await _send(channel, _event("", subtype="file_share", files=[clip]))
    assert "2 MB" in client.texts()[0]
    assert client.downloaded == [] and _asked(answers) == []


async def test_proactive_message_goes_to_the_remembered_dm(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, client = await _channel(store_dsn)
    await _send(channel, _event("hi"))
    client.calls.clear()
    await channel.send_plain(CHAT_ID, "**Brief** ready")
    assert client.of("send") == [(DM, "*Brief* ready", [])]

    other, other_client = await _channel(store_dsn)
    await other.send_plain(slack_chat_id("U0999ZZZZ"), "lost")
    assert other_client.calls == []


def test_connect_prompt_mentions_the_login_word() -> None:
    assert "`login`" in SLACK_CONNECT_PROMPT
