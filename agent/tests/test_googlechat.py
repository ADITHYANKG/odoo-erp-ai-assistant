"""Tests for the Google Chat channel: ids, token verification, events, sending."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from ont_agent import callback_server, googlechat_bot
from ont_agent.bot import ERROR_REPLY, BotReply
from ont_agent.channels import (
    GOOGLE_CHAT_OFFSET,
    WHATSAPP_OFFSET,
    googlechat_chat_id,
    is_google_chat,
    is_whatsapp,
    whatsapp_chat_id,
)
from ont_agent.config import Settings, load_settings
from ont_agent.googlechat import (
    KIND_ADDED,
    KIND_TAP,
    KIND_TEXT,
    AttachmentTooLarge,
    GoogleChatClient,
    GoogleChatError,
    GoogleChatVerifier,
    button_cards,
    parse_event,
)
from ont_agent.googlechat_bot import (
    DIRECT_ONLY_REPLY,
    GC_CONNECT_PROMPT,
    NO_FILES_REPLY,
    GoogleChatChannel,
)
from ont_agent.googlechat_format import html_to_googlechat, render_googlechat
from ont_agent.render import Artifact
from ont_agent.store import Binding, BindingStore

USER = "users/104512345678901234567"
SPACE = "spaces/AAAA1"
ENDPOINT = "https://agent.example.com/webhook/googlechat"
PROJECT = "123456789012"
SENDER = f"service-{PROJECT}@gcp-sa-gsuiteaddons.iam.gserviceaccount.com"
CHAT_ID = googlechat_chat_id(USER)
TRACE = "b" * 32

SETTINGS = Settings(
    mcp_url="http://odoo:8160/mcp",
    mcp_bearer_key="k",
    oauth_client_id="ont_client123",
    googlechat_project_number=PROJECT,
    googlechat_credentials_file="unused.json",
    googlechat_endpoint_url=ENDPOINT,
)


def _key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _pem(key: rsa.RSAPrivateKey) -> str:
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


# ---- ids -------------------------------------------------------------------


def test_googlechat_chat_id_is_stable_and_in_its_own_range() -> None:
    assert googlechat_chat_id(USER) == googlechat_chat_id(USER)
    assert googlechat_chat_id(USER) != googlechat_chat_id("users/2")
    assert is_google_chat(CHAT_ID)
    assert not is_whatsapp(CHAT_ID)
    assert CHAT_ID < 2**63  # fits BIGINT


def test_ranges_do_not_overlap() -> None:
    widest_whatsapp = whatsapp_chat_id("9" * 15)
    assert is_whatsapp(widest_whatsapp) and not is_google_chat(widest_whatsapp)
    assert WHATSAPP_OFFSET + 10**15 < GOOGLE_CHAT_OFFSET
    assert not is_google_chat(8967803452)


@pytest.mark.parametrize("bad", ["", "users/", "spaces/1", "1234"])
def test_googlechat_chat_id_rejects_bad_names(bad: str) -> None:
    with pytest.raises(ValueError):
        googlechat_chat_id(bad)


# ---- token verification ----------------------------------------------------


class _Certs:
    """Serves one signing key the way Google's JWKS endpoint does."""

    def __init__(self) -> None:
        self.key = _key()
        self.hits = 0
        jwk = jwt.algorithms.RSAAlgorithm.to_jwk(self.key.public_key(), as_dict=True)
        self.jwks = {"keys": [{**jwk, "kid": "k1", "alg": "RS256", "use": "sig"}]}

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.hits += 1
        return httpx.Response(200, json=self.jwks)

    def token(self, **overrides: Any) -> str:
        claims: dict[str, Any] = {
            "iss": "https://accounts.google.com",
            "aud": ENDPOINT,
            "email": SENDER,
            "email_verified": True,
            "exp": int(time.time()) + 300,
        }
        claims.update(overrides)
        kid = claims.pop("_kid", "k1")
        return jwt.encode(claims, _pem(self.key), algorithm="RS256", headers={"kid": kid})


def _verifier(certs: _Certs) -> GoogleChatVerifier:
    http = httpx.AsyncClient(transport=httpx.MockTransport(certs.handler))
    return GoogleChatVerifier(ENDPOINT, SENDER, http)


async def test_verifier_accepts_a_genuine_token() -> None:
    certs = _Certs()
    assert await _verifier(certs).verify(f"Bearer {certs.token()}")


@pytest.mark.parametrize(
    "overrides",
    [
        {"aud": "https://evil.example.com/hook"},
        {"email": "attacker@example.com"},
        {"email_verified": False},
        {"exp": int(time.time()) - 10},
        {"iss": "https://evil.example.com"},
        {"_kid": "unknown"},
    ],
    ids=["audience", "email", "unverified", "expired", "issuer", "kid"],
)
async def test_verifier_rejects_bad_tokens(overrides: dict[str, Any]) -> None:
    certs = _Certs()
    assert not await _verifier(certs).verify(f"Bearer {certs.token(**overrides)}")


async def test_verifier_rejects_wrong_key_and_missing_header() -> None:
    certs, other = _Certs(), _Certs()
    verifier = _verifier(certs)
    assert not await verifier.verify(f"Bearer {other.token()}")
    for header in (None, "", "Bearer ", "Bearer not-a-jwt", "Basic abc"):
        assert not await verifier.verify(header)


async def test_verifier_caches_keys_and_limits_refetching() -> None:
    certs = _Certs()
    verifier = _verifier(certs)
    assert await verifier.verify(f"Bearer {certs.token()}")
    assert await verifier.verify(f"Bearer {certs.token()}")
    assert not await verifier.verify(f"Bearer {certs.token(_kid='nope')}")
    assert certs.hits == 1


# ---- event parsing ---------------------------------------------------------


def _event(section: str, body: dict[str, Any], **user: Any) -> dict[str, Any]:
    return {
        "commonEventObject": {"hostApp": "CHAT"},
        "chat": {
            "user": {
                "name": USER,
                "displayName": "Priya",
                "email": "Priya@Acme.com",
                "type": "HUMAN",
                **user,
            },
            "eventTime": "2026-10-13T08:30:00.123456789Z",
            section: {"space": {"name": SPACE, "spaceType": "DIRECT_MESSAGE"}, **body},
        },
    }


def _text_event(text: str = "hello", **message: Any) -> dict[str, Any]:
    return _event(
        "messagePayload",
        {"message": {"name": f"{SPACE}/messages/{uuid.uuid4().hex}", "text": text, **message}},
    )


def test_parse_text_message() -> None:
    event = parse_event(_text_event("  what is overdue?  "))
    assert event is not None
    assert (event.kind, event.text, event.space, event.direct) == (
        KIND_TEXT,
        "what is overdue?",
        SPACE,
        True,
    )
    assert event.user_name == USER and event.email == "priya@acme.com"
    assert event.timestamp > 0


def test_parse_added_and_tap() -> None:
    added = parse_event(_event("addedToSpacePayload", {}))
    assert added is not None and added.kind == KIND_ADDED

    payload = _event(
        "buttonClickedPayload",
        {"message": {"name": f"{SPACE}/messages/m1", "text": "Approve?"}},
    )
    payload["commonEventObject"]["parameters"] = {"data": "confirm:approve"}
    tap = parse_event(payload)
    assert tap is not None
    assert (tap.kind, tap.tap, tap.message_name, tap.text) == (
        KIND_TAP,
        "confirm:approve",
        f"{SPACE}/messages/m1",
        "Approve?",
    )


def test_parse_app_command_uses_the_message_text() -> None:
    payload = _event(
        "appCommandPayload", {"message": {"name": f"{SPACE}/messages/m2", "text": "/help"}}
    )
    event = parse_event(payload)
    assert event is not None and event.text == "/help"


def test_parse_flags_spaces_that_are_not_direct() -> None:
    payload = _text_event()
    payload["chat"]["messagePayload"]["space"]["spaceType"] = "SPACE"
    event = parse_event(payload)
    assert event is not None and not event.direct


def test_parse_attachments() -> None:
    upload = {
        "contentName": "bill.pdf",
        "contentType": "application/pdf",
        "source": "UPLOADED_CONTENT",
        "attachmentDataRef": {"resourceName": "res/abc"},
    }
    drive = {"contentName": "sheet", "source": "DRIVE_FILE"}
    event = parse_event(_text_event("", attachment=[upload, drive]))
    assert event is not None
    assert [a.resource_name for a in event.attachments] == ["res/abc"]
    assert event.attachments[0].content_type == "application/pdf"
    assert event.has_unreadable_file


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"chat": {}},
        _event("removedFromSpacePayload", {}),
        _event("messagePayload", {"message": {"name": "m", "text": "x"}}, type="BOT"),
        _event(
            "messagePayload", {"message": {"name": "m", "text": "x"}}, name="spaces/not-a-user"
        ),
        {"chat": {"user": {"name": USER}, "messagePayload": {"message": {"name": "m"}}}},
    ],
    ids=["empty", "no-payload", "removed", "bot", "bad-user", "no-space"],
)
def test_parse_ignores_what_needs_no_answer(payload: dict[str, Any]) -> None:
    assert parse_event(payload) is None


# ---- formatting ------------------------------------------------------------


def test_format_bold_italic_code_and_links() -> None:
    markup = (
        '<b>Total</b> and <i>note</i>, see <a href="https://odoo/x?a=1&amp;b=2">INV/1</a> '
        '<a href="https://u.rl">https://u.rl</a> <code>a*b</code>'
    )
    assert html_to_googlechat(markup) == (
        "*Total* and _note_, see <https://odoo/x?a=1&b=2|INV/1> https://u.rl `a*b`"
    )


def test_format_code_block_is_literal() -> None:
    assert html_to_googlechat("<pre><code>x *y* &lt;z&gt;</code></pre>") == "```\nx *y* <z>\n```"


def test_render_googlechat_turns_markdown_bold() -> None:
    assert "*Total*" in render_googlechat("**Total** is 5")


def test_link_label_cannot_break_the_syntax() -> None:
    assert html_to_googlechat('<a href="https://x.y">a|b&gt;c</a>') == "<https://x.y|a/bc>"


# ---- buttons ---------------------------------------------------------------


def test_button_cards_carry_the_tap_data_back_to_the_endpoint() -> None:
    [card] = button_cards([("fu:chart", "Chart")], ENDPOINT, links=[("https://o/login", "Log in")])
    buttons = card["card"]["sections"][0]["widgets"][0]["buttonList"]["buttons"]
    assert buttons[0]["onClick"]["action"] == {
        "function": ENDPOINT,
        "parameters": [{"key": "data", "value": "fu:chart"}],
    }
    assert buttons[1]["onClick"] == {"openLink": {"url": "https://o/login"}}
    assert button_cards([], ENDPOINT) == []


# ---- client ----------------------------------------------------------------


def _credentials(tmp_path: Path) -> Path:
    path = tmp_path / "sa.json"
    path.write_text(
        json.dumps(
            {
                "client_email": "bot@proj.iam.gserviceaccount.com",
                "private_key": _pem(_key()),
                "token_uri": "https://oauth2.example/token",
            }
        )
    )
    return path


class _Google:
    """Stands in for the OAuth token endpoint and the Chat API."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.status = 200

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.host == "oauth2.example":
            return httpx.Response(200, json={"access_token": "ya29.tok", "expires_in": 3600})
        if request.url.path.startswith("/v1/media/"):
            return httpx.Response(self.status, content=b"x" * 50)
        return httpx.Response(self.status, json={"name": f"{SPACE}/messages/new"})

    def chat_calls(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.host == "chat.googleapis.com"]


def _client(tmp_path: Path, google: _Google) -> GoogleChatClient:
    settings = SETTINGS.model_copy(
        update={"googlechat_credentials_file": str(_credentials(tmp_path))}
    )
    return GoogleChatClient(
        settings, httpx.AsyncClient(transport=httpx.MockTransport(google.handler))
    )


async def test_client_signs_in_once_and_sends_with_a_bearer_token(tmp_path: Path) -> None:
    google = _Google()
    client = _client(tmp_path, google)
    name = await client.send_message(SPACE, "hi", [{"cardId": "c"}])
    await client.send_message(SPACE, "again")

    assert name == f"{SPACE}/messages/new"
    token_calls = [r for r in google.requests if r.url.host == "oauth2.example"]
    assert len(token_calls) == 1
    assert b"jwt-bearer" in token_calls[0].content
    first = google.chat_calls()[0]
    assert first.headers["Authorization"] == "Bearer ya29.tok"
    assert first.url.path == f"/v1/{SPACE}/messages"
    assert json.loads(first.content) == {"text": "hi", "cardsV2": [{"cardId": "c"}]}


async def test_client_update_replaces_text_and_buttons(tmp_path: Path) -> None:
    google = _Google()
    await _client(tmp_path, google).update_message(f"{SPACE}/messages/m1", "done")
    [call] = google.chat_calls()
    assert call.method == "PATCH"
    assert call.url.params["updateMask"] == "text,cardsV2"
    assert json.loads(call.content) == {"text": "done", "cardsV2": []}


async def test_client_clamps_text_and_reports_api_errors(tmp_path: Path) -> None:
    google = _Google()
    client = _client(tmp_path, google)
    await client.send_message(SPACE, "x" * 9000)
    assert len(json.loads(google.chat_calls()[0].content)["text"]) == 4096
    google.status = 403
    with pytest.raises(GoogleChatError, match="403"):
        await client.send_message(SPACE, "hi")


async def test_client_download_enforces_the_size_limit(tmp_path: Path) -> None:
    google = _Google()
    client = _client(tmp_path, google)
    assert await client.download("res/abc", 100) == b"x" * 50
    assert google.chat_calls()[0].url.params["alt"] == "media"
    with pytest.raises(AttachmentTooLarge):
        await client.download("res/abc", 10)


# ---- config ----------------------------------------------------------------


@pytest.fixture
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "GOOGLE_CHAT_PROJECT_NUMBER",
        "GOOGLE_CHAT_CREDENTIALS_FILE",
        "GOOGLE_CHAT_ENDPOINT_URL",
        "AGENT_ALLOWED_GOOGLE_CHAT_EMAILS",
    ):
        monkeypatch.delenv(key, raising=False)


def _env(tmp_path: Path, extra: str) -> Path:
    path = tmp_path / ".env"
    path.write_text(
        "AGENT_DB_URL=postgresql://u:p@127.0.0.1:5566/db\nMCP_BEARER_KEY=k\n" + extra,
        encoding="utf-8",
    )
    return path


@pytest.mark.usefixtures("_clean_env")
def test_config_enables_google_chat(tmp_path: Path) -> None:
    creds = _credentials(tmp_path)
    settings = load_settings(
        _env(
            tmp_path,
            f"GOOGLE_CHAT_PROJECT_NUMBER={PROJECT}\nGOOGLE_CHAT_CREDENTIALS_FILE={creds}\n"
            "AGENT_PUBLIC_URL=https://agent.example.com/\n"
            "AGENT_ALLOWED_GOOGLE_CHAT_EMAILS= Priya@Acme.com , bob@acme.com\n",
        )
    )
    assert settings.googlechat_enabled
    assert settings.googlechat_sender == SENDER
    assert settings.googlechat_audience == ENDPOINT
    assert settings.allowed_googlechat_emails == ["priya@acme.com", "bob@acme.com"]


@pytest.mark.usefixtures("_clean_env")
def test_config_is_off_by_default_and_audience_can_be_overridden(tmp_path: Path) -> None:
    assert not load_settings(_env(tmp_path, "")).googlechat_enabled
    assert (
        SETTINGS.model_copy(
            update={"googlechat_endpoint_url": "https://x.y/z/"}
        ).googlechat_audience
        == "https://x.y/z/"
    )


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.parametrize(
    "extra",
    [
        f"GOOGLE_CHAT_PROJECT_NUMBER={PROJECT}\n",
        "GOOGLE_CHAT_PROJECT_NUMBER=my-project\nGOOGLE_CHAT_CREDENTIALS_FILE=missing.json\n",
        f"GOOGLE_CHAT_PROJECT_NUMBER={PROJECT}\nGOOGLE_CHAT_CREDENTIALS_FILE=missing.json\n",
    ],
    ids=["no-file", "bad-number", "unreadable-file"],
)
def test_config_rejects_half_configured_google_chat(tmp_path: Path, extra: str) -> None:
    with pytest.raises(SystemExit):
        load_settings(_env(tmp_path, extra))


# ---- webhook route ---------------------------------------------------------


def _webhook(accept: bool) -> tuple[httpx.AsyncClient, list[dict[str, Any]]]:
    received: list[dict[str, Any]] = []

    async def verify(header: str | None) -> bool:
        return accept and header == "Bearer good"

    app = callback_server.build_app(
        None,  # type: ignore[arg-type]
        SETTINGS,
        on_googlechat=lambda payload: received.append(dict(payload)),
        verify_googlechat=verify,
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t"), received


async def test_webhook_acknowledges_verified_events() -> None:
    client, received = _webhook(accept=True)
    async with client:
        response = await client.post(
            "/webhook/googlechat", json={"chat": {}}, headers={"Authorization": "Bearer good"}
        )
    assert (response.status_code, response.json()) == (200, {})
    assert received == [{"chat": {}}]


@pytest.mark.parametrize("header", [None, "Bearer forged"], ids=["missing", "forged"])
async def test_webhook_refuses_unverified_events(header: str | None) -> None:
    client, received = _webhook(accept=True)
    async with client:
        headers = {"Authorization": header} if header else {}
        response = await client.post("/webhook/googlechat", json={"chat": {}}, headers=headers)
    assert response.status_code == 401
    assert received == []


async def test_webhook_is_closed_when_the_channel_is_off() -> None:
    app = callback_server.build_app(None, SETTINGS)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        response = await c.post(
            "/webhook/googlechat", json={}, headers={"Authorization": "Bearer x"}
        )
    assert response.status_code == 401


async def test_webhook_rejects_malformed_bodies() -> None:
    client, received = _webhook(accept=True)
    async with client:
        headers = {"Authorization": "Bearer good"}
        bad_json = await client.post("/webhook/googlechat", content=b"{", headers=headers)
        not_object = await client.post("/webhook/googlechat", json=[1], headers=headers)
    assert (bad_json.status_code, not_object.status_code) == (400, 400)
    assert received == []


# ---- the channel -----------------------------------------------------------


class FakeChat:
    """Records what the channel would have sent to Google."""

    endpoint = ENDPOINT

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fail_update = False

    async def send_message(self, space: str, text: str, cards: Any = ()) -> str:
        self.calls.append(("send", (space, text, list(cards))))
        return f"{space}/messages/{len(self.calls)}"

    async def update_message(self, name: str, text: str, cards: Any = ()) -> None:
        if self.fail_update:
            raise GoogleChatError("nope")
        self.calls.append(("update", (name, text, list(cards))))

    async def download(self, resource_name: str, max_bytes: int) -> bytes:
        return b"%PDF"

    async def aclose(self) -> None:
        pass

    def of(self, kind: str) -> list[tuple[Any, ...]]:
        return [args for name, args in self.calls if name == kind]

    def texts(self) -> list[str]:
        return [args[1] for _, args in self.calls]


class FakeVerifier:
    async def verify(self, header: str | None) -> bool:
        return True

    async def aclose(self) -> None:
        pass


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
) -> tuple[GoogleChatChannel, FakeChat]:
    store = BindingStore(store_dsn)
    await store.__aenter__()
    if bound:
        await store.save_binding(_binding())
    client = FakeChat()
    channel = GoogleChatChannel(
        store,
        settings,
        client,  # type: ignore[arg-type]
        FakeVerifier(),  # type: ignore[arg-type]
    )
    return channel, client


async def _drain(channel: GoogleChatChannel) -> None:
    while channel._tasks:
        await asyncio.gather(*list(channel._tasks))


async def _send(channel: GoogleChatChannel, payload: dict[str, Any]) -> int:
    scheduled = channel.handle_payload(payload)
    await _drain(channel)
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

    monkeypatch.setattr(googlechat_bot, "answer_question", fake_answer)
    monkeypatch.setattr(googlechat_bot, "resume_pending", fake_resume)
    calls.append({"kind": "replies", "set": replies})
    return calls


def _asked(answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for c in answers if c["kind"] != "replies"]


async def test_question_from_a_bound_user_is_answered(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, client = await _channel(store_dsn)
    assert await _send(channel, _text_event("what is overdue?")) == 1
    assert [(c["chat_id"], c["text"]) for c in _asked(answers)] == [(CHAT_ID, "what is overdue?")]
    [sent] = client.of("send")
    assert sent[0] == SPACE and "Total is 42" in sent[1]
    assert await channel.store.get_chat_space(CHAT_ID) == SPACE


async def test_retried_deliveries_are_answered_once(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, _ = await _channel(store_dsn)
    payload = _text_event("hello")
    assert await _send(channel, payload) == 1
    assert await _send(channel, payload) == 0
    assert len(_asked(answers)) == 1


async def test_unbound_user_is_asked_to_connect(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, client = await _channel(store_dsn, bound=False)
    await _send(channel, _text_event("what is overdue?"))
    assert client.texts() == [GC_CONNECT_PROMPT]
    assert _asked(answers) == []


async def test_start_offers_a_login_button(store_dsn: str) -> None:
    settings = SETTINGS.model_copy(update={"odoo_public_url": "https://odoo.example.com"})
    channel, client = await _channel(store_dsn, bound=False, settings=settings)
    await _send(channel, _text_event("/start"))
    [(_, text, cards)] = client.of("send")
    assert "connect you to Odoo" in text
    button = cards[0]["card"]["sections"][0]["widgets"][0]["buttonList"]["buttons"][0]
    assert button["onClick"]["openLink"]["url"].startswith("https://odoo.example.com/")


async def test_start_over_plain_http_falls_back_to_a_text_link(store_dsn: str) -> None:
    settings = SETTINGS.model_copy(update={"odoo_public_url": "http://odoo:8160"})
    channel, client = await _channel(store_dsn, bound=False, settings=settings)
    await _send(channel, _text_event("/start"))
    [(_, text, cards)] = client.of("send")
    assert "http://odoo:8160/mcp/oauth/authorize" in text and cards == []


async def test_spaces_get_one_private_only_notice(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, client = await _channel(store_dsn)
    for _ in range(2):
        payload = _text_event("hi")
        payload["chat"]["messagePayload"]["space"]["spaceType"] = "SPACE"
        await _send(channel, payload)
    assert client.texts() == [DIRECT_ONLY_REPLY]
    assert _asked(answers) == []


async def test_allowlist_blocks_silently(store_dsn: str, answers: list[dict[str, Any]]) -> None:
    settings = SETTINGS.model_copy(update={"allowed_googlechat_emails": ["boss@acme.com"]})
    channel, client = await _channel(store_dsn, settings=settings)
    await _send(channel, _text_event("hi"))
    assert client.calls == [] and _asked(answers) == []


async def test_stale_events_are_dropped(store_dsn: str, answers: list[dict[str, Any]]) -> None:
    channel, client = await _channel(store_dsn)
    old = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    await _send(channel, _text_event("hi", createTime=old))
    assert client.calls == [] and _asked(answers) == []


async def test_agent_failure_is_reported(store_dsn: str, monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(*args: Any, **kwargs: Any) -> BotReply:
        raise RuntimeError("model down")

    monkeypatch.setattr(googlechat_bot, "answer_question", boom)
    channel, client = await _channel(store_dsn)
    await _send(channel, _text_event("hi"))
    assert ERROR_REPLY in client.texts()[-1]


async def test_slow_answer_replaces_the_working_message(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(googlechat_bot, "_WORKING_AFTER_SECONDS", 0.01)

    async def slow(*args: Any, **kwargs: Any) -> BotReply:
        await asyncio.sleep(0.1)
        return BotReply(text="Done")

    monkeypatch.setattr(googlechat_bot, "answer_question", slow)
    channel, client = await _channel(store_dsn)
    await _send(channel, _text_event("hi"))
    [working] = client.of("send")
    [edit] = client.of("update")
    assert "Working" in working[1]
    assert edit[0] == f"{SPACE}/messages/1" and "Done" in edit[1]


async def test_answer_falls_back_to_a_new_message_when_the_edit_fails(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(googlechat_bot, "_WORKING_AFTER_SECONDS", 0.01)

    async def slow(*args: Any, **kwargs: Any) -> BotReply:
        await asyncio.sleep(0.1)
        return BotReply(text="Done")

    monkeypatch.setattr(googlechat_bot, "answer_question", slow)
    channel, client = await _channel(store_dsn)
    client.fail_update = True
    await _send(channel, _text_event("hi"))
    assert "Done" in client.texts()[-1]


async def test_grounded_answer_carries_follow_up_and_feedback_buttons(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    answers[0]["set"][0] = BotReply(
        text="| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |", grounded=True, trace_id=TRACE
    )
    channel, client = await _channel(store_dsn)
    await _send(channel, _text_event("list them"))
    [(_, _, cards)] = client.of("send")
    buttons = cards[0]["card"]["sections"][0]["widgets"][0]["buttonList"]["buttons"]
    data = [b["onClick"]["action"]["parameters"][0]["value"] for b in buttons]
    assert "fu:chart" in data and f"fb:1:{TRACE}" in data and f"fb:0:{TRACE}" in data


async def test_follow_up_tap_asks_the_follow_up_question(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, _ = await _channel(store_dsn)
    payload = _event(
        "buttonClickedPayload", {"message": {"name": f"{SPACE}/messages/m1", "text": "x"}}
    )
    payload["commonEventObject"]["parameters"] = {"data": "fu:chart"}
    await _send(channel, payload)
    assert [c["text"] for c in _asked(answers)] == ["Show that as a chart."]


def _tap(data: str, text: str = "Approval needed") -> dict[str, Any]:
    payload = _event(
        "buttonClickedPayload",
        {"message": {"name": f"{SPACE}/messages/m1", "text": text}},
    )
    payload["commonEventObject"]["parameters"] = {"data": data}
    return payload


async def test_approve_tap_removes_the_buttons_and_resumes(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, client = await _channel(store_dsn)
    await _send(channel, _tap(f"confirm:approve:{TRACE}"))
    [edit] = client.of("update")
    assert edit[0] == f"{SPACE}/messages/m1" and edit[2] == []
    assert edit[1].endswith("✅ Approved.") and edit[1].startswith("Approval needed")
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
    await _send(channel, _text_event("make it 5 units"))
    [call] = _asked(answers)
    assert (call["kind"], call["revision"]) == ("resume", "make it 5 units")
    assert CHAT_ID not in channel.awaiting_edit


async def test_files_cannot_be_attached_so_the_user_is_told(
    store_dsn: str, answers: list[dict[str, Any]], tmp_path: Path
) -> None:
    chart = tmp_path / "chart.png"
    chart.write_bytes(b"png")
    answers[0]["set"][0] = BotReply(text="Here", attachments=[Artifact(chart, "photo", "c")])
    channel, client = await _channel(store_dsn)
    await _send(channel, _text_event("chart it"))
    assert NO_FILES_REPLY in client.texts()
    assert not chart.exists()


async def test_pdf_upload_is_read_with_the_caption_as_question(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, _ = await _channel(store_dsn)
    upload = {
        "contentName": "bill.pdf",
        "contentType": "application/pdf",
        "source": "UPLOADED_CONTENT",
        "attachmentDataRef": {"resourceName": "res/abc"},
    }
    await _send(channel, _text_event("what is the total?", attachment=[upload]))
    [call] = _asked(answers)
    assert call["text"] == "what is the total?"
    assert (call["attachment"].filename, call["attachment"].data) == ("bill.pdf", b"%PDF")


async def test_unsupported_upload_is_declined_before_download(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, client = await _channel(store_dsn)
    upload = {
        "contentType": "video/mp4",
        "source": "UPLOADED_CONTENT",
        "attachmentDataRef": {"resourceName": "res/v"},
    }
    await _send(channel, _text_event("", attachment=[upload]))
    assert "video" in client.texts()[-1] and _asked(answers) == []


async def test_proactive_message_goes_to_the_remembered_space(
    store_dsn: str, answers: list[dict[str, Any]]
) -> None:
    channel, client = await _channel(store_dsn)
    await _send(channel, _text_event("hi"))
    client.calls.clear()
    await channel.send_plain(CHAT_ID, "**Brief** ready")
    assert client.of("send") == [(SPACE, "*Brief* ready", [])]

    other, other_client = await _channel(store_dsn)
    await other.send_plain(googlechat_chat_id("users/999"), "lost")
    assert other_client.calls == []
