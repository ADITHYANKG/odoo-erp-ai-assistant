"""Unit tests for ont_agent.channels and ont_agent.whatsapp (no network)."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from ont_agent.channels import WHATSAPP_OFFSET, is_whatsapp, wa_id_of, whatsapp_chat_id
from ont_agent.config import Settings
from ont_agent.whatsapp import (
    KIND_AUDIO,
    KIND_DOCUMENT,
    KIND_IMAGE,
    KIND_REPLY,
    KIND_TEXT,
    KIND_UNSUPPORTED,
    ListRow,
    WhatsAppClient,
    WhatsAppError,
    parse_events,
    verify_signature,
)

SETTINGS = Settings(
    mcp_url="http://odoo:8160/mcp",
    mcp_bearer_key="k",
    whatsapp_access_token="tok",
    whatsapp_phone_number_id="555",
    whatsapp_app_secret="s3cret",
    whatsapp_verify_token="vt",
)


def _payload(*messages: dict[str, Any], phone_number_id: str = "555") -> dict[str, Any]:
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "waba",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"phone_number_id": phone_number_id},
                            "messages": list(messages),
                        },
                    }
                ],
            }
        ],
    }


# ---- channels ------------------------------------------------------------


def test_whatsapp_chat_id_roundtrip() -> None:
    chat_id = whatsapp_chat_id("919876543210")
    assert chat_id == WHATSAPP_OFFSET + 919876543210
    assert is_whatsapp(chat_id)
    assert wa_id_of(chat_id) == "919876543210"


def test_telegram_ids_are_not_whatsapp() -> None:
    assert not is_whatsapp(8967803452)
    assert not is_whatsapp(-1001234567890)
    with pytest.raises(ValueError):
        wa_id_of(8967803452)


@pytest.mark.parametrize("bad", ["", "+9198", "12ab", "1" * 16])
def test_whatsapp_chat_id_rejects_bad_numbers(bad: str) -> None:
    with pytest.raises(ValueError):
        whatsapp_chat_id(bad)


# ---- signature -----------------------------------------------------------


def _sign(body: bytes, secret: str = "s3cret") -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_signature_accepts_valid() -> None:
    body = b'{"a": 1}'
    assert verify_signature("s3cret", body, _sign(body))


@pytest.mark.parametrize("header", [None, "", "sha256=", "md5=abc", "sha256=" + "0" * 64], ids=str)
def test_signature_rejects_invalid(header: str | None) -> None:
    assert not verify_signature("s3cret", b'{"a": 1}', header)


def test_signature_rejects_wrong_body_and_empty_secret() -> None:
    assert not verify_signature("s3cret", b"tampered", _sign(b'{"a": 1}'))
    assert not verify_signature("", b"x", _sign(b"x", secret=""))


# ---- parse_events --------------------------------------------------------


def _raw(kind: str, message_id: str = "a", **body: Any) -> dict[str, Any]:
    return {"from": "1", "id": message_id, "timestamp": "1", "type": kind, **body}


def test_parse_text_message() -> None:
    raw = _raw("text", "wamid.1", text={"body": "hello"})
    raw.update({"from": "919876543210", "timestamp": "1700000000"})
    [message] = parse_events(_payload(raw), "555")
    assert (message.kind, message.text, message.wa_id) == (KIND_TEXT, "hello", "919876543210")
    assert message.message_id == "wamid.1"
    assert message.timestamp == 1700000000.0


def test_parse_button_and_list_replies() -> None:
    button = _raw(
        "interactive",
        "a",
        interactive={
            "type": "button_reply",
            "button_reply": {"id": "confirm:approve", "title": "Approve"},
        },
    )
    listed = _raw(
        "interactive",
        "b",
        interactive={"type": "list_reply", "list_reply": {"id": "fu:chart", "title": "Chart"}},
    )
    first, second = parse_events(_payload(button, listed), "555")
    assert (first.kind, first.reply_id) == (KIND_REPLY, "confirm:approve")
    assert (second.kind, second.reply_id) == (KIND_REPLY, "fu:chart")


def test_parse_media_messages() -> None:
    audio = _raw("audio", "a", audio={"id": "m1", "mime_type": "audio/ogg; codecs=opus"})
    image = _raw(
        "image", "b", image={"id": "m2", "mime_type": "image/jpeg", "caption": "this bill"}
    )
    doc = _raw(
        "document",
        "c",
        document={"id": "m3", "mime_type": "application/pdf", "filename": "inv.pdf"},
    )
    a, i, d = parse_events(_payload(audio, image, doc), "555")
    assert (a.kind, a.media_id, a.mime) == (KIND_AUDIO, "m1", "audio/ogg")
    assert (i.kind, i.text, i.mime) == (KIND_IMAGE, "this bill", "image/jpeg")
    assert (d.kind, d.filename, d.mime) == (KIND_DOCUMENT, "inv.pdf", "application/pdf")


def test_parse_unsupported_type() -> None:
    [message] = parse_events(_payload(_raw("sticker", sticker={})), "555")
    assert message.kind == KIND_UNSUPPORTED


def test_parse_ignores_statuses_and_other_numbers_and_junk() -> None:
    value = {"metadata": {"phone_number_id": "555"}, "statuses": [{"id": "x", "status": "read"}]}
    statuses = {"entry": [{"changes": [{"value": value}]}]}
    assert parse_events(statuses, "555") == []
    text = _raw("text", text={"body": "hi"})
    assert parse_events(_payload(text, phone_number_id="999"), "555") == []
    junk = {"from": "not-digits", "id": "a", "type": "text"}
    assert parse_events(_payload(junk), "555") == []
    assert parse_events({}, "555") == []


# ---- client --------------------------------------------------------------


def _client(handler: Any) -> WhatsAppClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return WhatsAppClient(SETTINGS, http)


async def test_send_text_payload_and_auth() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"messages": [{"id": "x"}]})

    await _client(handler).send_text("919876543210", "hi")
    [request] = seen
    assert request.url.path.endswith("/555/messages")
    assert request.headers["Authorization"] == "Bearer tok"
    body = json.loads(request.content)
    assert body["to"] == "919876543210"
    assert body["type"] == "text"
    assert body["text"]["body"] == "hi"


async def test_send_buttons_enforces_meta_limits() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={})

    buttons = [(f"id{i}", "A very long button title indeed") for i in range(5)]
    await _client(handler).send_buttons("1", "x" * 2000, buttons)
    interactive = bodies[0]["interactive"]
    assert len(interactive["body"]["text"]) <= 1024
    sent = interactive["action"]["buttons"]
    assert len(sent) == 3
    assert all(len(b["reply"]["title"]) <= 20 for b in sent)


async def test_send_list_caps_rows_and_titles() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={})

    rows = [ListRow(f"r{i}", "T" * 40, "D" * 100) for i in range(8)]
    await _client(handler).send_list("1", "pick", "Options", [("A", rows), ("B", rows)])
    sections = bodies[0]["interactive"]["action"]["sections"]
    assert sum(len(s["rows"]) for s in sections) == 10
    assert all(len(r["title"]) <= 24 for s in sections for r in s["rows"])
    assert all(len(r["description"]) <= 72 for s in sections for r in s["rows"])


async def test_send_list_with_no_rows_sends_nothing() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={})

    await _client(handler).send_list("1", "pick", "Options", [("A", [])])
    assert calls == []


async def test_api_error_does_not_leak_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad"}})

    with pytest.raises(WhatsAppError) as caught:
        await _client(handler).send_text("1", "hi")
    assert "400" in str(caught.value)
    assert "tok" not in str(caught.value).replace("token", "")


async def test_media_info_then_download_uses_bearer() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.host == "graph.facebook.com":
            return httpx.Response(
                200,
                json={"url": "https://lookaside.example/file", "mime_type": "image/png",
                      "file_size": 123},
            )  # fmt: skip
        return httpx.Response(200, content=b"bytes")

    client = _client(handler)
    info = await client.media_info("m1")
    assert (info.mime, info.size) == ("image/png", 123)
    assert await client.download(info.url) == b"bytes"
    assert all(r.headers["Authorization"] == "Bearer tok" for r in seen)


async def test_send_file_uploads_then_sends_by_id(tmp_path: Path) -> None:
    chart = tmp_path / "chart.png"
    chart.write_bytes(b"\x89PNG")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/media"):
            return httpx.Response(200, json={"id": "media-9"})
        return httpx.Response(200, json={})

    await _client(handler).send_file("1", chart, as_image=True, caption="Sales")
    assert seen[0].url.path.endswith("/555/media")
    sent = json.loads(seen[1].content)
    assert sent["type"] == "image"
    assert sent["image"] == {"id": "media-9", "caption": "Sales"}


async def test_typing_indicator_failure_is_swallowed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="down")

    await _client(handler).show_typing("wamid.1")  # must not raise


async def test_reaction_payload_and_clearing() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={})

    client = _client(handler)
    await client.send_reaction("919876543210", "wamid.9", "⏳")
    await client.send_reaction("919876543210", "wamid.9", "")
    assert bodies[0]["type"] == "reaction"
    assert bodies[0]["reaction"] == {"message_id": "wamid.9", "emoji": "⏳"}
    assert bodies[1]["reaction"]["emoji"] == ""


async def test_reaction_failure_is_swallowed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"code": 100}})

    await _client(handler).send_reaction("1", "wamid.9", "⏳")  # must not raise


async def test_display_number_returns_digits_only() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/555")
        return httpx.Response(200, json={"display_phone_number": "+91 81390 45437", "id": "555"})

    assert await _client(handler).display_number() == "918139045437"


async def test_display_number_failure_returns_empty() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"code": 190}})

    assert await _client(handler).display_number() == ""
