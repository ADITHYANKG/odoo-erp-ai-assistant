"""Unit tests for ont_agent.callback_server (in-process ASGI, no sockets)."""

from __future__ import annotations

import hashlib
import hmac
import json

import httpx
import pytest
from fastapi import FastAPI

from ont_agent import callback_server
from ont_agent.auth import LinkCode, LoginError
from ont_agent.channels import whatsapp_chat_id
from ont_agent.config import Settings
from ont_agent.store import BindingStore

SETTINGS = Settings(
    mcp_url="http://odoo:8160/mcp",
    mcp_bearer_key="k",
    oauth_client_id="ont_client123",
    agent_public_url="http://cb:8180",
    odoo_public_url="http://odoo:8160",
)

SETTINGS_WHATSAPP = Settings(
    mcp_url="http://odoo:8160/mcp",
    mcp_bearer_key="k",
    whatsapp_verify_token="vt-secret",
    whatsapp_app_secret="app-secret",
    whatsapp_business_number="15551234567",
)


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _binding(chat_id: int = 42) -> LinkCode:
    return LinkCode(chat_id=chat_id, odoo_name="CFO Priya", code="482913")


async def test_error_param_shows_denied_page(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        app = callback_server.build_app(store, SETTINGS)
        async with _client(app) as client:
            resp = await client.get(
                "/oauth/callback", params={"error": "access_denied", "state": "s"}
            )
            assert resp.status_code == 400
            assert "did not authorize" in resp.text


async def test_missing_params_rejected(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        app = callback_server.build_app(store, SETTINGS)
        async with _client(app) as client:
            resp = await client.get("/oauth/callback")
            assert resp.status_code == 400


async def test_unknown_state_shows_login_error(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failing_complete(
        store: BindingStore, settings: Settings, state: str, code: str
    ) -> LinkCode:
        raise LoginError("Unknown or expired login attempt — start again from Telegram.")

    monkeypatch.setattr(callback_server, "complete_login", failing_complete)
    async with BindingStore(store_dsn) as store:
        app = callback_server.build_app(store, SETTINGS)
        async with _client(app) as client:
            resp = await client.get("/oauth/callback", params={"code": "c", "state": "bad"})
            assert resp.status_code == 400
            assert "start again" in resp.text.lower()


async def test_success_page_and_on_bound_notification(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    bound: list[LinkCode] = []

    async def fake_complete(
        store: BindingStore, settings: Settings, state: str, code: str
    ) -> LinkCode:
        assert (state, code) == ("st", "c0de")
        return _binding()

    async def on_bound(binding: LinkCode) -> None:
        bound.append(binding)

    monkeypatch.setattr(callback_server, "complete_login", fake_complete)
    async with BindingStore(store_dsn) as store:
        app = callback_server.build_app(store, SETTINGS, on_bound)
        async with _client(app) as client:
            resp = await client.get("/oauth/callback", params={"code": "c0de", "state": "st"})
            assert resp.status_code == 200
            body = resp.text
            assert "CFO Priya" in body
            assert "Telegram" in body
            assert "t.me" not in body  # no username known → no deep link
    assert len(bound) == 1
    assert bound[0].chat_id == 42


async def test_repeated_success_callback_is_idempotent(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    completions = 0
    notifications: list[LinkCode] = []

    async def fake_complete(
        store: BindingStore, settings: Settings, state: str, code: str
    ) -> LinkCode:
        nonlocal completions
        completions += 1
        return _binding()

    async def on_bound(binding: LinkCode) -> None:
        notifications.append(binding)

    monkeypatch.setattr(callback_server, "complete_login", fake_complete)
    async with BindingStore(store_dsn) as store:
        app = callback_server.build_app(store, SETTINGS, on_bound)
        async with _client(app) as client:
            first = await client.get("/oauth/callback", params={"code": "first", "state": "s"})
            repeated = await client.get("/oauth/callback", params={"code": "second", "state": "s"})

    assert first.status_code == 200
    assert repeated.status_code == 200
    assert "Almost connected" in repeated.text
    assert completions == 1
    assert len(notifications) == 1


async def test_success_page_offers_return_to_telegram_button(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_complete(
        store: BindingStore, settings: Settings, state: str, code: str
    ) -> LinkCode:
        return _binding()

    monkeypatch.setattr(callback_server, "complete_login", fake_complete)
    async with BindingStore(store_dsn) as store:
        app = callback_server.build_app(store, SETTINGS, bot_username="ont_agent_bot")
        async with _client(app) as client:
            resp = await client.get("/oauth/callback", params={"code": "c", "state": "s"})
            body = resp.text
            # The code must stay readable, so no auto-redirect away from it.
            assert 'href="https://t.me/ont_agent_bot"' in body
            assert "http-equiv" not in body
            assert "482913" in body
            assert "Return to Telegram" in body


async def test_code_page_polls_for_the_chat_confirmation(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_complete(
        store: BindingStore, settings: Settings, state: str, code: str
    ) -> LinkCode:
        return LinkCode(chat_id=42, odoo_name="CFO Priya", code="482913", status_token="tok-1")

    monkeypatch.setattr(callback_server, "complete_login", fake_complete)
    async with BindingStore(store_dsn) as store:
        app = callback_server.build_app(store, SETTINGS)
        async with _client(app) as client:
            resp = await client.get("/oauth/callback", params={"code": "c", "state": "s"})
            assert 'link-status?t=" + encodeURIComponent(token)' in resp.text
            assert '"tok-1"' in resp.text


async def test_link_status_endpoint_reports_the_outcome(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_status(store: BindingStore, token: str) -> str:
        return "connected" if token == "tok-1" else "expired"

    monkeypatch.setattr(callback_server, "link_status", fake_status)
    async with BindingStore(store_dsn) as store:
        app = callback_server.build_app(store, SETTINGS)
        async with _client(app) as client:
            resp = await client.get("/oauth/link-status", params={"t": "tok-1"})
            assert resp.json() == {"status": "connected"}
            assert resp.headers["cache-control"] == "no-store"
            resp = await client.get("/oauth/link-status")
            assert resp.json() == {"status": "expired"}


async def test_error_pages_have_no_redirect(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failing_complete(
        store: BindingStore, settings: Settings, state: str, code: str
    ) -> LinkCode:
        raise LoginError("nope")

    monkeypatch.setattr(callback_server, "complete_login", failing_complete)
    async with BindingStore(store_dsn) as store:
        app = callback_server.build_app(store, SETTINGS, bot_username="ont_agent_bot")
        async with _client(app) as client:
            resp = await client.get("/oauth/callback", params={"code": "c", "state": "bad"})
            body = resp.text
            assert resp.status_code == 400
            assert "t.me" not in body  # failures must not bounce the user away


async def test_health_ok(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        app = callback_server.build_app(store, SETTINGS)
        async with _client(app) as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}


async def test_avatar_served_as_png(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        app = callback_server.build_app(store, SETTINGS)
        async with _client(app) as client:
            resp = await client.get("/static/avatar.png")
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "image/png"
            assert resp.content.startswith(b"\x89PNG")


async def test_whatsapp_verify_echoes_challenge(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        app = callback_server.build_app(store, SETTINGS_WHATSAPP)
        async with _client(app) as client:
            resp = await client.get(
                "/webhook/whatsapp",
                params={
                    "hub.mode": "subscribe",
                    "hub.verify_token": "vt-secret",
                    "hub.challenge": "12345",
                },
            )
            assert resp.status_code == 200
            assert resp.text == "12345"  # Meta requires the raw challenge back


async def test_whatsapp_verify_wrong_token_403(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        app = callback_server.build_app(store, SETTINGS_WHATSAPP)
        async with _client(app) as client:
            resp = await client.get(
                "/webhook/whatsapp",
                params={
                    "hub.mode": "subscribe",
                    "hub.verify_token": "guess",
                    "hub.challenge": "12345",
                },
            )
            assert resp.status_code == 403


async def test_whatsapp_verify_403_when_token_unset(store_dsn: str) -> None:
    # An unset token means "webhook disabled" — it must never match an empty
    # hub.verify_token, or anyone could complete the handshake.
    async with BindingStore(store_dsn) as store:
        app = callback_server.build_app(store, SETTINGS)
        async with _client(app) as client:
            resp = await client.get(
                "/webhook/whatsapp",
                params={"hub.mode": "subscribe", "hub.verify_token": "", "hub.challenge": "x"},
            )
            assert resp.status_code == 403


async def test_whatsapp_post_without_signature_refused(store_dsn: str) -> None:
    seen: list[object] = []
    async with BindingStore(store_dsn) as store:
        app = callback_server.build_app(store, SETTINGS_WHATSAPP, on_whatsapp=seen.append)
        async with _client(app) as client:
            resp = await client.post("/webhook/whatsapp", json={"entry": []})
            assert resp.status_code == 403
    assert seen == []


async def test_whatsapp_post_bad_signature_refused(store_dsn: str) -> None:
    seen: list[object] = []
    async with BindingStore(store_dsn) as store:
        app = callback_server.build_app(store, SETTINGS_WHATSAPP, on_whatsapp=seen.append)
        async with _client(app) as client:
            resp = await client.post(
                "/webhook/whatsapp",
                content=b'{"entry": []}',
                headers={"X-Hub-Signature-256": "sha256=" + "0" * 64},
            )
            assert resp.status_code == 403
    assert seen == []


async def test_whatsapp_post_signed_is_handed_to_channel(store_dsn: str) -> None:
    seen: list[object] = []
    body = json.dumps({"entry": [{"id": "1"}]}).encode()
    signature = hmac.new(b"app-secret", body, hashlib.sha256).hexdigest()
    async with BindingStore(store_dsn) as store:
        app = callback_server.build_app(store, SETTINGS_WHATSAPP, on_whatsapp=seen.append)
        async with _client(app) as client:
            resp = await client.post(
                "/webhook/whatsapp",
                content=body,
                headers={"X-Hub-Signature-256": f"sha256={signature}"},
            )
            assert resp.status_code == 200
    assert seen == [{"entry": [{"id": "1"}]}]


async def test_whatsapp_login_success_page_returns_to_whatsapp(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_complete(
        store: BindingStore, settings: Settings, state: str, code: str
    ) -> LinkCode:
        return _binding(chat_id=whatsapp_chat_id("919876543210"))

    monkeypatch.setattr(callback_server, "complete_login", fake_complete)
    async with BindingStore(store_dsn) as store:
        app = callback_server.build_app(store, SETTINGS_WHATSAPP, bot_username="ont_agent_bot")
        async with _client(app) as client:
            resp = await client.get("/oauth/callback", params={"code": "c", "state": "s"})
            body = resp.text
            assert 'href="https://wa.me/15551234567"' in body
            assert "Return to WhatsApp" in body
            assert "t.me" not in body


async def test_whatsapp_return_link_uses_number_looked_up_from_meta(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_complete(
        store: BindingStore, settings: Settings, state: str, code: str
    ) -> LinkCode:
        return _binding(chat_id=whatsapp_chat_id("919876543210"))

    monkeypatch.setattr(callback_server, "complete_login", fake_complete)
    no_number = SETTINGS_WHATSAPP.model_copy(update={"whatsapp_business_number": ""})
    async with BindingStore(store_dsn) as store:
        app = callback_server.build_app(store, no_number, whatsapp_number="15551556791")
        async with _client(app) as client:
            resp = await client.get("/oauth/callback", params={"code": "c", "state": "s"})
            assert 'href="https://wa.me/15551556791"' in resp.text


async def test_configured_whatsapp_number_beats_the_looked_up_one(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_complete(
        store: BindingStore, settings: Settings, state: str, code: str
    ) -> LinkCode:
        return _binding(chat_id=whatsapp_chat_id("919876543210"))

    monkeypatch.setattr(callback_server, "complete_login", fake_complete)
    async with BindingStore(store_dsn) as store:
        app = callback_server.build_app(store, SETTINGS_WHATSAPP, whatsapp_number="15550000000")
        async with _client(app) as client:
            resp = await client.get("/oauth/callback", params={"code": "c", "state": "s"})
            assert 'href="https://wa.me/15551234567"' in resp.text


async def test_start_server_binds_and_stops(store_dsn: str) -> None:
    # Real socket lifecycle: ephemeral port, confirm it listens, clean stop.
    async with BindingStore(store_dsn) as store:
        app = callback_server.build_app(store, SETTINGS)
        handle = await callback_server.start_server(app, "127.0.0.1", 0)
        try:
            assert handle.server.started
        finally:
            await handle.stop()
        assert handle.task.done()
