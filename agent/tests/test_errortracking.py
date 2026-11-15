"""Unit tests for ont_agent.errortracking (no Sentry server; in-memory transport)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import pytest
import sentry_sdk
from sentry_sdk.envelope import Envelope
from sentry_sdk.transport import Transport

from ont_agent import errortracking
from ont_agent.config import Settings
from ont_agent.errortracking import EMAIL_MARKER, init_error_tracking, scrub, scrub_text

DISABLED = Settings(mcp_url="http://odoo:8160/mcp", mcp_bearer_key="k")
ENABLED = Settings(
    mcp_url="http://odoo:8160/mcp",
    mcp_bearer_key="k",
    sentry_dsn="https://public@sentry.example.com/1",
    sentry_environment="test",
)
SECRET = "sk-or-v1-" + "a" * 40


class CaptureTransport(Transport):
    """Keeps every event in memory instead of sending it."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def capture_envelope(self, envelope: Envelope) -> None:
        for item in envelope.items:
            if item.type == "event" and item.payload.json is not None:
                self.events.append(item.payload.json)


@pytest.fixture
def reset_sentry() -> Iterator[None]:
    yield
    client = sentry_sdk.get_client()
    if client.is_active():
        client.close()
    sentry_sdk.get_global_scope().set_client(None)


def test_settings_sentry_enabled_requires_dsn() -> None:
    assert not DISABLED.sentry_enabled
    assert ENABLED.sentry_enabled


def test_scrub_text_removes_secrets_and_emails() -> None:
    text = f"send to jane.doe@example.com failed, password={SECRET}"
    cleaned = scrub_text(text)
    assert "jane.doe@example.com" not in cleaned
    assert SECRET not in cleaned
    assert EMAIL_MARKER in cleaned


def test_scrub_walks_nested_structures() -> None:
    event = {
        "message": "a@b.com",
        "exception": {"values": [{"value": f"key {SECRET}"}]},
        "tags": ("x@y.org",),
        "level": 40,
    }
    cleaned = scrub(event)
    assert cleaned["message"] == EMAIL_MARKER
    assert SECRET not in cleaned["exception"]["values"][0]["value"]
    assert cleaned["tags"] == (EMAIL_MARKER,)
    assert cleaned["level"] == 40


def test_disabled_without_dsn_never_calls_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**_: Any) -> None:
        raise AssertionError("sentry_sdk.init must not run without a DSN")

    monkeypatch.setattr(sentry_sdk, "init", boom)
    assert init_error_tracking(DISABLED) is False


def test_init_uses_privacy_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(sentry_sdk, "init", lambda **kw: seen.update(kw))
    monkeypatch.setenv("SENTRY_RELEASE", "abc123")

    assert init_error_tracking(ENABLED) is True
    assert seen["dsn"] == ENABLED.sentry_dsn
    assert seen["environment"] == "test"
    assert seen["release"] == "abc123"
    collection = seen["data_collection"]
    assert collection["user_info"] is False
    assert collection["http_bodies"] == []
    assert collection["stack_frame_variables"] is False
    assert collection["gen_ai"] == {"inputs": False, "outputs": False}
    assert collection["http_headers"] == {"request": {"mode": "off"}}
    assert seen["max_request_body_size"] == "never"
    assert seen["traces_sample_rate"] is None


def test_init_failure_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(**_: Any) -> None:
        raise ValueError("bad DSN")

    monkeypatch.setattr(sentry_sdk, "init", broken)
    assert init_error_tracking(ENABLED) is False


def test_logged_error_reaches_sentry_scrubbed(
    monkeypatch: pytest.MonkeyPatch, reset_sentry: None
) -> None:
    transport = CaptureTransport()
    real_init = sentry_sdk.init
    monkeypatch.setattr(sentry_sdk, "init", lambda **kw: real_init(transport=transport, **kw))
    assert init_error_tracking(ENABLED) is True

    log = logging.getLogger("ont_agent.bot")
    try:
        raise RuntimeError(f"Odoo refused jane.doe@example.com with {SECRET}")
    except RuntimeError:
        log.error("Failed answering chat for jane.doe@example.com", exc_info=True)
    sentry_sdk.flush()

    assert len(transport.events) == 1
    event = transport.events[0]
    assert event["environment"] == "test"
    assert event["exception"]["values"][-1]["type"] == "RuntimeError"
    body = repr(event)
    assert "jane.doe@example.com" not in body
    assert SECRET not in body
    frames = event["exception"]["values"][-1]["stacktrace"]["frames"]
    assert all("vars" not in frame for frame in frames)


def test_warning_is_not_an_event(monkeypatch: pytest.MonkeyPatch, reset_sentry: None) -> None:
    transport = CaptureTransport()
    real_init = sentry_sdk.init
    monkeypatch.setattr(sentry_sdk, "init", lambda **kw: real_init(transport=transport, **kw))
    init_error_tracking(ENABLED)

    logging.getLogger("ont_agent.bot").warning("Tool returned nothing")
    sentry_sdk.flush()
    assert transport.events == []


def test_module_never_imports_sdk_when_disabled() -> None:
    # The module itself must not import sentry_sdk at import time.
    assert "sentry_sdk" not in vars(errortracking)
