"""Unit tests for ont_agent.transcribe (httpx.MockTransport, no network)."""

from __future__ import annotations

from typing import Any

import httpx

from ont_agent.config import Settings
from ont_agent.transcribe import Transcript, transcribe

SETTINGS = Settings(
    mcp_url="http://odoo:8160/mcp",
    mcp_bearer_key="k",
    openrouter_api_key="sk-or-test",
)


def _client(handler: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=handler)


async def test_posts_to_the_transcription_endpoint_not_chat_completions() -> None:
    """OpenRouter rejects a transcription model on /chat/completions outright:
    "openai/gpt-transcribe is a transcription model and cannot be used with
    the chat/completions endpoint" (verified live 2026-09-01). Pin the URL so
    nobody "simplifies" this onto the chat client.
    """
    seen: dict[str, Any] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = request.content
        return httpx.Response(200, json={"text": "hello there", "usage": {"cost": 0.0004}})

    async with _client(httpx.MockTransport(handle)) as client:
        result = await transcribe(SETTINGS, b"fake-ogg-bytes", client=client)

    assert seen["url"].endswith("/audio/transcriptions")
    assert "chat/completions" not in seen["url"]
    assert seen["auth"] == "Bearer sk-or-test"
    # Multipart upload carrying the audio and the configured model.
    assert b"fake-ogg-bytes" in seen["body"]
    assert b"openai/gpt-transcribe" in seen["body"]
    assert result == Transcript(text="hello there", cost_usd=0.0004)


async def test_uses_the_configured_model() -> None:
    seen: dict[str, Any] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content
        return httpx.Response(200, json={"text": "x"})

    other = SETTINGS.model_copy(update={"transcribe_model": "some/other-model"})
    async with _client(httpx.MockTransport(handle)) as client:
        await transcribe(other, b"audio", client=client)

    assert b"some/other-model" in seen["body"]


async def test_sends_ogg_because_that_is_what_telegram_records() -> None:
    # Verified live: OpenRouter accepts Telegram's OGG/Opus as-is, so there
    # is no ffmpeg transcoding step to go wrong.
    seen: dict[str, Any] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content
        return httpx.Response(200, json={"text": "x"})

    async with _client(httpx.MockTransport(handle)) as client:
        await transcribe(SETTINGS, b"audio", client=client)

    assert b"voice.ogg" in seen["body"]
    assert b"audio/ogg" in seen["body"]


async def test_an_http_error_degrades_to_empty_never_raises() -> None:
    # The person is in a car. A failure must become "I couldn't make that
    # out", never a stack trace.
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream exploded")

    async with _client(httpx.MockTransport(handle)) as client:
        assert await transcribe(SETTINGS, b"audio", client=client) == Transcript()


async def test_a_transport_failure_degrades_to_empty() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    async with _client(httpx.MockTransport(handle)) as client:
        assert await transcribe(SETTINGS, b"audio", client=client) == Transcript()


async def test_a_non_json_body_degrades_to_empty() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>gateway</html>")

    async with _client(httpx.MockTransport(handle)) as client:
        assert await transcribe(SETTINGS, b"audio", client=client) == Transcript()


async def test_silence_transcribes_to_empty_but_still_reports_its_cost() -> None:
    # Audio with no speech returns text:"" and is still billed — recording
    # the cost anyway keeps the spend ledger honest.
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "", "usage": {"seconds": 2, "cost": 0.00015}})

    async with _client(httpx.MockTransport(handle)) as client:
        result = await transcribe(SETTINGS, b"audio", client=client)

    assert result.text == ""
    assert result.cost_usd == 0.00015


async def test_a_missing_or_odd_cost_is_zero_not_a_crash() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "hi", "usage": {"cost": "free"}})

    async with _client(httpx.MockTransport(handle)) as client:
        result = await transcribe(SETTINGS, b"audio", client=client)

    assert result == Transcript(text="hi", cost_usd=0.0)


async def test_no_key_or_no_audio_never_calls_out() -> None:
    called = False

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"text": "x"})

    keyless = SETTINGS.model_copy(update={"openrouter_api_key": ""})
    async with _client(httpx.MockTransport(handle)) as client:
        assert await transcribe(keyless, b"audio", client=client) == Transcript()
        assert await transcribe(SETTINGS, b"", client=client) == Transcript()

    assert called is False
