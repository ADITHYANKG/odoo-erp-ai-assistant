"""Unit tests for ont_agent.oauth_flow (httpx.MockTransport, no network)."""

from __future__ import annotations

import base64
import hashlib
import re
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from ont_agent.config import Settings
from ont_agent.oauth_flow import (
    OAuthError,
    build_authorize_url,
    exchange_code,
    grants_write,
    make_pkce,
    refresh,
    revoke,
)

SETTINGS = Settings(
    mcp_url="http://odoo:8160/mcp",
    mcp_bearer_key="k",
    oauth_client_id="ont_client123",
    agent_public_url="http://cb:8180",
    odoo_public_url="http://odoo:8160",
)


def test_pkce_matches_server_side_s256_rules() -> None:
    # Mirror of ont_mcp_oauth/tools/pkce.py: 43-128 urlsafe chars, and
    # challenge = b64url(sha256(verifier)) without padding.
    verifier, challenge = make_pkce()
    assert 43 <= len(verifier) <= 128
    assert re.fullmatch(r"[A-Za-z0-9\-._~]+", verifier)
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    )
    assert challenge == expected
    assert "=" not in challenge


def test_pkce_is_unique_per_call() -> None:
    assert make_pkce() != make_pkce()


def test_authorize_url_shape() -> None:
    url = build_authorize_url(SETTINGS, state="st4te", challenge="ch4llenge")
    parts = urlsplit(url)
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == "http://odoo:8160/mcp/oauth/authorize"
    params = parse_qs(parts.query)
    assert params["response_type"] == ["code"]
    assert params["client_id"] == ["ont_client123"]
    assert params["redirect_uri"] == ["http://cb:8180/oauth/callback"]
    # Phase 5: write is requested too; the consent screen lets the user untick it.
    assert params["scope"] == ["mcp:read mcp:write"]
    assert params["state"] == ["st4te"]
    assert params["code_challenge"] == ["ch4llenge"]
    assert params["code_challenge_method"] == ["S256"]
    assert "resource" not in params  # validated against the frozen issuer — omit


def _client(handler: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=handler)


async def test_exchange_code_posts_form_and_parses_tokens() -> None:
    seen: dict[str, str] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://odoo:8160/mcp/oauth/token"
        assert request.headers["content-type"].startswith("application/x-www-form-urlencoded")
        seen.update({k: v[0] for k, v in parse_qs(request.content.decode()).items()})
        return httpx.Response(
            200,
            json={"access_token": "jwt-x", "refresh_token": "ref-x", "expires_in": 900},
        )

    async with _client(httpx.MockTransport(handle)) as client:
        tokens = await exchange_code(SETTINGS, code="c0de", verifier="v3rifier", client=client)

    assert seen["grant_type"] == "authorization_code"
    assert seen["code"] == "c0de"
    assert seen["code_verifier"] == "v3rifier"
    assert seen["client_id"] == "ont_client123"
    assert seen["redirect_uri"] == "http://cb:8180/oauth/callback"
    assert tokens.access_token == "jwt-x"
    assert tokens.refresh_token == "ref-x"
    # 900s minus the 300s refresh margin. The stored value is when to
    # REFRESH, not when the server expires it — the gap has to outlast a
    # whole turn, because the token is checked once at the start and then
    # used for every tool call in it, and nothing recovers from a 401
    # partway through.
    remaining = tokens.expires_at - datetime.now(UTC)
    assert timedelta(seconds=560) < remaining < timedelta(seconds=605)


async def test_refresh_uses_rotation_grant() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        assert form["grant_type"] == "refresh_token"
        assert form["refresh_token"] == "old-ref"
        return httpx.Response(
            200,
            json={"access_token": "jwt-2", "refresh_token": "NEW-ref", "expires_in": 900},
        )

    async with _client(httpx.MockTransport(handle)) as client:
        tokens = await refresh(SETTINGS, "old-ref", client=client)
    assert tokens.refresh_token == "NEW-ref"  # rotated — caller must persist


def test_grants_write_needs_an_explicit_write_scope() -> None:
    # Fails closed, matching the server rule in ont_mcp_oauth/models/ir_http.py:
    # anything short of an explicit mcp:write is read-only.
    assert grants_write("mcp:read mcp:write") is True
    assert grants_write("mcp:write") is True
    assert grants_write("mcp:read") is False
    assert grants_write("") is False
    # Substrings must not count — "mcp:write_all" is a different scope.
    assert grants_write("mcp:write_all") is False


async def test_token_response_carries_the_granted_scope() -> None:
    # The GRANT, not the request: a user without "MCP: Allow Write", or one
    # who unticks the box on the consent screen, gets a narrowed scope back.
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "access_token": "jwt-x",
                "refresh_token": "ref-x",
                "expires_in": 900,
                "scope": "mcp:read",
            },
        )

    async with _client(httpx.MockTransport(handle)) as client:
        tokens = await exchange_code(SETTINGS, code="c", verifier="v", client=client)
    assert tokens.scope == "mcp:read"
    assert grants_write(tokens.scope) is False


async def test_token_response_without_scope_is_read_only() -> None:
    # An older server that omits the field must not be read as write-capable.
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"access_token": "a", "refresh_token": "r", "expires_in": 900}
        )

    async with _client(httpx.MockTransport(handle)) as client:
        tokens = await refresh(SETTINGS, "old", client=client)
    assert tokens.scope == ""
    assert grants_write(tokens.scope) is False


async def test_error_response_raises_oauth_error() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid_grant", "error_description": "expired"})

    async with _client(httpx.MockTransport(handle)) as client:
        with pytest.raises(OAuthError) as excinfo:
            await refresh(SETTINGS, "dead-ref", client=client)
    assert excinfo.value.code == "invalid_grant"


async def test_short_lifetime_margin_never_goes_negative() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"access_token": "a", "refresh_token": "r", "expires_in": 10}
        )

    async with _client(httpx.MockTransport(handle)) as client:
        tokens = await refresh(SETTINGS, "x", client=client)

    # The margin (300s) is thirty times this token's whole life, so without
    # the half-life clamp every token would be born already due for refresh
    # and the bot would refresh in a loop. The server's own bounds allow an
    # access-token TTL as low as 60s, so this is reachable in practice.
    remaining = tokens.expires_at - datetime.now(UTC)
    assert timedelta(seconds=0) < remaining <= timedelta(seconds=5)


async def test_revoke_posts_token() -> None:
    posted: dict[str, str] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://odoo:8160/mcp/oauth/revoke"
        posted.update({k: v[0] for k, v in parse_qs(request.content.decode()).items()})
        return httpx.Response(200)

    async with _client(httpx.MockTransport(handle)) as client:
        await revoke(SETTINGS, "ref-to-kill", client=client)
    assert posted == {"token": "ref-to-kill"}
