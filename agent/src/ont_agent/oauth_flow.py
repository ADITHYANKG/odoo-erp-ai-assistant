"""OAuth 2.1 client flow against Odoo's ont_mcp_oauth server.

Authorization-code + PKCE (S256), refresh with rotation, revocation. Pure
logic plus httpx calls — everything testable offline via httpx.MockTransport.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlencode

import httpx

from ont_agent.config import Settings
from ont_agent.log import get_logger

logger = get_logger(__name__)

# Phase 5: request write too — the consent screen shows both as checkboxes
# the user can untick, and the client's server-side ceiling still applies.
SCOPE = "mcp:read mcp:write"


class OAuthError(Exception):
    """A token-endpoint failure; ``code`` is the RFC 6749 error string."""

    def __init__(self, code: str, description: str = "") -> None:
        self.code = code
        self.description = description
        super().__init__(f"{code}: {description}" if description else code)


@dataclass(frozen=True)
class TokenSet:
    access_token: str
    refresh_token: str
    expires_at: datetime
    #: Scope actually GRANTED (requested ∩ client ceiling ∩ user ceiling ∩
    #: whatever the user left ticked on the consent screen) — not what we
    #: asked for above. Present on every token response, so a permission
    #: change reaches us at the next refresh without a re-login.
    scope: str = ""


def grants_write(scope: str) -> bool:
    """Whether a granted scope string permits write-category MCP tools.

    Fails closed, mirroring the server's own rule in
    ``ont_mcp_oauth/models/ir_http.py``: only an explicit ``mcp:write``
    grants write; an empty or garbled scope is read-only.
    """
    return "mcp:write" in scope.split()


def make_pkce() -> tuple[str, str]:
    """Return (verifier, S256 challenge) per RFC 7636."""
    verifier = secrets.token_urlsafe(64)[:96]  # 43..128 chars required
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def build_authorize_url(settings: Settings, state: str, challenge: str) -> str:
    """Browser URL that starts the login+consent flow in Odoo.

    Deliberately no ``resource`` parameter: the server validates it against
    its (frozen) issuer, which need not match the host the browser uses.
    """
    params = {
        "response_type": "code",
        "client_id": settings.oauth_client_id,
        "redirect_uri": settings.redirect_uri,
        "scope": SCOPE,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    return f"{settings.authorize_url}?{urlencode(params)}"


def _token_set(settings: Settings, payload: dict[str, Any]) -> TokenSet:
    """Parse a token response, storing when we should REFRESH rather than
    when the server expires it.

    The margin is clamped to half the token's life: the server allows an
    access-token TTL as low as 60 seconds, and a margin wider than the token
    would mean every token was born already due for refresh.
    """
    expires_in = int(payload.get("expires_in") or 0)
    margin = min(settings.token_refresh_margin_seconds, expires_in // 2)
    return TokenSet(
        access_token=str(payload["access_token"]),
        refresh_token=str(payload["refresh_token"]),
        expires_at=datetime.now(UTC) + timedelta(seconds=expires_in - margin),
        scope=str(payload.get("scope") or ""),
    )


async def _post_token(
    settings: Settings, form: dict[str, str], client: httpx.AsyncClient | None = None
) -> TokenSet:
    async def call(c: httpx.AsyncClient) -> httpx.Response:
        return await c.post(settings.token_url, data=form)

    if client is not None:
        response = await call(client)
    else:
        async with httpx.AsyncClient(timeout=30.0) as c:
            response = await call(c)

    payload: dict[str, Any] = response.json()
    if response.status_code != 200 or "error" in payload:
        error = str(payload.get("error") or f"http_{response.status_code}")
        description = str(payload.get("error_description") or "")
        logger.warning("Token endpoint error: %s %s", error, description)
        raise OAuthError(error, description)
    return _token_set(settings, payload)


async def exchange_code(
    settings: Settings,
    code: str,
    verifier: str,
    client: httpx.AsyncClient | None = None,
) -> TokenSet:
    """Redeem an authorization code (with its PKCE verifier) for tokens."""
    return await _post_token(
        settings,
        {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": settings.oauth_client_id,
            "redirect_uri": settings.redirect_uri,
            "code_verifier": verifier,
        },
        client,
    )


async def refresh(
    settings: Settings,
    refresh_token: str,
    client: httpx.AsyncClient | None = None,
) -> TokenSet:
    """Rotate the refresh token; the caller MUST persist the returned pair."""
    return await _post_token(
        settings,
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": settings.oauth_client_id,
        },
        client,
    )


async def revoke(
    settings: Settings,
    refresh_token: str,
    client: httpx.AsyncClient | None = None,
) -> None:
    """Revoke the whole token family (server answers 200 regardless)."""
    form = {"token": refresh_token}
    if client is not None:
        await client.post(settings.revoke_url, data=form)
    else:
        async with httpx.AsyncClient(timeout=30.0) as c:
            await c.post(settings.revoke_url, data=form)
    logger.info("Token family revoked")
