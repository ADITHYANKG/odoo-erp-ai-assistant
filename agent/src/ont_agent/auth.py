"""Binding orchestration: Telegram chat ↔ Odoo user, via OAuth.

Sits between the bot/callback layers and the low-level pieces (store,
oauth_flow, MCP tools): starts logins, completes them (capturing the Odoo
identity via a one-off ``whoami`` call with the fresh token), and keeps
access tokens alive through refresh rotation.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from ont_agent import oauth_flow
from ont_agent.config import Settings
from ont_agent.log import get_logger
from ont_agent.oauth_flow import OAuthError, TokenSet
from ont_agent.store import Binding, BindingStore, LinkStatus, binding_id
from ont_agent.tools import get_odoo_tools

logger = get_logger(__name__)

#: How long the code shown on the login page stays valid.
LINK_CODE_TTL = timedelta(minutes=10)
#: Wrong codes allowed before the pending login is thrown away.
MAX_LINK_ATTEMPTS = 5
_LINK_CODE = re.compile(r"\d{6}")

LINK_CODE_PROMPT = (
    "Login received. To finish connecting, reply here with the 6-digit code "
    "shown on the login page."
)


class LoginError(Exception):
    """A login attempt that cannot be completed (bad state, exchange failure)."""


@dataclass(frozen=True)
class LinkCode:
    """A browser login that the chat must confirm with ``code``.

    Binds the login to the person holding the chat: a forwarded login link
    shows the code to whoever opened it, not to whoever sent it.
    """

    chat_id: int
    odoo_name: str
    code: str
    #: Lets the login page poll for the outcome without knowing the code's
    #: chat; only the page that showed the code holds it.
    status_token: str = ""


def _hash_code(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


async def begin_login(store: BindingStore, settings: Settings, chat_id: int) -> str:
    """Create a pending login and return the browser URL for the user."""
    state = secrets.token_urlsafe(32)
    verifier, challenge = oauth_flow.make_pkce()
    await store.create_pending(state, chat_id, verifier)
    url = oauth_flow.build_authorize_url(settings, state, challenge)
    logger.info("Login started for chat %s", binding_id(chat_id))
    return url


#: Group display name whoami reports for users allowed to use call_method.
#: UX hint only — Odoo enforces the gate server-side either way.
METHOD_CALL_GROUP = "MCP: Allow Method Calls"

#: Odoo's SYSTEM administrator group (base.group_system), exactly as Odoo
#: 17's whoami names it. Matched exactly on purpose: many other groups are
#: called "<App> / Administrator" (Sales, Accounting, Purchase...), and a
#: loose match would hand the cross-user audit view to a sales manager.
ADMIN_GROUP = "Administration / Settings"


@dataclass(frozen=True)
class Identity:
    """Who the fresh token belongs to, per Odoo's own whoami.

    ``groups`` carries the full list because access decisions are made from
    it (the call_method hint, and the configurable group required to bind a
    chat at all) — a tuple of flags would need extending every time.
    """

    login: str = "unknown"
    name: str = "Unknown user"
    can_call_methods: bool = False
    groups: list[str] = field(default_factory=list)
    #: Odoo's timezone for this user, e.g. "Asia/Kolkata". Decides when the
    #: morning brief is morning FOR THEM; may be empty.
    tz: str = ""
    #: Member of Odoo's system-administrator group. Gates the /audit view.
    is_admin: bool = False


async def fetch_identity(mcp_url: str, access_token: str) -> Identity:
    """Call whoami with a fresh token and return who it belongs to."""
    tools = await get_odoo_tools(mcp_url, access_token)
    whoami = next((tool for tool in tools if tool.name == "whoami"), None)
    if whoami is None:
        return Identity()
    result = await whoami.ainvoke({})
    return parse_identity(result)


def parse_identity(result: Any) -> Identity:
    """Extract the Odoo identity from a whoami result, defensively."""
    text = ""
    if isinstance(result, str):
        text = result
    elif isinstance(result, list):
        for block in result:
            if isinstance(block, dict) and block.get("type") == "text":
                text = str(block.get("text", ""))
                break
    try:
        data = json.loads(text)
        raw_groups = data.get("groups") or []
        groups = [str(group) for group in raw_groups] if isinstance(raw_groups, list) else []
        return Identity(
            login=str(data.get("login", "unknown")),
            name=str(data.get("name", "Unknown user")),
            can_call_methods=METHOD_CALL_GROUP in groups,
            groups=groups,
            tz=str(data.get("tz") or ""),
            is_admin=ADMIN_GROUP in groups,
        )
    except (json.JSONDecodeError, TypeError):
        logger.warning("Could not parse whoami result")
        return Identity()


async def complete_login(
    store: BindingStore, settings: Settings, state: str, code: str
) -> LinkCode:
    """Redeem the callback's code and hold the binding until the chat confirms it."""
    pending = await store.pop_pending(state)
    if pending is None:
        raise LoginError("Unknown or expired login attempt — start again from Telegram.")
    # Second gate on the private-chat rule, independent of the Telegram
    # handler chain: this runs from the HTTP callback, which never passes
    # through bot.guard. Telegram ids answer it with no lookup — a private
    # chat id IS the user's own id and is positive, while group, supergroup
    # and channel ids are negative. Binding a group would give every member
    # of it one person's Odoo identity and audit trail.
    if pending.chat_id < 0:
        logger.warning("Login refused for chat %s: not a private chat", pending.chat_id)
        raise LoginError("This assistant only works in a private chat. Message the bot directly.")
    try:
        tokens = await oauth_flow.exchange_code(settings, code, pending.verifier)
    except OAuthError as exc:
        raise LoginError(
            f"Odoo rejected the login ({exc.code}). Start again from Telegram."
        ) from exc

    identity = await fetch_identity(settings.mcp_url, tokens.access_token)
    # Access gate: refuse BEFORE the binding is saved, so a rejected user
    # leaves nothing behind. Matched against the group's full name, which is
    # what whoami reports ("Category / Name" for categorised groups, the
    # plain name otherwise). Empty setting = any internal Odoo user.
    required = settings.required_odoo_group
    if required and required not in identity.groups:
        logger.warning("Login refused for %r: not a member of %r", identity.login, required)
        raise LoginError(
            f"Your Odoo account is not authorised to use this assistant. "
            f'Ask an administrator to add you to the "{required}" group.'
        )

    binding = Binding(
        chat_id=pending.chat_id,
        odoo_login=identity.login,
        odoo_name=identity.name,
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        expires_at=tokens.expires_at,
        can_call_methods=identity.can_call_methods,
        can_write=oauth_flow.grants_write(tokens.scope),
        tz=identity.tz,
        is_admin=identity.is_admin,
    )
    replaced = await store.get_pending_link(pending.chat_id)
    if replaced is not None:
        await _discard_link(store, settings, replaced.binding)
    link_code = f"{secrets.randbelow(10**6):06d}"
    status_token = secrets.token_urlsafe(32)
    await store.save_pending_link(binding, _hash_code(link_code))
    await store.create_link_status(_hash_code(status_token), pending.chat_id)
    logger.info(
        "Chat %s: login for Odoo user %r awaits chat confirmation",
        binding_id(pending.chat_id),
        identity.login,
    )
    return LinkCode(
        chat_id=pending.chat_id,
        odoo_name=identity.name,
        code=link_code,
        status_token=status_token,
    )


async def _discard_link(store: BindingStore, settings: Settings, binding: Binding) -> None:
    """Drop an unconfirmed login and revoke its tokens."""
    await store.delete_pending_link(binding.chat_id)
    await store.settle_link_status(binding.chat_id, "expired")
    try:
        await oauth_flow.revoke(settings, binding.refresh_token)
    except Exception:
        logger.warning("Could not revoke an unconfirmed login", exc_info=True)


async def confirm_link(
    store: BindingStore, settings: Settings, chat_id: int, text: str
) -> str | None:
    """Finish a pending login from a chat message.

    :return: the reply to send, or None when no login is waiting for this chat.
    """
    pending = await store.get_pending_link(chat_id)
    if pending is None:
        return None
    if datetime.now(UTC) - pending.created_at > LINK_CODE_TTL:
        await _discard_link(store, settings, pending.binding)
        return "That login expired. Start the login again to connect."
    candidate = re.sub(r"\s", "", text)
    if not _LINK_CODE.fullmatch(candidate):
        return LINK_CODE_PROMPT
    if not hmac.compare_digest(_hash_code(candidate), pending.code_hash):
        if pending.attempts + 1 >= MAX_LINK_ATTEMPTS:
            await _discard_link(store, settings, pending.binding)
            logger.warning("Chat %s: too many wrong link codes", binding_id(chat_id))
            return "Too many wrong codes. Start the login again to connect."
        await store.bump_link_attempts(chat_id)
        return "That code doesn't match. Check the login page and try again."
    await store.save_binding(pending.binding)
    await store.delete_pending_link(chat_id)
    await store.settle_link_status(chat_id, "connected")
    logger.info("Chat %s bound to Odoo user %r", binding_id(chat_id), pending.binding.odoo_login)
    return (
        f"✅ Connected as {pending.binding.odoo_name} ({pending.binding.odoo_login}). "
        "Ask me anything about your Odoo data."
    )


async def link_status(store: BindingStore, status_token: str) -> LinkStatus:
    """The outcome the login page polls for; unknown tokens read as expired."""
    row = await store.get_link_status(_hash_code(status_token)) if status_token else None
    if row is None:
        return "expired"
    status, created_at = row
    # Nobody typed anything before the code ran out: confirm_link only
    # discards on the next chat message, so the page must not wait for it.
    if status == "pending" and datetime.now(UTC) - created_at > LINK_CODE_TTL:
        return "expired"
    return status


#: One lock per chat, so a refresh cannot run twice concurrently. The only
#: real concurrency here is the proactive ticker, which is its own asyncio
#: task and can reach a chat at the same moment a handler does.
#:
#: Refresh tokens ROTATE — redeeming one invalidates it — so a double
#: refresh is a replay from the server's point of view. ont_mcp_oauth
#: happens to forgive that inside a 30-second grace window, but the window
#: is a system parameter a client can set to 0, and the agent should not
#: depend on how someone else's Odoo is configured.
#:
#: A plain dict: one small object per chat this process has served, cleared
#: on restart. Bounded by user count, so not a leak worth weakening the
#: reasoning for.
_REFRESH_LOCKS: dict[int, asyncio.Lock] = {}


def _refresh_lock(chat_id: int) -> asyncio.Lock:
    lock = _REFRESH_LOCKS.get(chat_id)
    if lock is None:
        lock = _REFRESH_LOCKS[chat_id] = asyncio.Lock()
    return lock


async def ensure_access(store: BindingStore, settings: Settings, chat_id: int) -> Binding | None:
    """Return a binding with a live access token, refreshing if needed.

    Returns None when the chat is unbound, or when its grant is genuinely
    dead — revoked, expired, or a burnt token family — in which case the
    binding is removed and the caller should ask the user to /start again.

    :raise OAuthError: the refresh failed for a reason that is NOT the
        grant being dead — Odoo restarting, a 5xx through the tunnel. The
        binding is kept: telling a user to /start again because their ERP
        hiccuped would be both wrong and unhelpful, and the login still
        works once the server is back.
    """
    binding = await store.get_binding(chat_id)
    if binding is None:
        return None
    if binding.expires_at > datetime.now(UTC):
        return binding

    async with _refresh_lock(chat_id):
        # Re-read inside the lock, and this is the whole point of holding it:
        # a coroutine that waited here captured its binding BEFORE waiting, so
        # acting on that copy would refresh with a token the winner has
        # already spent.
        binding = await store.get_binding(chat_id)
        if binding is None:
            return None
        if binding.expires_at > datetime.now(UTC):
            return binding
        return await _refresh_binding(store, settings, binding)


async def _refresh_binding(
    store: BindingStore, settings: Settings, binding: Binding
) -> Binding | None:
    """Rotate the token pair and persist it. Caller holds the chat's lock."""
    chat_id = binding.chat_id
    logger.info("Refreshing access token for chat %s", binding_id(chat_id))
    try:
        tokens: TokenSet = await oauth_flow.refresh(settings, binding.refresh_token)
    except OAuthError as exc:
        # invalid_grant is the one code that means the family is genuinely
        # dead — revoked, expired, or burnt by a replay. Everything else
        # (server_error, unauthorized_client, an HTTP 5xx) is the server
        # having a bad moment, and deleting the binding for that would log
        # every refreshing user out during an Odoo restart.
        if exc.code == "invalid_grant":
            logger.warning("Refresh rejected (%s) — dropping binding", exc.code)
            await store.delete_binding(chat_id)
            return None
        logger.warning("Refresh failed transiently (%s) — keeping binding", exc.code)
        raise

    refreshed = Binding(
        chat_id=binding.chat_id,
        odoo_login=binding.odoo_login,
        odoo_name=binding.odoo_name,
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,  # rotation: ALWAYS persist the new one
        expires_at=tokens.expires_at,
        can_call_methods=binding.can_call_methods,  # carried forward, set at login
        tz=binding.tz,  # likewise: whoami only runs at login
        is_admin=binding.is_admin,
        brief_enabled=binding.brief_enabled,
        last_brief_date=binding.last_brief_date,
        # Unlike the group above (read from whoami only at login), the granted
        # scope rides EVERY token response — so re-read it here and a write
        # grant added or removed in Odoo lands without the user re-binding.
        can_write=oauth_flow.grants_write(tokens.scope),
    )
    await store.save_binding(refreshed)
    return refreshed


async def logout(store: BindingStore, settings: Settings, chat_id: int) -> bool:
    """Revoke the user's token family and forget the binding."""
    binding = await store.get_binding(chat_id)
    if binding is None:
        return False
    try:
        await oauth_flow.revoke(settings, binding.refresh_token)
    except Exception:
        # Disconnecting must never depend on Odoo being reachable.
        logger.warning("Chat %s: token revocation failed", binding_id(chat_id), exc_info=True)
    await store.delete_binding(chat_id)
    return True
