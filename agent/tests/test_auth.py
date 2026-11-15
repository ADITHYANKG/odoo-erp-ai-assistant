"""Unit tests for ont_agent.auth — orchestration with fakes."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from ont_agent import auth, oauth_flow
from ont_agent.auth import (
    MAX_LINK_ATTEMPTS,
    Identity,
    LoginError,
    begin_login,
    complete_login,
    confirm_link,
    ensure_access,
    link_status,
    logout,
    parse_identity,
)
from ont_agent.config import Settings
from ont_agent.oauth_flow import OAuthError, TokenSet
from ont_agent.store import Binding, BindingStore

SETTINGS = Settings(
    mcp_url="http://odoo:8160/mcp",
    mcp_bearer_key="k",
    oauth_client_id="ont_client123",
    agent_public_url="http://cb:8180",
    odoo_public_url="http://odoo:8160",
)


def _tokens(
    access: str = "jwt-1", refresh: str = "ref-1", scope: str = "mcp:read mcp:write"
) -> TokenSet:
    return TokenSet(
        access_token=access,
        refresh_token=refresh,
        expires_at=datetime.now(UTC) + timedelta(minutes=14),
        scope=scope,
    )


async def _login(store: BindingStore, settings: Settings, state: str, code: str) -> Binding:
    """Browser login, then the chat types the code it was shown."""
    link = await complete_login(store, settings, state=state, code=code)
    reply = await confirm_link(store, settings, link.chat_id, link.code)
    assert reply is not None and reply.startswith("✅")
    binding = await store.get_binding(link.chat_id)
    assert binding is not None
    return binding


@pytest.fixture
def fake_login(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A successful exchange + whoami; collects refresh tokens that get revoked."""
    revoked: list[str] = []

    async def fake_exchange(
        settings: Settings, code: str, verifier: str, client: object = None
    ) -> TokenSet:
        return _tokens(refresh=f"ref-{code}")

    async def fake_identity(mcp_url: str, access_token: str) -> Identity:
        return Identity("jane@acme", "Jane", False, [])

    async def fake_revoke(settings: Settings, token: str, client: object = None) -> None:
        revoked.append(token)

    monkeypatch.setattr(oauth_flow, "exchange_code", fake_exchange)
    monkeypatch.setattr(oauth_flow, "revoke", fake_revoke)
    monkeypatch.setattr(auth, "fetch_identity", fake_identity)
    return revoked


async def test_browser_login_alone_does_not_bind_the_chat(
    store_dsn: str, fake_login: list[str]
) -> None:
    # A forwarded login link: whoever opens it sees the code, the chat does not.
    async with BindingStore(store_dsn) as store:
        await store.create_pending("st", 901, "v")
        link = await complete_login(store, SETTINGS, state="st", code="c")
        assert await store.get_binding(901) is None
        assert len(link.code) == 6 and link.code.isdigit()

        reply = await confirm_link(store, SETTINGS, 901, "what are my invoices?")
        assert reply == auth.LINK_CODE_PROMPT
        assert await store.get_binding(901) is None


async def test_wrong_codes_discard_the_login_and_revoke_it(
    store_dsn: str, fake_login: list[str]
) -> None:
    async with BindingStore(store_dsn) as store:
        await store.create_pending("st", 902, "v")
        link = await complete_login(store, SETTINGS, state="st", code="c")
        wrong = "000000" if link.code != "000000" else "111111"
        for _ in range(MAX_LINK_ATTEMPTS - 1):
            reply = await confirm_link(store, SETTINGS, 902, wrong)
            assert reply is not None and "doesn't match" in reply
        reply = await confirm_link(store, SETTINGS, 902, wrong)
        assert reply is not None and "Too many" in reply

        assert await confirm_link(store, SETTINGS, 902, link.code) is None
        assert await store.get_binding(902) is None
    assert fake_login == ["ref-c"]


async def test_expired_link_code_is_refused(
    store_dsn: str, fake_login: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(auth, "LINK_CODE_TTL", timedelta(seconds=-1))
    async with BindingStore(store_dsn) as store:
        await store.create_pending("st", 903, "v")
        link = await complete_login(store, SETTINGS, state="st", code="c")
        reply = await confirm_link(store, SETTINGS, 903, link.code)
        assert reply is not None and "expired" in reply
        assert await store.get_binding(903) is None
    assert fake_login == ["ref-c"]


async def test_a_newer_login_replaces_and_revokes_the_older_one(
    store_dsn: str, fake_login: list[str]
) -> None:
    async with BindingStore(store_dsn) as store:
        await store.create_pending("s1", 904, "v")
        first = await complete_login(store, SETTINGS, state="s1", code="one")
        await store.create_pending("s2", 904, "v")
        second = await complete_login(store, SETTINGS, state="s2", code="two")
        assert fake_login == ["ref-one"]
        if first.code != second.code:
            reply = await confirm_link(store, SETTINGS, 904, first.code)
            assert reply is not None and "doesn't match" in reply
        reply = await confirm_link(store, SETTINGS, 904, f" {second.code} ")
        assert reply is not None and reply.startswith("✅")
        binding = await store.get_binding(904)
        assert binding is not None and binding.refresh_token == "ref-two"


async def test_login_page_status_follows_the_chat_confirmation(
    store_dsn: str, fake_login: list[str]
) -> None:
    async with BindingStore(store_dsn) as store:
        await store.create_pending("st", 905, "v")
        link = await complete_login(store, SETTINGS, state="st", code="c")
        assert link.status_token
        assert await link_status(store, link.status_token) == "pending"

        await confirm_link(store, SETTINGS, 905, link.code)
        assert await link_status(store, link.status_token) == "connected"
        assert await link_status(store, "not-a-real-token") == "expired"
        assert await link_status(store, "") == "expired"


async def test_login_page_status_expires_when_the_login_is_thrown_away(
    store_dsn: str, fake_login: list[str]
) -> None:
    async with BindingStore(store_dsn) as store:
        await store.create_pending("s1", 906, "v")
        first = await complete_login(store, SETTINGS, state="s1", code="one")
        await store.create_pending("s2", 906, "v")
        second = await complete_login(store, SETTINGS, state="s2", code="two")
        assert await link_status(store, first.status_token) == "expired"
        assert await link_status(store, second.status_token) == "pending"

        wrong = "000000" if second.code != "000000" else "111111"
        for _ in range(MAX_LINK_ATTEMPTS):
            await confirm_link(store, SETTINGS, 906, wrong)
        assert await link_status(store, second.status_token) == "expired"


async def test_login_page_status_expires_with_the_code(
    store_dsn: str, fake_login: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    async with BindingStore(store_dsn) as store:
        await store.create_pending("st", 907, "v")
        link = await complete_login(store, SETTINGS, state="st", code="c")
        monkeypatch.setattr(auth, "LINK_CODE_TTL", timedelta(seconds=-1))
        # Nothing typed in the chat yet, but the page must still stop waiting.
        assert await link_status(store, link.status_token) == "expired"


def test_parse_identity_variants() -> None:
    assert parse_identity('{"login": "a@b", "name": "A B"}') == Identity("a@b", "A B", False, [])
    blocks = [{"type": "text", "text": '{"login": "x", "name": "X"}'}]
    assert parse_identity(blocks) == Identity("x", "X", False, [])
    assert parse_identity("not json") == Identity()
    assert parse_identity([]) == Identity()


def test_parse_identity_detects_method_call_group() -> None:
    payload = (
        '{"login": "boss", "name": "Boss", '
        '"groups": ["Administration / Settings", "MCP: Allow Method Calls"]}'
    )
    identity = parse_identity(payload)
    assert (identity.login, identity.name, identity.can_call_methods) == ("boss", "Boss", True)
    # The full group list is kept, not just the one flag — the binding gate
    # reads it too, and a tuple of flags would need extending each time.
    assert "Administration / Settings" in identity.groups

    without = parse_identity('{"login": "cfo", "name": "CFO", "groups": ["MCP: Allow Write"]}')
    assert without.can_call_methods is False
    assert without.groups == ["MCP: Allow Write"]

    malformed = parse_identity('{"login": "u", "name": "U", "groups": "not-a-list"}')
    assert malformed == Identity("u", "U", False, [])  # defensive, never crashes


async def test_begin_login_stores_pending_matching_url(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        url = await begin_login(store, SETTINGS, chat_id=777)
        state = parse_qs(urlsplit(url).query)["state"][0]
        pending = await store.pop_pending(state)
        assert pending is not None
        assert pending.chat_id == 777
        assert len(pending.verifier) >= 43


async def test_complete_login_rejects_unknown_state(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        with pytest.raises(LoginError):
            await complete_login(store, SETTINGS, state="nope", code="c")


async def test_complete_login_happy_path(store_dsn: str, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_exchange(
        settings: Settings, code: str, verifier: str, client: object = None
    ) -> TokenSet:
        assert (code, verifier) == ("c0de", "the-verifier")
        return _tokens()

    async def fake_identity(mcp_url: str, access_token: str) -> Identity:
        assert access_token == "jwt-1"
        return Identity("priya@acme", "CFO Priya", True, ["MCP: Allow Method Calls"])

    monkeypatch.setattr(oauth_flow, "exchange_code", fake_exchange)
    monkeypatch.setattr(auth, "fetch_identity", fake_identity)

    async with BindingStore(store_dsn) as store:
        await store.create_pending("st", 555, "the-verifier")
        binding = await _login(store, SETTINGS, "st", "c0de")

        assert binding.odoo_login == "priya@acme"
        stored = await store.get_binding(555)
        assert stored is not None
        assert stored.access_token == "jwt-1"
        assert stored.can_call_methods is True  # learned from whoami, persisted
        assert stored.can_write is True  # granted scope included mcp:write


async def test_read_only_grant_is_stored_as_such(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A user without "MCP: Allow Write" gets a narrowed scope back. Recording
    # it is what stops the bot proposing writes that can only be refused.
    async def fake_exchange(
        settings: Settings, code: str, verifier: str, client: object = None
    ) -> TokenSet:
        return _tokens(scope="mcp:read")

    async def fake_identity(mcp_url: str, access_token: str) -> Identity:
        return Identity("daniel@acme", "Daniel", False, [])

    monkeypatch.setattr(oauth_flow, "exchange_code", fake_exchange)
    monkeypatch.setattr(auth, "fetch_identity", fake_identity)

    async with BindingStore(store_dsn) as store:
        await store.create_pending("st", 556, "v")
        binding = await _login(store, SETTINGS, "st", "c")

    assert binding.can_write is False


async def test_login_refused_without_the_required_group(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The access gate: valid Odoo credentials alone must not be enough to
    # point the agent at the company's data.
    async def fake_exchange(
        settings: Settings, code: str, verifier: str, client: object = None
    ) -> TokenSet:
        return _tokens()

    async def fake_identity(mcp_url: str, access_token: str) -> Identity:
        return Identity("stranger@acme", "Stranger", False, ["Some / Other Group"])

    monkeypatch.setattr(oauth_flow, "exchange_code", fake_exchange)
    monkeypatch.setattr(auth, "fetch_identity", fake_identity)
    gated = SETTINGS.model_copy(update={"required_odoo_group": "AI Assistant Users"})

    async with BindingStore(store_dsn) as store:
        await store.create_pending("st", 600, "v")
        with pytest.raises(LoginError) as excinfo:
            await complete_login(store, gated, state="st", code="c")

        # Nothing persisted: a refused user must leave no trace behind.
        assert await store.get_binding(600) is None

    assert "AI Assistant Users" in str(excinfo.value)


async def test_login_allowed_for_a_member_of_the_required_group(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_exchange(
        settings: Settings, code: str, verifier: str, client: object = None
    ) -> TokenSet:
        return _tokens()

    async def fake_identity(mcp_url: str, access_token: str) -> Identity:
        return Identity("priya@acme", "Priya", False, ["AI Assistant Users"])

    monkeypatch.setattr(oauth_flow, "exchange_code", fake_exchange)
    monkeypatch.setattr(auth, "fetch_identity", fake_identity)
    gated = SETTINGS.model_copy(update={"required_odoo_group": "AI Assistant Users"})

    async with BindingStore(store_dsn) as store:
        await store.create_pending("st", 601, "v")
        binding = await _login(store, gated, "st", "c")

    assert binding.odoo_login == "priya@acme"


async def test_no_required_group_means_any_internal_user_may_bind(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Empty setting keeps today's behaviour: the OAuth server's internal-user
    # check remains the only filter.
    async def fake_exchange(
        settings: Settings, code: str, verifier: str, client: object = None
    ) -> TokenSet:
        return _tokens()

    async def fake_identity(mcp_url: str, access_token: str) -> Identity:
        return Identity("anyone@acme", "Anyone", False, [])

    monkeypatch.setattr(oauth_flow, "exchange_code", fake_exchange)
    monkeypatch.setattr(auth, "fetch_identity", fake_identity)

    async with BindingStore(store_dsn) as store:
        await store.create_pending("st", 602, "v")
        binding = await _login(store, SETTINGS, "st", "c")

    assert binding.odoo_login == "anyone@acme"


async def test_ensure_access_returns_fresh_binding_untouched(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        fresh = Binding(
            chat_id=1,
            odoo_login="a",
            odoo_name="A",
            access_token="live-jwt",
            refresh_token="ref",
            expires_at=datetime.now(UTC) + timedelta(minutes=10),
        )
        await store.save_binding(fresh)
        result = await ensure_access(store, SETTINGS, 1)
        assert result is not None
        assert result.access_token == "live-jwt"


async def test_ensure_access_refreshes_and_persists_rotation(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_refresh(settings: Settings, token: str, client: object = None) -> TokenSet:
        assert token == "old-ref"
        return _tokens(access="jwt-new", refresh="ROTATED")

    monkeypatch.setattr(oauth_flow, "refresh", fake_refresh)

    async with BindingStore(store_dsn) as store:
        stale = Binding(
            chat_id=2,
            odoo_login="a",
            odoo_name="A",
            access_token="dead-jwt",
            refresh_token="old-ref",
            expires_at=datetime.now(UTC) - timedelta(seconds=5),
            can_call_methods=True,
        )
        await store.save_binding(stale)

        result = await ensure_access(store, SETTINGS, 2)

        assert result is not None
        assert result.access_token == "jwt-new"
        stored = await store.get_binding(2)
        assert stored is not None
        assert stored.refresh_token == "ROTATED"  # rotation persisted
        assert stored.can_call_methods is True  # identity flag survives refresh
        assert stored.can_write is True  # refreshed scope still grants write


async def test_refresh_downgrades_write_when_the_grant_narrows(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Revoking "MCP: Allow Write" in Odoo narrows the next token response.
    # Re-reading it here is what makes the change land without a /logout.
    async def fake_refresh(settings: Settings, token: str, client: object = None) -> TokenSet:
        return _tokens(access="jwt-new", refresh="ROTATED", scope="mcp:read")

    monkeypatch.setattr(oauth_flow, "refresh", fake_refresh)

    async with BindingStore(store_dsn) as store:
        await store.save_binding(
            Binding(
                chat_id=4,
                odoo_login="demoted",
                odoo_name="Demoted",
                access_token="dead",
                refresh_token="old-ref",
                expires_at=datetime.now(UTC) - timedelta(seconds=5),
                can_write=True,
            )
        )

        result = await ensure_access(store, SETTINGS, 4)

    assert result is not None
    assert result.can_write is False


async def test_ensure_access_drops_binding_on_dead_grant(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_refresh(settings: Settings, token: str, client: object = None) -> TokenSet:
        raise OAuthError("invalid_grant", "family revoked")

    monkeypatch.setattr(oauth_flow, "refresh", fake_refresh)

    async with BindingStore(store_dsn) as store:
        stale = Binding(
            chat_id=3,
            odoo_login="a",
            odoo_name="A",
            access_token="dead",
            refresh_token="revoked-ref",
            expires_at=datetime.now(UTC) - timedelta(seconds=5),
        )
        await store.save_binding(stale)

        assert await ensure_access(store, SETTINGS, 3) is None
        assert await store.get_binding(3) is None  # binding removed


async def test_ensure_access_unbound_chat(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        assert await ensure_access(store, SETTINGS, 999) is None


async def test_login_stores_the_odoo_timezone(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Drives when the morning brief is morning FOR THEM; one bot serves
    # users in different regions.
    async def fake_exchange(
        settings: Settings, code: str, verifier: str, client: object = None
    ) -> TokenSet:
        return _tokens()

    async def fake_identity(mcp_url: str, access_token: str) -> Identity:
        return Identity("priya@acme", "Priya", False, [], tz="Europe/London")

    monkeypatch.setattr(oauth_flow, "exchange_code", fake_exchange)
    monkeypatch.setattr(auth, "fetch_identity", fake_identity)

    async with BindingStore(store_dsn) as store:
        await store.create_pending("st", 700, "v")
        binding = await _login(store, SETTINGS, "st", "c")

    assert binding.tz == "Europe/London"


def test_identity_tolerates_a_missing_timezone() -> None:
    # Odoo users routinely have none set; it must parse to "" not crash.
    assert parse_identity('{"login": "a@b", "name": "A"}').tz == ""
    assert parse_identity('{"login": "a@b", "name": "A", "tz": "Asia/Kolkata"}').tz == (
        "Asia/Kolkata"
    )


def test_only_the_system_administrator_group_confers_admin() -> None:
    """21 Odoo groups are named "<App> / Administrator" — Sales, Accounting,
    Purchase and so on. Matching loosely would hand the cross-user audit
    view to a sales manager."""
    system = parse_identity('{"login": "a", "name": "A", "groups": ["Administration / Settings"]}')
    assert system.is_admin is True

    sales = parse_identity(
        '{"login": "b", "name": "B", "groups": ["Sales / Administrator",'
        ' "Accounting / Administrator"]}'
    )
    assert sales.is_admin is False

    assert parse_identity('{"login": "c", "name": "C"}').is_admin is False


async def test_login_records_administrator_status(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_exchange(
        settings: Settings, code: str, verifier: str, client: object = None
    ) -> TokenSet:
        return _tokens()

    async def fake_identity(mcp_url: str, access_token: str) -> Identity:
        return Identity("boss@acme", "Boss", False, ["Administration / Settings"], is_admin=True)

    monkeypatch.setattr(oauth_flow, "exchange_code", fake_exchange)
    monkeypatch.setattr(auth, "fetch_identity", fake_identity)

    async with BindingStore(store_dsn) as store:
        await store.create_pending("st", 800, "v")
        binding = await _login(store, SETTINGS, "st", "c")

    assert binding.is_admin is True


async def test_login_refused_for_a_group_chat(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second gate on the private-chat rule, independent of bot.guard: the
    binding is written from the HTTP callback, which never passes through
    the Telegram handler chain.

    Telegram ids answer it with no lookup — a private chat id IS the user's
    own id and is positive, while group, supergroup and channel ids are
    negative. Binding a group would give every member of it one person's
    Odoo identity, permissions and audit trail.
    """

    async def explode(*args: object, **kwargs: object) -> object:
        raise AssertionError("must refuse before redeeming the code")

    monkeypatch.setattr(oauth_flow, "exchange_code", explode)
    monkeypatch.setattr(auth, "fetch_identity", explode)

    group_chat_id = -1001234567890
    async with BindingStore(store_dsn) as store:
        await store.create_pending("st-group", group_chat_id, "v")
        with pytest.raises(LoginError) as excinfo:
            await complete_login(store, SETTINGS, state="st-group", code="c")

        # Nothing persisted: a refusal that still saved credentials would be
        # worse than no check at all.
        assert await store.get_binding(group_chat_id) is None

    assert "private chat" in str(excinfo.value)


async def test_a_transient_failure_raises_and_keeps_the_login(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Odoo restarting must not log everyone out.

    Only invalid_grant means the token family is genuinely dead. A
    server_error is the ERP having a bad moment, and deleting the binding
    for that would silently unbind every user whose token happened to need
    refreshing during a restart — each of whom would then be told to
    /start again for no reason they could see.
    """

    async def fake_refresh(settings: Settings, token: str, client: object = None) -> TokenSet:
        raise OAuthError("server_error", "Odoo is restarting")

    monkeypatch.setattr(oauth_flow, "refresh", fake_refresh)

    async with BindingStore(store_dsn) as store:
        await store.save_binding(
            Binding(
                chat_id=41,
                odoo_login="a",
                odoo_name="A",
                access_token="dead",
                refresh_token="still-good-ref",
                expires_at=datetime.now(UTC) - timedelta(seconds=5),
            )
        )

        with pytest.raises(OAuthError) as excinfo:
            await ensure_access(store, SETTINGS, 41)

        assert excinfo.value.code == "server_error"
        # The assertion that matters: the login survives, so the user's next
        # question works once Odoo is back — no /start needed.
        stored = await store.get_binding(41)
        assert stored is not None
        assert stored.refresh_token == "still-good-ref"


async def test_concurrent_callers_refresh_only_once(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refresh tokens ROTATE, so redeeming one invalidates it.

    The proactive ticker is its own asyncio task and can reach a chat at the
    same moment a handler does. Without the per-chat lock both would refresh,
    and the second would be a replay of a spent token — forgiven only by a
    server-side grace window the agent does not control.
    """
    calls: list[str] = []

    async def fake_refresh(settings: Settings, token: str, client: object = None) -> TokenSet:
        calls.append(token)
        await asyncio.sleep(0)  # yield, so the other caller really interleaves
        return _tokens(access="jwt-new", refresh="ROTATED")

    monkeypatch.setattr(oauth_flow, "refresh", fake_refresh)

    async with BindingStore(store_dsn) as store:
        await store.save_binding(
            Binding(
                chat_id=42,
                odoo_login="a",
                odoo_name="A",
                access_token="dead",
                refresh_token="old-ref",
                expires_at=datetime.now(UTC) - timedelta(seconds=5),
            )
        )

        first, second = await asyncio.gather(
            ensure_access(store, SETTINGS, 42),
            ensure_access(store, SETTINGS, 42),
        )

    assert calls == ["old-ref"], "the spent token must not be redeemed twice"
    assert first is not None and second is not None
    # The waiter re-read inside the lock rather than acting on the stale copy
    # it captured before waiting — which is the entire point of holding it.
    assert first.access_token == second.access_token == "jwt-new"


async def test_each_chat_refreshes_independently(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The lock is per chat, not global: one user's slow refresh must not
    # stall everybody else's.
    calls: list[str] = []

    async def fake_refresh(settings: Settings, token: str, client: object = None) -> TokenSet:
        calls.append(token)
        return _tokens(access="jwt-new", refresh="ROTATED")

    monkeypatch.setattr(oauth_flow, "refresh", fake_refresh)

    async with BindingStore(store_dsn) as store:
        for chat_id, ref in ((51, "ref-51"), (52, "ref-52")):
            await store.save_binding(
                Binding(
                    chat_id=chat_id,
                    odoo_login="a",
                    odoo_name="A",
                    access_token="dead",
                    refresh_token=ref,
                    expires_at=datetime.now(UTC) - timedelta(seconds=5),
                )
            )

        await asyncio.gather(
            ensure_access(store, SETTINGS, 51),
            ensure_access(store, SETTINGS, 52),
        )

    assert sorted(calls) == ["ref-51", "ref-52"]


async def test_logout_succeeds_when_odoo_is_unreachable(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unreachable(settings: Settings, token: str, client: object = None) -> None:
        raise httpx.ConnectError("getaddrinfo failed")

    monkeypatch.setattr(oauth_flow, "revoke", unreachable)
    async with BindingStore(store_dsn) as store:
        await store.save_binding(
            Binding(
                chat_id=950,
                odoo_login="a",
                odoo_name="A",
                access_token="jwt",
                refresh_token="ref",
                expires_at=datetime.now(UTC) + timedelta(minutes=10),
            )
        )
        assert await logout(store, SETTINGS, 950) is True
        assert await store.get_binding(950) is None


def test_token_calls_use_the_mcp_host_not_the_browser_url() -> None:
    # The browser URL is a tunnel that changes on every restart.
    settings = SETTINGS.model_copy(update={"odoo_public_url": "https://old-tunnel.example"})
    assert settings.token_url == "http://odoo:8160/mcp/oauth/token"
    assert settings.revoke_url == "http://odoo:8160/mcp/oauth/revoke"
    assert settings.authorize_url == "https://old-tunnel.example/mcp/oauth/authorize"
