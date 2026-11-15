"""Per-user persistence: Telegram chat ↔ Odoo OAuth token bindings.

PostgreSQL via psycopg. Six tables, of which two carry the weight:

* ``bindings`` — one row per Telegram chat, holding the Odoo identity and the
  current OAuth token pair. Refresh rotation MUST overwrite the stored refresh
  token every time. The two token columns are encrypted (see ``crypto.py``).
* ``pending_auth`` — in-flight login attempts, keyed by the OAuth ``state``;
  single-use (pop) and expired after 10 minutes.

Postgres rather than a SQLite file so that agent state lands inside a
client's existing backup regime, and so more than one agent process is
possible at all — SQLite is single-writer.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import Any, Literal, Self

import psycopg
from psycopg_pool import AsyncConnectionPool

from ont_agent.crypto import TokenCipher, TokenDecryptionError, is_encrypted
from ont_agent.log import get_logger

logger = get_logger(__name__)

PENDING_TTL = timedelta(minutes=10)
#: How long a login page's outcome stays answerable; the page stops polling
#: long before this.
LINK_STATUS_RETENTION = timedelta(days=1)

#: What a login page's poll reports: still waiting for the chat code, bound,
#: or thrown away (expired, too many wrong codes, or replaced by a newer login).
LinkStatus = Literal["pending", "connected", "expired"]

#: BIGINT for every chat_id, and this is load-bearing rather than tidy:
#: Telegram supergroup ids are below -2^31 and user ids now exceed 2^31.
#: SQLite's INTEGER is 64-bit so it never mattered; Postgres INTEGER is
#: 32-bit and would overflow on a real id — failing for SOME users only,
#: which is the worst way to fail.
#:
#: The boolean-ish columns stay INTEGER rather than becoming BOOLEAN: they
#: are read as bool(row[n]) throughout, so keeping the type leaves every
#: row-unpacking site untouched.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS bindings (
    chat_id BIGINT PRIMARY KEY,
    odoo_login TEXT NOT NULL,
    odoo_name TEXT NOT NULL,
    access_token TEXT NOT NULL,
    refresh_token TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    can_call_methods INTEGER NOT NULL DEFAULT 0,
    can_write INTEGER NOT NULL DEFAULT 1,
    tz TEXT NOT NULL DEFAULT '',
    is_admin INTEGER NOT NULL DEFAULT 0,
    brief_enabled INTEGER NOT NULL DEFAULT 1,
    last_brief_date TEXT NOT NULL DEFAULT '',
    last_digest_week TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS watchers (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    chat_id BIGINT NOT NULL,
    odoo_login TEXT NOT NULL,
    condition TEXT NOT NULL,
    created_at TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    last_checked_at TEXT NOT NULL DEFAULT '',
    fingerprint TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS action_log (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    chat_id BIGINT NOT NULL,
    odoo_login TEXT NOT NULL,
    tool TEXT NOT NULL,
    summary TEXT NOT NULL,
    outcome TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS action_log_when ON action_log (created_at);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pending_auth (
    state TEXT PRIMARY KEY,
    chat_id BIGINT NOT NULL,
    verifier TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_spaces (
    chat_id BIGINT PRIMARY KEY,
    space TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pending_links (
    chat_id BIGINT PRIMARY KEY,
    code_hash TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    odoo_login TEXT NOT NULL,
    odoo_name TEXT NOT NULL,
    access_token TEXT NOT NULL,
    refresh_token TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    can_call_methods INTEGER NOT NULL DEFAULT 0,
    can_write INTEGER NOT NULL DEFAULT 0,
    tz TEXT NOT NULL DEFAULT '',
    is_admin INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS link_status (
    token_hash TEXT PRIMARY KEY,
    chat_id BIGINT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS link_status_chat ON link_status (chat_id);
CREATE TABLE IF NOT EXISTS usage_daily (
    subject TEXT NOT NULL,
    day TEXT NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0,
    spend_usd DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    PRIMARY KEY (subject, day)
);
CREATE TABLE IF NOT EXISTS llm_calls (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at TEXT NOT NULL,
    generation_id TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL,
    ttft_ms DOUBLE PRECISION,
    generation_ms DOUBLE PRECISION,
    total_ms DOUBLE PRECISION NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS llm_calls_when ON llm_calls (created_at);
CREATE INDEX IF NOT EXISTS llm_calls_model ON llm_calls (model, created_at);
CREATE TABLE IF NOT EXISTS proactive_state (
    chat_id BIGINT PRIMARY KEY,
    brief_enabled INTEGER NOT NULL DEFAULT 1,
    last_brief_date TEXT NOT NULL DEFAULT '',
    last_digest_week TEXT NOT NULL DEFAULT ''
);
INSERT INTO proactive_state (chat_id, brief_enabled, last_brief_date, last_digest_week)
    SELECT chat_id, brief_enabled, last_brief_date, last_digest_week FROM bindings
    ON CONFLICT (chat_id) DO NOTHING;
"""
# proactive_state holds the brief/digest stamps and the /brief opt-out apart
# from `bindings`, because /logout deletes the binding row. Kept there, a
# logout-and-login reset them: a second brief the same morning (with the
# first-brief preamble), a repeated weekly digest, and a silently undone
# /brief off. The INSERT ... SELECT is the one-time copy from the old
# columns; DO NOTHING makes it a no-op once a chat has a row, so it is safe
# on every start. The old bindings columns are no longer read or written.

#: Usage rows are keyed by a prefixed SUBJECT rather than a chat id: the
#: same person in two chats must share one budget, and the bill needs a
#: total of its own. Prefixes make the three kinds collision-proof.
GLOBAL_SUBJECT = "global"


def user_subject(odoo_login: str) -> str:
    """Budget subject for a bound chat — the Odoo user, not the chat."""
    return f"user:{odoo_login}"


def chat_subject(chat_id: int) -> str:
    """Budget subject for a chat with no binding yet."""
    return f"chat:{chat_id}"


@dataclass(frozen=True)
class Binding:
    chat_id: int
    odoo_login: str
    odoo_name: str
    access_token: str
    refresh_token: str
    expires_at: datetime
    #: Learned from whoami at login: member of "MCP: Allow Method Calls".
    #: UX hint only — the server enforces the gate regardless.
    can_call_methods: bool = False
    #: Whether the GRANTED OAuth scope includes mcp:write. Refreshed from
    #: every token response, so a permission change lands within one token
    #: lifetime. Defaults True so a pre-existing binding is never silently
    #: demoted before its first refresh; the server enforces regardless.
    can_write: bool = True
    #: The user's Odoo timezone, from whoami at login. Decides when "their
    #: morning" is; empty falls back to Settings.default_timezone.
    tz: str = ""
    #: Member of Odoo 17's "Administration / Settings" group. Gates the cross-user
    #: audit view; the EXACT name matters, since 21 other groups are called
    #: "<App> / Administrator" and none of them is the system administrator.
    is_admin: bool = False
    #: Morning brief opt-out (/brief off).
    brief_enabled: bool = True
    #: ISO date of the last brief sent, so a restart cannot double-send.
    last_brief_date: str = ""
    #: ISO week of the last weekly digest, e.g. "2026-W36".
    last_digest_week: str = ""


@dataclass(frozen=True)
class UsageRow:
    """One chat's usage for one day; zeros when nothing was recorded yet."""

    message_count: int = 0
    spend_usd: float = 0.0


#: What became of a proposed write. These four are exhaustive because every
#: write tool is confirm-gated: nothing reaches Odoo without passing through
#: the approval the outcome describes.
ActionOutcome = Literal["done", "blocked", "cancelled", "revised"]


@dataclass(frozen=True)
class LoggedAction:
    """One write the bot proposed, and what happened to it."""

    id: int
    chat_id: int
    odoo_login: str
    tool: str
    summary: str
    outcome: str
    created_at: datetime


@dataclass(frozen=True)
class Watcher:
    """A standing condition, checked on a schedule and alerted on once.

    ``fingerprint`` identifies WHAT last triggered it (say, the set of
    invoice numbers). An alert fires only when the condition is true AND the
    fingerprint has changed, so the same overdue invoice does not shout at
    every check while a genuinely new one still does.
    """

    id: int
    chat_id: int
    odoo_login: str
    condition: str
    created_at: datetime
    last_checked_at: datetime | None = None
    fingerprint: str = ""


@dataclass(frozen=True)
class LlmCall:
    """One model call, and how long it took.

    Two timing sources, and they measure different things:

    * ``total_ms`` is OUR wall clock around the call and is always present.
    * ``ttft_ms`` and ``generation_ms`` are OpenRouter's own figures, read
      afterwards from ``GET /generation``; both are ``None`` when that
      lookup did not succeed.

    ``generation_ms`` is OpenRouter's whole request, measured at their edge,
    so decode time is ``generation_ms - ttft_ms`` and ``total_ms -
    generation_ms`` is everything that is not the model — our network hop
    and client overhead, 464-1494ms across the first four live calls.
    See ``metrics.GenerationStats`` for how that was pinned down, and why
    the spec's example values argue for the opposite and are wrong.

    ``model`` is the model that ANSWERED, taken from the response rather
    than from settings: with fallbacks configured the two differ exactly
    when it matters most.
    """

    model: str
    kind: str
    total_ms: float
    generation_id: str = ""
    provider: str = ""
    ttft_ms: float | None = None
    generation_ms: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    #: When the CALL ended, stamped by the handler — not when the row was
    #: written. The two are up to ~22 seconds apart, because the row waits on
    #: the OpenRouter lookup and its backoff, and rows can be written out of
    #: order when one call's stats take longer to appear than a later call's.
    #: A timing table whose own timestamp is the time we got round to it
    #: would misorder the last ten calls and straddle its own day boundary.
    ended_at: datetime | None = None


@dataclass(frozen=True)
class LatencyRow:
    """One model's latency over a window — medians, never means.

    A single 90-second outlier swamps a mean, and the question this answers
    is what a typical answer costs in waiting.
    """

    model: str
    calls: int
    ttft_samples: int
    median_ttft_ms: float | None
    p90_ttft_ms: float | None
    median_total_ms: float
    cost_usd: float


@dataclass(frozen=True)
class PendingAuth:
    state: str
    chat_id: int
    verifier: str


@dataclass(frozen=True)
class PendingLink:
    """A completed browser login waiting for its code to be typed in the chat."""

    binding: Binding
    code_hash: str
    attempts: int
    created_at: datetime


#: Connections BindingStore's own pool may open. Small on purpose: these
#: are short queries, while LangGraph's separate pool does the heavy
#: checkpoint writes. bot.py overrides it from Settings.
DEFAULT_POOL_MAX_SIZE = 4


class BindingStore:
    """Async store; use as ``async with BindingStore(dsn) as store:``."""

    def __init__(
        self,
        dsn: str,
        cipher: TokenCipher | None = None,
        max_size: int = DEFAULT_POOL_MAX_SIZE,
    ) -> None:
        """:param dsn: PostgreSQL connection string.
        :param cipher: encrypts the OAuth token columns; None stores them
        in plaintext, which is the behaviour before AGENT_TOKEN_KEY existed.
        :param max_size: connections the pool may open."""
        self._dsn = dsn
        self._pool: AsyncConnectionPool[psycopg.AsyncConnection[Any]] | None = None
        self._cipher = cipher
        self._max_size = max_size

    async def __aenter__(self) -> Self:
        # A POOL, not one connection, and that is a correctness requirement
        # rather than a performance one. Three tasks share this store on one
        # event loop: the Telegram handlers, the proactive ticker, and the
        # embedded callback server. A psycopg transaction is CONNECTION
        # scoped, so while record_usage held one open on a shared
        # connection, any other task's write joined it — measured live
        # 2026-09-04, and it does not fail quietly: a concurrent method that
        # commits raises ProgrammingError("Explicit commit() forbidden
        # within a Transaction context"), and one that does not raises
        # OutOfOrderTransactionNesting. A connection each makes that
        # impossible by construction.
        #
        # autocommit stays, for the reason it was introduced: in psycopg's
        # default mode a plain SELECT opens a transaction nothing closes, so
        # the connection sits "idle in transaction" — which stops VACUUM
        # reclaiming dead tuples and blocks CREATE INDEX CONCURRENTLY
        # outright. Now it applies to every pooled connection.
        self._pool = AsyncConnectionPool(
            self._dsn,
            min_size=1,
            max_size=self._max_size,
            open=False,
            kwargs={"autocommit": True},
        )
        await self._pool.open(wait=True)
        async with self.connection() as db:
            # Multi-statement DDL in one round trip; every CREATE is IF NOT
            # EXISTS, so this is safe to run on every start.
            await db.execute(_SCHEMA)
            await self._check_token_key(db)
        logger.debug("Binding store connected")
        return self

    async def _check_token_key(self, db: psycopg.AsyncConnection[Any]) -> None:
        """Notice a changed encryption key and say so in plain words.

        Without this, rotating AGENT_TOKEN_KEY looks like an unexplained
        mass logout. With it, the reason is stated once at startup. The same
        applies to REMOVING the key, which is otherwise indistinguishable
        from never having set one.
        """
        if self._cipher is None:
            async with await db.execute(
                "SELECT value FROM meta WHERE key = 'token_key_fingerprint'"
            ) as cursor:
                previous = await cursor.fetchone()
            if previous:
                logger.warning(
                    "AGENT_TOKEN_KEY was removed (was %s); existing logins cannot be "
                    "read and those users must reconnect with /start",
                    previous[0],
                )
            else:
                logger.warning("OAuth tokens are stored UNENCRYPTED (AGENT_TOKEN_KEY is not set)")
            return
        async with await db.execute(
            "SELECT value FROM meta WHERE key = 'token_key_fingerprint'"
        ) as cursor:
            row = await cursor.fetchone()
        current = self._cipher.fingerprint
        if row and row[0] != current:
            logger.warning(
                "AGENT_TOKEN_KEY has changed (%s -> %s); existing logins cannot be "
                "decrypted and those users must reconnect with /start",
                row[0],
                current,
            )
        if not row or row[0] != current:
            await db.execute(
                "INSERT INTO meta (key, value) VALUES ('token_key_fingerprint', %s)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (current,),
            )

    def _encrypt(self, value: str) -> str:
        return self._cipher.encrypt(value) if self._cipher else value

    def _decrypt(self, value: str) -> str:
        """:raise TokenDecryptionError: encrypted with a different key, or
        encrypted while no key is configured at all."""
        if self._cipher:
            return self._cipher.decrypt(value)
        if is_encrypted(value):
            # A removed AGENT_TOKEN_KEY. Without this the ciphertext would be
            # handed to Odoo as a bearer token and come back 401 — a confusing
            # auth failure instead of the actual cause, which is a config
            # change made minutes earlier.
            raise TokenDecryptionError(
                "Stored token is encrypted but AGENT_TOKEN_KEY is not set; "
                "restore the key, or affected users must reconnect with /start."
            )
        return value

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def connection(self) -> AbstractAsyncContextManager[psycopg.AsyncConnection[Any]]:
        """Borrow a connection for one unit of work.

        Every method holds one only for as long as its own statements take,
        so no two tasks ever share a transaction. Also the seam tests use to
        forge or age rows directly.
        """
        if self._pool is None:
            raise RuntimeError("BindingStore used outside 'async with'")
        return self._pool.connection()

    # -- bindings -------------------------------------------------------
    async def get_binding(self, chat_id: int) -> Binding | None:
        async with (
            self.connection() as db,
            await db.execute(
                "SELECT b.chat_id, b.odoo_login, b.odoo_name, b.access_token, b.refresh_token,"
                " b.expires_at, b.can_call_methods, b.can_write, b.tz,"
                " COALESCE(p.brief_enabled, 1), COALESCE(p.last_brief_date, ''), b.is_admin,"
                " COALESCE(p.last_digest_week, '')"
                " FROM bindings b LEFT JOIN proactive_state p ON p.chat_id = b.chat_id"
                " WHERE b.chat_id = %s",
                (chat_id,),
            ) as cursor,
        ):
            row = await cursor.fetchone()
        if row is None:
            return None
        try:
            access_token = self._decrypt(row[3])
            refresh_token = self._decrypt(row[4])
        except TokenDecryptionError:
            # Treated as "not connected" rather than fatal: one unreadable
            # row must not take the service down for everyone else, and the
            # user simply reconnects.
            logger.error(
                "Cannot decrypt tokens for chat %s — AGENT_TOKEN_KEY has most likely "
                "changed; this user must reconnect with /start",
                binding_id(row[0]),
            )
            return None
        return Binding(
            chat_id=row[0],
            odoo_login=row[1],
            odoo_name=row[2],
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=datetime.fromisoformat(row[5]),
            can_call_methods=bool(row[6]),
            can_write=bool(row[7]),
            tz=row[8],
            brief_enabled=bool(row[9]),
            last_brief_date=row[10],
            is_admin=bool(row[11]),
            last_digest_week=row[12],
        )

    async def save_binding(self, binding: Binding) -> None:
        async with self.connection() as db:
            await db.execute(
                "INSERT INTO bindings"
                " (chat_id, odoo_login, odoo_name, access_token, refresh_token, expires_at,"
                "  can_call_methods, can_write, tz, is_admin)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT(chat_id) DO UPDATE SET"
                "  odoo_login=excluded.odoo_login, odoo_name=excluded.odoo_name,"
                "  access_token=excluded.access_token, refresh_token=excluded.refresh_token,"
                "  expires_at=excluded.expires_at, can_call_methods=excluded.can_call_methods,"
                "  can_write=excluded.can_write, tz=excluded.tz,"
                "  is_admin=excluded.is_admin",
                (
                    binding.chat_id,
                    binding.odoo_login,
                    binding.odoo_name,
                    self._encrypt(binding.access_token),
                    self._encrypt(binding.refresh_token),
                    binding.expires_at.isoformat(),
                    int(binding.can_call_methods),
                    int(binding.can_write),
                    binding.tz,
                    int(binding.is_admin),
                ),
            )
        logger.info("Binding saved for chat %s (%s)", binding.chat_id, binding.odoo_login)

    async def delete_binding(self, chat_id: int) -> None:
        async with self.connection() as db:
            await db.execute("DELETE FROM bindings WHERE chat_id = %s", (chat_id,))
        logger.info("Binding deleted for chat %s", binding_id(chat_id))

    # -- logins awaiting chat confirmation ------------------------------
    async def save_pending_link(self, binding: Binding, code_hash: str) -> None:
        async with self.connection() as db:
            await db.execute(
                "INSERT INTO pending_links"
                " (chat_id, code_hash, attempts, created_at, odoo_login, odoo_name,"
                "  access_token, refresh_token, expires_at, can_call_methods, can_write,"
                "  tz, is_admin)"
                " VALUES (%s, %s, 0, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT(chat_id) DO UPDATE SET"
                "  code_hash=excluded.code_hash, attempts=0, created_at=excluded.created_at,"
                "  odoo_login=excluded.odoo_login, odoo_name=excluded.odoo_name,"
                "  access_token=excluded.access_token, refresh_token=excluded.refresh_token,"
                "  expires_at=excluded.expires_at, can_call_methods=excluded.can_call_methods,"
                "  can_write=excluded.can_write, tz=excluded.tz, is_admin=excluded.is_admin",
                (
                    binding.chat_id,
                    code_hash,
                    datetime.now(UTC).isoformat(),
                    binding.odoo_login,
                    binding.odoo_name,
                    self._encrypt(binding.access_token),
                    self._encrypt(binding.refresh_token),
                    binding.expires_at.isoformat(),
                    int(binding.can_call_methods),
                    int(binding.can_write),
                    binding.tz,
                    int(binding.is_admin),
                ),
            )
        logger.info("Login awaiting chat confirmation for chat %s", binding_id(binding.chat_id))

    async def get_pending_link(self, chat_id: int) -> PendingLink | None:
        async with (
            self.connection() as db,
            await db.execute(
                "SELECT code_hash, attempts, created_at, odoo_login, odoo_name, access_token,"
                " refresh_token, expires_at, can_call_methods, can_write, tz, is_admin"
                " FROM pending_links WHERE chat_id = %s",
                (chat_id,),
            ) as cursor,
        ):
            row = await cursor.fetchone()
        if row is None:
            return None
        try:
            access_token = self._decrypt(row[5])
            refresh_token = self._decrypt(row[6])
        except TokenDecryptionError:
            await self.delete_pending_link(chat_id)
            return None
        binding = Binding(
            chat_id=chat_id,
            odoo_login=row[3],
            odoo_name=row[4],
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=datetime.fromisoformat(row[7]),
            can_call_methods=bool(row[8]),
            can_write=bool(row[9]),
            tz=row[10],
            is_admin=bool(row[11]),
        )
        return PendingLink(
            binding=binding,
            code_hash=row[0],
            attempts=int(row[1]),
            created_at=datetime.fromisoformat(row[2]),
        )

    async def bump_link_attempts(self, chat_id: int) -> None:
        async with self.connection() as db:
            await db.execute(
                "UPDATE pending_links SET attempts = attempts + 1 WHERE chat_id = %s", (chat_id,)
            )

    async def delete_pending_link(self, chat_id: int) -> None:
        async with self.connection() as db:
            await db.execute("DELETE FROM pending_links WHERE chat_id = %s", (chat_id,))

    # -- login page status ---------------------------------------------
    async def create_link_status(self, token_hash: str, chat_id: int) -> None:
        """Start tracking a login page as ``pending``; old rows are purged here."""
        async with self.connection() as db:
            cutoff = (datetime.now(UTC) - LINK_STATUS_RETENTION).isoformat()
            await db.execute("DELETE FROM link_status WHERE created_at < %s", (cutoff,))
            await db.execute(
                "INSERT INTO link_status (token_hash, chat_id, status, created_at)"
                " VALUES (%s, %s, 'pending', %s) ON CONFLICT (token_hash) DO NOTHING",
                (token_hash, chat_id, datetime.now(UTC).isoformat()),
            )

    async def settle_link_status(self, chat_id: int, status: LinkStatus) -> None:
        """Record the outcome for every login page still pending for ``chat_id``."""
        async with self.connection() as db:
            await db.execute(
                "UPDATE link_status SET status = %s WHERE chat_id = %s AND status = 'pending'",
                (status, chat_id),
            )

    async def get_link_status(self, token_hash: str) -> tuple[LinkStatus, datetime] | None:
        async with (
            self.connection() as db,
            await db.execute(
                "SELECT status, created_at FROM link_status WHERE token_hash = %s",
                (token_hash,),
            ) as cursor,
        ):
            row = await cursor.fetchone()
        if row is None:
            return None
        return row[0], datetime.fromisoformat(row[1])

    # -- pending logins -------------------------------------------------
    async def create_pending(self, state: str, chat_id: int, verifier: str) -> None:
        async with self.connection() as db:
            await self._purge_expired(db)
            await db.execute(
                "INSERT INTO pending_auth (state, chat_id, verifier, created_at)"
                " VALUES (%s, %s, %s, %s)"
                # SQLite's INSERT OR REPLACE, spelled out: a repeated state is
                # a retried login and must supersede the first attempt.
                " ON CONFLICT (state) DO UPDATE SET"
                "  chat_id = excluded.chat_id,"
                "  verifier = excluded.verifier,"
                "  created_at = excluded.created_at",
                (state, chat_id, verifier, datetime.now(UTC).isoformat()),
            )

    async def pop_pending(self, state: str) -> PendingAuth | None:
        """Return and delete the pending login for ``state`` — single use."""
        async with self.connection() as db:
            await self._purge_expired(db)
            async with await db.execute(
                "SELECT state, chat_id, verifier FROM pending_auth WHERE state = %s",
                (state,),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return None
            await db.execute("DELETE FROM pending_auth WHERE state = %s", (state,))
        return PendingAuth(state=row[0], chat_id=row[1], verifier=row[2])

    async def _purge_expired(self, db: psycopg.AsyncConnection[Any]) -> None:
        """Takes the caller's connection, so this joins that unit of work
        rather than borrowing a second one for a single DELETE."""
        cutoff = (datetime.now(UTC) - PENDING_TTL).isoformat()
        await db.execute("DELETE FROM pending_auth WHERE created_at < %s", (cutoff,))

    # -- daily usage (rate limit + spend cap) ----------------------------
    async def get_usage(self, subject: str, day: str) -> UsageRow:
        async with (
            self.connection() as db,
            await db.execute(
                "SELECT message_count, spend_usd FROM usage_daily WHERE subject = %s AND day = %s",
                (subject, day),
            ) as cursor,
        ):
            row = await cursor.fetchone()
        if row is None:
            return UsageRow()
        return UsageRow(message_count=row[0], spend_usd=row[1])

    async def usage_between(self, subject: str, first_day: str, last_day: str) -> UsageRow:
        """Usage summed over ``first_day``..``last_day`` inclusive (ISO dates)."""
        async with (
            self.connection() as db,
            await db.execute(
                "SELECT COALESCE(SUM(message_count), 0), COALESCE(SUM(spend_usd), 0)"
                " FROM usage_daily WHERE subject = %s AND day >= %s AND day <= %s",
                (subject, first_day, last_day),
            ) as cursor,
        ):
            row = await cursor.fetchone()
        if row is None:
            return UsageRow()
        return UsageRow(message_count=int(row[0]), spend_usd=float(row[1]))

    async def record_usage(
        self, subject: str, day: str, spend_usd: float, count_message: bool = True
    ) -> None:
        """One more message (and its real cost) for this subject today.

        ``count_message=False`` records spend only — for unattended work
        (briefs, watchers) that nobody asked for, so the digest's "you asked
        N questions" and the daily message cap count people, not the ticker.

        Also increments the global row in the SAME transaction — one call
        site, so the total can never drift from the parts.
        """
        increment = 1 if count_message else 0
        # Explicit, because under autocommit these would otherwise be two
        # separate transactions — and the whole point of one call site is
        # that the global total can never drift from the parts.
        # On its OWN pooled connection, which is what makes the transaction
        # safe. Held on a connection shared with the ticker, any concurrent
        # write joined it: one that commits raised ProgrammingError, one
        # that does not raised OutOfOrderTransactionNesting.
        async with self.connection() as db, db.transaction():
            for key in (subject, GLOBAL_SUBJECT):
                await db.execute(
                    "INSERT INTO usage_daily (subject, day, message_count, spend_usd)"
                    " VALUES (%s, %s, %s, %s)"
                    " ON CONFLICT(subject, day) DO UPDATE SET"
                    # Qualified, because Postgres cannot tell the existing
                    # row's column from the one being set; SQLite allowed
                    # the bare name.
                    "  message_count = usage_daily.message_count + excluded.message_count,"
                    "  spend_usd = usage_daily.spend_usd + excluded.spend_usd",
                    (key, day, increment, spend_usd),
                )

    async def prune_usage(self, cutoff_day: str) -> int:
        """Drop usage rows for days before ``cutoff_day``; returns how many.

        Safe as long as retention exceeds a week: the limit checks ask about
        today, and the weekly digest sums the last seven days. Nothing reads
        further back, so nothing can be broken by its absence.
        """
        async with self.connection() as db:
            cursor = await db.execute("DELETE FROM usage_daily WHERE day < %s", (cutoff_day,))
            return cursor.rowcount

    # -- per-call latency (metrics.py) -----------------------------------
    async def record_llm_call(self, call: LlmCall, when: datetime | None = None) -> None:
        """Store one model call's timings. Never raises.

        Measurement must never be able to break an answer, and by the time
        this runs the answer has already been sent — so a failure here is
        logged and dropped rather than propagated to the worker loop.
        """
        # The call's own end time wins: `when` is only for tests, and the
        # clock here is up to ~22s later than the thing being timed.
        moment = (call.ended_at or when or datetime.now(UTC)).isoformat()
        try:
            async with self.connection() as db:
                await db.execute(
                    "INSERT INTO llm_calls (created_at, generation_id, model, provider, kind,"
                    " ttft_ms, generation_ms, total_ms, prompt_tokens, completion_tokens,"
                    " cached_tokens, cost_usd)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        moment,
                        call.generation_id,
                        call.model,
                        call.provider,
                        call.kind,
                        call.ttft_ms,
                        call.generation_ms,
                        call.total_ms,
                        call.prompt_tokens,
                        call.completion_tokens,
                        call.cached_tokens,
                        call.cost_usd,
                    ),
                )
        except psycopg.Error:
            logger.warning("Could not record latency for %s", call.model, exc_info=True)

    async def prune_llm_calls(self, cutoff: str) -> int:
        """Drop latency rows created before ``cutoff``; returns how many."""
        async with self.connection() as db:
            cursor = await db.execute("DELETE FROM llm_calls WHERE created_at < %s", (cutoff,))
            return cursor.rowcount

    async def llm_latency_report(self, since: str, kind: str = "") -> list[LatencyRow]:
        """Latency per model since ``since``, slowest median first.

        ``ttft_samples`` is counted separately from ``calls`` because the
        OpenRouter lookup can fail: a median over three samples out of two
        hundred calls is a very different claim from one over all of them,
        and the row has to be able to say so.
        """
        clause = " AND kind = %s" if kind else ""
        params: tuple[Any, ...] = (since, kind) if kind else (since,)
        async with (
            self.connection() as db,
            await db.execute(
                "SELECT model, COUNT(*), COUNT(ttft_ms),"
                " PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ttft_ms),"
                " PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY ttft_ms),"
                " PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_ms),"
                " SUM(cost_usd)"
                f" FROM llm_calls WHERE created_at >= %s{clause}"
                " GROUP BY model ORDER BY 4 DESC NULLS LAST, 6 DESC",
                params,
            ) as cursor,
        ):
            rows = await cursor.fetchall()
        return [
            LatencyRow(
                model=str(row[0]),
                calls=int(row[1]),
                ttft_samples=int(row[2]),
                median_ttft_ms=None if row[3] is None else float(row[3]),
                p90_ttft_ms=None if row[4] is None else float(row[4]),
                median_total_ms=float(row[5] or 0.0),
                cost_usd=float(row[6] or 0.0),
            )
            for row in rows
        ]

    async def idle_threads(self, cutoff: str) -> list[str]:
        """Conversation threads whose newest checkpoint predates ``cutoff``.

        ``ts`` is a documented field of LangGraph's Checkpoint and is stored
        whole into the jsonb column, so this reads their data rather than
        reverse-engineering it. Deleting is left to the checkpointer's own
        ``adelete_thread`` — the only prune API it implements for Postgres,
        and the only one that also clears blobs and writes.

        Fails safe: if that field ever moves, this matches nothing and
        retention quietly stops rather than deleting the wrong threads.
        """
        async with (
            self.connection() as db,
            await db.execute(
                "SELECT thread_id FROM checkpoints GROUP BY thread_id"
                " HAVING MAX(checkpoint->>'ts') < %s",
                (cutoff,),
            ) as cursor,
        ):
            return [str(row[0]) for row in await cursor.fetchall()]

    async def count_checkpoints(self, thread_id: str) -> int:
        """How many checkpoints one conversation thread holds.

        Read straight from LangGraph's table, the same way ``idle_threads``
        does, and indexed by the primary key. Counting is the honest measure
        of the thing rotation bounds: a turn writes three checkpoints or a
        dozen depending on how many tools it calls, so turns are a proxy and
        this is not.

        Fails safe like ``idle_threads``: if the table cannot be read, this
        returns 0 and rotation quietly never fires. Losing rotation costs
        disk; raising here would break every question.
        """
        try:
            async with (
                self.connection() as db,
                await db.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE thread_id = %s", (thread_id,)
                ) as cursor,
            ):
                row = await cursor.fetchone()
        except psycopg.Error:
            logger.warning("Could not count checkpoints for thread %s", thread_id, exc_info=True)
            return 0
        return int(row[0]) if row else 0

    # -- chat spaces (where a Google Chat user's DM lives) ----------------
    async def set_chat_space(self, chat_id: int, space: str) -> None:
        async with self.connection() as db:
            await db.execute(
                "INSERT INTO chat_spaces (chat_id, space) VALUES (%s, %s)"
                " ON CONFLICT(chat_id) DO UPDATE SET space = excluded.space",
                (chat_id, space),
            )

    async def get_chat_space(self, chat_id: int) -> str | None:
        async with (
            self.connection() as db,
            await db.execute("SELECT space FROM chat_spaces WHERE chat_id = %s", (chat_id,)) as c,
        ):
            row = await c.fetchone()
        return str(row[0]) if row else None

    # -- meta (small key/value bookkeeping) ------------------------------
    async def get_meta(self, key: str) -> str | None:
        async with (
            self.connection() as db,
            await db.execute("SELECT value FROM meta WHERE key = %s", (key,)) as cursor,
        ):
            row = await cursor.fetchone()
        return str(row[0]) if row else None

    async def set_meta(self, key: str, value: str) -> None:
        async with self.connection() as db:
            await db.execute(
                "INSERT INTO meta (key, value) VALUES (%s, %s)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # -- action log ----------------------------------------------------
    async def log_action(
        self,
        chat_id: int,
        odoo_login: str,
        tool: str,
        summary: str,
        outcome: ActionOutcome,
        retention_days: int = 90,
    ) -> None:
        """Record one proposed write and its outcome.

        Written after the decision resolves, so it records what HAPPENED
        rather than what was intended — an approved action the server then
        refused is logged as "blocked", not "done".
        """
        async with self.connection() as db:
            await db.execute(
                "INSERT INTO action_log (chat_id, odoo_login, tool, summary, outcome, created_at)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                (chat_id, odoo_login, tool, summary, outcome, datetime.now(UTC).isoformat()),
            )
            if retention_days > 0:
                cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
                await db.execute("DELETE FROM action_log WHERE created_at < %s", (cutoff,))
        logger.info("Action logged: %s %s for %s", tool, outcome, odoo_login)

    async def actions_for(
        self, odoo_login: str, since: datetime, limit: int = 200
    ) -> list[LoggedAction]:
        """One person's actions since a moment, newest first."""
        async with (
            self.connection() as db,
            await db.execute(
                "SELECT id, chat_id, odoo_login, tool, summary, outcome, created_at"
                " FROM action_log WHERE odoo_login = %s AND created_at >= %s"
                " ORDER BY created_at DESC LIMIT %s",
                (odoo_login, since.isoformat(), limit),
            ) as cursor,
        ):
            rows = await cursor.fetchall()
        return [_action(row) for row in rows]

    async def all_actions(self, since: datetime, limit: int = 200) -> list[LoggedAction]:
        """Every user's actions since a moment — the operator view."""
        async with (
            self.connection() as db,
            await db.execute(
                "SELECT id, chat_id, odoo_login, tool, summary, outcome, created_at"
                " FROM action_log WHERE created_at >= %s ORDER BY created_at DESC LIMIT %s",
                (since.isoformat(), limit),
            ) as cursor,
        ):
            rows = await cursor.fetchall()
        return [_action(row) for row in rows]

    async def count_actions(self, since: datetime, odoo_login: str | None = None) -> int:
        """How many in the window — so truncation can be STATED, not silent."""
        query = "SELECT COUNT(*) FROM action_log WHERE created_at >= %s"
        params: tuple[str, ...] = (since.isoformat(),)
        if odoo_login is not None:
            query = "SELECT COUNT(*) FROM action_log WHERE odoo_login = %s AND created_at >= %s"
            params = (odoo_login, since.isoformat())
        async with self.connection() as db, await db.execute(query, params) as cursor:
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    # -- proactive: brief state ----------------------------------------
    async def list_bindings(self) -> list[Binding]:
        """Every bound chat — the fan-out list for scheduled work."""
        async with (
            self.connection() as db,
            await db.execute("SELECT chat_id FROM bindings") as cursor,
        ):
            rows = await cursor.fetchall()
        bindings = [await self.get_binding(row[0]) for row in rows]
        return [binding for binding in bindings if binding is not None]

    async def _set_proactive(self, chat_id: int, column: str, value: int | str) -> None:
        """Upsert one proactive_state column. ``column`` is always one of
        the literals below, never caller input."""
        async with self.connection() as db:
            await db.execute(
                f"INSERT INTO proactive_state (chat_id, {column}) VALUES (%s, %s)"
                f" ON CONFLICT (chat_id) DO UPDATE SET {column} = excluded.{column}",
                (chat_id, value),
            )

    async def set_brief_enabled(self, chat_id: int, enabled: bool) -> None:
        await self._set_proactive(chat_id, "brief_enabled", int(enabled))

    async def mark_digest_sent(self, chat_id: int, week: str) -> None:
        """Stamp the week so a digest cannot go out twice — a re-login too."""
        await self._set_proactive(chat_id, "last_digest_week", week)

    async def mark_brief_sent(self, chat_id: int, day: str) -> None:
        """Stamp today's brief so a restart or re-login cannot send it twice."""
        await self._set_proactive(chat_id, "last_brief_date", day)

    # -- proactive: watchers -------------------------------------------
    async def count_watchers(self, chat_id: int) -> int:
        async with (
            self.connection() as db,
            await db.execute(
                "SELECT COUNT(*) FROM watchers WHERE chat_id = %s AND active = 1", (chat_id,)
            ) as cursor,
        ):
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def add_watcher(self, chat_id: int, odoo_login: str, condition: str) -> int:
        async with self.connection() as db:
            cursor = await db.execute(
                "INSERT INTO watchers (chat_id, odoo_login, condition, created_at)"
                " VALUES (%s, %s, %s, %s) RETURNING id",
                (chat_id, odoo_login, condition, datetime.now(UTC).isoformat()),
            )
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def list_watchers(self, chat_id: int) -> list[Watcher]:
        async with (
            self.connection() as db,
            await db.execute(
                "SELECT id, chat_id, odoo_login, condition, created_at, last_checked_at,"
                " fingerprint FROM watchers WHERE chat_id = %s AND active = 1 ORDER BY id",
                (chat_id,),
            ) as cursor,
        ):
            rows = await cursor.fetchall()
        return [_watcher(row) for row in rows]

    async def deactivate_watcher(self, watcher_id: int, chat_id: int) -> bool:
        """Retire one watcher. Scoped to the owning chat: a watcher id is a
        small integer, so without this one chat could delete another's."""
        async with self.connection() as db:
            cursor = await db.execute(
                "UPDATE watchers SET active = 0 WHERE id = %s AND chat_id = %s AND active = 1",
                (watcher_id, chat_id),
            )
            return cursor.rowcount > 0

    async def due_watchers(self, cutoff: datetime) -> list[Watcher]:
        """Active watchers not checked since ``cutoff`` (never-checked first)."""
        async with (
            self.connection() as db,
            await db.execute(
                "SELECT id, chat_id, odoo_login, condition, created_at, last_checked_at,"
                " fingerprint FROM watchers"
                " WHERE active = 1 AND (last_checked_at = '' OR last_checked_at < %s)"
                " ORDER BY last_checked_at, id",
                (cutoff.isoformat(),),
            ) as cursor,
        ):
            rows = await cursor.fetchall()
        return [_watcher(row) for row in rows]

    async def record_watcher_check(self, watcher_id: int, fingerprint: str) -> None:
        """Stamp the check. The fingerprint is what makes the next identical
        result silent; passing "" (not triggered) clears it so the SAME
        condition becoming true again is treated as news."""
        async with self.connection() as db:
            await db.execute(
                "UPDATE watchers SET last_checked_at = %s, fingerprint = %s WHERE id = %s",
                (datetime.now(UTC).isoformat(), fingerprint, watcher_id),
            )


def _action(row: Any) -> LoggedAction:
    return LoggedAction(
        id=row[0],
        chat_id=row[1],
        odoo_login=row[2],
        tool=row[3],
        summary=row[4],
        outcome=row[5],
        created_at=datetime.fromisoformat(row[6]),
    )


def _watcher(row: Any) -> Watcher:
    return Watcher(
        id=row[0],
        chat_id=row[1],
        odoo_login=row[2],
        condition=row[3],
        created_at=datetime.fromisoformat(row[4]),
        last_checked_at=datetime.fromisoformat(row[5]) if row[5] else None,
        fingerprint=row[6],
    )


def binding_id(chat_id: int) -> str:
    """Loggable form of a chat id (avoid leaking full ids at INFO level)."""
    text = str(chat_id)
    return f"…{text[-4:]}" if len(text) > 4 else text
