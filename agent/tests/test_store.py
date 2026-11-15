"""Unit tests for ont_agent.store (tmp SQLite, no network)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from ont_agent.crypto import TokenCipher
from ont_agent.store import GLOBAL_SUBJECT, Binding, BindingStore, UsageRow


def _binding(chat_id: int = 111, refresh: str = "r1") -> Binding:
    return Binding(
        chat_id=chat_id,
        odoo_login="priya@acme",
        odoo_name="CFO Priya",
        access_token="jwt-a",
        refresh_token=refresh,
        expires_at=datetime.now(UTC) + timedelta(minutes=14),
    )


async def test_binding_roundtrip_and_delete(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        assert await store.get_binding(111) is None

        await store.save_binding(_binding())
        loaded = await store.get_binding(111)
        assert loaded is not None
        assert loaded.odoo_login == "priya@acme"
        assert loaded.expires_at.tzinfo is not None  # tz survives the roundtrip

        await store.delete_binding(111)
        assert await store.get_binding(111) is None


async def test_save_replaces_rotated_refresh_token(store_dsn: str) -> None:
    # Refresh rotation: persisting the NEW refresh token is load-bearing.
    async with BindingStore(store_dsn) as store:
        await store.save_binding(_binding(refresh="old-refresh"))
        await store.save_binding(_binding(refresh="new-refresh"))
        loaded = await store.get_binding(111)
        assert loaded is not None
        assert loaded.refresh_token == "new-refresh"


async def test_pending_is_single_use(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        await store.create_pending("state-1", 222, "verifier-1")

        first = await store.pop_pending("state-1")
        assert first is not None
        assert (first.chat_id, first.verifier) == (222, "verifier-1")

        assert await store.pop_pending("state-1") is None  # consumed
        assert await store.pop_pending("never-existed") is None


async def test_expired_pending_rejected(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        stale = (datetime.now(UTC) - timedelta(minutes=11)).isoformat()
        async with store.connection() as db:
            await db.execute(
                "INSERT INTO pending_auth (state, chat_id, verifier, created_at)"
                " VALUES (%s, %s, %s, %s)",
                ("old-state", 333, "v", stale),
            )
        assert await store.pop_pending("old-state") is None


async def test_bindings_are_per_chat(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        await store.save_binding(_binding(chat_id=1))
        await store.save_binding(_binding(chat_id=2, refresh="other"))
        one = await store.get_binding(1)
        two = await store.get_binding(2)
        assert one is not None and two is not None
        assert one.refresh_token != two.refresh_token


async def test_can_call_methods_roundtrip(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        await store.save_binding(
            Binding(
                chat_id=9,
                odoo_login="boss",
                odoo_name="Boss",
                access_token="jwt",
                refresh_token="ref",
                expires_at=datetime.now(UTC) + timedelta(minutes=5),
                can_call_methods=True,
            )
        )
        loaded = await store.get_binding(9)
        assert loaded is not None
        assert loaded.can_call_methods is True
        # Default is False — least privilege when unknown.
        await store.save_binding(_binding(chat_id=10))
        loaded_default = await store.get_binding(10)
        assert loaded_default is not None
        assert loaded_default.can_call_methods is False


async def test_can_write_roundtrip(store_dsn: str) -> None:
    # Mirrors the granted OAuth scope; a read-only grant must survive the
    # roundtrip as False so write tools stay withheld from the model.
    async with BindingStore(store_dsn) as store:
        await store.save_binding(
            Binding(
                chat_id=11,
                odoo_login="reader",
                odoo_name="Read Only",
                access_token="jwt",
                refresh_token="ref",
                expires_at=datetime.now(UTC) + timedelta(minutes=5),
                can_write=False,
            )
        )
        loaded = await store.get_binding(11)
        assert loaded is not None
        assert loaded.can_write is False

        # A later refresh that regains write must overwrite it, not stick.
        await store.save_binding(_binding(chat_id=11))
        regained = await store.get_binding(11)
        assert regained is not None
        assert regained.can_write is True


async def test_usage_unseen_subject_and_day_is_zero(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        assert await store.get_usage("user:nobody", "2026-08-24") == UsageRow()


async def test_record_usage_upserts_across_repeated_calls(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        await store.record_usage("user:a", "2026-08-24", 0.01)
        await store.record_usage("user:a", "2026-08-24", 0.02)
        usage = await store.get_usage("user:a", "2026-08-24")
        assert usage.message_count == 2
        assert round(usage.spend_usd, 5) == 0.03


async def test_usage_is_per_subject_and_per_day(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        await store.record_usage("user:a", "2026-08-24", 1.0)
        await store.record_usage("user:b", "2026-08-24", 5.0)  # different user
        await store.record_usage("user:a", "2026-08-25", 9.0)  # different day

        today_a = await store.get_usage("user:a", "2026-08-24")
        assert today_a.message_count == 1
        assert today_a.spend_usd == 1.0

        today_b = await store.get_usage("user:b", "2026-08-24")
        assert today_b.spend_usd == 5.0

        # A new day starts fresh — yesterday's spend does not carry over.
        tomorrow_a = await store.get_usage("user:a", "2026-08-25")
        assert tomorrow_a.message_count == 1
        assert tomorrow_a.spend_usd == 9.0


async def test_record_usage_also_totals_into_the_global_row(store_dsn: str) -> None:
    # Written in the same call, so the total can never drift from the parts.
    async with BindingStore(store_dsn) as store:
        await store.record_usage("user:a", "2026-08-24", 1.50)
        await store.record_usage("user:b", "2026-08-24", 2.25)

        total = await store.get_usage(GLOBAL_SUBJECT, "2026-08-24")
        assert total.message_count == 2
        assert round(total.spend_usd, 5) == 3.75
        # And the global row is per-day like any other.
        assert await store.get_usage(GLOBAL_SUBJECT, "2026-08-25") == UsageRow()


async def test_spend_only_usage_is_not_counted_as_a_message(store_dsn: str) -> None:
    # Unattended work (briefs, watchers) costs money but asks no question.
    async with BindingStore(store_dsn) as store:
        await store.record_usage("user:a", "2026-08-24", 0.10)
        await store.record_usage("user:a", "2026-08-24", 0.05, count_message=False)

        usage = await store.get_usage("user:a", "2026-08-24")
        assert usage.message_count == 1
        assert round(usage.spend_usd, 5) == 0.15
        total = await store.get_usage(GLOBAL_SUBJECT, "2026-08-24")
        assert total.message_count == 1


async def test_usage_between_sums_an_inclusive_range(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        await store.record_usage("user:a", "2026-08-17", 9.0)  # before the range
        await store.record_usage("user:a", "2026-08-18", 1.0)
        await store.record_usage("user:a", "2026-08-21", 2.0)
        await store.record_usage("user:a", "2026-08-24", 4.0)
        await store.record_usage("user:b", "2026-08-21", 50.0)  # someone else

        week = await store.usage_between("user:a", "2026-08-18", "2026-08-24")
        assert week.message_count == 3
        assert week.spend_usd == 7.0
        assert await store.usage_between("user:nobody", "2026-08-18", "2026-08-24") == UsageRow()


async def test_brief_flags_roundtrip_and_survive_a_token_refresh(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        await store.save_binding(_binding(chat_id=5))
        loaded = await store.get_binding(5)
        assert loaded is not None
        assert loaded.brief_enabled is True  # opt-out default
        assert loaded.last_brief_date == ""

        await store.set_brief_enabled(5, False)
        await store.mark_brief_sent(5, "2026-09-01")

        # A refresh re-saves the binding; the opt-out and the stamp must NOT
        # be reset by it, or a token refresh would resurrect a brief the user
        # switched off.
        await store.save_binding(_binding(chat_id=5, refresh="rotated"))
        after = await store.get_binding(5)
        assert after is not None
        assert after.refresh_token == "rotated"
        assert after.brief_enabled is False
        assert after.last_brief_date == "2026-09-01"


async def test_brief_state_survives_logout_and_login_again(store_dsn: str) -> None:
    """Logout deletes the binding row. The brief/digest stamps and the
    opt-out must outlive it, or re-linking sends a second brief the same
    morning, repeats the week's digest, and undoes /brief off."""
    async with BindingStore(store_dsn) as store:
        await store.save_binding(_binding(chat_id=5))
        await store.set_brief_enabled(5, False)
        await store.mark_brief_sent(5, "2026-09-01")
        await store.mark_digest_sent(5, "2026-W36")

        await store.delete_binding(5)  # /logout
        assert await store.get_binding(5) is None
        await store.save_binding(_binding(chat_id=5))  # log in again

        again = await store.get_binding(5)
        assert again is not None
        assert again.brief_enabled is False
        assert again.last_brief_date == "2026-09-01"
        assert again.last_digest_week == "2026-W36"


async def test_existing_brief_state_is_carried_over_from_the_bindings_table(
    store_dsn: str,
) -> None:
    # A deployment from before proactive_state kept these on `bindings`.
    async with BindingStore(store_dsn) as store:
        await store.save_binding(_binding(chat_id=7))
        async with store.connection() as db:
            await db.execute("DELETE FROM proactive_state WHERE chat_id = 7")
            await db.execute(
                "UPDATE bindings SET brief_enabled = 0, last_brief_date = '2026-09-01',"
                " last_digest_week = '2026-W36' WHERE chat_id = 7"
            )

    async with BindingStore(store_dsn) as store:  # restart runs the schema again
        carried = await store.get_binding(7)
        assert carried is not None
        assert carried.brief_enabled is False
        assert carried.last_brief_date == "2026-09-01"
        assert carried.last_digest_week == "2026-W36"


async def test_list_bindings_returns_every_bound_chat(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        assert await store.list_bindings() == []
        await store.save_binding(_binding(chat_id=1))
        await store.save_binding(_binding(chat_id=2))
        assert {b.chat_id for b in await store.list_bindings()} == {1, 2}


async def test_watcher_crud(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        assert await store.count_watchers(5) == 0
        first = await store.add_watcher(5, "priya@acme", "invoices over 60 days")
        await store.add_watcher(5, "priya@acme", "deals over 50 lakh")

        watchers = await store.list_watchers(5)
        assert [w.condition for w in watchers] == [
            "invoices over 60 days",
            "deals over 50 lakh",
        ]
        assert await store.count_watchers(5) == 2
        assert watchers[0].id == first
        assert watchers[0].last_checked_at is None

        assert await store.deactivate_watcher(first, 5) is True
        assert await store.count_watchers(5) == 1
        # Retiring is idempotent, not an error on the second attempt.
        assert await store.deactivate_watcher(first, 5) is False


async def test_a_watcher_cannot_be_deleted_from_another_chat(store_dsn: str) -> None:
    # Watcher ids are small integers, so without the chat scope one user
    # could retire another user's alerts by guessing.
    async with BindingStore(store_dsn) as store:
        watcher_id = await store.add_watcher(5, "priya@acme", "mine")
        assert await store.deactivate_watcher(watcher_id, 999) is False
        assert await store.count_watchers(5) == 1


async def test_due_watchers_respects_the_cutoff(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        watcher_id = await store.add_watcher(5, "priya@acme", "anything")
        future = datetime.now(UTC) + timedelta(minutes=1)

        # Never checked -> always due.
        assert [w.id for w in await store.due_watchers(future)] == [watcher_id]

        await store.record_watcher_check(watcher_id, "INV/1")
        # Just checked -> not due until the cutoff moves past it.
        assert await store.due_watchers(datetime.now(UTC) - timedelta(hours=1)) == []
        assert [w.id for w in await store.due_watchers(future)] == [watcher_id]

        stored = (await store.list_watchers(5))[0]
        assert stored.fingerprint == "INV/1"
        assert stored.last_checked_at is not None


async def test_actions_round_trip_newest_first(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        await store.log_action(5, "priya@acme", "send_email", "send_email\n  to: a@b", "done")
        await store.log_action(5, "priya@acme", "update_records", "update_records", "cancelled")

        actions = await store.actions_for("priya@acme", datetime.now(UTC) - timedelta(days=1))
        assert [a.tool for a in actions] == ["update_records", "send_email"]
        assert [a.outcome for a in actions] == ["cancelled", "done"]
        assert actions[0].created_at.tzinfo is not None


async def test_one_users_actions_are_not_anothers(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        await store.log_action(5, "priya@acme", "send_email", "x", "done")
        await store.log_action(6, "vikram@acme", "send_email", "y", "done")

        since = datetime.now(UTC) - timedelta(days=1)
        assert len(await store.actions_for("priya@acme", since)) == 1
        # The operator view spans everyone; the per-user view does not.
        assert len(await store.all_actions(since)) == 2


async def test_actions_outside_the_window_are_excluded(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        await store.log_action(5, "priya@acme", "send_email", "x", "done")
        future = datetime.now(UTC) + timedelta(minutes=1)
        assert await store.actions_for("priya@acme", future) == []
        assert await store.count_actions(future) == 0


async def test_the_log_is_pruned_past_retention(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        await store.log_action(5, "priya@acme", "send_email", "old", "done")
        # Age it beyond the window, then write again to trigger the prune.
        old = (datetime.now(UTC) - timedelta(days=120)).isoformat()
        async with store.connection() as db:
            await db.execute("UPDATE action_log SET created_at = %s", (old,))

        await store.log_action(5, "priya@acme", "send_email", "new", "done", retention_days=90)
        remaining = await store.all_actions(datetime.now(UTC) - timedelta(days=365))
        assert [a.summary for a in remaining] == ["new"]


async def test_count_supports_stated_truncation(store_dsn: str) -> None:
    # /audit shows a page but must be able to say how many it did not show.
    async with BindingStore(store_dsn) as store:
        for index in range(30):
            await store.log_action(5, "priya@acme", "send_email", f"mail {index}", "done")
        since = datetime.now(UTC) - timedelta(days=1)
        assert await store.count_actions(since) == 30
        assert await store.count_actions(since, "priya@acme") == 30
        assert await store.count_actions(since, "nobody@acme") == 0
        assert len(await store.all_actions(since, limit=25)) == 25


async def test_digest_week_stamp_roundtrips(store_dsn: str) -> None:
    async with BindingStore(store_dsn) as store:
        await store.save_binding(_binding(chat_id=5))
        loaded = await store.get_binding(5)
        assert loaded is not None and loaded.last_digest_week == ""

        await store.mark_digest_sent(5, "2026-W36")
        # A token refresh re-saves the binding and must NOT reset the stamp,
        # or the digest would go out again the same week.
        await store.save_binding(_binding(chat_id=5, refresh="rotated"))
        after = await store.get_binding(5)
        assert after is not None and after.last_digest_week == "2026-W36"


# --- token encryption at rest ----------------------------------------------


def _cipher() -> TokenCipher:
    from cryptography.fernet import Fernet

    return TokenCipher(Fernet.generate_key().decode())


async def _raw_tokens(dsn: str) -> tuple[str, str]:
    """Read the columns the way someone holding a database dump would —
    straight from the table, bypassing the store and its cipher."""
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn) as connection:
        cursor = await connection.execute("SELECT access_token, refresh_token FROM bindings")
        row = await cursor.fetchone()
    assert row is not None
    return str(row[0]), str(row[1])


@contextlib.contextmanager
def caplog_at_warning() -> Iterator[list[str]]:
    """Capture ont_agent.store warnings as formatted strings."""
    records: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())  # getMessage already applies args

    logger = logging.getLogger("ont_agent.store")
    handler = _Collect(level=logging.WARNING)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


#: Realistic token values, and the length is the point. An earlier version
#: of the test below searched the ciphertext for the two-character string
#: "r1". Fernet output is base64, whose alphabet is 64 symbols, so any given
#: pair of characters turns up by chance about once in 4096 positions — the
#: assertion failed roughly one run in forty, for no reason connected to the
#: encryption it was meant to be checking.
LONG_ACCESS = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.header-payload-signature-abcdef"
LONG_REFRESH = "refresh-QmFzZTY0LWlzaC1yZWZyZXNoLXRva2VuLXZhbHVl-0123456789"


async def test_tokens_are_encrypted_in_the_file(store_dsn: str) -> None:
    """The whole point: a copy of agent.db must not be a copy of everyone's
    Odoo access. Checked by reading the raw column, not through the API."""
    path = store_dsn
    binding = _binding(chat_id=5)
    binding = replace(binding, access_token=LONG_ACCESS, refresh_token=LONG_REFRESH)
    async with BindingStore(path, _cipher()) as store:
        await store.save_binding(binding)
        loaded = await store.get_binding(5)

    assert loaded is not None
    assert loaded.access_token == LONG_ACCESS  # round-trips through the API
    assert loaded.refresh_token == LONG_REFRESH

    access, refresh = await _raw_tokens(path)
    assert LONG_ACCESS not in access
    assert LONG_REFRESH not in refresh
    assert access.startswith("enc:v1:")
    assert refresh.startswith("enc:v1:")


async def test_without_a_key_the_file_still_holds_plaintext(store_dsn: str) -> None:
    # Pins the pre-encryption behaviour, so the default is never in doubt.
    path = store_dsn
    async with BindingStore(path) as store:
        await store.save_binding(_binding(chat_id=5))
    access, _ = await _raw_tokens(path)
    assert access == "jwt-a"


async def test_a_legacy_plaintext_row_keeps_working_and_then_converts(
    store_dsn: str,
) -> None:
    """Rows written before encryption must not lock anyone out. They read
    fine and encrypt themselves on the next save — which happens at every
    token refresh, so migration needs nothing run by hand."""
    path = store_dsn
    async with BindingStore(path) as plain:  # written before encryption
        await plain.save_binding(_binding(chat_id=5))

    cipher = _cipher()
    async with BindingStore(path, cipher) as store:
        loaded = await store.get_binding(5)
        assert loaded is not None
        assert loaded.access_token == "jwt-a"  # still readable

        await store.save_binding(_binding(chat_id=5, refresh="rotated"))

    access, _ = await _raw_tokens(path)
    assert access.startswith("enc:v1:")  # converted


async def test_a_changed_key_logs_out_that_user_rather_than_crashing(
    store_dsn: str,
) -> None:
    """One unreadable row must not take the service down for everyone: the
    binding reads as absent, so that user is simply asked to /start."""
    path = store_dsn
    async with BindingStore(path, _cipher()) as store:
        await store.save_binding(_binding(chat_id=5))

    async with BindingStore(path, _cipher()) as rotated:  # different key
        assert await rotated.get_binding(5) is None
        # And the fan-out skips them instead of raising.
        assert await rotated.list_bindings() == []


async def test_the_key_fingerprint_is_remembered(store_dsn: str) -> None:
    # What lets startup say "the key changed" rather than leaving a mass
    # logout looking like a bug.
    import psycopg

    cipher = _cipher()
    async with BindingStore(store_dsn, cipher):
        pass

    async with await psycopg.AsyncConnection.connect(store_dsn) as connection:
        cursor = await connection.execute(
            "SELECT value FROM meta WHERE key = 'token_key_fingerprint'"
        )
        row = await cursor.fetchone()
    assert row is not None
    stored = row[0]
    assert stored == cipher.fingerprint


async def test_a_removed_key_logs_out_rather_than_sending_ciphertext_to_odoo(
    store_dsn: str,
) -> None:
    """Deleting AGENT_TOKEN_KEY must not hand `enc:v1:...` to Odoo as a bearer
    token: that returns a 401 that looks like a random auth failure instead of
    the config change made minutes earlier."""
    path = store_dsn
    async with BindingStore(path, _cipher()) as store:
        await store.save_binding(_binding(chat_id=6))

    async with BindingStore(path) as without_key:  # key removed from .env
        assert await without_key.get_binding(6) is None
        assert await without_key.list_bindings() == []


async def test_removing_the_key_is_distinguished_from_never_setting_one(
    store_dsn: str,
    other_store_dsn: str,
) -> None:
    # Identical state otherwise, so only the remembered fingerprint can tell
    # the operator which of the two happened — which is why this needs two
    # separate databases: one that HAD a key, and one that never did.
    async with BindingStore(store_dsn, _cipher()):
        pass

    with caplog_at_warning() as records:
        async with BindingStore(store_dsn):
            pass
    assert any("AGENT_TOKEN_KEY was removed" in r for r in records)

    with caplog_at_warning() as records:
        async with BindingStore(other_store_dsn):
            pass
    assert any("stored UNENCRYPTED" in r for r in records)


# --- one connection per unit of work ----------------------------------------


async def test_a_transaction_does_not_break_another_tasks_write(store_dsn: str) -> None:
    """The bug the pool exists to prevent.

    A psycopg transaction is CONNECTION scoped, and three tasks share this
    store on one event loop: the Telegram handlers, the proactive ticker and
    the callback server. Measured live 2026-09-04 on the single-connection
    version, a concurrent write during ``record_usage``'s transaction did
    not fail quietly — one that commits raised ProgrammingError("Explicit
    commit() forbidden within a Transaction context"), one that does not
    raised OutOfOrderTransactionNesting.
    """
    async with BindingStore(store_dsn) as store:
        opened = asyncio.Event()

        async def rolls_back() -> None:
            async with store.connection() as db, db.transaction():
                await db.execute(
                    "INSERT INTO meta (key, value) VALUES ('doomed', '1')"
                    " ON CONFLICT(key) DO UPDATE SET value = excluded.value"
                )
                opened.set()
                await asyncio.sleep(0.2)
                raise RuntimeError("boom")

        async def innocent() -> None:
            await opened.wait()
            await store.set_meta("innocent", "kept")

        outcomes = await asyncio.gather(rolls_back(), innocent(), return_exceptions=True)

    assert isinstance(outcomes[0], RuntimeError)  # its own failure, as asked for
    assert outcomes[1] is None  # the bystander survived it
    async with BindingStore(store_dsn) as store:
        assert await store.get_meta("innocent") == "kept"
        assert await store.get_meta("doomed") is None  # rolled back, as asked for


async def test_the_pool_is_closed_on_exit(store_dsn: str) -> None:
    # Without this the suite would leak a pool per store, and production
    # would leak one per restart of the callback server.
    store = BindingStore(store_dsn)
    async with store:
        assert await store.get_meta("anything") is None
    with pytest.raises(RuntimeError, match="outside 'async with'"):
        store.connection()


async def test_concurrent_readers_do_not_serialise_behind_each_other(store_dsn: str) -> None:
    # More than one connection is genuinely available: with a single shared
    # connection these would queue, and with max_size=1 this would deadlock.
    async with BindingStore(store_dsn, max_size=4) as store:
        await store.set_meta("k", "v")
        results = await asyncio.gather(*(store.get_meta("k") for _ in range(8)))
    assert results == ["v"] * 8
