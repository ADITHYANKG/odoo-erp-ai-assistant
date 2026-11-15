"""Test fixtures — chiefly an isolated Postgres schema per test.

`agent.db` used to be SQLite, and `tmp_path` gave every test a private
database for free. Postgres has no equivalent, so this builds one: each
test gets its own SCHEMA, created before and dropped after.

A schema rather than a database because it is cheap enough to do ~130
times a run, and because a test that opens the same DSN twice — the
key-rotation and fingerprint tests do exactly that — still lands on the
same tables, which is the behaviour those tests were written against.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from collections.abc import AsyncIterator, Iterator

import psycopg
import pytest

# psycopg's async mode refuses Windows' default ProactorEventLoop outright,
# and pytest-asyncio builds its loop from whatever policy is installed. The
# bot sets the same policy in main(); without it here every store test dies
# with an InterfaceError that says nothing about the real cause.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

#: The local `ont-agent-db-17` from agent/docker-compose.yaml (port 5566).
#:
#: 127.0.0.1 rather than localhost, deliberately: Windows resolves localhost
#: to ::1 first, Docker publishes the port on IPv4 only, and the connection
#: then hangs until it times out with no useful error.
DEFAULT_TEST_DSN = "postgresql://ont_agent:ont_agent@127.0.0.1:5566/ont_agent"

SKIP_MESSAGE = (
    "No Postgres for the tests. The agent stores everything in Postgres now, "
    "so the suite needs it running:\n"
    "    docker compose -f agent/docker-compose.yaml up -d\n"
    "Override the connection with AGENT_TEST_DB_URL."
)


@pytest.fixture(scope="session")
def base_dsn() -> Iterator[str]:
    """The database the per-test schemas are created in.

    Skips the whole suite with one readable sentence rather than letting
    every store test fail with its own connection error.
    """
    dsn = os.environ.get("AGENT_TEST_DB_URL") or DEFAULT_TEST_DSN
    try:
        with psycopg.connect(dsn, connect_timeout=5) as connection:
            connection.execute("SELECT 1")
    except Exception:
        pytest.skip(SKIP_MESSAGE, allow_module_level=True)
    yield dsn


async def _isolated_schema(base_dsn: str) -> AsyncIterator[str]:
    schema = f"t_{uuid.uuid4().hex[:12]}"
    async with await psycopg.AsyncConnection.connect(base_dsn, autocommit=True) as setup:
        await setup.execute(f'CREATE SCHEMA "{schema}"')
    # search_path makes unqualified CREATE/SELECT land in this schema, so the
    # store's own SQL needs no knowledge that any of this is happening.
    yield f"{base_dsn}?options=-csearch_path%3D{schema}"
    async with await psycopg.AsyncConnection.connect(base_dsn, autocommit=True) as teardown:
        await teardown.execute(f'DROP SCHEMA "{schema}" CASCADE')


@pytest.fixture
async def store_dsn(base_dsn: str) -> AsyncIterator[str]:
    """A DSN pointing at a schema private to this test.

    Yields the same string shape `BindingStore` takes in production, so
    tests exercise the real connection path rather than a stub.
    """
    async for dsn in _isolated_schema(base_dsn):
        yield dsn


@pytest.fixture
async def other_store_dsn(base_dsn: str) -> AsyncIterator[str]:
    """A SECOND independent database, for the handful of tests that compare
    one against another — "a store that had a key" versus "a store that
    never did", which cannot be told apart inside a single schema."""
    async for dsn in _isolated_schema(base_dsn):
        yield dsn


@pytest.fixture
async def checkpoint_dsn(store_dsn: str) -> str:
    """`store_dsn` with LangGraph's own checkpoint tables already created.

    Anything that counts, prunes or writes conversation threads needs them.
    The bot gets them from the saver's `setup()` at startup; tests that talk
    to the `checkpoints` table have to ask for them explicitly.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row

    async with await psycopg.AsyncConnection.connect(
        store_dsn, autocommit=True, row_factory=dict_row
    ) as connection:
        await AsyncPostgresSaver(connection).setup()
    return store_dsn
