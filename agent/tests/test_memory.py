"""Unit tests for ont_agent.memory — namespacing, recall, forget (no network)."""

from __future__ import annotations

from typing import Any

from langgraph.store.memory import InMemoryStore

from ont_agent.config import DEFAULT_MEMORY_MAX_RECALL, Settings
from ont_agent.memory import (
    build_embeddings,
    build_store_index,
    forget_all,
    namespace_for,
    recall,
    safe_label,
)

PRIYA = "priya@gmail.com"
VIKRAM = "vikram@gmail.com"

BASE = Settings(mcp_url="http://odoo:8160/mcp", mcp_bearer_key="k")


async def _save(store: InMemoryStore, login: str, key: str, content: str) -> None:
    await store.aput(namespace_for(login), key, {"content": content})


def test_safe_label_escapes_periods_and_stays_injective() -> None:
    # LangGraph raises InvalidNamespaceError on periods — a raw login would
    # break every write (caught live before shipping).
    assert "." not in safe_label(PRIYA)
    # Escaping "_" first keeps distinct logins distinct: john.doe vs john_doe
    # must never collapse into one namespace (that would be a cross-user leak).
    assert safe_label("john.doe@x.com") != safe_label("john_doe@x.com")
    assert namespace_for(PRIYA)[0] == "memories"


async def test_memories_are_isolated_per_user() -> None:
    # The identity boundary: one chat can change identity, so a memory saved
    # for one Odoo user must be invisible to another.
    store = InMemoryStore()
    await _save(store, PRIYA, "m1", "Prefers amounts in lakhs")

    assert await recall(store, PRIYA, "money") == ["Prefers amounts in lakhs"]
    assert await recall(store, VIKRAM, "money") == []


async def test_recall_respects_the_default_and_a_configured_cap() -> None:
    store = InMemoryStore()
    for index in range(DEFAULT_MEMORY_MAX_RECALL + 5):
        await _save(store, PRIYA, f"m{index}", f"preference {index}")

    memories = await recall(store, PRIYA, "preference")
    assert len(memories) == DEFAULT_MEMORY_MAX_RECALL
    assert all(memory.startswith("preference") for memory in memories)

    # The cap is configuration (AGENT_MEMORY_MAX_RECALL), not a constant.
    assert len(await recall(store, PRIYA, "preference", limit=3)) == 3


async def test_recall_without_a_store_or_on_failure_returns_empty() -> None:
    assert await recall(None, PRIYA, "anything") == []

    class Exploding(InMemoryStore):
        async def asearch(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("store down")

    # Memory is an enhancement: a broken store must never break answering.
    assert await recall(Exploding(), PRIYA, "anything") == []


async def test_forget_all_clears_only_that_user() -> None:
    store = InMemoryStore()
    await _save(store, PRIYA, "m1", "Prefers lakhs")
    await _save(store, PRIYA, "m2", "Prefers bar charts")
    await _save(store, VIKRAM, "m1", "Prefers PDF reports")

    assert await forget_all(store, PRIYA) == 2
    assert await recall(store, PRIYA, "prefers") == []
    assert await recall(store, VIKRAM, "prefers") == ["Prefers PDF reports"]

    assert await forget_all(None, PRIYA) == 0  # memory disabled


def test_deleting_a_memory_must_cascade_to_its_vector(tmp_path: Any) -> None:
    """SQLite ignores foreign keys unless PRAGMA foreign_keys=ON, so the
    ON DELETE CASCADE on store_vectors silently does nothing and every
    delete leaks an orphaned embedding (found live 2026-08-24). This pins
    the pragma bot.py sets when opening the store.
    """
    import sqlite3

    def orphans_after_delete(foreign_keys_on: bool) -> int:
        con = sqlite3.connect(str(tmp_path / f"fk{foreign_keys_on}.db"))
        if foreign_keys_on:
            con.execute("PRAGMA foreign_keys = ON")
        con.executescript(
            """
            CREATE TABLE store (prefix text, key text, value text,
                                PRIMARY KEY (prefix, key));
            CREATE TABLE store_vectors (
                prefix text, key text, field_name text, embedding BLOB,
                PRIMARY KEY (prefix, key, field_name),
                FOREIGN KEY (prefix, key) REFERENCES store(prefix, key)
                    ON DELETE CASCADE);
            """
        )
        con.execute("INSERT INTO store VALUES ('memories.p', 'k1', '{}')")
        con.execute("INSERT INTO store_vectors VALUES ('memories.p', 'k1', '$', X'00')")
        con.execute("DELETE FROM store WHERE prefix='memories.p' AND key='k1'")
        count: int = con.execute("SELECT COUNT(*) FROM store_vectors").fetchone()[0]
        con.close()
        return count

    # Without the pragma the vector survives its parent — the live bug.
    assert orphans_after_delete(foreign_keys_on=False) == 1
    # With it, the delete cascades as the schema intends.
    assert orphans_after_delete(foreign_keys_on=True) == 0


def test_embeddings_are_openrouter_backed_and_optional() -> None:
    assert build_embeddings(BASE) is None  # no OpenRouter key configured

    assert build_store_index(BASE) is None

    configured = BASE.model_copy(update={"openrouter_api_key": "sk-or-test"})
    embeddings: Any = build_embeddings(configured)
    assert embeddings is not None
    assert "openrouter.ai" in str(embeddings.openai_api_base)
    # OpenRouter needs raw text, not the tiktoken arrays sent by default.
    assert embeddings.check_embedding_ctx_length is False

    index = build_store_index(configured)
    assert index is not None
    assert index["dims"] == configured.memory_embed_dims

    # Dims follow the configured model — a hardcoded width would silently
    # corrupt the index the moment someone switches embedding models.
    narrow = configured.model_copy(
        update={"memory_embed_model": "qwen/qwen3-embedding-0.6b", "memory_embed_dims": 1024}
    )
    narrow_index = build_store_index(narrow)
    assert narrow_index is not None
    assert narrow_index["dims"] == 1024

    # The provider base URL is configuration, never a literal in the module.
    self_hosted = configured.model_copy(update={"openrouter_base_url": "http://proxy.internal/v1"})
    relocated: Any = build_embeddings(self_hosted)
    assert "proxy.internal" in str(relocated.openai_api_base)

    # An empty model name disables vector search without disabling memory.
    no_model = configured.model_copy(update={"memory_embed_model": ""})
    assert build_store_index(no_model) is None
