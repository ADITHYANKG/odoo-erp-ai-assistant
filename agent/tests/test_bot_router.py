"""Integration tests verifying semantic routing within answer_question."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ont_agent import bot
from ont_agent.config import Settings
from ont_agent.router import GREETINGS_RESPONSE, OUT_OF_SCOPE_RESPONSE
from ont_agent.store import Binding, BindingStore


def _binding(chat_id: int) -> Binding:
    return Binding(
        chat_id=chat_id,
        odoo_login="admin",
        odoo_name="Mitchell Admin",
        access_token="user-jwt",
        refresh_token="ref-123",
        expires_at=datetime.now(UTC) + timedelta(minutes=10),
        can_write=True,
    )


async def test_answer_question_semantic_router_greetings(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_ensure(store: BindingStore, settings: Settings, chat_id: int) -> Binding:
        return _binding(chat_id)

    def explode(*args: object, **kwargs: object) -> object:
        raise AssertionError("LLM agent should not be built or called for greetings!")

    monkeypatch.setattr(bot, "ensure_access", fake_ensure)
    monkeypatch.setattr(bot, "get_odoo_tools", explode)
    monkeypatch.setattr(bot, "build_agent", explode)

    settings = Settings(
        mcp_url="http://localhost:8069/mcp",
        mcp_bearer_key="k",
        openrouter_api_key="k",
        agent_db_url=store_dsn,
        semantic_router=True,
    )

    async with BindingStore(store_dsn) as store:
        await store.save_binding(_binding(chat_id=5))
        reply = await bot.answer_question(store, settings, chat_id=5, text="Hello there!")

    assert reply.text == GREETINGS_RESPONSE


async def test_answer_question_semantic_router_out_of_scope(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_ensure(store: BindingStore, settings: Settings, chat_id: int) -> Binding:
        return _binding(chat_id)

    def explode(*args: object, **kwargs: object) -> object:
        raise AssertionError("LLM agent should not be built or called for out-of-scope query!")

    monkeypatch.setattr(bot, "ensure_access", fake_ensure)
    monkeypatch.setattr(bot, "get_odoo_tools", explode)
    monkeypatch.setattr(bot, "build_agent", explode)

    settings = Settings(
        mcp_url="http://localhost:8069/mcp",
        mcp_bearer_key="k",
        openrouter_api_key="k",
        agent_db_url=store_dsn,
        semantic_router=True,
    )

    async with BindingStore(store_dsn) as store:
        await store.save_binding(_binding(chat_id=5))
        reply = await bot.answer_question(store, settings, chat_id=5, text="tell me a joke")

    assert reply.text == OUT_OF_SCOPE_RESPONSE


async def test_on_text_routed_greeting_skips_draft_and_placeholder(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_bot import _FakeContext, _FakeUpdate

    async def fake_ensure(store: BindingStore, settings: Settings, chat_id: int) -> Binding:
        return _binding(chat_id)

    monkeypatch.setattr(bot, "ensure_access", fake_ensure)

    settings = Settings(
        mcp_url="http://localhost:8069/mcp",
        mcp_bearer_key="k",
        openrouter_api_key="k",
        agent_db_url=store_dsn,
        semantic_router=True,
    )

    async with BindingStore(store_dsn) as store:
        await store.save_binding(_binding(chat_id=5))
        context = _FakeContext({"store": store, "settings": settings})
        update = _FakeUpdate(5, "hi")
        await bot.on_text(update, context)  # type: ignore[arg-type]

    # No ephemeral draft, no spinner placeholder: the answer is the only message.
    assert context.bot.drafts == []
    assert update.message.edits == []
    assert update.message.replies == [GREETINGS_RESPONSE]


async def test_semantic_router_answer_is_traced(
    store_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Router answers skip the model; without their own trace Langfuse showed
    # only the questions that reached the LLM.
    import contextlib
    from typing import Any

    import langfuse

    from tests.test_bot import _fake_tracing

    tracing, client = _fake_tracing()
    propagated: list[dict[str, Any]] = []

    @contextlib.contextmanager
    def fake_propagate(**kwargs: Any) -> Any:
        propagated.append(kwargs)
        yield

    async def fake_ensure(store: BindingStore, settings: Settings, chat_id: int) -> Binding:
        return _binding(chat_id)

    monkeypatch.setattr(langfuse, "propagate_attributes", fake_propagate)
    monkeypatch.setattr(bot, "ensure_access", fake_ensure)
    monkeypatch.setattr(
        bot, "fast_route", lambda settings, text: ("greetings", GREETINGS_RESPONSE)
    )

    settings = Settings(
        mcp_url="http://localhost:8069/mcp",
        mcp_bearer_key="k",
        openrouter_api_key="k",
        agent_db_url=store_dsn,
        semantic_router=True,
    )
    async with BindingStore(store_dsn) as store:
        reply = await bot.answer_question(store, settings, 5, "hi", tracing=tracing)

    assert reply.text == GREETINGS_RESPONSE
    (span,) = client.spans
    assert span["name"] == "telegram-ask"
    (attributes,) = propagated
    assert attributes["tags"] == ["telegram", bot.ROUTER_TRACE_TAG]
    (first, last) = client.span_objects[0].updates
    assert first["input"] == "hi"
    assert first["metadata"]["route"] == "greetings"
    assert last == {"output": GREETINGS_RESPONSE}
