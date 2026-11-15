"""Unit tests for ont_agent.tracing (fake Langfuse client, no server)."""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any

import pytest

from ont_agent import tracing
from ont_agent.config import Settings
from ont_agent.tracing import Tracing, init_tracing
from tests.fakes import FakeLangfuseClient as FakeClient

DISABLED = Settings(mcp_url="http://odoo:8160/mcp", mcp_bearer_key="k")
ENABLED = Settings(
    mcp_url="http://odoo:8160/mcp",
    mcp_bearer_key="k",
    langfuse_public_key="pk-lf-x",
    langfuse_secret_key="sk-lf-x",
)


def test_settings_langfuse_enabled_requires_both_keys() -> None:
    assert not DISABLED.langfuse_enabled
    assert ENABLED.langfuse_enabled
    half = Settings(mcp_url="http://x/mcp", mcp_bearer_key="k", langfuse_public_key="pk")
    assert not half.langfuse_enabled


def test_init_tracing_disabled_without_keys() -> None:
    assert init_tracing(DISABLED) is None


def test_init_tracing_stays_on_when_langfuse_is_not_up_yet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # After a host reboot the bot can start before Langfuse answers. The
    # failed check must not disable tracing for the life of the process.
    import langfuse

    class StartingLangfuse:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def auth_check(self) -> bool:
            raise TimeoutError("timed out")

    monkeypatch.setattr(langfuse, "Langfuse", StartingLangfuse)
    assert isinstance(init_tracing(ENABLED), Tracing)


def test_init_tracing_disabled_when_the_client_cannot_be_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import langfuse

    def broken(**kwargs: Any) -> Any:
        raise ValueError("bad config")

    monkeypatch.setattr(langfuse, "Langfuse", broken)
    assert init_tracing(ENABLED) is None


def test_turn_yields_hex_trace_id_and_opens_span() -> None:
    client = FakeClient()
    t = Tracing(client, handler=object())  # type: ignore[arg-type]
    with t.turn(
        trace_name="telegram-ask", user_id="ceo@acme", session_id="42", input_text="revenue?"
    ) as turn:
        assert re.fullmatch(r"[0-9a-f]{32}", turn.trace_id)
        turn.set_output("42k this month")
    (span,) = client.spans
    assert span["name"] == "telegram-ask"
    assert span["trace_context"] == {"trace_id": turn.trace_id}
    # Question and answer land on the root span — what UI evaluators map to.
    (span_object,) = client.span_objects
    assert {"input": "revenue?"} in span_object.updates
    assert {"output": "42k this month"} in span_object.updates


def test_turn_survives_span_setup_failure() -> None:
    class ExplodingClient(FakeClient):
        @contextmanager
        def start_as_current_observation(self, **kwargs: Any) -> Any:
            raise RuntimeError("no otel for you")
            yield  # pragma: no cover

    t = Tracing(ExplodingClient(), handler=object())  # type: ignore[arg-type]
    with t.turn(trace_name="telegram-ask", user_id="u", session_id="1") as turn:
        ran = True  # the turn itself must still run, just untraced
        turn.set_output("still fine")  # inert Turn: must not raise
    assert ran
    assert turn.trace_id == ""


def test_record_feedback_scores_the_trace() -> None:
    client = FakeClient()
    t = Tracing(client, handler=object())  # type: ignore[arg-type]
    t.record_feedback("a" * 32, positive=True)
    t.record_feedback("b" * 32, positive=False)
    assert client.scores == [
        {"trace_id": "a" * 32, "name": "user-feedback", "value": 1, "data_type": "NUMERIC"},
        {"trace_id": "b" * 32, "name": "user-feedback", "value": 0, "data_type": "NUMERIC"},
    ]


def test_record_decision_scores_the_proposing_trace() -> None:
    client = FakeClient()
    t = Tracing(client, handler=object())  # type: ignore[arg-type]
    t.record_decision("a" * 32, "approved")
    t.record_decision("b" * 32, "cancelled")
    t.record_decision("c" * 32, "edit-requested")
    assert [(s["name"], s["value"], s["comment"]) for s in client.scores] == [
        ("action-decision", 1, "approved"),
        ("action-decision", 0, "cancelled"),
        ("action-decision", 0, "edit-requested"),
    ]


def test_record_score_is_generic_and_never_raises() -> None:
    client = FakeClient()
    t = Tracing(client, handler=object())  # type: ignore[arg-type]
    t.record_score("d" * 32, "custom-metric", 1, comment="any score name works")
    (score,) = client.scores
    assert score["name"] == "custom-metric"
    Tracing(FakeClient(fail_scores=True), handler=object()).record_score(  # type: ignore[arg-type]
        "e" * 32, "x", 0
    )  # must not raise


def test_record_feedback_swallows_failures() -> None:
    t = Tracing(FakeClient(fail_scores=True), handler=object())  # type: ignore[arg-type]
    t.record_feedback("c" * 32, positive=True)  # must not raise


def test_flush_never_raises() -> None:
    client = FakeClient()
    t = Tracing(client, handler=object())  # type: ignore[arg-type]
    t.flush()
    assert client.flushed == 1

    class BadFlush(FakeClient):
        def flush(self) -> None:
            raise RuntimeError("boom")

    Tracing(BadFlush(), handler=object()).flush()  # type: ignore[arg-type]


def _llm_result(metadata: dict[str, Any]) -> Any:
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    message = AIMessage(content="hi", response_metadata=metadata)
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_extract_cost_details_reads_openrouter_cost() -> None:
    result = tracing.extract_cost_details(_llm_result({"cost": 0.000279375}))
    assert result == {"total": 0.000279375}


def test_extract_cost_details_handles_missing_or_bad_cost() -> None:
    assert tracing.extract_cost_details(_llm_result({})) is None
    assert tracing.extract_cost_details(_llm_result({"cost": "free"})) is None
    assert tracing.extract_cost_details(_llm_result({"cost": -1})) is None

    from langchain_core.outputs import Generation, LLMResult

    plain = LLMResult(generations=[[Generation(text="no message attr")]])
    assert tracing.extract_cost_details(plain) is None
    assert tracing.extract_cost_details(LLMResult(generations=[])) is None


def test_cost_aware_handler_ingests_cost_then_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from uuid import uuid4

    from langfuse.langchain import CallbackHandler

    delegated: list[Any] = []
    monkeypatch.setattr(
        CallbackHandler,
        "on_llm_end",
        lambda self, response, *, run_id, parent_run_id=None, **kw: delegated.append(run_id),
    )

    class RecorderRun:
        def __init__(self) -> None:
            self.updates: list[dict[str, Any]] = []

        def update(self, **kwargs: Any) -> RecorderRun:
            self.updates.append(kwargs)
            return self

    handler = tracing._build_cost_aware_handler()
    run_id = uuid4()
    run = RecorderRun()
    handler._runs[run_id] = run  # type: ignore[attr-defined]

    handler.on_llm_end(_llm_result({"cost": 0.001}), run_id=run_id)
    assert run.updates == [{"cost_details": {"total": 0.001}}]
    assert delegated == [run_id]

    # No cost → no update, but ALWAYS delegates.
    handler.on_llm_end(_llm_result({}), run_id=run_id)
    assert run.updates == [{"cost_details": {"total": 0.001}}]
    assert delegated == [run_id, run_id]

    # A poisoned response must neither raise nor block delegation.
    handler.on_llm_end(object(), run_id=run_id)  # type: ignore[arg-type]
    assert delegated == [run_id, run_id, run_id]


def test_module_disabled_path_does_not_require_langfuse_import() -> None:
    # The disabled path returns before touching the SDK — assert the module
    # itself has no top-level langfuse import (lazy imports only).
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(tracing))
    top_level_imports = {
        name.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import | ast.ImportFrom)
        for name in (node.names if isinstance(node, ast.Import) else [])
    } | {
        node.module.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "langfuse" not in top_level_imports
