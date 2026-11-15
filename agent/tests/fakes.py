"""Shared test fakes — no network, no Odoo, no LLM, no Langfuse."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, ClassVar


class FakeTool:
    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, args: dict[str, Any]) -> str:
        self.calls.append(args)
        return f"result-from-{self.name}"


class FakeMCPClient:
    """Stands in for MultiServerMCPClient; records the connection config."""

    tools_to_return: ClassVar[list[FakeTool]] = []
    last_connections: ClassVar[dict[str, Any] | None] = None

    def __init__(self, connections: dict[str, Any]) -> None:
        FakeMCPClient.last_connections = connections
        self._tools = self.tools_to_return  # set per-test on the class

    async def get_tools(self) -> list[FakeTool]:
        return self._tools


class FakeSpan:
    """Records span.update calls (input/output enrichment)."""

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class FakeLangfuseClient:
    """Stands in for the Langfuse SDK client; records spans/scores/flushes."""

    def __init__(self, fail_scores: bool = False) -> None:
        self.scores: list[dict[str, Any]] = []
        self.spans: list[dict[str, Any]] = []
        self.span_objects: list[FakeSpan] = []
        self.flushed = 0
        self.fail_scores = fail_scores

    @contextmanager
    def start_as_current_observation(self, **kwargs: Any) -> Any:
        self.spans.append(kwargs)
        span = FakeSpan()
        self.span_objects.append(span)
        yield span

    def create_score(self, **kwargs: Any) -> None:
        if self.fail_scores:
            raise RuntimeError("langfuse down")
        self.scores.append(kwargs)

    def flush(self) -> None:
        self.flushed += 1
