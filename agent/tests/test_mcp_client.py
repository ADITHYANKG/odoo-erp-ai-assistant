"""Unit tests for ont_agent.mcp_client — fakes only, no network."""

from __future__ import annotations

import pytest

from ont_agent import mcp_client
from ont_agent.config import Settings
from tests.fakes import FakeTool


@pytest.fixture
def fake_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    settings = Settings(mcp_url="http://fake:1/mcp", mcp_bearer_key="sekret")
    monkeypatch.setattr(mcp_client, "load_settings", lambda: settings)
    return settings


def _patch_tools(monkeypatch: pytest.MonkeyPatch, tools: list[FakeTool]) -> None:
    async def fake_get_odoo_tools(mcp_url: str, bearer: str) -> list[FakeTool]:
        return tools

    monkeypatch.setattr(mcp_client, "get_odoo_tools", fake_get_odoo_tools)


def test_first_line_helper() -> None:
    assert mcp_client._first_line("hello\nworld") == "hello"
    assert mcp_client._first_line("  padded  \nrest") == "padded"
    assert mcp_client._first_line("") == ""
    assert mcp_client._first_line(None) == ""


async def test_run_lists_tools_and_calls_whoami(
    fake_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    whoami = FakeTool("whoami", "Identify the current user")
    _patch_tools(monkeypatch, [FakeTool("search_read", "Search records"), whoami])

    await mcp_client.run()

    out = capsys.readouterr().out
    assert "2 tools" in out
    assert "search_read" in out
    assert "whoami" in out
    assert whoami.calls == [{}]  # exactly one authenticated call
    assert "result-from-whoami" in out


async def test_run_without_whoami_degrades_gracefully(
    fake_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    only = FakeTool("search_read")
    _patch_tools(monkeypatch, [only])

    await mcp_client.run()  # must not raise

    out = capsys.readouterr().out
    assert "whoami tool not found" in out
    assert only.calls == []  # nothing was invoked


def test_main_wraps_failures_in_clean_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom() -> None:
        raise RuntimeError("connection refused")

    monkeypatch.setattr(mcp_client, "run", boom)
    with pytest.raises(SystemExit) as excinfo:
        mcp_client.main()
    assert "connection refused" in str(excinfo.value)
