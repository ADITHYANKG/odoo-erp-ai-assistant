"""Unit tests for ont_agent.tools — fakes only, no network."""

from __future__ import annotations

import base64
import json
from typing import Any, ClassVar

import pytest
from langchain_core.tools import tool
from mcp.types import Tool as MCPTool

from ont_agent import tools as tools_mod
from ont_agent.render import Artifact, discard
from ont_agent.tools import (
    CONFIRM_TOOLS,
    METHOD_TOOLS,
    WRITE_TOOLS,
    Evidence,
    bridge_export_tool,
    filter_for_bot,
    filter_read_only,
    get_odoo_tools,
)
from tests.fakes import FakeTool

ALL_TOOL_NAMES = [
    "call_method",
    "create_records",
    "delete_records",
    "describe_model",
    "export_records",
    "get_access_rights",
    "get_messages",
    "list_languages",
    "list_models",
    "list_modules",
    "post_message",
    "print_report",
    "read_group",
    "read_records",
    "read_resource",
    "search_count",
    "search_read",
    "system_info",
    "update_records",
    "whoami",
]


def _definition(name: str) -> MCPTool:
    return MCPTool(name=name, description="", inputSchema={"type": "object", "properties": {}})


@pytest.fixture
def fake_discovery(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Counts tools/list round-trips and records each built tool's connection."""
    tools_mod.clear_odoo_tools_cache()
    seen: dict[str, Any] = {"lists": 0, "connections": []}

    async def fake_list(connection: dict[str, Any]) -> list[MCPTool]:
        seen["lists"] += 1
        return [_definition("whoami")]

    real_convert = tools_mod.convert_mcp_tool_to_langchain_tool

    def recording_convert(session: Any, definition: MCPTool, **kwargs: Any) -> Any:
        seen["connections"].append(kwargs["connection"])
        return real_convert(session, definition, **kwargs)

    monkeypatch.setattr(tools_mod, "_list_definitions", fake_list)
    monkeypatch.setattr(tools_mod, "convert_mcp_tool_to_langchain_tool", recording_convert)
    return seen


async def test_get_odoo_tools_builds_bearer_connection(fake_discovery: dict[str, Any]) -> None:
    fetched = await get_odoo_tools("http://fake:1/mcp", "sekret")

    assert [tool.name for tool in fetched] == ["whoami"]
    conn = fake_discovery["connections"][-1]
    assert conn["transport"] == "streamable_http"
    assert conn["url"] == "http://fake:1/mcp"
    assert conn["headers"]["Authorization"] == "Bearer sekret"


async def test_get_odoo_tools_caches_discovery_not_tokens(fake_discovery: dict[str, Any]) -> None:
    # Discovery happens once per URL; every caller's tools still send its own
    # bearer — a re-issued token, a rotation or another user is never served
    # tools bound to someone else's token.
    await get_odoo_tools("http://fake-cache:1/mcp", "alice-token")
    bob = await get_odoo_tools("http://fake-cache:1/mcp", "bob-token")

    assert fake_discovery["lists"] == 1
    assert [tool.name for tool in bob] == ["whoami"]
    bearers = [conn["headers"]["Authorization"] for conn in fake_discovery["connections"]]
    assert bearers == ["Bearer alice-token", "Bearer bob-token"]

    await get_odoo_tools("http://fake-cache:1/mcp", "bob-token", force_refresh=True)
    assert fake_discovery["lists"] == 2


def test_filter_read_only_drops_exactly_the_write_tools() -> None:
    everything = [FakeTool(name) for name in ALL_TOOL_NAMES]

    kept = filter_read_only(everything)

    kept_names = {tool.name for tool in kept}
    assert len(kept) == 15
    assert kept_names.isdisjoint(WRITE_TOOLS)
    assert kept_names == set(ALL_TOOL_NAMES) - WRITE_TOOLS


def test_filter_read_only_preserves_order_and_handles_no_writes() -> None:
    safe = [FakeTool("whoami"), FakeTool("search_read"), FakeTool("read_group")]
    assert filter_read_only(safe) == safe


def test_write_tools_set_matches_server_write_surface() -> None:
    # muk_mcp's five write tools + ont_mcp_mail's send_email; the set must
    # never silently shrink (call_method is the widest write surface).
    assert {
        "call_method",
        "create_records",
        "update_records",
        "delete_records",
        "post_message",
        "send_email",
    } == WRITE_TOOLS


def test_filter_for_bot_hides_call_method_by_default() -> None:
    # Non-members of "MCP: Allow Method Calls" never even see call_method;
    # every other write tool stays in (approval happens at the interrupt).
    everything = [FakeTool(name) for name in ALL_TOOL_NAMES]

    kept_names = {tool.name for tool in filter_for_bot(everything)}

    assert {"call_method"} == METHOD_TOOLS
    assert "call_method" not in kept_names
    assert kept_names == set(ALL_TOOL_NAMES) - {"call_method"}


def test_filter_for_bot_offers_call_method_to_members() -> None:
    everything = [FakeTool(name) for name in ALL_TOOL_NAMES]
    kept_names = {tool.name for tool in filter_for_bot(everything, allow_method_calls=True)}
    assert kept_names == set(ALL_TOOL_NAMES)  # full surface, incl. call_method


def test_filter_for_bot_withholds_writes_from_a_read_only_grant() -> None:
    # Without mcp:write the server refuses every write tool anyway, so
    # offering them only produces approval prompts that cannot succeed
    # (live incident 2026-08-31). Withhold them and the model says so
    # up front instead.
    everything = [FakeTool(name) for name in ALL_TOOL_NAMES]

    kept_names = {tool.name for tool in filter_for_bot(everything, allow_writes=False)}

    assert not (kept_names & WRITE_TOOLS)
    assert kept_names == set(ALL_TOOL_NAMES) - WRITE_TOOLS
    # Read tools are untouched — a read-only user still gets full answers.
    assert "search_read" in kept_names


def test_read_only_grant_beats_the_method_call_group() -> None:
    # call_method is a write tool: the group permits it, but a read-only
    # grant must still withhold it. The narrower limit wins.
    everything = [FakeTool(name) for name in ALL_TOOL_NAMES]

    kept_names = {
        tool.name
        for tool in filter_for_bot(everything, allow_method_calls=True, allow_writes=False)
    }

    assert "call_method" not in kept_names
    assert not (kept_names & WRITE_TOOLS)


def test_every_write_tool_is_confirm_gated() -> None:
    # call_method included: ALL writes pause for the human Approve tap.
    assert CONFIRM_TOOLS == WRITE_TOOLS
    assert METHOD_TOOLS <= CONFIRM_TOOLS


# --- export bridge: base64 file payloads become chat attachments -----------


def _fake_export_tool(response: str) -> object:
    @tool
    async def export_records(model: str, fields: list[str]) -> str:
        """Export records as CSV or XLSX, returned as base64."""
        return response

    return export_records


def _payload(content: bytes, filename: str = "account_move.csv", rows: int = 11) -> str:
    return json.dumps(
        {
            "filename": filename,
            "mimetype": "text/csv",
            "row_count": rows,
            "content_base64": base64.b64encode(content).decode(),
        }
    )


async def test_bridge_attaches_decoded_file_and_hides_base64() -> None:
    csv_bytes = b'"Invoice","Customer"\n"INV/1","Northwind"\n'
    collector: list[Artifact] = []
    (bridged,) = bridge_export_tool([_fake_export_tool(_payload(csv_bytes))], collector)  # type: ignore[list-item]

    assert bridged.name == "export_records"  # model-facing identity unchanged
    message = await bridged.ainvoke({"model": "account.move", "fields": ["name"]})

    assert "account_move.csv" in message and "11 rows" in message
    assert "content_base64" not in message  # the blob never reaches the model
    (artifact,) = collector
    assert artifact.kind == "document"
    assert artifact.path.read_bytes() == csv_bytes  # byte-for-byte the export
    assert artifact.path.suffix == ".csv"
    assert "11 rows" in artifact.caption
    discard(collector)


async def test_bridge_passes_error_text_through_unchanged() -> None:
    denial = "Model 'ir.config_parameter' is not accessible via MCP."
    collector: list[Artifact] = []
    (bridged,) = bridge_export_tool([_fake_export_tool(denial)], collector)  # type: ignore[list-item]
    message = await bridged.ainvoke({"model": "x", "fields": ["y"]})
    assert message == denial  # the model must still see denials verbatim
    assert collector == []


async def test_bridge_clamps_oversized_non_json_text() -> None:
    collector: list[Artifact] = []
    (bridged,) = bridge_export_tool([_fake_export_tool("x" * 5000)], collector)  # type: ignore[list-item]
    message = await bridged.ainvoke({"model": "x", "fields": ["y"]})
    assert len(message) <= 2000


async def test_bridge_unwraps_content_and_artifact_tuples() -> None:
    # MCP adapter tools return (content_blocks, artifacts) on direct ainvoke
    # — the live bug: the bridge saw a tuple, failed to parse, and the model
    # bluffed an attachment. Every layer must unwrap.
    csv_bytes = b"a,b\n1,2\n"
    payload = _payload(csv_bytes, filename="sale_order.csv", rows=2)

    @tool("export_records", response_format="content_and_artifact")
    async def export_records_ca(model: str, fields: list[str]) -> tuple[object, None]:
        """Export records as CSV or XLSX, returned as base64."""
        return [{"type": "text", "text": payload}], None

    collector: list[Artifact] = []
    (bridged,) = bridge_export_tool([export_records_ca], collector)
    message = await bridged.ainvoke({"model": "sale.order", "fields": ["name"]})

    assert "sale_order.csv" in message and "content_base64" not in message
    (artifact,) = collector
    assert artifact.path.read_bytes() == csv_bytes
    discard(collector)


def test_bridge_leaves_other_tools_untouched() -> None:
    others: list[object] = [FakeTool("search_read"), FakeTool("whoami")]
    collector: list[Artifact] = []
    bridged: list[object] = list(bridge_export_tool(others, collector))  # type: ignore[arg-type]
    assert all(b is o for b, o in zip(bridged, others, strict=True))  # same objects


# --- deep-link harvesting --------------------------------------------------


def test_record_url_uses_odoo_17_scheme() -> None:
    from ont_agent.tools import record_url

    assert record_url("https://odoo.test", "account.move", 26) == (
        "https://odoo.test/web#model=account.move&id=26&view_type=form"
    )
    # A trailing slash on the configured URL must not double up.
    assert record_url("https://odoo.test/", "res.partner", 7) == (
        "https://odoo.test/web#model=res.partner&id=7&view_type=form"
    )
    # Dotless models take the m- prefix the web router expects.
    assert record_url("https://odoo.test", "website", 1) == (
        "https://odoo.test/web#model=website&id=1&view_type=form"
    )


class _RecordTool:
    """A fake search_read returning MCP-shaped content blocks."""

    name = "search_read"
    description = "search"
    args_schema: ClassVar[dict[str, object]] = {}

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    async def ainvoke(self, kwargs: dict[str, object]) -> object:
        self.calls.append(kwargs)
        return [{"type": "text", "text": self.payload}]


async def test_links_are_harvested_without_touching_the_result() -> None:
    from ont_agent.tools import bridge_record_links

    payload = '[{"id": 26, "display_name": "INV/2026/00026"}, {"id": 7, "name": "Acme Corp"}]'
    original = _RecordTool(payload)
    evidence = Evidence()

    wrapped = bridge_record_links([original], "https://odoo.test", evidence)[0]
    result = await wrapped.ainvoke({"model": "account.move", "domain": "[]"})

    # The model sees exactly what it would have seen unwrapped.
    assert result == [{"type": "text", "text": payload}]
    # Rows are counted as they pass, which is what makes "19 invoices" a
    # counted fact rather than something read back out of the prose.
    assert evidence.counts == {"account.move": 2}
    assert evidence.record_count == 2
    assert evidence.links == {
        "INV/2026/00026": "https://odoo.test/web#model=account.move&id=26&view_type=form",
        "Acme Corp": "https://odoo.test/web#model=account.move&id=7&view_type=form",
    }


async def test_short_or_numeric_names_are_never_linked() -> None:
    # "SO" or a bare figure in prose must not turn into a link.
    from ont_agent.tools import bridge_record_links

    payload = (
        '[{"id": 1, "name": "SO"}, {"id": 2, "name": "12345"}, {"id": 3, "name": "Real Name"}]'
    )
    evidence = Evidence()
    wrapped = bridge_record_links([_RecordTool(payload)], "https://odoo.test", evidence)[0]
    await wrapped.ainvoke({"model": "sale.order"})

    assert list(evidence.links) == ["Real Name"]
    # All three rows still COUNT — they exist, they are merely not linkable.
    assert evidence.counts == {"sale.order": 3}


async def test_a_malformed_result_yields_no_links_and_no_error() -> None:
    from ont_agent.tools import bridge_record_links

    evidence = Evidence()
    wrapped = bridge_record_links([_RecordTool("not json at all")], "https://odoo.test", evidence)[
        0
    ]
    result = await wrapped.ainvoke({"model": "sale.order"})

    assert evidence.links == {}
    assert evidence.counts == {}  # no count rather than a wrong one
    assert result == [{"type": "text", "text": "not json at all"}]


async def test_only_record_returning_tools_are_wrapped() -> None:
    from ont_agent.tools import bridge_record_links

    others = [FakeTool("read_group"), FakeTool("search_count"), FakeTool("send_email")]
    wrapped = bridge_record_links(others, "https://odoo.test", Evidence())
    assert wrapped == others  # untouched objects, not copies


def test_no_public_url_disables_linking_entirely() -> None:
    from ont_agent.tools import bridge_record_links

    tools = [FakeTool("search_read")]
    assert bridge_record_links(tools, "", Evidence()) == tools


# --- document search is a READ tool and must survive every filter ----------


def test_search_documents_survives_the_read_only_user_filter() -> None:
    """A user without mcp:write loses all six write tools. Losing document
    search with them would leave policy questions unanswerable for exactly
    the people most likely to ask them."""
    from ont_agent.tools import filter_for_bot

    tools = [FakeTool("search_read"), FakeTool("search_documents"), FakeTool("create_records")]
    kept = {tool.name for tool in filter_for_bot(tools, allow_writes=False)}

    assert "search_documents" in kept
    assert "create_records" not in kept


def test_search_documents_survives_the_proactive_filter() -> None:
    # Morning briefs and watchers run read-only, with nobody present to
    # approve anything; a watcher on a policy change still needs to search.
    from ont_agent.tools import filter_read_only

    tools = [FakeTool("search_documents"), FakeTool("delete_records")]
    kept = {tool.name for tool in filter_read_only(tools)}

    assert kept == {"search_documents"}


# --- credential redaction on tool results -----------------------------------


async def test_a_clean_result_is_passed_through_untouched() -> None:
    """The object itself, not a rebuilt one. This is the common path on
    essentially every call, so it must involve no shape conversion."""
    from ont_agent.tools import redact_tool_results

    payload = '[{"id": 7, "name": "Acme", "comment": "prefers email"}]'
    tool = _RecordTool(payload)
    wrapped = redact_tool_results([tool])[0]  # type: ignore[list-item]

    result = await wrapped.ainvoke({"model": "res.partner"})

    assert result == [{"type": "text", "text": payload}]


async def test_a_credential_in_a_record_never_reaches_the_model() -> None:
    # The hook that CONTAINS the leak: redacting here keeps the key out of
    # the reply, the conversation checkpoint and the Langfuse trace at once.
    from ont_agent.tools import redact_tool_results

    secret = "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcd"
    tool = _RecordTool(f'[{{"id": 7, "comment": "api {secret}"}}]')
    wrapped = redact_tool_results([tool])[0]  # type: ignore[list-item]

    result = await wrapped.ainvoke({"model": "res.partner"})

    assert secret not in str(result)
    assert "[redacted: OpenAI-style key]" in str(result)
    assert '"id": 7' in str(result)  # the rest of the record survives


async def test_redaction_can_be_switched_off() -> None:
    from ont_agent.tools import redact_tool_results

    tool = _RecordTool("sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcd")
    assert redact_tool_results([tool], enabled=False) == [tool]  # type: ignore[list-item,comparison-overlap]


async def test_tool_discovery_that_hangs_times_out_and_frees_the_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The fetch runs under a lock every chat shares: a hang there must not
    # block every other chat's next fetch.
    import asyncio

    async def hang(connection: Any) -> list[MCPTool]:
        await asyncio.Event().wait()
        return []

    tools_mod.clear_odoo_tools_cache()
    monkeypatch.setattr(tools_mod, "_list_definitions", hang)
    monkeypatch.setattr(tools_mod, "TOOLS_FETCH_TIMEOUT_SECONDS", 0.05)

    with pytest.raises(TimeoutError):
        await get_odoo_tools("http://fake:hang/mcp", "sekret")
    assert not tools_mod._TOOLS_CACHE_LOCK.locked()
