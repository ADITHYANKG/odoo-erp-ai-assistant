"""Odoo MCP tool loading and phase-scoped guardrails.

One connection path shared by every entry point (the lister and the agent),
plus a client-side write filter: phase 2 is read-only, so write-capable tools
are never even offered to the model — defense in depth on top of the
read-scope bearer key, whose enforcement lives server-side in muk_mcp.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StreamableHttpConnection
from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool
from mcp.types import Tool as MCPTool

from ont_agent.log import get_logger
from ont_agent.redaction import redact_secrets
from ont_agent.render import Artifact, temp_file_path

logger = get_logger(__name__)

#: Tools that mutate Odoo state or act on the outside world. Kept in sync
#: with the server's write-category tools: muk_mcp's five plus our
#: ont_mcp_mail send_email.
WRITE_TOOLS: frozenset[str] = frozenset(
    {
        "call_method",
        "create_records",
        "update_records",
        "delete_records",
        "post_message",
        "send_email",
    }
)

#: Write tools the bot offers the model — each one interrupts the graph for a
#: human Approve/Cancel in Telegram before executing (phase 5 decision).
CONFIRM_TOOLS: frozenset[str] = WRITE_TOOLS

#: Confirm-gated tools whose approval prompt also offers ✏️ Edit (revise via
#: a plain-language instruction). Only mail: record mutations are
#: approve-or-cancel to keep their semantics unambiguous.
EDITABLE_TOOLS: frozenset[str] = frozenset({"send_email"})

#: Offered only when the bound user is in "MCP: Allow Method Calls"
#: (administrators by default): call_method invokes ANY public method — the
#: widest write surface. The server enforces the same gate regardless.
METHOD_TOOLS: frozenset[str] = frozenset({"call_method"})

#: Actions with no compensating action: a sent email cannot be unsent and a
#: deleted record cannot be restored. Everything else can be corrected after
#: the fact — an update rewritten, a created record removed. The approval
#: button turns red for these, and it is the same property that made the
#: planned Undo hold actions rather than try to reverse them.
IRREVERSIBLE_TOOLS: frozenset[str] = frozenset({"delete_records", "send_email"})


class _NamedTool(Protocol):
    name: str


#: Tool DEFINITIONS (name, description, schema) per MCP URL -> (cached_at, defs).
#: Only schemas are shared: Odoo lists tools under sudo, identical for every
#: user, while each call carries the caller's own bearer (see get_odoo_tools).
_DEFINITIONS_CACHE: dict[str, tuple[float, list[MCPTool]]] = {}
_TOOLS_CACHE_LOCK = asyncio.Lock()
DEFAULT_TOOLS_CACHE_TTL_SECONDS = 3600
#: Ceiling on one tools/list round trip; a healthy Odoo answers in about a second.
TOOLS_FETCH_TIMEOUT_SECONDS = 30
#: Server name the tools are registered under (matches the old client config).
_SERVER = "odoo"


def clear_odoo_tools_cache() -> None:
    """Clear cached MCP tool definitions (useful for tests or forced reload)."""
    _DEFINITIONS_CACHE.clear()


async def _list_definitions(connection: StreamableHttpConnection) -> list[MCPTool]:
    client = MultiServerMCPClient({_SERVER: connection})
    definitions: list[MCPTool] = []
    cursor: str | None = None
    async with client.session(_SERVER) as session:
        while True:
            page = await session.list_tools(cursor=cursor)
            definitions.extend(page.tools)
            cursor = page.nextCursor
            if not cursor:
                return definitions


async def get_odoo_tools(
    mcp_url: str,
    bearer: str,
    *,
    ttl: float = DEFAULT_TOOLS_CACHE_TTL_SECONDS,
    force_refresh: bool = False,
) -> list[BaseTool]:
    """Connect to the Odoo MCP server and return its tools as LangChain tools.

    ``bearer`` may be a muk_mcp key (the shared dev key) or a per-user OAuth
    JSON Web Token — the transport is identical.

    Discovery (``tools/list``) is cached per ``mcp_url`` for ``ttl`` seconds;
    the returned tools are rebuilt on every call (no network) so each one
    sends THIS ``bearer``, never a cached token.
    """
    connection: StreamableHttpConnection = {
        "transport": "streamable_http",
        "url": mcp_url,
        "headers": {"Authorization": f"Bearer {bearer}"},
    }
    cached = _DEFINITIONS_CACHE.get(mcp_url)
    if force_refresh or cached is None or time.monotonic() - cached[0] >= ttl:
        async with _TOOLS_CACHE_LOCK:
            cached = _DEFINITIONS_CACHE.get(mcp_url)
            if force_refresh or cached is None or time.monotonic() - cached[0] >= ttl:
                logger.info("Fetching tool definitions from %s", mcp_url)
                # Bounded: this runs under a lock every chat shares, and an MCP
                # session can wait forever on a response that never comes.
                try:
                    async with asyncio.timeout(TOOLS_FETCH_TIMEOUT_SECONDS):
                        definitions = await _list_definitions(connection)
                except TimeoutError:
                    logger.error(
                        "Tool discovery from %s timed out after %ss",
                        mcp_url,
                        TOOLS_FETCH_TIMEOUT_SECONDS,
                    )
                    raise
                logger.info(
                    "Fetched %d tools from %s (caching definitions for %.0fs)",
                    len(definitions),
                    mcp_url,
                    ttl,
                )
                cached = (time.monotonic(), definitions)
                _DEFINITIONS_CACHE[mcp_url] = cached
    return [
        convert_mcp_tool_to_langchain_tool(
            None, definition, connection=connection, server_name=_SERVER
        )
        for definition in cached[1]
    ]


#: The muk_mcp tool whose result is a base64 file payload, not prose.
EXPORT_TOOL = "export_records"
#: Non-file passthrough (e.g. access-denied text) is clamped defensively.
_PASSTHROUGH_LIMIT = 2000


def bridge_export_tool(tools: Sequence[BaseTool], collector: list[Artifact]) -> list[BaseTool]:
    """Replace ``export_records`` so its file reaches the CHAT, not the model.

    The raw tool returns ``{filename, mimetype, row_count, content_base64}``;
    unbridged, the base64 floods the model context and the file never reaches
    the user (live incident: the model then *claimed* a file was delivered).
    The bridge decodes the payload outside the model's context, appends a
    document :class:`Artifact` to the per-message collector (the same
    pipeline charts and PDFs ride), and hands the model a short confirmation
    instead. Error/denial strings pass through unchanged so the model can
    react to them.
    """
    return [_wrap_export(tool, collector) if tool.name == EXPORT_TOOL else tool for tool in tools]


def _content_text(raw: Any) -> str:
    """Text out of an MCP adapter tool result.

    Direct ``ainvoke`` on adapter tools (response_format
    ``content_and_artifact``) returns a ``(content, artifacts)`` tuple, and
    the content itself may be a list of content BLOCKS
    (``{"type": "text", "text": ...}``) rather than a plain string — unwrap
    every layer down to text.
    """
    if isinstance(raw, tuple) and raw:
        raw = raw[0]
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
            else:
                text_attr = getattr(item, "text", None)
                parts.append(str(text_attr) if text_attr is not None else str(item))
        return "\n".join(parts)
    return str(raw)


def _wrap_export(original: BaseTool, collector: list[Artifact]) -> BaseTool:
    async def export_and_attach(**kwargs: Any) -> str:
        text = _content_text(await original.ainvoke(kwargs))
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text[:_PASSTHROUGH_LIMIT]
        if not isinstance(payload, dict) or "content_base64" not in payload:
            return text[:_PASSTHROUGH_LIMIT]

        filename = str(payload.get("filename") or "export.csv")
        row_count = payload.get("row_count", "?")
        try:
            content = base64.b64decode(payload["content_base64"])
            stem, _, extension = filename.rpartition(".")
            path = temp_file_path(stem or "export", f".{extension}" if extension else ".csv")
            path.write_bytes(content)
        except Exception:
            logger.error("Export file could not be prepared", exc_info=True)
            return "Error: the export file could not be prepared."
        collector.append(
            Artifact(path=path, kind="document", caption=f"{filename} · {row_count} rows")
        )
        logger.info("Export bridged: %s (%s rows, %d bytes)", filename, row_count, len(content))
        return (
            f"Export '{filename}' ({row_count} rows) created; it will be attached to "
            "your reply as a downloadable file. Keep the accompanying text to one or "
            "two sentences and do not repeat the file contents."
        )

    return StructuredTool(
        name=original.name,
        description=(original.description or "")
        + " The exported file is attached to the user's chat automatically - "
        "never print its contents.",
        args_schema=original.args_schema if original.args_schema is not None else {},
        coroutine=export_and_attach,
    )


#: Tools whose results are lists of records carrying ids — the ones worth
#: watching for deep links. read_group returns aggregates, not records.
#: Tools that return PROSE from documents rather than rows from Odoo. A
#: turn grounded only in these is a real, cited answer — but there is
#: nothing in it to chart or break down by customer, because a policy is
#: not a dataset.
DOCUMENT_TOOLS: frozenset[str] = frozenset({"search_documents"})

LINKABLE_TOOLS: frozenset[str] = frozenset({"search_read", "read_records"})

#: Shorter names are too collision-prone to link ("SO", "Ltd"), and a purely
#: numeric name would turn stray figures in prose into links.
_MIN_LINKABLE_NAME = 4


def record_url(odoo_public_url: str, model: str, record_id: int) -> str:
    """Odoo 17 deep link to one record.

    Odoo 17's web client routes on the URL fragment:
    ``/web#model={model}&id={id}&view_type=form``. (The path-based
    ``/odoo/{model}/{id}`` scheme only arrived in Odoo 18.)
    """
    base = odoo_public_url.rstrip("/")
    return f"{base}/web#model={model}&id={record_id}&view_type=form"


@dataclass
class Evidence:
    """What the tools actually saw this turn.

    Collected as a side effect of calls the model was making anyway, so it
    costs nothing. Three features read it: deep links use ``links``, the
    freshness stamp uses ``counts``, and "show your working" uses both plus
    the tool-call trace the agent already records.
    """

    #: Record display name -> Odoo URL.
    links: dict[str, str] = field(default_factory=dict)
    #: Model -> how many rows came back. This is what makes "19 invoices" a
    #: counted fact rather than something inferred from the prose.
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def record_count(self) -> int:
        return sum(self.counts.values())


def _harvest(payload: Any, model: str, base_url: str, evidence: Evidence) -> None:
    """Record names, URLs and row counts from one tool result."""
    rows = payload if isinstance(payload, list) else [payload]
    countable = [row for row in rows if isinstance(row, dict)]
    if countable:
        evidence.counts[model] = evidence.counts.get(model, 0) + len(countable)
    for row in countable:
        record_id = row.get("id")
        name = row.get("display_name") or row.get("name")
        if not isinstance(record_id, int) or not isinstance(name, str):
            continue
        cleaned = name.strip()
        if len(cleaned) < _MIN_LINKABLE_NAME or cleaned.replace(".", "").isdigit():
            continue
        evidence.links.setdefault(cleaned, record_url(base_url, model, record_id))


def bridge_record_links[ToolT: _NamedTool](
    tools: Sequence[ToolT], odoo_public_url: str, evidence: Evidence
) -> list[ToolT]:
    """Watch read tools for record ids, so answers can link back to Odoo.

    A pure observer: results pass through untouched and the model never sees
    a difference. The ids have already flowed past us, so linking costs no
    extra tool call — nothing is looked up.
    """
    if not odoo_public_url:
        return list(tools)
    return [
        _wrap_links(tool, odoo_public_url, evidence) if tool.name in LINKABLE_TOOLS else tool
        for tool in tools
    ]


def _wrap_links(original: Any, base_url: str, evidence: Evidence) -> Any:
    async def observe(**kwargs: Any) -> Any:
        result = await original.ainvoke(kwargs)
        model = kwargs.get("model")
        if isinstance(model, str):
            # An unparseable result simply yields no links; observing must
            # never be able to fail an answer.
            with contextlib.suppress(json.JSONDecodeError, TypeError, ValueError):
                _harvest(json.loads(_content_text(result)), model, base_url, evidence)
        return result

    return StructuredTool(
        name=original.name,
        description=original.description or "",
        args_schema=original.args_schema if original.args_schema is not None else {},
        coroutine=observe,
    )


def redact_tool_results(tools: Sequence[BaseTool], enabled: bool = True) -> list[BaseTool]:
    """Strip credentials out of tool results BEFORE the model sees them.

    This is the hook that contains the leak rather than merely hiding it. A
    key pasted into a res.partner note otherwise reaches three places at
    once: the reply, the conversation checkpoint (stored as a ToolMessage
    for up to 30 days) and the Langfuse trace. Redacting here means it
    reaches none of them.

    Applied OUTERMOST over the Odoo tools, so bridge_record_links still
    harvests ids and bridge_export_tool still bridges from the original
    text; redaction is the last thing before the model.
    """
    if not enabled:
        return list(tools)
    return [_wrap_redaction(tool) for tool in tools]


def _wrap_redaction(original: BaseTool) -> BaseTool:
    async def scrub(**kwargs: Any) -> Any:
        result = await original.ainvoke(kwargs)
        clean, kinds = redact_secrets(_content_text(result))
        if not kinds:
            # The original OBJECT, not a rebuilt one: on essentially every
            # call this must be a no-op, with no shape conversion and no
            # behaviour change.
            return result
        # The kinds, never the values — logging the secret would move it
        # into bot.log rather than remove it.
        logger.warning("Redacted %s from %s result", ", ".join(kinds), original.name)
        return clean

    # getattr rather than attribute access: this wraps EVERY Odoo tool,
    # including any that does not implement the full BaseTool surface, where
    # _wrap_links only ever sees the handful in LINKABLE_TOOLS.
    return StructuredTool(
        name=original.name,
        description=getattr(original, "description", "") or "",
        args_schema=getattr(original, "args_schema", None) or {},
        coroutine=scrub,
    )


def filter_read_only[ToolT: _NamedTool](tools: Sequence[ToolT]) -> list[ToolT]:
    """Drop ALL write-capable tools — for entry points with no confirmation UI
    (the one-shot CLI), which stay read-only by design."""
    kept = [tool for tool in tools if tool.name not in WRITE_TOOLS]
    dropped = sorted({tool.name for tool in tools} & WRITE_TOOLS)
    if dropped:
        logger.info("Write tools withheld from the model: %s", ", ".join(dropped))
    return kept


def filter_for_bot[ToolT: _NamedTool](
    tools: Sequence[ToolT], *, allow_method_calls: bool = False, allow_writes: bool = True
) -> list[ToolT]:
    """Per-user tool surface for the bot.

    ``allow_writes`` reflects the GRANTED OAuth scope. Without ``mcp:write``
    every write tool would be refused by the server anyway, so offering them
    only produces approval prompts that cannot succeed (live incident
    2026-08-31: a read-only user approved a ``send_email`` the server then
    denied). Withholding them lets the model say so up front instead.

    Write tools a user *can* execute always stay in — the bot pauses for
    human approval before any of them runs. ``call_method`` is narrower
    still: offered only to users allowed to call methods, so everyone else's
    model never even sees it.
    """
    if not allow_writes:
        tools = filter_read_only(tools)
    if allow_method_calls:
        return list(tools)
    kept = [tool for tool in tools if tool.name not in METHOD_TOOLS]
    dropped = sorted({tool.name for tool in tools} & METHOD_TOOLS)
    if dropped:
        logger.info("Method tools withheld from the model: %s", ", ".join(dropped))
    return kept
