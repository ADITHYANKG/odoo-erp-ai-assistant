"""Phase 1: prove the agent can talk to Odoo's MCP endpoint.

Connects to muk_mcp over Streamable HTTP with a bearer key, lists every tool
the server exposes, then makes one real authenticated tool call (``whoami``)
so the round trip is proven end to end — schema listing alone would not
exercise auth on tools/call.

Diagnostics go to the log (stderr, LOG_LEVEL env); the report itself prints
to stdout. Run from ``agent/``:

    uv run python -m ont_agent.mcp_client
    uv run ont-agent-list-tools
"""

from __future__ import annotations

import asyncio
import time

from ont_agent.config import load_settings
from ont_agent.log import get_logger, setup_logging
from ont_agent.tools import get_odoo_tools

logger = get_logger(__name__)


def _first_line(text: str | None) -> str:
    lines = (text or "").strip().splitlines()
    return lines[0].strip() if lines else ""


async def run() -> None:
    settings = load_settings()
    print(f"Connecting to {settings.mcp_url} ...")

    started = time.monotonic()
    tools = await get_odoo_tools(settings.mcp_url, settings.mcp_bearer_key)
    logger.info("Fetched %d tools in %.2fs", len(tools), time.monotonic() - started)

    print(f"\n{len(tools)} tools exposed by the Odoo MCP server:\n")
    width = max(len(tool.name) for tool in tools)
    for tool in sorted(tools, key=lambda t: t.name):
        print(f"  {tool.name:<{width}}  {_first_line(tool.description)}")

    whoami = next((tool for tool in tools if tool.name == "whoami"), None)
    if whoami is None:
        logger.warning("whoami tool absent — skipping the tool-call smoke test")
        print("\nwhoami tool not found — cannot smoke-test a tool call.")
        return

    print("\nCalling whoami() to prove an authenticated tool call ...")
    started = time.monotonic()
    result = await whoami.ainvoke({})
    logger.info("whoami call completed in %.2fs", time.monotonic() - started)
    print(f"whoami -> {result}")


def main() -> None:
    setup_logging()
    try:
        asyncio.run(run())
    except Exception as exc:  # surface auth/transport failures readably
        logger.error("MCP connection failed", exc_info=True)
        raise SystemExit(f"MCP connection failed: {exc!r}") from exc


if __name__ == "__main__":
    main()
