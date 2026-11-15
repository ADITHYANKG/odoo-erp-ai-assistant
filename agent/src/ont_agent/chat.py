"""Phase 2 CLI: ask one natural-language question over the Odoo data.

    uv run python -m ont_agent.chat "How many contacts do we have?"
    uv run ont-agent-chat "How many contacts do we have?"

Diagnostics (tool trace, timings) go to the log on stderr; the answer itself
prints to stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import time

from ont_agent.agent import ask, build_agent
from ont_agent.config import load_settings
from ont_agent.llm import build_model
from ont_agent.log import get_logger, setup_logging
from ont_agent.tools import filter_read_only, get_odoo_tools

logger = get_logger(__name__)

DEFAULT_QUESTION = "How many contacts do we have, and what are the names of the first five?"


async def run(question: str) -> None:
    settings = load_settings(require_llm=True)

    started = time.monotonic()
    tools = filter_read_only(await get_odoo_tools(settings.mcp_url, settings.mcp_bearer_key))
    logger.info("%d read-only tools ready in %.2fs", len(tools), time.monotonic() - started)

    agent = build_agent(build_model(settings), tools)

    print(f"Q: {question}")
    started = time.monotonic()
    result = await ask(agent, question)
    logger.info(
        "Answered in %.2fs after %d tool call(s)",
        time.monotonic() - started,
        len(result.tool_calls),
    )

    print("\n=== Answer ===")
    print(result.answer or "(the model returned no text)")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()
    try:
        asyncio.run(run(args.question))
    except Exception as exc:  # surface transport/model failures readably
        logger.error("Chat run failed", exc_info=True)
        raise SystemExit(f"Chat run failed: {exc!r}") from exc


if __name__ == "__main__":
    main()
