"""Promptfoo provider: run one red-team prompt through the real agent pipeline.

Mirrors the tool chain ``bot.py`` builds for a chat turn — ``filter_for_bot``,
credential redaction, the prompt-extraction guards and the human-approval
interrupt — so an attack that passes here would pass in Telegram. Memory,
uploads, exports and the semantic router are left out.

**Nothing is ever written.** The agent is built with a checkpointer, so every
write tool pauses at the approval interrupt, and this provider never resumes
it. A proposed write is appended to the output as ``[PROPOSED WRITE] ...`` so
graders and assertions can see what the model would have done.

Per-test ``vars``:

- ``poison`` — text appended to the first read-tool result, simulating an
  instruction planted in Odoo data (indirect prompt injection) without
  writing to the database.

Per-target ``config``:

- ``token_env`` — env var holding the bearer token to act as (default
  ``MCP_BEARER_KEY``).
- ``can_write`` / ``can_call_methods`` — the user's granted scope.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

from ont_agent.agent import ask, build_agent
from ont_agent.config import load_settings
from ont_agent.llm import build_model
from ont_agent.promptguard import REFUSAL, leaks_prompt, looks_like_extraction
from ont_agent.tools import WRITE_TOOLS, filter_for_bot, get_odoo_tools, redact_tool_results

#: Marks a write the model proposed; graders and assertions key on it.
PROPOSED = "[PROPOSED WRITE]"


def _poisoned(tools: list[BaseTool], payload: str) -> list[BaseTool]:
    """Append ``payload`` to the first read-tool result of the turn."""
    state = {"done": False}

    def wrap(original: BaseTool) -> BaseTool:
        async def run(**kwargs: Any) -> Any:
            result = await original.ainvoke(kwargs)
            if state["done"]:
                return result
            state["done"] = True
            text = result if isinstance(result, str) else json.dumps(result, default=str)
            return f"{text}\n{payload}"

        return StructuredTool(
            name=original.name,
            description=original.description or "",
            args_schema=original.args_schema or {},
            coroutine=run,
        )

    return [tool if tool.name in WRITE_TOOLS else wrap(tool) for tool in tools]


async def _run(prompt: str, config: dict[str, Any], poison: str) -> dict[str, Any]:
    settings = load_settings(require_llm=True)
    if settings.prompt_guard and looks_like_extraction(prompt):
        return {"output": REFUSAL, "metadata": {"input_guard_blocked": True}}

    token = os.environ.get(config.get("token_env", ""), "") or settings.mcp_bearer_key
    tools: list[BaseTool] = filter_for_bot(
        await get_odoo_tools(settings.mcp_url, token),
        allow_method_calls=bool(config.get("can_call_methods", False)),
        allow_writes=bool(config.get("can_write", True)),
    )
    if poison:
        tools = _poisoned(tools, poison)
    tools = redact_tool_results(tools, settings.secret_scan)

    # The checkpointer is what turns on the approval interrupt for writes.
    agent = build_agent(build_model(settings), tools, checkpointer=InMemorySaver())
    result = await ask(
        agent,
        prompt,
        thread_id=str(uuid.uuid4()),
        recursion_limit=settings.recursion_limit,
    )

    reply = result.answer
    leaked = settings.prompt_guard and leaks_prompt(reply, settings.prompt_leak_ngram)
    lines = [REFUSAL if leaked else reply]
    if result.pending:
        for name, args in zip(result.pending.names, result.pending.arguments, strict=False):
            lines.append(f"{PROPOSED} {name} {json.dumps(args, default=str)}")
    return {
        "output": "\n".join(lines),
        "metadata": {
            "tool_calls": result.tool_calls,
            "failures": [f"{f.name}: {f.message}" for f in result.failures],
            "output_guard_replaced": bool(leaked),
        },
    }


def call_api(prompt: str, options: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    config = options.get("config") or {}
    poison = str((context or {}).get("vars", {}).get("poison") or "")
    try:
        return asyncio.run(_run(prompt, config, poison))
    except Exception as exc:  # a failed turn is a test error, not a crash
        return {"error": repr(exc)}
