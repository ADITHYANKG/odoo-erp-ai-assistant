# Telegram Bot Streaming Knowledge for Copilot

Use this guide when implementing streamed LLM responses in an asynchronous Python Telegram bot. It is designed for `python-telegram-bot` and an OpenAI-compatible async client, including OpenRouter.

## Goal

Make a Telegram response appear progressively while the language model is generating it, without sending one Telegram message per token.

The preferred flow is:

1. Receive the user's message.
2. Start an LLM request with `stream=True`.
3. Accumulate content deltas into one response string.
4. Update one Telegram draft or placeholder message at a throttled interval.
5. Accumulate fragmented tool-call deltas when tools are enabled.
6. Execute requested tools and continue the model/tool loop.
7. Send or finalize one permanent answer.
8. Roll back incomplete conversation state when the turn fails.

## Important Distinction

There are two unrelated uses of the word "streaming":

- LLM response streaming: partial text and tool-call deltas arrive from the model. This creates the live typing effect.
- MCP streamable HTTP: the transport used to communicate with an MCP server. A tool result can still arrive as one complete result and is not automatically streamed to Telegram.

## Core Design Rules

- Use asynchronous APIs end to end. Do not block the Telegram event loop with synchronous HTTP calls.
- Enable model streaming with `stream=True`.
- Accumulate deltas because each chunk contains only the newly generated fragment.
- Update Telegram every 1-2 seconds, not for every token.
- Keep one stable draft ID or one stable placeholder message ID for the whole turn.
- Treat live updates as best-effort. A failed preview update must not discard a valid final answer.
- Respect Telegram's 4,096-character text limit.
- Keep the final accumulated answer for conversation history, not only the most recent chunk.
- Serialize turns per chat with an `asyncio.Lock` so two messages cannot corrupt shared history.
- Do not log API keys, Telegram tokens, tool credentials, or full sensitive tool results.

## Dependencies

```text
openai>=1.40
python-telegram-bot
```

Before using `Bot.send_message_draft`, verify that the installed `python-telegram-bot` version exposes that method and that the deployed Telegram Bot API supports it. Otherwise, use the placeholder-message fallback described below.

## Reusable Streaming Function

OpenAI-compatible providers can stream both text and tool calls. Tool names and JSON arguments may be split across several chunks, so they must be assembled by tool-call index.

```python
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from openai import AsyncOpenAI


PartialTextCallback = Callable[[str], Awaitable[None]]


async def stream_completion(
    client: AsyncOpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: Sequence[dict[str, Any]],
    on_partial_text: PartialTextCallback,
    tool_choice: str | None = None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": True,
    }
    if tool_choice is not None:
        request["tool_choice"] = tool_choice

    stream = await client.chat.completions.create(**request)
    content = ""
    tool_calls: dict[int, dict[str, Any]] = {}

    async for chunk in stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        if delta.content:
            content += delta.content
            await on_partial_text(content)

        for delta_call in delta.tool_calls or []:
            call = tool_calls.setdefault(
                delta_call.index,
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            if delta_call.id:
                call["id"] = delta_call.id
            if delta_call.function and delta_call.function.name:
                call["function"]["name"] += delta_call.function.name
            if delta_call.function and delta_call.function.arguments:
                call["function"]["arguments"] += delta_call.function.arguments

    message: dict[str, Any] = {
        "role": "assistant",
        "content": content or None,
    }
    if tool_calls:
        message["tool_calls"] = [
            tool_calls[index] for index in sorted(tool_calls)
        ]
    return message
```

Do not parse tool arguments until the stream has ended. Partial JSON fragments are normally invalid JSON.

## Throttled Telegram Draft Updates

Telegram must not be called for every model chunk. The callback below publishes the accumulated response at most once every two seconds.

```python
import random
import time

from telegram.ext import ContextTypes


TELEGRAM_TEXT_LIMIT = 4096
DRAFT_UPDATE_INTERVAL = 2.0


def make_draft_callbacks(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
):
    draft_id = random.randint(1, 2_147_483_647)
    last_update = 0.0

    async def send_draft(text: str) -> None:
        try:
            await context.bot.send_message_draft(
                chat_id=chat_id,
                draft_id=draft_id,
                text=text[-TELEGRAM_TEXT_LIMIT:],
            )
        except Exception:
            # A preview failure must not fail the completed response.
            pass

    async def on_partial_text(accumulated_text: str) -> None:
        nonlocal last_update
        now = time.monotonic()
        if now - last_update < DRAFT_UPDATE_INTERVAL:
            return
        last_update = now
        await send_draft(accumulated_text)

    async def on_status(status: str) -> None:
        nonlocal last_update
        last_update = time.monotonic()
        await send_draft(status)

    return on_partial_text, on_status
```

Use `time.monotonic()` for elapsed-time checks. System clock changes can make `time.time()` unsuitable for throttling.

## Compatible Placeholder Fallback

If `send_message_draft` is unavailable, send one placeholder and edit it repeatedly:

```python
import time

from telegram import Message


async def stream_with_placeholder(update, context, run_model) -> str:
    placeholder: Message = await update.message.reply_text("Thinking...")
    last_update = 0.0
    last_rendered = ""

    async def on_partial_text(accumulated_text: str) -> None:
        nonlocal last_update, last_rendered
        now = time.monotonic()
        rendered = accumulated_text[-4096:] or "Thinking..."
        if now - last_update < 2.0 or rendered == last_rendered:
            return
        last_update = now
        last_rendered = rendered
        try:
            await context.bot.edit_message_text(
                chat_id=placeholder.chat_id,
                message_id=placeholder.message_id,
                text=rendered,
            )
        except Exception:
            pass

    answer = await run_model(on_partial_text)
    final_text = answer[:4096] or "(no reply)"
    if final_text != last_rendered:
        await placeholder.edit_text(final_text)
    return answer
```

Avoid editing with identical text because Telegram may reject it with a "message is not modified" error.

## Model and Tool Loop

When tools are enabled, one user turn can require multiple streamed model completions:

```python
import json


MAX_TOOL_ROUNDS = 8


async def run_chat_turn(
    client,
    session,
    model,
    messages,
    tools,
    on_partial_text,
    on_status,
):
    seen_calls: dict[str, int] = {}

    for _ in range(MAX_TOOL_ROUNDS):
        reply = await stream_completion(
            client,
            model=model,
            messages=messages,
            tools=tools,
            on_partial_text=on_partial_text,
        )
        messages.append(reply)
        calls = reply.get("tool_calls")
        if not calls:
            return reply.get("content") or "(no reply)"

        for call in calls:
            function = call["function"]
            arguments = json.loads(function.get("arguments") or "{}")
            signature = function["name"] + function.get("arguments", "")
            seen_calls[signature] = seen_calls.get(signature, 0) + 1

            await on_status(f"Checking data: {function['name']}...")
            if seen_calls[signature] > 2:
                result = "Stop repeating this tool call and answer with available data."
            else:
                result = await session.call_tool(
                    function["name"], arguments=arguments
                )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": str(result)[:50000],
                }
            )

    messages.append(
        {
            "role": "user",
            "content": "Stop using tools. Answer from the information above.",
        }
    )
    final_reply = await stream_completion(
        client,
        model=model,
        messages=messages,
        tools=tools,
        on_partial_text=on_partial_text,
        tool_choice="none",
    )
    messages.append(final_reply)
    return final_reply.get("content") or "(no reply)"
```

Production code should normalize the MCP result into text rather than relying on `str(result)`.

## Safe Telegram Handler Pattern

Use one lock per chat. Record the conversation rollback point before adding assistant or tool messages.

```python
import asyncio
from collections import defaultdict

from telegram.constants import ChatAction


chat_locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)


async def on_message(update, context):
    if not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    async with chat_locks[chat_id]:
        messages = get_history(chat_id)
        rollback_index = len(messages)
        messages.append({"role": "user", "content": update.message.text})
        on_partial_text, on_status = make_draft_callbacks(context, chat_id)

        await context.bot.send_chat_action(
            chat_id=chat_id,
            action=ChatAction.TYPING,
        )

        try:
            await on_status("Thinking...")
            answer = await run_model_turn(
                messages=messages,
                on_partial_text=on_partial_text,
                on_status=on_status,
            )
        except Exception as error:
            del messages[rollback_index:]
            error_text = user_safe_error(error)[:4096]
            await on_status(error_text)
            await update.message.reply_text(error_text)
            return

        await update.message.reply_text(answer[:4096])
```

The rollback index must be captured before appending the user's message if the whole failed turn should be removed. If it is captured afterward, the failed user message remains in history.

## Long Responses

Choose one explicit policy for responses longer than 4,096 characters:

- Split the final response into multiple Telegram messages, preferably on paragraph boundaries.
- Ask the model to produce concise responses below the limit.
- Show only the last 4,096 characters in the live preview, then split the complete final answer.

Do not preview the last 4,096 characters and then silently publish only the first 4,096 characters. That produces an inconsistent user experience.

## Error Handling

- Convert provider authentication, rate limit, timeout, and connectivity errors into safe user-facing messages.
- Never expose secrets or raw internal stack traces in Telegram.
- Roll back incomplete assistant/tool messages after a failed turn.
- Let preview updates fail independently from model generation.
- Add one application-level error handler for failures that escape the message handler.
- Use a fresh external-tool session per turn if a broken long-lived session can poison future requests.

## Verification Checklist

Ask Copilot to verify all of the following:

- A plain response visibly updates before generation completes.
- Updates are throttled and do not occur for every token.
- Only one draft or placeholder is used per turn.
- The permanent final message contains the full accumulated answer.
- A tool-calling response correctly assembles fragmented function arguments.
- Status text remains visible while a tool executes.
- Two rapid messages in the same chat are serialized.
- Different chats can still run concurrently.
- Draft/edit failures do not prevent the final answer.
- Provider and tool failures leave valid conversation history.
- Long responses follow the selected 4,096-character policy.
- Logs contain no credentials or sensitive tool output.

## Prompt to Give Copilot in Another Project

```text
Implement streaming responses for this asynchronous Python Telegram bot.

Follow TELEGRAM_STREAMING_KNOWLEDGE.md. Stream the OpenAI-compatible model with
stream=True, accumulate text and fragmented tool-call deltas, and update one
Telegram draft or placeholder message no more than once every two seconds. Use
Telegram drafts when the installed library supports send_message_draft;
otherwise send one placeholder and edit it. Keep live updates best-effort,
respect the 4,096-character limit, serialize turns per chat, preserve concurrency
between different chats, roll back incomplete history on failure, and finalize
one permanent answer. Reuse the project's existing client, handlers, tool loop,
error mapping, and coding conventions. Add focused tests for throttling, chunk
accumulation, fragmented tool calls, concurrent messages, and failed updates.
```

## Reference Implementation in This Repository

The original implementation is in `chatbot/telegram_bot.py`:

- `stream_completion`: streams and accumulates text and tool calls.
- `chat_turn`: runs the model/tool loop.
- `on_message`: creates draft callbacks and publishes the final answer.
- `open_mcp_session`: isolates tool transport failures per user turn.

The reusable guide intentionally adds a per-chat lock and calls out the long-response consistency policy, which should be included in new implementations.