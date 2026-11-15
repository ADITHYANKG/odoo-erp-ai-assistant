"""Force every watcher to be evaluated NOW, for a demo.

Watchers normally run on an 8-hour cycle, which is right for cost and wrong
for standing in front of an audience. This runs the same code path the
scheduler uses - the real evaluation, the real alert text, the real
fingerprint dedupe - and sends the result straight to the user's chat.

It is honest, not a mock: if the condition is false you will see nothing
fire, which is the correct behaviour and worth saying out loud in a demo.

Usage (from the repository root, with the bot ALREADY RUNNING or stopped -
either is fine, it talks to Telegram directly):

    uv run python scripts/14_fire_watchers_now.py
"""

from __future__ import annotations

import asyncio
import sys
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from telegram import Bot  # noqa: E402

from ont_agent.config import load_settings  # noqa: E402
from ont_agent.crypto import TokenCipher  # noqa: E402
from ont_agent.proactive import check_watcher, refresh_binding  # noqa: E402
from ont_agent.store import BindingStore  # noqa: E402
from ont_agent.bot import BotReply, prepare_reply_text  # noqa: E402


# psycopg's async mode refuses Windows' default ProactorEventLoop.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def main() -> None:
    settings = load_settings()
    bot = Bot(settings.telegram_bot_token)

    async def send(chat_id: int, text: str) -> None:
        body, parse_mode = prepare_reply_text(BotReply(text=text))
        await bot.send_message(chat_id=chat_id, text=body, parse_mode=parse_mode)

    # The token columns are encrypted, so the cipher is not optional here:
    # without it get_binding refuses the row and every chat looks unbound.
    cipher = TokenCipher.from_key(settings.token_key)
    async with BindingStore(settings.agent_db_url, cipher) as store:
        bindings = await store.list_bindings()
        if not bindings:
            print("No bound chats — run /start in Telegram first.")
            return

        for binding in bindings:
            watchers = await store.list_watchers(binding.chat_id)
            print(f"\n{binding.odoo_login}: {len(watchers)} watcher(s)")
            if not watchers:
                print("  (none — add one with: /watch any invoice more than 60 days overdue)")
                continue

            live = await refresh_binding(store, settings, binding.chat_id)
            if live is None:
                print("  binding is dead — that user needs /start")
                continue

            for watcher in watchers:
                print(f"  checking: {watcher.condition!r}")
                fired = await check_watcher(store, settings, live, watcher, send)
                print("    -> ALERT SENT" if fired else "    -> quiet (condition false, or unchanged since last check)")


asyncio.run(main())
