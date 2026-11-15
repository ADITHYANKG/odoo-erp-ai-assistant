"""Send the weekly "what I did" digest NOW, for a demo.

The real one goes out on Monday morning, once per ISO week. This builds the
IDENTICAL text from the same action log and usage counters and sends it to
each bound chat.

It deliberately does NOT stamp the week as sent, so it is safe to re-run and
it cannot swallow the genuine Monday digest.

Usage (from the repository root):

    uv run python scripts/15_send_digest_now.py
"""

from __future__ import annotations

import asyncio
import sys
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from telegram import Bot  # noqa: E402

from ont_agent.bot import BotReply, prepare_reply_text  # noqa: E402
from ont_agent.budget import subject_for  # noqa: E402
from ont_agent.config import load_settings  # noqa: E402
from ont_agent.crypto import TokenCipher  # noqa: E402
from ont_agent.proactive import compose_digest  # noqa: E402
from ont_agent.store import BindingStore  # noqa: E402


# psycopg's async mode refuses Windows' default ProactorEventLoop.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def main() -> None:
    settings = load_settings()
    bot = Bot(settings.telegram_bot_token)
    now = datetime.now(UTC)

    async with BindingStore(settings.agent_db_url, TokenCipher.from_key(settings.token_key)) as store:
        bindings = await store.list_bindings()
        if not bindings:
            print("No bound chats - run /start in Telegram first.")
            return

        for binding in bindings:
            actions = await store.actions_for(binding.odoo_login, now - timedelta(days=7))
            usage = await store.get_usage(
                subject_for(binding, binding.chat_id), now.date().isoformat()
            )
            text = compose_digest(
                actions, usage.message_count, usage.spend_usd, binding.odoo_name
            )
            if not text:
                print(f"{binding.odoo_login}: nothing to report - no actions, no questions")
                continue
            body, parse_mode = prepare_reply_text(BotReply(text=text))
            await bot.send_message(chat_id=binding.chat_id, text=body, parse_mode=parse_mode)
            print(f"{binding.odoo_login}: sent ({len(actions)} action(s))")


asyncio.run(main())
