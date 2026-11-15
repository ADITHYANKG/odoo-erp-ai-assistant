"""Telegram-style HTML / Markdown -> WhatsApp text.

WhatsApp has no HTML: bold is ``*x*``, italic ``_x_``, monospace ```` ```x``` ````.
Rather than a second Markdown renderer, answers go through the existing
Telegram renderer (tables, bullets, headers already solved) and its small
tag set (b, i, code, pre, a) is translated here. The same translation
handles the prebuilt HTML of approval prompts.
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping

from ont_agent.telegram_format import render_markdown

TEXT_LIMIT = 4096

_PRE = re.compile(r"<pre>(?:<code[^>]*>)?(.*?)(?:</code>)?</pre>", re.S)
_ANCHOR = re.compile(r'<a href="([^"]*)">(.*?)</a>', re.S)
_BOLD = re.compile(r"<b>(.*?)</b>", re.S)
_ITALIC = re.compile(r"<i>(.*?)</i>", re.S)
_CODE = re.compile(r"<code>(.*?)</code>", re.S)
_ANY_TAG = re.compile(r"</?[a-zA-Z][^>]*>")


def wrap_lines(marker: str, inner: str) -> str:
    # WhatsApp refuses markers hugging whitespace and does not span lines, so
    # each non-empty line is wrapped on its own.
    lines = []
    for line in inner.split("\n"):
        stripped = line.strip()
        lines.append(f"{marker}{stripped}{marker}" if stripped else line)
    return "\n".join(lines)


def html_to_whatsapp(markup: str) -> str:
    """Translate the Telegram HTML subset to WhatsApp formatting."""
    # Code blocks first and stashed: their content is literal, so nothing
    # inside may be reinterpreted as bold or a link.
    blocks: list[str] = []

    def stash(match: re.Match[str]) -> str:
        blocks.append("```\n" + html.unescape(match.group(1)).strip("\n") + "\n```")
        return f"\x00{len(blocks) - 1}\x00"

    text = _PRE.sub(stash, markup)
    text = _CODE.sub(lambda m: _stash_inline(m.group(1), blocks), text)
    text = _ANCHOR.sub(
        lambda m: m.group(2) if m.group(1) == m.group(2) else f"{m.group(2)} ({m.group(1)})",
        text,
    )
    text = _BOLD.sub(lambda m: wrap_lines("*", m.group(1)), text)
    text = _ITALIC.sub(lambda m: wrap_lines("_", m.group(1)), text)
    text = _ANY_TAG.sub("", text)
    text = html.unescape(text)
    return re.sub(r"\x00(\d+)\x00", lambda m: blocks[int(m.group(1))], text)


def _stash_inline(code: str, blocks: list[str]) -> str:
    # Stashed so a `*` inside inline code cannot pair with a bold marker.
    blocks.append(f"`{html.unescape(code)}`")
    return f"\x00{len(blocks) - 1}\x00"


def render_whatsapp(text: str, links: Mapping[str, str] | None = None) -> str:
    """Model Markdown -> WhatsApp text."""
    return html_to_whatsapp(render_markdown(text, links))


def italic(text: str) -> str:
    return wrap_lines("_", text)


#: WhatsApp interactive messages (buttons, lists) hold at most this much body text.
BODY_LIMIT = 1024


def split_for_body(text: str, limit: int = BODY_LIMIT) -> tuple[str, str] | None:
    """Split ``text`` into (head, tail) so ``tail`` fits an interactive message body.

    Lets buttons share one bubble with the end of the answer instead of arriving
    as a separate message. Cuts only at a line break, and never inside a code
    block. ``None`` when no clean cut exists.
    """
    if len(text) <= limit:
        return "", text
    position = len(text) - limit
    # The earliest clean cut keeps the most of the answer next to the buttons.
    while (cut := text.find("\n", position)) != -1:
        head = text[:cut].rstrip()
        tail = text[cut + 1 :].lstrip()
        if tail and head.count("```") % 2 == 0 and tail.count("```") % 2 == 0:
            return head, tail
        position = cut + 1
    return None


def clamp(body: str, footer: str | None = None, limit: int = TEXT_LIMIT) -> str:
    """Fit ``body`` plus an optional footer under ``limit``, cutting the body."""
    suffix = f"\n\n{footer}" if footer else ""
    if len(body) + len(suffix) <= limit:
        return body + suffix
    keep = max(0, limit - len(suffix) - 1)
    return body[:keep].rstrip() + "…" + suffix
