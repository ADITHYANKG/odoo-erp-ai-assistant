"""Telegram-style HTML / Markdown -> Google Chat text.

Chat's text syntax is close to WhatsApp's (``*bold*``, ``_italic_``, backtick
code) but links are ``<url|label>``. Answers go through the existing Telegram
renderer (tables, bullets, headers already solved) and its small tag set
(b, i, code, pre, a) is translated here.
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping

from ont_agent.telegram_format import render_markdown
from ont_agent.whatsapp_format import wrap_lines

_PRE = re.compile(r"<pre>(?:<code[^>]*>)?(.*?)(?:</code>)?</pre>", re.S)
_ANCHOR = re.compile(r'<a href="([^"]*)">(.*?)</a>', re.S)
_BOLD = re.compile(r"<b>(.*?)</b>", re.S)
_ITALIC = re.compile(r"<i>(.*?)</i>", re.S)
_CODE = re.compile(r"<code>(.*?)</code>", re.S)
_ANY_TAG = re.compile(r"</?[a-zA-Z][^>]*>")


def html_to_googlechat(markup: str) -> str:
    """Translate the Telegram HTML subset to Google Chat formatting."""
    # Literal spans are stashed so nothing inside them is reinterpreted.
    stash: list[str] = []

    def keep(value: str) -> str:
        stash.append(value)
        return f"\x00{len(stash) - 1}\x00"

    text = _PRE.sub(
        lambda m: keep("```\n" + html.unescape(m.group(1)).strip("\n") + "\n```"), markup
    )
    text = _CODE.sub(lambda m: keep(f"`{html.unescape(m.group(1))}`"), text)

    def link(match: re.Match[str]) -> str:
        url = html.unescape(match.group(1))
        label = html.unescape(_ANY_TAG.sub("", match.group(2))).replace("|", "/").replace(">", "")
        return keep(url if label == url else f"<{url}|{label}>")

    text = _ANCHOR.sub(link, text)
    text = _BOLD.sub(lambda m: wrap_lines("*", m.group(1)), text)
    text = _ITALIC.sub(lambda m: wrap_lines("_", m.group(1)), text)
    text = html.unescape(_ANY_TAG.sub("", text))
    return re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], text)


def render_googlechat(text: str, links: Mapping[str, str] | None = None) -> str:
    """Model Markdown -> Google Chat text."""
    return html_to_googlechat(render_markdown(text, links))
