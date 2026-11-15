"""Telegram-style HTML / Markdown -> Slack mrkdwn.

Slack's syntax is ``*bold*``, ``_italic_``, backtick code and ``<url|label>``
links, and it reserves ``&``, ``<`` and ``>``: unescaped, a model answer
containing ``<!channel>`` would notify the whole workspace. Answers go through
the existing Telegram renderer and its small tag set (b, i, code, pre, a) is
translated here, escaping everything else.
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
#: Slack's own markup in text a person typed: <url|label>, <@U123>, <#C123|name>, <!here>.
_MARKUP = re.compile(r"<([^<>|]+)(?:\|([^<>]*))?>")


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def html_to_slack(markup: str) -> str:
    """Translate the Telegram HTML subset to Slack mrkdwn."""
    # Literal spans are stashed so nothing inside them is reinterpreted.
    stash: list[str] = []

    def keep(value: str) -> str:
        stash.append(value)
        return f"\x00{len(stash) - 1}\x00"

    text = _PRE.sub(
        lambda m: keep("```\n" + escape(html.unescape(m.group(1))).strip("\n") + "\n```"), markup
    )
    text = _CODE.sub(lambda m: keep(f"`{escape(html.unescape(m.group(1)))}`"), text)

    def link(match: re.Match[str]) -> str:
        url = html.unescape(match.group(1))
        label = html.unescape(_ANY_TAG.sub("", match.group(2)))
        safe_url = escape(url.replace("|", "%7C"))
        if label == url:
            return keep(f"<{safe_url}>")
        return keep(f"<{safe_url}|{escape(label.replace('|', '/'))}>")

    text = _ANCHOR.sub(link, text)
    text = _BOLD.sub(lambda m: wrap_lines("*", m.group(1)), text)
    text = _ITALIC.sub(lambda m: wrap_lines("_", m.group(1)), text)
    text = escape(html.unescape(_ANY_TAG.sub("", text)))
    return re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], text)


def render_slack(text: str, links: Mapping[str, str] | None = None) -> str:
    """Model Markdown -> Slack mrkdwn."""
    return html_to_slack(render_markdown(text, links))


def plain_from_slack(text: str) -> str:
    """What a person typed, with Slack's markup and entities undone."""

    def replace(match: re.Match[str]) -> str:
        target, label = match.group(1), match.group(2)
        if target.startswith(("@", "#")):
            return f"{target[0]}{label or target[1:]}"
        if target.startswith("!"):
            return label or f"@{target[1:].split('^')[0]}"
        return label or target

    text = _MARKUP.sub(replace, text)
    return text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&").strip()
