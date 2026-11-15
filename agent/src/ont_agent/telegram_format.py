"""Render the model's Markdown answers as Telegram HTML.

Telegram does not render Markdown: ``### headers`` and ``| tables |`` arrive
as raw syntax in the chat. Telegram DOES support a small HTML subset (b, i,
code, pre) — so the bot converts the common Markdown the model produces:

- ``### Heading``            -> bold line
- ``**bold**``               -> <b>bold</b>
- ```inline code```          -> <code>…</code>
- fenced ``` blocks          -> <pre>…</pre>
- ``- `` / ``* `` bullets    -> ``• `` bullets
- Markdown tables            -> aligned monospace <pre> blocks
- ``---`` rules              -> dropped

Everything is HTML-escaped first, so model output can never inject markup.

Record names can also be turned into links back into Odoo. Linking happens
AFTER escaping, inside the inline pipeline, so it composes with bold/italic
and still cannot be used to inject markup. Telegram does not render links
inside <pre>, and tables are monospace <pre> blocks — so a record that only
ever appears in a table gets a compact "Open in Odoo" footer instead of an
inline link. Without that fallback the feature would silently do nothing on
exactly the answers where record numbers cluster.
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping

from ont_agent.latex import strip_latex

_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_HEADER = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")
_BULLET = re.compile(r"^(\s*)[-*]\s+(.*)$")
_RULE = re.compile(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$")
#: ***both*** must be matched BEFORE ** and *, or the bold pass eats two
#: of the three stars and the italic pass then closes across the bold tag,
#: producing "<b><i>x</b></i>" - crossed tags that Telegram rejects
#: outright, which drops the whole reply to unformatted text.
_BOLD_ITALIC = re.compile(r"\*\*\*(.+?)\*\*\*")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_INLINE_CODE = re.compile(r"`([^`\n]+)`")

#: Records linked in the table footer. Beyond a handful it stops being
#: navigation and becomes a wall of blue text.
_FOOTER_LIMIT = 8


def _mono(text: str) -> str:
    """A monospace block with the language PINNED to "text" — Telegram
    Desktop otherwise runs language auto-detection on bare <pre> blocks and
    slaps a guessed label like "sql" on data tables (live incident)."""
    return '<pre><code class="language-text">' + html.escape(text) + "</code></pre>"


def _cells(row: str) -> list[str]:
    # Bold/code markers can't render inside a monospace <pre> block — strip
    # them so cells never show literal ** or ` characters.
    return [
        cell.strip().replace("**", "").replace("`", "")
        for cell in row.strip().strip("|").split("|")
    ]


def _render_table(rows: list[str]) -> str:
    """Aligned monospace table; the separator row is dropped."""
    parsed = [_cells(row) for row in rows if not _TABLE_SEPARATOR.match(row)]
    if not parsed:
        return ""
    columns = max(len(row) for row in parsed)
    padded = [row + [""] * (columns - len(row)) for row in parsed]
    widths = [max(len(row[i]) for row in padded) for i in range(columns)]
    lines = ["  ".join(row[i].ljust(widths[i]) for i in range(columns)).rstrip() for row in padded]
    return _mono("\n".join(lines))


class _Linker:
    """Turns known record names into Odoo links, at most once each.

    Longest name first, so a short name can never carve up a longer one that
    contains it ("Acme" inside "Acme Corporation"). Once each, because a
    link on every repetition of a customer's name is visual noise rather
    than navigation.
    """

    def __init__(self, links: Mapping[str, str] | None) -> None:
        # Match against the ESCAPED name: a customer called "Smith & Co"
        # reaches this point as "Smith &amp; Co".
        self._by_escaped = {
            html.escape(name, quote=False): url for name, url in (links or {}).items()
        }
        self.used: set[str] = set()
        self._pattern = None
        if self._by_escaped:
            names = sorted(self._by_escaped, key=len, reverse=True)
            self._pattern = re.compile("|".join(re.escape(name) for name in names))

    def apply(self, escaped: str) -> str:
        if self._pattern is None:
            return escaped

        def replace(match: re.Match[str]) -> str:
            name = match.group(0)
            if name in self.used:
                return name
            self.used.add(name)
            return f'<a href="{html.escape(self._by_escaped[name], quote=True)}">{name}</a>'

        return self._pattern.sub(replace, escaped)

    def unused_in(self, text: str) -> dict[str, str]:
        """Known records that appear in the text but were never linked —
        because they only ever showed up inside a table.

        A name contained inside a longer one that WAS linked is skipped:
        "Acme" living inside an already-linked "Acme Corporation" is not a
        second record the reader is missing, it is the same one.
        """
        escaped_text = html.escape(text, quote=False)
        return {
            name: url
            for name, url in self._by_escaped.items()
            if name not in self.used
            and name in escaped_text
            and not any(name in longer for longer in self.used)
        }


def _render_inline(line: str, linker: _Linker | None = None) -> str:
    escaped = html.escape(line, quote=False)
    escaped = _INLINE_CODE.sub(r"<code>\1</code>", escaped)
    escaped = _BOLD_ITALIC.sub(r"<b><i>\1</i></b>", escaped)
    escaped = _BOLD.sub(r"<b>\1</b>", escaped)
    # Single-asterisk italics AFTER bold, so ** pairs are already consumed.
    escaped = _ITALIC.sub(r"<i>\1</i>", escaped)
    return linker.apply(escaped) if linker is not None else escaped


def plain_text(text: str) -> str:
    """Readable prose with the Markdown markers removed.

    The last resort when Telegram refuses the rendered HTML. Sending the
    ORIGINAL text there means the reader gets every ``**`` and ``*`` in
    the answer - which is how a malformed-tag bug surfaced as "a lot of
    annoying asterisks" rather than as an error anyone could see (live,
    2026-09-02). A formatting failure should cost formatting, not
    legibility.
    """
    text = strip_latex(text)
    source = text.split("\n")
    lines: list[str] = []
    i = 0
    while i < len(source):
        line = source[i]
        if _TABLE_ROW.match(line):
            rows: list[str] = []
            while i < len(source) and _TABLE_ROW.match(source[i]):
                if not _TABLE_SEPARATOR.match(source[i]):
                    rows.append(source[i])
                i += 1
            parsed = [_cells(row) for row in rows]
            if parsed:
                lines.append(" — ".join(parsed[0]))
                lines.extend("• " + " — ".join(row) for row in parsed[1:])
            continue
        if _RULE.match(line):
            i += 1
            continue
        header = _HEADER.match(line)
        if header:
            line = header.group(1)
        bullet = _BULLET.match(line)
        if bullet:
            line = f"{bullet.group(1)}• {bullet.group(2)}"
        for pattern in (_BOLD_ITALIC, _BOLD, _ITALIC, _INLINE_CODE):
            line = pattern.sub(r"\1", line)
        lines.append(line)
        i += 1
    return "\n".join(lines)


def render_markdown(text: str, links: Mapping[str, str] | None = None) -> str:
    """Telegram-HTML rendering of a Markdown answer (safe to parse as HTML).

    :param links: record name -> Odoo URL. Names are linked in prose the
        first time each appears; any that occur only inside a table are
        listed in a short footer, since Telegram will not render a link
        inside the monospace block a table becomes.
    """
    linker = _Linker(links)
    output: list[str] = []
    # Before anything else: the model sometimes emits LaTeX, which Telegram
    # shows to the reader as markup. Done here so EVERY user-facing message
    # is covered — answers, briefs, watcher alerts, digests.
    lines = strip_latex(text).split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]

        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):  # fenced code block (both styles)
            fence_chars = stripped[:3]
            block: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith(fence_chars):
                block.append(lines[i])
                i += 1
            i += 1  # closing fence
            output.append(_mono("\n".join(block)))
            continue

        if _TABLE_ROW.match(line):
            rows = []
            while i < len(lines) and _TABLE_ROW.match(lines[i]):
                rows.append(lines[i])
                i += 1
            output.append(_render_table(rows))
            continue

        if _RULE.match(line):
            i += 1
            continue

        header = _HEADER.match(line)
        if header:
            # A heading is already bold; "### **Total**" would otherwise
            # nest <b> inside <b>, which Telegram also refuses.
            title = header.group(1).strip()
            if title.startswith("**") and title.endswith("**") and len(title) > 4:
                title = title[2:-2]
            output.append("<b>" + _render_inline(title, linker) + "</b>")
            i += 1
            continue

        bullet = _BULLET.match(line)
        if bullet:
            output.append(bullet.group(1) + "• " + _render_inline(bullet.group(2), linker))
            i += 1
            continue

        output.append(_render_inline(line, linker))
        i += 1

    leftover = linker.unused_in(text)
    if leftover:
        # Table-only records: Telegram renders no link inside <pre>, so give
        # them somewhere tappable rather than dropping them silently.
        footer = ", ".join(
            f'<a href="{html.escape(url, quote=True)}">{name}</a>'
            for name, url in list(leftover.items())[:_FOOTER_LIMIT]
        )
        output.append(f"\n🔗 Open in Odoo: {footer}")

    rendered = "\n".join(output)
    # Collapse the blank lines that Markdown needs but chat does not.
    return re.sub(r"\n{3,}", "\n\n", rendered).strip()
