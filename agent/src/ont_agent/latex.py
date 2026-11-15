"""Turn stray LaTeX in a model answer into text a reader can actually read.

Gemini reaches for LaTeX whenever an answer compares numbers, unbidden and
regardless of the prompt. A policy answer about travel caps arrived in
Telegram reading:

    Premium Economy for segments $\\ge$ 6 hours
    $> \\$200$ requires VP & Finance approval

Telegram renders none of that: the reader sees the markup. It is the same
class of leak as the raw ``<b>`` tags in the very first morning brief —
something intended as formatting reaching the user as characters.

Symbols are CONVERTED rather than deleted. The sentence means "at least six
hours"; dropping the operator would quietly change what the policy says,
which on a reimbursement rule is worse than showing the markup.

This is deliberately not a maths renderer and does not try to be.
"""

from __future__ import annotations

import re

#: LaTeX commands worth translating: the ones that carry meaning in business
#: prose. Anything else keeps its word and loses its backslash.
_SYMBOLS: dict[str, str] = {
    "ge": "≥",
    "geq": "≥",
    "le": "≤",
    "leq": "≤",
    "ne": "≠",
    "neq": "≠",
    "approx": "≈",
    "pm": "±",
    "times": "×",  # noqa: RUF001 - a multiplication sign is the point
    "div": "÷",
    "cdot": "·",
    "rightarrow": "→",
    "to": "→",
    "leftarrow": "←",
    "infty": "∞",
    "ldots": "…",
    "dots": "…",
}

#: Escaped punctuation: the backslash goes, the character stays.
_ESCAPED = re.compile(r"\\([%$&_#{}])")

#: A named command, e.g. \ge or \textbf.
_COMMAND = re.compile(r"\\([A-Za-z]+)")

#: Inline and display maths: $$..$$, $..$, \(..\), \[..\].
#:
#: The inline arm must allow an ESCAPED dollar inside, or the live answer
#: "$> \$200$ requires VP approval" closes the span at the escaped dollar
#: and leaves a stray backslash in the reader's text.
_MATH_SPAN = re.compile(
    r"\$\$(?P<display>.+?)\$\$"
    r"|\$(?P<inline>(?:\\\$|[^$\n])+?)\$"
    r"|\\\((?P<paren>.+?)\\\)"
    r"|\\\[(?P<bracket>.+?)\\\]",
    re.S,
)


def _translate(fragment: str) -> str:
    """Replace commands inside one fragment; unknown ones keep their word."""
    fragment = _ESCAPED.sub(r"\1", fragment)
    return _COMMAND.sub(lambda m: _SYMBOLS.get(m.group(1), m.group(1)), fragment)


def strip_latex(text: str) -> str:
    """Remove maths delimiters and translate the symbols inside them.

    Cheap to call on every reply: text with neither a dollar sign nor a
    backslash is returned untouched, which is almost all of them.
    """
    if "$" not in text and "\\" not in text:
        return text

    def unwrap(match: re.Match[str]) -> str:
        inner = next(value for value in match.groupdict().values() if value is not None)
        return _translate(inner).strip()

    return _translate(_MATH_SPAN.sub(unwrap, text))
