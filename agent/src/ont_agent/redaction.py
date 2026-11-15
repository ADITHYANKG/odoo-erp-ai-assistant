"""Redact credentials out of tool results and replies.

Named ``redaction`` rather than ``secrets`` on purpose: ``auth.py`` does
``import secrets`` for ``token_urlsafe``, and a module of ours by that name
inside the package is a shadowing trap waiting for the first relative
import.

**The leak that matters is not the reply.** A key pasted into a
``res.partner`` note does not merely get shown once — the tool result is
stored as a ToolMessage in the conversation checkpoint (up to 30 days), and
traced to Langfuse. So the primary hook is tool results, BEFORE the model
sees them: redact there and the secret reaches neither the thread, the
trace, nor the answer. The reply scan is a second net, and earns its keep
because an uploaded PDF reaches the model without passing a tool.

**Structural patterns only — no entropy heuristics, and that is the
decision this stands or falls on.** Odoo is full of long opaque strings:
sha1 attachment checksums, UUIDs, SKUs, INV/2026/00015, IBANs, base64 image
references. Anything scoring "this looks random" redacts those and makes
answers unreadable. Every rule below is anchored on a vendor prefix, which
is essentially unambiguous.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class _Rule:
    kind: str
    pattern: re.Pattern[str]
    #: Which group holds the value. 0 redacts the whole match; a positive
    #: group keeps the surrounding text, so "password=..." stays readable as
    #: "password=[redacted: ...]" rather than vanishing entirely.
    group: int = 0


#: Order matters: the specific vendors come before the generic ``sk-`` rule,
#: so the reported kind is the most precise one and the generic rule never
#: sees text another has already replaced.
_RULES: tuple[_Rule, ...] = (
    _Rule(
        "private key",
        re.compile(
            r"-----BEGIN [A-Z ]{0,24}PRIVATE KEY-----[\s\S]*?-----END [A-Z ]{0,24}PRIVATE KEY-----"
        ),
    ),
    _Rule("private key", re.compile(r"-----BEGIN [A-Z ]{0,24}PRIVATE KEY-----")),
    # Catches this bot's OWN OAuth access and refresh tokens, which are the
    # most damaging thing that could ever appear in a reply.
    _Rule(
        "JSON Web Token",
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    ),
    _Rule("OpenRouter key", re.compile(r"\bsk-or-v1-[0-9a-fA-F]{16,}")),
    _Rule("Anthropic key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}")),
    _Rule("Stripe key", re.compile(r"\b[rs]k_(?:live|test)_[0-9A-Za-z]{16,}")),
    _Rule("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    _Rule("AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    _Rule("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}")),
    _Rule("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    _Rule("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}")),
    # The only rule not anchored on a vendor prefix, and so the only one with
    # real false-positive potential. The required ':' or '=' is what makes it
    # safe: "Password reset requested" does not match, "password=hunter2xyz"
    # does. First thing to drop if a client's data ever trips it.
    _Rule(
        "credential assignment",
        re.compile(
            r"(?i)\b(?:password|passwd|api[_-]?key|apikey|secret|"
            r"access[_-]?token|auth[_-]?token)\s*[:=]\s*[\"']?([^\s\"',;]{8,})"
        ),
        group=1,
    ),
)


def redact_secrets(text: str) -> tuple[str, list[str]]:
    """Replace credentials with a marker naming their KIND, never the value.

    :return: the cleaned text, and the kinds found in the order found —
        which is what a caller may safely log. Logging the value would move
        the secret into ``bot.log`` rather than remove it.

    Surgical rather than wholesale, unlike the system-prompt guard: a note
    containing one stray key still has a legitimate answer in it, and
    destroying that answer would be the worse outcome.
    """
    found: list[str] = []
    for rule in _RULES:

        def replace(match: re.Match[str], rule: _Rule = rule) -> str:
            if rule.kind not in found:
                found.append(rule.kind)
            marker = f"[redacted: {rule.kind}]"
            if rule.group == 0:
                return marker
            whole = match.group(0)
            offset = match.start(0)
            prefix = whole[: match.start(rule.group) - offset]
            suffix = whole[match.end(rule.group) - offset :]
            return f"{prefix}{marker}{suffix}"

        text = rule.pattern.sub(replace, text)
    return text, found
