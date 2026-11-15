"""Guardrail against system-prompt extraction.

The existing injection rules in ``agent.SYSTEM_PROMPT`` point the other way:
they stop *content* (tool results, uploaded files) from issuing orders, and
one of them says outright that the person chatting is the legitimate source
of instructions. Nothing stopped that person asking for the instructions
themselves.

**What this does and does not claim.** It raises the cost of casual
extraction and makes attempts visible in the log. It does not make the
prompt secret, and the security model does not ask it to — every real
control is server-side (OAuth scope, the muk_mcp_access allowlist, Odoo
record rules, ``tools.filter_for_bot``, and the human approval gate on every
write). Someone who reads the whole prompt still cannot bypass one of them.

Three layers, none of which costs a model call:

1. A confidentiality rule in the system prompt itself — the only layer that
   handles paraphrase.
2. ``looks_like_extraction`` on the way in, refusing before anything is
   built or billed.
3. ``leaks_prompt`` on the way out. Layers 1 and 2 are bypassable by
   rephrasing; this one is not, because it inspects the text rather than the
   request. It catches verbatim reproduction, which is what "the
   instructions become a public artifact" actually means.
"""

from __future__ import annotations

import re
from functools import lru_cache

from ont_agent.agent import SYSTEM_PROMPT

#: Sent for both a detected attempt and a detected leak. Deliberately
#: neither an accusation nor a lecture: the commonest reason a real user
#: sees this is a false positive, so it says how to get help instead.
#:
#: It must itself contain nothing from the prompt — a guardrail that leaks
#: in its own refusal would be absurd, and a test pins it.
REFUSAL = (
    "I can't share my own setup or instructions. If you were asking about "
    "something in your Odoo data, put it another way and I'll help."
)

#: Second person is load-bearing throughout. In an ERP, "the system prompt
#: for onboarding", "prompt payment discount terms" and "the instructions in
#: the HR policy" are ordinary questions; "YOUR system prompt" is not.
_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    for pattern in (
        # Asking for the configuration directly.
        r"\byour\s+(own\s+)?(system\s+|initial\s+|original\s+|internal\s+)?"
        r"(prompt|instructions|rules|guidelines|configuration|directives|system\s+message)\b",
        # Override framings.
        r"\b(ignore|disregard|forget|override)\s+(all\s+|any\s+|the\s+)?"
        r"(previous|prior|above|preceding|earlier|your)\b",
        # Asking for the context to be reproduced. "show" is left out on
        # purpose — "show me the ones above" is a plausible business phrase,
        # and an actual request is caught by the self-reference rule anyway.
        r"\b(repeat|print|output|reveal|reproduce|display|dump|recite)\b"
        r"[^.?!]{0,40}\b(above|preceding|verbatim|word\s+for\s+word)\b",
        # Role-play and jailbreak framings.
        r"\bdeveloper\s+mode\b",
        r"\bdan\s+mode\b",
        r"\b(pretend|act\s+as\s+if|imagine)\b[^.?!]{0,40}"
        r"\b(no\s+(rules|restrictions|limits|guidelines)|unrestricted|jailbroken)\b",
        r"\byou\s+are\s+now\s+(a|an|in)\b",
        # Probing for the prompt's delimiters.
        r"<\s*/?\s*system\s*>",
        r"\[\s*system\s*\]",
        r"^#{1,6}\s*system\b",
        # Provenance questions.
        r"\bwhat\s+(were|was)\s+you\s+(told|instructed|programmed|configured)\b",
        r"\bwho\s+(told|instructed|programmed)\s+you\b",
        r"\b(first|initial)\s+message\s+(in\s+)?(this\s+)?(conversation|context|thread)\b",
        # Asking for the same thing through a transformation.
        r"\b(base64|rot13|encode|encrypt|translate|spell\s+out)\b[^.?!]{0,40}"
        r"\byour\s+(prompt|instructions|rules|setup)\b",
    )
)


def looks_like_extraction(text: str) -> bool:
    """Whether this message is asking for the agent's own configuration.

    Checked before the agent is built, so a refused probe costs nothing —
    and the caller still records the message, so repeated probing burns the
    sender's daily allowance rather than being free.
    """
    return any(pattern.search(text) for pattern in _PATTERNS)


def _words(text: str) -> list[str]:
    """Lowercased words with edge punctuation stripped.

    Normalising both sides is what makes the comparison survive a leak that
    has been re-wrapped, re-cased or re-punctuated on the way out — which
    reformatting into a chat reply routinely does.
    """
    return [
        word
        for word in (part.strip(".,;:!?\"'`*_()[]{}-") for part in text.lower().split())
        if word
    ]


@lru_cache(maxsize=8)
def _windows(ngram: int) -> frozenset[str]:
    """Every run of ``ngram`` consecutive words in the live SYSTEM_PROMPT.

    Derived from the prompt at call time rather than hand-listed, so editing
    the prompt can never leave the fingerprints stale — which is exactly how
    a hand-written list of "secret phrases" rots.
    """
    words = _words(SYSTEM_PROMPT)
    return frozenset(
        " ".join(words[index : index + ngram]) for index in range(len(words) - ngram + 1)
    )


def leaks_prompt(reply: str, ngram: int) -> str | None:
    """The first run of prompt text found in ``reply``, or None.

    ``ngram`` is the whole trade-off. Too low and an honest answer trips it:
    the prompt spells out what "overdue" means, and a reply explaining that
    filter could legitimately echo the wording. Too high and partial
    reproduction slips through. A real dump matches dozens of windows; a
    coincidental echo matches one run at most.
    """
    if ngram <= 0:
        return None
    words = _words(reply)
    if len(words) < ngram:
        return None
    known = _windows(ngram)
    for index in range(len(words) - ngram + 1):
        window = " ".join(words[index : index + ngram])
        if window in known:
            return window
    return None
