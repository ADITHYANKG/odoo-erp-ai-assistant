"""PKCE (RFC 7636) helpers — S256 only, ``plain`` is never accepted."""
from __future__ import annotations

import base64
import hashlib
import hmac
import re

# RFC 7636: code_verifier = 43*128 unreserved chars [A-Za-z0-9-._~]
_VERIFIER_RE = re.compile(r'^[A-Za-z0-9\-._~]{43,128}$')
# A base64url-encoded SHA-256 is always 43 chars, unpadded.
_CHALLENGE_RE = re.compile(r'^[A-Za-z0-9\-_]{43}$')


def is_valid_verifier(verifier: object) -> bool:
    """True iff ``verifier`` is a syntactically valid PKCE code_verifier."""
    return isinstance(verifier, str) and bool(_VERIFIER_RE.match(verifier))


def is_valid_challenge(challenge: object) -> bool:
    """True iff ``challenge`` looks like a base64url S256 code_challenge."""
    return isinstance(challenge, str) and bool(_CHALLENGE_RE.match(challenge))


def compute_challenge(verifier: str) -> str:
    """Return the S256 code_challenge for a verifier (base64url, unpadded)."""
    digest = hashlib.sha256(verifier.encode('ascii')).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b'=').decode('ascii')


def verify(verifier: object, challenge: object) -> bool:
    """Constant-time check that ``verifier`` matches the stored ``challenge``.

    Returns False (never raises) for any malformed input.
    """
    if not is_valid_verifier(verifier) or not is_valid_challenge(challenge):
        return False
    return hmac.compare_digest(compute_challenge(verifier), challenge)  # type: ignore[arg-type]
