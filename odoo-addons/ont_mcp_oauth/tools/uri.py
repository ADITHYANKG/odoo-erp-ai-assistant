"""Redirect-URI validation for OAuth clients.

Enforces the OAuth 2.1 / MCP client rules: HTTPS for remote clients,
``http://localhost`` (and 127.0.0.1 / [::1]) for native loopback, and
custom-scheme native URIs. Fragments are always rejected. Matching against a
client's registered set is EXACT-string only — never prefix or wildcard — to
make open-redirect and ``good.com.evil.com`` tricks impossible.
"""
from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

_LOOPBACK_HOSTS = {'localhost', '127.0.0.1', '::1'}
_SCHEME_RE = re.compile(r'^[a-z][a-z0-9+.\-]*$')
_MAX_URI_LEN = 2000


def is_valid_redirect_uri(uri: object) -> bool:
    """True iff ``uri`` is an acceptable OAuth redirect target."""
    if not isinstance(uri, str) or not uri or len(uri) > _MAX_URI_LEN:
        return False
    # No control chars / CR / LF anywhere (header-injection defense).
    if any(ord(c) < 0x20 or ord(c) == 0x7f for c in uri):
        return False
    try:
        parts = urlsplit(uri)
    except ValueError:
        return False
    # Fragments are forbidden by RFC 6749 for redirect URIs.
    if parts.fragment or uri.rstrip().endswith('#'):
        return False
    scheme = parts.scheme.lower()
    if not scheme or not _SCHEME_RE.match(scheme):
        return False

    if scheme == 'https':
        return bool(parts.hostname)

    if scheme == 'http':
        # Loopback only, per OAuth 2.1 (RFC 8252) for native apps.
        host = (parts.hostname or '').lower()
        if host in _LOOPBACK_HOSTS:
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    # Custom-scheme native URI (e.g. "com.example.app:/cb"). Must not be a
    # dangerous pseudo-scheme; require a private-use style scheme with a body.
    if scheme in ('javascript', 'data', 'vbscript', 'file'):
        return False
    return bool(parts.netloc or parts.path)


_DEFAULT_PORTS = {'https': 443, 'http': 80}


def canonical_resource(uri: object) -> str | None:
    """Canonicalize an RFC 8707 resource indicator for comparison.

    Lowercases scheme and host, strips a default port and any trailing slash,
    so ``https://Host:443/mcp/`` compares equal to ``https://host/mcp``.
    Returns None for values that do not parse as an absolute URI.
    """
    if not isinstance(uri, str) or not uri:
        return None
    try:
        parts = urlsplit(uri)
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if not scheme or not parts.hostname:
        return None
    host = parts.hostname.lower()
    try:
        port = parts.port
    except ValueError:
        return None
    netloc = host if port in (None, _DEFAULT_PORTS.get(scheme)) else f'{host}:{port}'
    return f"{scheme}://{netloc}{parts.path.rstrip('/')}"


def matches_registered(uri: object, registered: list[str]) -> bool:
    """Exact-match ``uri`` against the client's registered redirect URIs."""
    return isinstance(uri, str) and uri in registered


def parse_redirect_uris(text: object) -> list[str]:
    """Split a newline-separated redirect_uris textarea into a clean list."""
    if not isinstance(text, str):
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]
