"""Minimal, hardened RS256 JWT sign/verify built directly on ``cryptography``.

Deliberately supports exactly one algorithm (RS256). Any token whose header
declares a different ``alg`` (including ``none`` or an HMAC family) is rejected
before signature verification even runs, which structurally eliminates the
alg-confusion and alg:none downgrade attacks. No external JWT library is used.
"""
from __future__ import annotations

import base64
import hmac
import json
import re
import time
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.rsa import (
    RSAPrivateKey,
    RSAPublicKey,
)


class JWTError(Exception):
    """Raised when a token cannot be verified for any reason."""


# The exact base64url alphabet (RFC 4648 §5), no padding — JWS compact
# serialization forbids '=' padding in its segments. Matched with fullmatch,
# not ^...$: '$' also matches just before a trailing newline, which would let
# "abcd\n" through as an alias of "abcd".
_B64URL_RE = re.compile(r'[A-Za-z0-9_-]*')


def b64url_encode(data: bytes) -> str:
    """URL-safe base64 without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('ascii')


def b64url_decode(value: str) -> bytes:
    """Decode URL-safe base64 that may be missing its ``=`` padding.

    Characters outside the base64url alphabet are rejected rather than silently
    discarded (Python's decoder is lenient by default). Without this, one token
    has many distinct string spellings that all decode to the same payload,
    which would let an attacker mint unlimited aliases of a single token to
    evade anything keyed on the raw token string (caches, rate-limit buckets,
    log correlation).
    """
    if not isinstance(value, str):
        raise JWTError('segment is not a string')
    if not _B64URL_RE.fullmatch(value):
        raise JWTError('segment is not valid base64url')
    pad = -len(value) % 4
    try:
        return base64.urlsafe_b64decode(value + ('=' * pad))
    except (ValueError, TypeError) as exc:
        raise JWTError('invalid base64url segment') from exc


# ----------------------------------------------------------------------------
# Key generation / serialization
# ----------------------------------------------------------------------------

def generate_rsa_keypair() -> tuple[str, str]:
    """Generate a 2048-bit RSA keypair, returned as (private_pem, public_pem)."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode('ascii')
    public_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode('ascii')
    return private_pem, public_pem


def load_private_key(private_pem: str) -> RSAPrivateKey:
    key = serialization.load_pem_private_key(private_pem.encode('ascii'), password=None)
    if not isinstance(key, RSAPrivateKey):
        raise JWTError('not an RSA private key')
    return key


def load_public_key(public_pem: str) -> RSAPublicKey:
    key = serialization.load_pem_public_key(public_pem.encode('ascii'))
    if not isinstance(key, RSAPublicKey):
        raise JWTError('not an RSA public key')
    return key


def public_jwk(public_pem: str, kid: str) -> dict[str, str]:
    """Return the RFC 7517 JWK dict for an RSA public key."""
    numbers = load_public_key(public_pem).public_numbers()
    n = numbers.n.to_bytes((numbers.n.bit_length() + 7) // 8, 'big')
    e = numbers.e.to_bytes((numbers.e.bit_length() + 7) // 8, 'big')
    return {
        'kty': 'RSA',
        'use': 'sig',
        'alg': 'RS256',
        'kid': kid,
        'n': b64url_encode(n),
        'e': b64url_encode(e),
    }


# ----------------------------------------------------------------------------
# Sign / verify
# ----------------------------------------------------------------------------

def encode(claims: dict[str, Any], private_pem: str, kid: str) -> str:
    """Sign ``claims`` into a compact RS256 JWT."""
    header = {'alg': 'RS256', 'typ': 'JWT', 'kid': kid}
    segments = [
        b64url_encode(json.dumps(header, separators=(',', ':')).encode('utf-8')),
        b64url_encode(json.dumps(claims, separators=(',', ':')).encode('utf-8')),
    ]
    signing_input = '.'.join(segments).encode('ascii')
    signature = load_private_key(private_pem).sign(
        signing_input, padding.PKCS1v15(), hashes.SHA256(),
    )
    segments.append(b64url_encode(signature))
    return '.'.join(segments)


def peek_header(token: str) -> dict[str, Any]:
    """Parse the JWT header WITHOUT verifying (needed to find the kid).

    :raise JWTError: on a structurally malformed token.
    """
    parts = token.split('.')
    if len(parts) != 3:
        raise JWTError('token does not have three segments')
    try:
        header = json.loads(b64url_decode(parts[0]))
    except (ValueError, TypeError) as exc:
        raise JWTError('header is not valid JSON') from exc
    if not isinstance(header, dict):
        raise JWTError('header is not an object')
    return header


def decode(
    token: str,
    public_pem: str,
    *,
    issuer: str,
    audience: str,
    leeway: int = 60,
    now: int | None = None,
) -> dict[str, Any]:
    """Verify signature and standard claims, returning the payload.

    Enforces: RS256 only, valid signature, ``iss`` exact match, ``aud`` contains
    ``audience``, and ``exp``/``nbf`` within ``leeway`` seconds. ``jti`` presence
    is required here; revocation lookup is the caller's job.

    :raise JWTError: on any verification failure.
    """
    parts = token.split('.')
    if len(parts) != 3:
        raise JWTError('token does not have three segments')

    try:
        header = json.loads(b64url_decode(parts[0]))
    except (ValueError, TypeError) as exc:
        raise JWTError('header is not valid JSON') from exc
    if not isinstance(header, dict) or header.get('alg') != 'RS256':
        raise JWTError('unexpected or missing alg (RS256 required)')

    signing_input = f'{parts[0]}.{parts[1]}'.encode('ascii')
    signature = b64url_decode(parts[2])
    try:
        load_public_key(public_pem).verify(
            signature, signing_input, padding.PKCS1v15(), hashes.SHA256(),
        )
    except InvalidSignature as exc:
        raise JWTError('signature verification failed') from exc

    try:
        payload = json.loads(b64url_decode(parts[1]))
    except (ValueError, TypeError) as exc:
        raise JWTError('payload is not valid JSON') from exc
    if not isinstance(payload, dict):
        raise JWTError('payload is not an object')

    ts = int(now if now is not None else time.time())

    if not hmac.compare_digest(str(payload.get('iss', '')), issuer):
        raise JWTError('issuer mismatch')

    aud = payload.get('aud')
    aud_list = aud if isinstance(aud, list) else [aud]
    if not any(isinstance(a, str) and hmac.compare_digest(a, audience) for a in aud_list):
        raise JWTError('audience mismatch')

    exp = payload.get('exp')
    if not isinstance(exp, int) or ts > exp + leeway:
        raise JWTError('token expired')

    nbf = payload.get('nbf')
    if isinstance(nbf, int) and ts < nbf - leeway:
        raise JWTError('token not yet valid')

    if not payload.get('jti'):
        raise JWTError('missing jti')

    return payload
