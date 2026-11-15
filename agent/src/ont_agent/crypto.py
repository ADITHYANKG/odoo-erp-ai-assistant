"""Encryption for the OAuth tokens held in ``agent.db``.

Those tokens are live credentials: anyone holding them can act as that user
against Odoo until they expire. The database is a file, and files travel —
into backups, disk images, cloud-sync folders and volumes on hosts you do
not own. Encrypting the two columns means a copy of the file is not a copy
of everyone's Odoo access.

**Be clear about what this does and does not buy.** A key stored beside the
database protects backups, snapshots and sync copies. It does not protect
against an attacker already on the running host, who can read the key as
easily as the database. Real separation means the key arrives from
somewhere else at runtime.

Uses Fernet (authenticated encryption from ``cryptography``, already a
transitive dependency), so a wrong key or a corrupted value fails loudly
rather than decrypting to plausible garbage.
"""

from __future__ import annotations

import hashlib
from typing import Self

from cryptography.fernet import Fernet, InvalidToken

from ont_agent.log import get_logger

logger = get_logger(__name__)

#: Marks a stored value as encrypted. This is what keeps "written before
#: encryption was switched on" distinguishable from "cannot be decrypted" —
#: the whole failure design rests on telling those two apart, and inferring
#: it from Fernet's own format would conflate them. The version leaves room
#: to change scheme later without a second guessing game.
ENCRYPTED_PREFIX = "enc:v1:"


class TokenDecryptionError(Exception):
    """A stored token could not be decrypted with the configured key."""


class TokenCipher:
    """Encrypts and decrypts token columns with one symmetric key."""

    def __init__(self, key: str) -> None:
        """:raise ValueError: the key is not a valid Fernet key.

        Raised at construction on purpose: a malformed key is a
        configuration mistake, and it should stop startup rather than
        surface on somebody's first question of the day.
        """
        try:
            self._fernet = Fernet(key.encode("utf-8"))
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "AGENT_TOKEN_KEY is not a valid Fernet key. Generate one with: "
                'python -c "from cryptography.fernet import Fernet; '
                'print(Fernet.generate_key().decode())"'
            ) from exc
        # A short digest, never the key itself — enough to notice the key
        # changed, useless to anyone who reads the database or the log.
        self.fingerprint = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]

    @classmethod
    def from_key(cls, key: str) -> Self | None:
        """A cipher, or None when no key is configured (plaintext, as before)."""
        return cls(key) if key else None

    def encrypt(self, value: str) -> str:
        """Encrypt one token for storage, tagged with the scheme prefix."""
        if not value:
            return value
        return ENCRYPTED_PREFIX + self._fernet.encrypt(value.encode("utf-8")).decode("ascii")

    def decrypt(self, value: str) -> str:
        """Decrypt a stored token.

        A value without the prefix predates encryption and is returned
        untouched, so existing rows keep working and convert themselves on
        the next write.

        :raise TokenDecryptionError: the value is encrypted but this key
            cannot open it — almost always a changed or wrong key.
        """
        if not value or not value.startswith(ENCRYPTED_PREFIX):
            return value
        payload = value[len(ENCRYPTED_PREFIX) :]
        try:
            return self._fernet.decrypt(payload.encode("ascii")).decode("utf-8")
        except (InvalidToken, ValueError) as exc:
            raise TokenDecryptionError(
                "Stored token could not be decrypted; AGENT_TOKEN_KEY has most "
                "likely changed. The affected user must reconnect with /start."
            ) from exc


def is_encrypted(value: str) -> bool:
    """Whether a stored value carries the encryption marker."""
    return value.startswith(ENCRYPTED_PREFIX)
