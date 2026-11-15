"""Unit tests for ont_agent.crypto — token encryption at rest."""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from ont_agent.crypto import (
    ENCRYPTED_PREFIX,
    TokenCipher,
    TokenDecryptionError,
    is_encrypted,
)

#: A realistic-looking access token: the thing actually being protected.
TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIyIiwic2NvcGUiOiJtY3A6cmVhZCJ9.sig"


def _key() -> str:
    return Fernet.generate_key().decode()


def test_a_token_round_trips() -> None:
    cipher = TokenCipher(_key())
    assert cipher.decrypt(cipher.encrypt(TOKEN)) == TOKEN


def test_the_plaintext_is_not_recoverable_from_the_ciphertext() -> None:
    """The entire point: a copy of the database must not be a copy of
    everyone's Odoo access."""
    encrypted = TokenCipher(_key()).encrypt(TOKEN)
    assert TOKEN not in encrypted
    # Not even the recognisable JWT header.
    assert "eyJhbGciOiJSUzI1Ni" not in encrypted


def test_encrypted_values_carry_the_scheme_marker() -> None:
    # The marker is what keeps "written before encryption" distinguishable
    # from "cannot be decrypted" — the failure design depends on it.
    encrypted = TokenCipher(_key()).encrypt(TOKEN)
    assert encrypted.startswith(ENCRYPTED_PREFIX)
    assert is_encrypted(encrypted)
    assert not is_encrypted(TOKEN)


def test_the_same_token_encrypts_differently_each_time() -> None:
    # Fernet is randomised, so equal tokens do not produce equal
    # ciphertext — someone reading the file cannot tell two users share one.
    cipher = TokenCipher(_key())
    assert cipher.encrypt(TOKEN) != cipher.encrypt(TOKEN)


def test_a_wrong_key_raises_rather_than_returning_garbage() -> None:
    """Fernet is authenticated, so a wrong key fails loudly instead of
    decrypting to something plausible that would then be sent to Odoo."""
    encrypted = TokenCipher(_key()).encrypt(TOKEN)
    with pytest.raises(TokenDecryptionError):
        TokenCipher(_key()).decrypt(encrypted)


def test_a_tampered_value_raises() -> None:
    cipher = TokenCipher(_key())
    encrypted = cipher.encrypt(TOKEN)
    with pytest.raises(TokenDecryptionError):
        cipher.decrypt(encrypted[:-4] + "AAAA")


def test_a_malformed_key_fails_at_construction() -> None:
    """Raised here so a bad key stops startup, rather than surfacing on
    somebody's first question of the day."""
    with pytest.raises(ValueError) as excinfo:
        TokenCipher("not-a-valid-fernet-key")

    message = str(excinfo.value)
    assert "AGENT_TOKEN_KEY" in message
    assert "Fernet.generate_key" in message  # actionable, not just "invalid"


def test_legacy_plaintext_passes_through_untouched() -> None:
    """Rows written before encryption keep working and convert themselves
    on the next save — no migration step to run."""
    assert TokenCipher(_key()).decrypt(TOKEN) == TOKEN


def test_no_key_means_no_cipher() -> None:
    assert TokenCipher.from_key("") is None
    assert TokenCipher.from_key(_key()) is not None


def test_fingerprints_identify_a_key_without_revealing_it() -> None:
    key = _key()
    same = TokenCipher(key).fingerprint
    assert TokenCipher(key).fingerprint == same
    assert TokenCipher(_key()).fingerprint != same
    # Short digest only — safe to log, useless to an attacker.
    assert key not in same
    assert len(same) == 12


def test_empty_values_are_left_alone() -> None:
    cipher = TokenCipher(_key())
    assert cipher.encrypt("") == ""
    assert cipher.decrypt("") == ""
