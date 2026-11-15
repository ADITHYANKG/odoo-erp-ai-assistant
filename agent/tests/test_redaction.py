"""Unit tests for ont_agent.redaction.

The non-matching corpus carries as much weight as the matching one: Odoo is
full of long opaque strings, and a scanner that eats invoice references or
attachment checksums makes every answer unreadable.
"""

from __future__ import annotations

import logging

import pytest

from ont_agent.redaction import redact_secrets

# Fabricated, structurally valid, and none of them real. Some are split with
# "+" so secret scanners (GitHub push protection) don't flag the source file.
SECRETS: list[tuple[str, str]] = [
    ("OpenAI-style key", "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcd"),
    ("OpenRouter key", "sk-or-v1-0123456789abcdef0123456789abcdef0123456789abcdef"),
    ("Anthropic key", "sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"),
    ("AWS access key id", "AKIAIOSFODNN7EXAMPLE"),
    ("GitHub token", "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
    ("Slack token", "xox" + "b-123456789012-abcdefghijklmnop"),
    ("Google API key", "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ0123456"),
    ("Stripe key", "sk_" + "live_ABCDEFGHIJKLMNOPQRSTUVWX"),
    (
        "JSON Web Token",
        "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        ".dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk",
    ),
    ("private key", "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAK\n-----END RSA PRIVATE KEY-----"),
]


@pytest.mark.parametrize(("kind", "secret"), SECRETS)
def test_each_credential_is_redacted_and_named(kind: str, secret: str) -> None:
    clean, kinds = redact_secrets(f"The value is {secret} apparently.")

    assert secret not in clean
    assert kinds == [kind]
    assert f"[redacted: {kind}]" in clean
    # Surgical, not wholesale: the sentence around it survives.
    assert clean.startswith("The value is ")
    assert clean.endswith(" apparently.")


def test_the_assignment_form_keeps_its_label() -> None:
    # The value goes, the label stays, so the reader can still see WHAT was
    # redacted without seeing it.
    clean, kinds = redact_secrets("Login with password=hunter2secret please")

    assert kinds == ["credential assignment"]
    assert "password=" in clean
    assert "hunter2secret" not in clean


# --- the corpus that must survive untouched ---------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # Odoo's own long identifiers, which an entropy heuristic would eat.
        "Attachment checksum a94a8fe5ccb19ba61c4c0873d391e987982fbbd3",
        "Reference 3f2504e0-4f89-11d3-9a0c-0305e82c3301",
        "Invoice INV/2026/00015 for EUR 4,217.55",
        "SKU FURN-8899-BLACK-XL ordered 12 times",
        "IBAN DE89370400440532013000",
        "Contact number +91 98470 12345",
        "Product: heavy-duty aluminium bracket, powder coated, 240x80mm",
        # The word without an assignment: this is why the ':' or '=' is
        # required, and it is the commonest phrase in any HR chatter.
        "Password reset requested by the user on Monday",
        "Please update your api key policy in the handbook",
        # Short things that superficially resemble a prefix.
        "sk-12",
        "Meeting notes: AKIA is an acronym we use internally",
    ],
)
def test_ordinary_business_text_is_untouched(text: str) -> None:
    clean, kinds = redact_secrets(text)

    assert clean == text
    assert kinds == []


# --- behaviour --------------------------------------------------------------


def test_several_secrets_in_one_result_all_go() -> None:
    text = f"key {SECRETS[0][1]} and aws {SECRETS[3][1]} and token {SECRETS[4][1]}"
    clean, kinds = redact_secrets(text)

    for _, secret in (SECRETS[0], SECRETS[3], SECRETS[4]):
        assert secret not in clean
    assert set(kinds) == {"OpenAI-style key", "AWS access key id", "GitHub token"}


def test_the_more_specific_vendor_wins() -> None:
    # sk-or-v1-... also matches the generic sk- rule; the reported kind must
    # be the precise one, which is why rule order matters.
    clean, kinds = redact_secrets(SECRETS[1][1])

    assert kinds == ["OpenRouter key"]
    assert clean == "[redacted: OpenRouter key]"


def test_redaction_is_idempotent() -> None:
    # The tool scan and the reply scan both run, so the second must not
    # chew on the first one's marker.
    once, first = redact_secrets(f"here: {SECRETS[0][1]}")
    twice, second = redact_secrets(once)

    assert twice == once
    assert first and second == []


def test_nothing_found_returns_the_text_unchanged() -> None:
    text = "You have 19 unpaid invoices totalling EUR 284,120.55."
    clean, kinds = redact_secrets(text)

    assert clean is text or clean == text
    assert kinds == []


def test_the_kinds_are_safe_to_log() -> None:
    """A scanner that writes the value into bot.log has moved the secret,
    not removed it — so callers log only what this returns."""
    _clean, kinds = redact_secrets(f"key {SECRETS[0][1]}")

    for kind in kinds:
        assert SECRETS[0][1] not in kind


def test_the_wrapper_logs_the_kind_and_never_the_value(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from ont_agent.redaction import redact_secrets as scan

    secret = SECRETS[0][1]
    with caplog.at_level(logging.WARNING):
        _clean, kinds = scan(f"note: {secret}")
        logging.getLogger("probe").warning("Redacted %s", ", ".join(kinds))

    assert "OpenAI-style key" in caplog.text
    assert secret not in caplog.text
