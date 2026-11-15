"""Which messaging channel a chat id belongs to.

Bindings, threads, watchers and usage are all keyed by one BIGINT chat id.
Telegram ids stay far below ``WHATSAPP_OFFSET``; a WhatsApp user is stored as
the offset plus their phone number (at most 15 digits), and a Google Chat user
as ``GOOGLE_CHAT_OFFSET`` plus a 56-bit hash of their user id, a Slack user
likewise from ``SLACK_OFFSET``. The ranges can never collide and no schema
change is needed.
"""

from __future__ import annotations

import hashlib
import re

WHATSAPP_OFFSET = 10**17
#: Above the whole WhatsApp range (10**17 + 15 digits) and, with the 56-bit
#: hash below, still far inside BIGINT.
GOOGLE_CHAT_OFFSET = 2 * 10**18
#: Above the whole Google Chat range (2*10**18 + 2**56) and inside BIGINT.
SLACK_OFFSET = 3 * 10**18

#: E.164 allows at most 15 digits.
_MAX_WA_DIGITS = 15


def whatsapp_chat_id(wa_id: str) -> int:
    """Chat id for a WhatsApp phone number (digits, country code included)."""
    if not wa_id.isdigit() or len(wa_id) > _MAX_WA_DIGITS:
        raise ValueError(f"not a WhatsApp id: {wa_id!r}")
    return WHATSAPP_OFFSET + int(wa_id)


def is_whatsapp(chat_id: int) -> bool:
    return WHATSAPP_OFFSET <= chat_id < GOOGLE_CHAT_OFFSET


def wa_id_of(chat_id: int) -> str:
    """The phone number a WhatsApp chat id was built from."""
    if not is_whatsapp(chat_id):
        raise ValueError(f"not a WhatsApp chat id: {chat_id}")
    return str(chat_id - WHATSAPP_OFFSET)


def googlechat_chat_id(user_name: str) -> int:
    """Chat id for a Google Chat user resource name such as ``users/1234``.

    Google user ids have 21 digits and overflow BIGINT, so the id is hashed;
    the reverse lookup (chat id -> space) lives in the store.
    """
    if not user_name.startswith("users/") or len(user_name) <= len("users/"):
        raise ValueError(f"not a Google Chat user: {user_name!r}")
    digest = hashlib.sha256(user_name.encode()).digest()
    return GOOGLE_CHAT_OFFSET + int.from_bytes(digest[:7])


def is_google_chat(chat_id: int) -> bool:
    return GOOGLE_CHAT_OFFSET <= chat_id < SLACK_OFFSET


def slack_chat_id(user_id: str) -> int:
    """Chat id for a Slack member id such as ``U0123ABCD`` (or ``W…`` on Enterprise Grid).

    Hashed so the id fits BIGINT; the reverse lookup (chat id -> DM channel)
    lives in the store.
    """
    if not re.fullmatch(r"[UW][A-Z0-9]{2,}", user_id):
        raise ValueError(f"not a Slack user id: {user_id!r}")
    digest = hashlib.sha256(f"slack:{user_id}".encode()).digest()
    return SLACK_OFFSET + int.from_bytes(digest[:7])


def is_slack(chat_id: int) -> bool:
    return chat_id >= SLACK_OFFSET


def channel_name(chat_id: int) -> str:
    """Short name of the chat's channel, as used in trace names and tags."""
    if is_slack(chat_id):
        return "slack"
    if is_google_chat(chat_id):
        return "googlechat"
    if is_whatsapp(chat_id):
        return "whatsapp"
    return "telegram"
