"""WhatsApp Cloud API plumbing: webhook signature, inbound parsing, sending.

No agent logic lives here — ``whatsapp_bot`` decides what to say; this module
only knows Meta's wire formats and limits.
"""

from __future__ import annotations

import hashlib
import hmac
import mimetypes
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ont_agent.config import Settings
from ont_agent.log import get_logger

logger = get_logger(__name__)

#: Meta's hard limits for outbound messages.
TEXT_LIMIT = 4096
BUTTON_BODY_LIMIT = 1024
BUTTON_TITLE_LIMIT = 20
MAX_BUTTONS = 3
LIST_ROW_TITLE_LIMIT = 24
LIST_ROW_DESCRIPTION_LIMIT = 72
LIST_BUTTON_LABEL_LIMIT = 20
LIST_SECTION_TITLE_LIMIT = 24
MAX_LIST_ROWS = 10
ID_LIMIT = 200

KIND_TEXT = "text"
KIND_REPLY = "reply"  # a button or list tap
KIND_AUDIO = "audio"
KIND_IMAGE = "image"
KIND_DOCUMENT = "document"
KIND_UNSUPPORTED = "unsupported"


class WhatsAppError(Exception):
    """A Graph API call failed; the message never contains the access token."""


def verify_signature(app_secret: str, body: bytes, header: str | None) -> bool:
    """True when ``X-Hub-Signature-256`` matches the HMAC of the raw body."""
    if not app_secret or not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header.removeprefix("sha256="))


@dataclass(frozen=True)
class InboundMessage:
    wa_id: str
    message_id: str
    timestamp: float
    kind: str
    text: str = ""
    reply_id: str = ""
    media_id: str = ""
    mime: str = ""
    filename: str = ""


def _message(raw: Mapping[str, Any]) -> InboundMessage | None:
    wa_id = str(raw.get("from") or "")
    message_id = str(raw.get("id") or "")
    if not wa_id.isdigit() or not message_id:
        return None
    try:
        timestamp = float(raw.get("timestamp") or 0)
    except (TypeError, ValueError):
        timestamp = 0.0

    def build(kind: str, **fields: str) -> InboundMessage:
        return InboundMessage(
            wa_id=wa_id, message_id=message_id, timestamp=timestamp, kind=kind, **fields
        )

    kind = raw.get("type")
    if kind == "text":
        body = (raw.get("text") or {}).get("body") or ""
        return build(KIND_TEXT, text=str(body))
    if kind == "interactive":
        interactive = raw.get("interactive") or {}
        tap = interactive.get("button_reply") or interactive.get("list_reply") or {}
        return build(
            KIND_REPLY, reply_id=str(tap.get("id") or ""), text=str(tap.get("title") or "")
        )
    if kind in ("audio", "image", "document"):
        media = raw.get(kind) or {}
        return build(
            kind,
            media_id=str(media.get("id") or ""),
            mime=str(media.get("mime_type") or "").split(";")[0].strip().lower(),
            filename=str(media.get("filename") or ""),
            text=str(media.get("caption") or ""),
        )
    return build(KIND_UNSUPPORTED)


def parse_events(payload: Mapping[str, Any], phone_number_id: str) -> list[InboundMessage]:
    """Inbound user messages in a webhook payload.

    Delivery/read receipts are dropped, and so is anything addressed to a
    different phone number than ours.
    """
    messages: list[InboundMessage] = []
    for entry in payload.get("entry") or []:
        for change in entry.get("changes") or []:
            value = change.get("value") or {}
            target = (value.get("metadata") or {}).get("phone_number_id")
            if target and str(target) != phone_number_id:
                continue
            for raw in value.get("messages") or []:
                parsed = _message(raw)
                if parsed is not None:
                    messages.append(parsed)
    return messages


@dataclass(frozen=True)
class MediaInfo:
    url: str
    mime: str
    size: int


@dataclass(frozen=True)
class ListRow:
    id: str
    title: str
    description: str = ""


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


class WhatsAppClient:
    """Thin async wrapper over the Cloud API calls the bot needs."""

    def __init__(self, settings: Settings, http: httpx.AsyncClient | None = None) -> None:
        self._base = settings.whatsapp_graph_url
        self._phone_id = settings.whatsapp_phone_number_id
        self._headers = {"Authorization": f"Bearer {settings.whatsapp_access_token}"}
        self._http = http or httpx.AsyncClient(timeout=30.0)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._http.request(method, url, headers=self._headers, **kwargs)
        except httpx.HTTPError as exc:
            raise WhatsAppError(f"WhatsApp request failed: {type(exc).__name__}") from exc
        if response.status_code >= 400:
            raise WhatsAppError(
                f"WhatsApp API returned {response.status_code}: {response.text[:300]}"
            )
        return response

    async def _send(self, to: str, kind: str, body: dict[str, Any]) -> None:
        await self._request(
            "POST",
            f"{self._base}/{self._phone_id}/messages",
            json={
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": to,
                "type": kind,
                kind: body,
            },
        )

    async def send_text(self, to: str, text: str) -> None:
        await self._send(to, "text", {"body": _clip(text, TEXT_LIMIT), "preview_url": False})

    async def send_buttons(self, to: str, body: str, buttons: Sequence[tuple[str, str]]) -> None:
        """Up to three reply buttons, as (id, title) pairs."""
        await self._send(
            to,
            "interactive",
            {
                "type": "button",
                "body": {"text": _clip(body, BUTTON_BODY_LIMIT)},
                "action": {
                    "buttons": [
                        {
                            "type": "reply",
                            "reply": {
                                "id": button_id[:ID_LIMIT],
                                "title": _clip(title, BUTTON_TITLE_LIMIT),
                            },
                        }
                        for button_id, title in buttons[:MAX_BUTTONS]
                    ]
                },
            },
        )

    async def send_cta_url(self, to: str, body: str, label: str, url: str) -> None:
        """A message with one tappable link button (https only)."""
        await self._send(
            to,
            "interactive",
            {
                "type": "cta_url",
                "body": {"text": _clip(body, BUTTON_BODY_LIMIT)},
                "action": {
                    "name": "cta_url",
                    "parameters": {"display_text": _clip(label, BUTTON_TITLE_LIMIT), "url": url},
                },
            },
        )

    async def send_list(
        self,
        to: str,
        body: str,
        button_label: str,
        sections: Sequence[tuple[str, Sequence[ListRow]]],
    ) -> None:
        """A list message; rows beyond Meta's ten-row total are dropped."""
        remaining = MAX_LIST_ROWS
        rendered: list[dict[str, Any]] = []
        for title, rows in sections:
            kept = list(rows)[:remaining]
            if not kept:
                continue
            remaining -= len(kept)
            rendered.append(
                {
                    "title": _clip(title, LIST_SECTION_TITLE_LIMIT),
                    "rows": [
                        {
                            "id": row.id[:ID_LIMIT],
                            "title": _clip(row.title, LIST_ROW_TITLE_LIMIT),
                            **(
                                {"description": _clip(row.description, LIST_ROW_DESCRIPTION_LIMIT)}
                                if row.description
                                else {}
                            ),
                        }
                        for row in kept
                    ],
                }
            )
        if not rendered:
            return
        await self._send(
            to,
            "interactive",
            {
                "type": "list",
                "body": {"text": _clip(body, BUTTON_BODY_LIMIT)},
                "action": {
                    "button": _clip(button_label, LIST_BUTTON_LABEL_LIMIT),
                    "sections": rendered,
                },
            },
        )

    async def upload_media(self, path: Path, mime: str) -> str:
        response = await self._request(
            "POST",
            f"{self._base}/{self._phone_id}/media",
            data={"messaging_product": "whatsapp", "type": mime},
            files={"file": (path.name, path.read_bytes(), mime)},
        )
        media_id = str(response.json().get("id") or "")
        if not media_id:
            raise WhatsAppError("WhatsApp media upload returned no id")
        return media_id

    async def send_file(self, to: str, path: Path, *, as_image: bool, caption: str = "") -> None:
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        media_id = await self.upload_media(path, mime)
        if as_image:
            body: dict[str, Any] = {"id": media_id}
            if caption:
                body["caption"] = _clip(caption, 1024)
            await self._send(to, "image", body)
            return
        body = {"id": media_id, "filename": path.name}
        if caption:
            body["caption"] = _clip(caption, 1024)
        await self._send(to, "document", body)

    async def display_number(self) -> str:
        """Digits of this sender's phone number, or "" when Meta cannot say.

        Used for the wa.me return link, so a failure must not stop startup.
        """
        try:
            response = await self._request(
                "GET", f"{self._base}/{self._phone_id}", params={"fields": "display_phone_number"}
            )
            shown = str(response.json().get("display_phone_number") or "")
        except (WhatsAppError, ValueError):
            logger.warning("Could not read the WhatsApp number from Meta", exc_info=True)
            return ""
        return "".join(ch for ch in shown if ch.isdigit())

    async def media_info(self, media_id: str) -> MediaInfo:
        response = await self._request("GET", f"{self._base}/{media_id}")
        data = response.json()
        return MediaInfo(
            url=str(data.get("url") or ""),
            mime=str(data.get("mime_type") or "").split(";")[0].strip().lower(),
            size=int(data.get("file_size") or 0),
        )

    async def download(self, url: str) -> bytes:
        return (await self._request("GET", url)).content

    async def send_reaction(self, to: str, message_id: str, emoji: str) -> None:
        """Put an emoji reaction on the user's message; "" clears it.

        Best-effort: a status flourish must never stop the answer.
        """
        try:
            await self._send(to, "reaction", {"message_id": message_id, "emoji": emoji})
        except WhatsAppError:
            logger.debug("Reaction failed", exc_info=True)

    async def show_typing(self, message_id: str) -> None:
        """Mark read and show "typing…" for up to ~25 s or until we reply.

        Best-effort: a failure here must never stop the answer.
        """
        try:
            await self._request(
                "POST",
                f"{self._base}/{self._phone_id}/messages",
                json={
                    "messaging_product": "whatsapp",
                    "status": "read",
                    "message_id": message_id,
                    "typing_indicator": {"type": "text"},
                },
            )
        except WhatsAppError:
            logger.debug("Typing indicator failed", exc_info=True)
