"""Google Chat plumbing: event verification, inbound parsing, sending.

No agent logic lives here — ``googlechat_bot`` decides what to say; this module
only knows Google's wire formats and limits. The app is an HTTP-endpoint Chat
app (Workspace add-on event format): Google POSTs each interaction to our
endpoint with an ID token, and we answer asynchronously through the Chat REST
API, authenticated as the app's service account (``chat.bot`` scope).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import jwt

from ont_agent.config import Settings
from ont_agent.log import get_logger

logger = get_logger(__name__)

CHAT_API = "https://chat.googleapis.com/v1"
CHAT_SCOPE = "https://www.googleapis.com/auth/chat.bot"
GOOGLE_CERTS_URL = "https://www.googleapis.com/oauth2/v3/certs"
_ISSUERS = ["https://accounts.google.com", "accounts.google.com"]

#: Chat's limit for the ``text`` field of one message.
TEXT_LIMIT = 4096
#: Re-read Google's signing keys at most this often...
_CERTS_TTL = 3600.0
#: ...except when a token names a key we do not have, which is allowed this often.
_CERTS_MIN_REFETCH = 60.0

KIND_TEXT = "text"
KIND_TAP = "tap"
KIND_ADDED = "added"
KIND_UNSUPPORTED = "unsupported"

TAP_PARAMETER = "data"


class GoogleChatError(Exception):
    """A Chat API call failed; the message never contains a credential."""


class AttachmentTooLarge(GoogleChatError):
    """An attachment exceeded the size the caller is willing to hold in memory."""


class GoogleChatVerifier:
    """Proves a POST came from Google Chat for OUR app.

    Google signs an OIDC ID token for the app's per-project service account,
    with the endpoint URL as audience. Without this check anybody who finds
    the URL could impersonate any user to the bot.
    """

    def __init__(
        self, audience: str, service_account: str, http: httpx.AsyncClient | None = None
    ) -> None:
        self._audience = audience
        self._service_account = service_account
        self._http = http or httpx.AsyncClient(timeout=10.0)
        self._keys: dict[str, Any] = {}
        self._fetched_at = 0.0

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _key(self, kid: str) -> Any | None:
        now = time.monotonic()
        age = now - self._fetched_at
        if kid in self._keys and age < _CERTS_TTL:
            return self._keys[kid]
        if age < _CERTS_MIN_REFETCH:  # a forged kid must not make us hammer Google
            return self._keys.get(kid)
        try:
            response = await self._http.get(GOOGLE_CERTS_URL)
            response.raise_for_status()
            jwks = response.json().get("keys") or []
        except (httpx.HTTPError, ValueError):
            logger.warning("Could not fetch Google's signing keys", exc_info=True)
            return self._keys.get(kid)
        keys: dict[str, Any] = {}
        for jwk in jwks:
            try:
                keys[str(jwk["kid"])] = jwt.PyJWK.from_dict(jwk).key
            except (KeyError, jwt.PyJWTError):
                continue
        self._keys, self._fetched_at = keys, now
        return keys.get(kid)

    async def verify(self, authorization: str | None) -> bool:
        """True when ``Authorization: Bearer <token>`` is Google's, for us."""
        if not authorization or not authorization.lower().startswith("bearer "):
            return False
        token = authorization[7:].strip()
        try:
            kid = jwt.get_unverified_header(token).get("kid")
        except jwt.PyJWTError:
            return False
        key = await self._key(str(kid)) if kid else None
        if key is None:
            return False
        try:
            claims = jwt.decode(
                token,
                key,
                algorithms=["RS256"],
                audience=self._audience,
                issuer=_ISSUERS,
                options={"require": ["exp", "aud", "iss"]},
            )
        except jwt.PyJWTError:
            return False
        return (
            claims.get("email") == self._service_account and claims.get("email_verified") is True
        )


# ---- inbound events --------------------------------------------------------


@dataclass(frozen=True)
class ChatAttachment:
    resource_name: str
    content_type: str
    filename: str


@dataclass(frozen=True)
class ChatEvent:
    kind: str
    #: ``users/123`` — Google's id for the sender.
    user_name: str
    email: str
    display_name: str
    space: str
    #: True for a 1:1 chat with the app; the bot answers nowhere else.
    direct: bool
    #: Unique per delivery, so Google's retries can be recognised.
    event_id: str
    timestamp: float
    message_name: str = ""
    text: str = ""
    tap: str = ""
    attachments: tuple[ChatAttachment, ...] = field(default_factory=tuple)
    #: A file was attached that the app cannot download (a Drive link).
    has_unreadable_file: bool = False


_FRACTION = re.compile(r"(\.\d{6})\d+")


def _epoch(stamp: object) -> float:
    """Seconds since the epoch for an RFC 3339 timestamp; 0.0 when absent or odd."""
    if not isinstance(stamp, str) or not stamp:
        return 0.0
    try:
        return datetime.fromisoformat(_FRACTION.sub(r"\1", stamp)).timestamp()
    except ValueError:
        return 0.0


def _is_direct(space: Mapping[str, Any]) -> bool:
    return (
        space.get("spaceType") == "DIRECT_MESSAGE"
        or space.get("type") == "DM"
        or bool(space.get("singleUserBotDm"))
    )


def parse_event(payload: Mapping[str, Any]) -> ChatEvent | None:
    """The interaction in a verified event body, or None when it needs no answer.

    Events from other apps' bots, removals and anything without a sender are
    dropped here so the channel only ever sees a human talking to us.
    """
    chat = payload.get("chat")
    if not isinstance(chat, Mapping):
        return None
    user = chat.get("user") or {}
    user_name = str(user.get("name") or "")
    if not user_name.startswith("users/") or user.get("type") == "BOT":
        return None

    sections = {
        key: chat[key]
        for key in (
            "messagePayload",
            "appCommandPayload",
            "buttonClickedPayload",
            "addedToSpacePayload",
        )
        if isinstance(chat.get(key), Mapping)
    }
    if not sections:
        return None
    key, section = next(iter(sections.items()))
    space = section.get("space") or {}
    space_name = str(space.get("name") or "")
    if not space_name:
        return None
    message = section.get("message") or {}
    event_time = chat.get("eventTime")

    def build(kind: str, event_id: str, **fields: Any) -> ChatEvent:
        return ChatEvent(
            kind=kind,
            user_name=user_name,
            email=str(user.get("email") or "").lower(),
            display_name=str(user.get("displayName") or ""),
            space=space_name,
            direct=_is_direct(space),
            event_id=event_id,
            timestamp=_epoch(message.get("createTime")) or _epoch(event_time),
            **fields,
        )

    if key == "addedToSpacePayload":
        return build(KIND_ADDED, f"added:{space_name}:{user_name}:{event_time}")

    if key == "buttonClickedPayload":
        parameters = (payload.get("commonEventObject") or {}).get("parameters") or {}
        tap = str(parameters.get(TAP_PARAMETER) or "")
        message_name = str(message.get("name") or "")
        return build(
            KIND_TAP,
            f"tap:{message_name}:{user_name}:{event_time}:{tap}",
            message_name=message_name,
            text=str(message.get("text") or ""),
            tap=tap,
        )

    message_name = str(message.get("name") or "")
    if not message_name:
        return None
    attachments: list[ChatAttachment] = []
    unreadable = False
    for raw in message.get("attachment") or []:
        resource = (raw.get("attachmentDataRef") or {}).get("resourceName")
        if raw.get("source") == "UPLOADED_CONTENT" and resource:
            attachments.append(
                ChatAttachment(
                    resource_name=str(resource),
                    content_type=str(raw.get("contentType") or "").split(";")[0].strip().lower(),
                    filename=str(raw.get("contentName") or ""),
                )
            )
        else:
            unreadable = True
    return build(
        KIND_TEXT if (message.get("text") or attachments or unreadable) else KIND_UNSUPPORTED,
        message_name,
        message_name=message_name,
        text=str(message.get("text") or "").strip(),
        attachments=tuple(attachments),
        has_unreadable_file=unreadable,
    )


# ---- outbound ---------------------------------------------------------------


def button_cards(
    buttons: Sequence[tuple[str, str]], endpoint: str, links: Sequence[tuple[str, str]] = ()
) -> list[dict[str, Any]]:
    """One card of buttons under a message.

    ``buttons`` are (data, title): a tap comes back to ``endpoint`` carrying
    ``data``. ``links`` are (url, title) buttons that open a page instead.
    """
    widgets: list[dict[str, Any]] = [
        {
            "text": title,
            "onClick": {
                "action": {
                    "function": endpoint,
                    "parameters": [{"key": TAP_PARAMETER, "value": data}],
                }
            },
        }
        for data, title in buttons
    ]
    widgets += [{"text": title, "onClick": {"openLink": {"url": url}}} for url, title in links]
    if not widgets:
        return []
    section = {"widgets": [{"buttonList": {"buttons": widgets}}]}
    return [{"cardId": "actions", "card": {"sections": [section]}}]


class GoogleChatClient:
    """Thin async wrapper over the Chat API calls the bot needs."""

    def __init__(self, settings: Settings, http: httpx.AsyncClient | None = None) -> None:
        info = json.loads(Path(settings.googlechat_credentials_file).read_text(encoding="utf-8"))
        self._email = str(info["client_email"])
        self._private_key = str(info["private_key"])
        self._token_uri = str(info["token_uri"])
        self._endpoint = settings.googlechat_audience
        self._http = http or httpx.AsyncClient(timeout=30.0)
        self._token = ""
        self._expires = 0.0
        self._lock = asyncio.Lock()

    @property
    def endpoint(self) -> str:
        return self._endpoint

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _access_token(self) -> str:
        async with self._lock:
            if self._token and time.time() < self._expires - 60:
                return self._token
            now = int(time.time())
            assertion = jwt.encode(
                {
                    "iss": self._email,
                    "scope": CHAT_SCOPE,
                    "aud": self._token_uri,
                    "iat": now,
                    "exp": now + 3600,
                },
                self._private_key,
                algorithm="RS256",
            )
            try:
                response = await self._http.post(
                    self._token_uri,
                    data={
                        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                        "assertion": assertion,
                    },
                )
            except httpx.HTTPError as exc:
                raise GoogleChatError(
                    f"Google token request failed: {type(exc).__name__}"
                ) from exc
            if response.status_code >= 400:
                raise GoogleChatError(f"Google token request returned {response.status_code}")
            data = response.json()
            self._token = str(data["access_token"])
            self._expires = time.time() + float(data.get("expires_in") or 3600)
            return self._token

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        headers = {"Authorization": f"Bearer {await self._access_token()}"}
        try:
            response = await self._http.request(method, url, headers=headers, **kwargs)
        except httpx.HTTPError as exc:
            raise GoogleChatError(f"Google Chat request failed: {type(exc).__name__}") from exc
        if response.status_code >= 400:
            raise GoogleChatError(
                f"Google Chat API returned {response.status_code}: {response.text[:300]}"
            )
        return response

    def _body(self, text: str, cards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {"text": text[:TEXT_LIMIT], "cardsV2": list(cards)}

    async def send_message(
        self, space: str, text: str, cards: Sequence[Mapping[str, Any]] = ()
    ) -> str:
        """Post a message into ``space``; returns its resource name."""
        response = await self._request(
            "POST", f"{CHAT_API}/{space}/messages", json=self._body(text, cards)
        )
        return str(response.json().get("name") or "")

    async def update_message(
        self, name: str, text: str, cards: Sequence[Mapping[str, Any]] = ()
    ) -> None:
        """Replace an app message's text and buttons (empty ``cards`` removes them)."""
        await self._request(
            "PATCH",
            f"{CHAT_API}/{name}",
            params={"updateMask": "text,cardsV2"},
            json=self._body(text, cards),
        )

    async def delete_message(self, name: str) -> None:
        await self._request("DELETE", f"{CHAT_API}/{name}")

    async def download(self, resource_name: str, max_bytes: int) -> bytes:
        """An uploaded attachment's bytes; raises when it exceeds ``max_bytes``."""
        headers = {"Authorization": f"Bearer {await self._access_token()}"}
        url = f"{CHAT_API}/media/{quote(resource_name, safe='/')}"
        try:
            async with self._http.stream(
                "GET", url, params={"alt": "media"}, headers=headers
            ) as response:
                if response.status_code >= 400:
                    raise GoogleChatError(
                        f"Google Chat media download returned {response.status_code}"
                    )
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data += chunk
                    if len(data) > max_bytes:
                        raise AttachmentTooLarge("attachment is larger than the allowed size")
        except httpx.HTTPError as exc:
            raise GoogleChatError(f"Google Chat download failed: {type(exc).__name__}") from exc
        return bytes(data)
