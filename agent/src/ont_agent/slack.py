"""Slack plumbing: request verification, inbound parsing, sending, Socket Mode.

No agent logic lives here — ``slack_bot`` decides what to say; this module only
knows Slack's wire formats and limits. Events reach the agent one of two ways
that share the same payload shapes: Socket Mode (the agent connects out and
Slack pushes envelopes over the socket) or HTTP (Slack POSTs to
``/webhook/slack``, authenticated by the signing secret).
"""

from __future__ import annotations

import hashlib
import hmac
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from slack_sdk.errors import SlackApiError
from slack_sdk.http_retry.builtin_async_handlers import AsyncRateLimitErrorRetryHandler
from slack_sdk.socket_mode.aiohttp import SocketModeClient
from slack_sdk.socket_mode.async_client import AsyncBaseSocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.web.async_client import AsyncWebClient

from ont_agent.config import Settings
from ont_agent.log import get_logger
from ont_agent.slack_format import plain_from_slack

logger = get_logger(__name__)

#: Slack suggests keeping ``text`` under this; longer is truncated.
TEXT_LIMIT = 4000
#: A section block's text may hold at most 3000 characters.
SECTION_LIMIT = 3000
_BUTTON_TITLE_LIMIT = 75
_MAX_BUTTONS = 25
#: Slack signs requests with a timestamp; older than this is a replay.
SIGNATURE_TOLERANCE_SECONDS = 300

KIND_TEXT = "text"
KIND_MENTION = "mention"
KIND_COMMAND = "command"
KIND_TAP = "tap"

#: Envelope types, as Socket Mode names them; HTTP mode maps onto the same names.
ENVELOPE_EVENTS = "events_api"
ENVELOPE_COMMAND = "slash_commands"
ENVELOPE_INTERACTIVE = "interactive"

_USER_ID = re.compile(r"[UW][A-Z0-9]{2,}")
_TRUSTED_FILE_HOSTS = (".slack.com", ".slack-files.com")
_AUTH_FAILURES = {
    "invalid_auth",
    "token_revoked",
    "account_inactive",
    "not_authed",
    "token_expired",
}


class SlackError(Exception):
    """A Slack call failed; the message never contains a credential."""


class SlackAuthError(SlackError):
    """Slack refused the token itself."""


class FileTooLarge(SlackError):
    """A file exceeded the size the caller is willing to hold in memory."""


# ---- request verification --------------------------------------------------


def verify_signature(
    secret: str,
    body: bytes,
    timestamp: str | None,
    signature: str | None,
    *,
    now: float | None = None,
) -> bool:
    """True when an HTTP-mode POST was signed by Slack with ``secret``."""
    if not secret or not timestamp or not signature:
        return False
    try:
        age = abs((time.time() if now is None else now) - int(timestamp))
    except ValueError:
        return False
    if age > SIGNATURE_TOLERANCE_SECONDS:
        return False
    base = b"v0:" + timestamp.encode() + b":" + body
    expected = "v0=" + hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


# ---- inbound events --------------------------------------------------------


@dataclass(frozen=True)
class SlackFile:
    url: str
    mime: str
    name: str
    size: int
    is_audio: bool


@dataclass(frozen=True)
class SlackEvent:
    kind: str
    #: ``U...`` (``W...`` on Enterprise Grid): Slack's id for the sender.
    user_id: str
    team_id: str
    #: The conversation to answer in.
    channel: str
    #: True for a 1:1 DM with the app; the bot answers nowhere else.
    direct: bool
    #: Unique per delivery, so Slack's retries can be recognised.
    event_id: str
    #: Seconds since the epoch; 0.0 when the payload carries none.
    timestamp: float
    text: str = ""
    #: A tap: the ``ts`` of the message holding the button.
    message_ts: str = ""
    tap: str = ""
    files: tuple[SlackFile, ...] = field(default_factory=tuple)
    #: A file was shared that the app cannot read (e.g. from another workspace).
    has_unreadable_file: bool = False
    #: Slash commands only: where an ephemeral reply can be posted from any channel.
    response_url: str = ""


def _seconds(value: object) -> float:
    try:
        return float(str(value))
    except ValueError:
        return 0.0


def _is_dm(channel: str, channel_type: str = "") -> bool:
    return channel_type == "im" or channel.startswith("D")


def _parse_files(raw_files: object) -> tuple[tuple[SlackFile, ...], bool]:
    files: list[SlackFile] = []
    unreadable = False
    for raw in raw_files if isinstance(raw_files, list) else []:
        if not isinstance(raw, Mapping):
            continue
        url = str(raw.get("url_private_download") or raw.get("url_private") or "")
        if raw.get("mode") == "tombstone":
            continue
        if not url or raw.get("file_access") == "check_file_info":
            unreadable = True
            continue
        mime = str(raw.get("mimetype") or "").split(";")[0].strip().lower()
        files.append(
            SlackFile(
                url=url,
                mime=mime,
                name=str(raw.get("name") or raw.get("title") or ""),
                size=int(raw.get("size") or 0),
                is_audio=mime.startswith("audio/") or raw.get("subtype") == "slack_audio",
            )
        )
    return tuple(files), unreadable


def parse_event(envelope: str, payload: Mapping[str, Any]) -> SlackEvent | None:
    """The interaction in an envelope, or None when it needs no answer.

    Bots' own messages, edits, deletions and anything without a human sender
    are dropped here so the channel only ever sees a person talking to us.
    """
    if envelope == ENVELOPE_EVENTS:
        return _parse_message(payload)
    if envelope == ENVELOPE_COMMAND:
        user_id = str(payload.get("user_id") or "")
        channel = str(payload.get("channel_id") or "")
        if not _USER_ID.fullmatch(user_id) or not channel:
            return None
        return SlackEvent(
            kind=KIND_COMMAND,
            user_id=user_id,
            team_id=str(payload.get("team_id") or ""),
            channel=channel,
            direct=_is_dm(channel),
            event_id=f"cmd:{payload.get('trigger_id') or time.time_ns()}",
            timestamp=0.0,
            text=plain_from_slack(str(payload.get("text") or "")),
            response_url=str(payload.get("response_url") or ""),
        )
    if envelope == ENVELOPE_INTERACTIVE:
        return _parse_tap(payload)
    return None


def _parse_message(payload: Mapping[str, Any]) -> SlackEvent | None:
    event = payload.get("event")
    if not isinstance(event, Mapping):
        return None
    event_type = event.get("type")
    if event_type not in ("message", "app_mention"):
        return None
    user_id = str(event.get("user") or "")
    channel = str(event.get("channel") or "")
    ts = str(event.get("ts") or "")
    if event.get("bot_id") or not _USER_ID.fullmatch(user_id) or not channel or not ts:
        return None

    files: tuple[SlackFile, ...] = ()
    if event_type == "app_mention":
        kind, direct, unreadable = KIND_MENTION, False, False
    else:
        if event.get("subtype") not in (None, "file_share"):
            return None
        if not _is_dm(channel, str(event.get("channel_type") or "")):
            return None  # only the DM is served; channel chatter is not ours to answer
        kind, direct = KIND_TEXT, True
        files, unreadable = _parse_files(event.get("files"))

    return SlackEvent(
        kind=kind,
        user_id=user_id,
        team_id=str(event.get("user_team") or event.get("team") or payload.get("team_id") or ""),
        channel=channel,
        direct=direct,
        event_id=str(payload.get("event_id") or f"{channel}:{ts}"),
        timestamp=_seconds(ts),
        text=plain_from_slack(str(event.get("text") or "")),
        message_ts=ts,
        files=files,
        has_unreadable_file=unreadable,
    )


def _parse_tap(payload: Mapping[str, Any]) -> SlackEvent | None:
    if payload.get("type") != "block_actions":
        return None
    actions = payload.get("actions")
    action = actions[0] if isinstance(actions, list) and actions else None
    # Link buttons also report a click, but carry no value to act on.
    if not isinstance(action, Mapping) or not action.get("value"):
        return None
    user = payload.get("user") or {}
    user_id = str(user.get("id") or "")
    container = payload.get("container") or {}
    channel = str(container.get("channel_id") or (payload.get("channel") or {}).get("id") or "")
    message = payload.get("message") or {}
    if not _USER_ID.fullmatch(user_id) or not channel:
        return None
    return SlackEvent(
        kind=KIND_TAP,
        user_id=user_id,
        team_id=str(user.get("team_id") or (payload.get("team") or {}).get("id") or ""),
        channel=channel,
        direct=_is_dm(channel),
        event_id=f"tap:{payload.get('trigger_id') or time.time_ns()}",
        timestamp=_seconds(action.get("action_ts")),
        text=str(message.get("text") or ""),
        message_ts=str(message.get("ts") or container.get("message_ts") or ""),
        tap=str(action["value"]),
    )


# ---- outbound --------------------------------------------------------------


def _style(data: str) -> str | None:
    if data.startswith("confirm:approve"):
        return "primary"
    if data.startswith("confirm:reject"):
        return "danger"
    return None


def split_text(text: str, limit: int = SECTION_LIMIT) -> list[str]:
    """``text`` in pieces a section block accepts, cut at line breaks.

    A code fence open at a cut is closed there and reopened in the next piece
    so the formatting survives.
    """
    pieces: list[str] = []
    current = ""
    room = limit - 8  # headroom for a closing and a reopening fence
    for line in text.split("\n"):
        while len(line) > room:  # a single line longer than a block: hard cut
            if current:
                pieces.append(current)
                current = ""
            pieces.append(line[:room])
            line = line[room:]
        if current and len(current) + 1 + len(line) > room:
            pieces.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current or not pieces:
        pieces.append(current)

    fenced: list[str] = []
    reopen = False
    for piece in pieces:
        if reopen:
            piece = "```\n" + piece
        reopen = piece.count("```") % 2 == 1
        if reopen:
            piece += "\n```"
        fenced.append(piece)
    return fenced


def message_blocks(
    text: str,
    buttons: Sequence[tuple[str, str]],
    links: Sequence[tuple[str, str]] = (),
) -> list[dict[str, Any]]:
    """The answer plus one row of buttons; empty when there are no buttons.

    ``buttons`` are (data, title): a click comes back carrying ``data``.
    ``links`` are (url, title) buttons that open a page instead.
    """
    elements: list[dict[str, Any]] = []
    for index, (data, title) in enumerate(buttons):
        element: dict[str, Any] = {
            "type": "button",
            "action_id": f"b{index}",
            "text": {"type": "plain_text", "text": title[:_BUTTON_TITLE_LIMIT], "emoji": True},
            "value": data,
        }
        if style := _style(data):
            element["style"] = style
        elements.append(element)
    for index, (url, title) in enumerate(links):
        elements.append(
            {
                "type": "button",
                "action_id": f"link{index}",
                "text": {"type": "plain_text", "text": title[:_BUTTON_TITLE_LIMIT], "emoji": True},
                "url": url,
                "style": "primary",
            }
        )
    if not elements:
        return []
    blocks: list[dict[str, Any]] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": piece}}
        for piece in split_text(text)
    ]
    blocks.append({"type": "actions", "elements": elements[:_MAX_BUTTONS]})
    return blocks


@dataclass(frozen=True)
class AuthInfo:
    team_id: str
    user_id: str


class SlackClient:
    """Thin async wrapper over the Slack Web API calls the bot needs."""

    def __init__(
        self,
        settings: Settings,
        web: AsyncWebClient | None = None,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._token = settings.slack_bot_token
        if web is None:
            web = AsyncWebClient(token=self._token)
            web.retry_handlers.append(AsyncRateLimitErrorRetryHandler(max_retry_count=2))
        self.web = web
        self._http = http or httpx.AsyncClient(timeout=30.0)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _call(self, method: str, **kwargs: Any) -> Mapping[str, Any]:
        try:
            response = await getattr(self.web, method)(**kwargs)
        except SlackApiError as exc:
            code = str(exc.response.get("error") or "unknown")
            error = SlackAuthError if code in _AUTH_FAILURES else SlackError
            raise error(f"Slack {method} failed: {code}") from exc
        except Exception as exc:
            raise SlackError(f"Slack {method} failed: {type(exc).__name__}") from exc
        data = response.data
        return data if isinstance(data, Mapping) else {}

    async def auth_test(self) -> AuthInfo:
        """The workspace and bot user behind the token; raises when it is refused."""
        data = await self._call("auth_test")
        return AuthInfo(
            team_id=str(data.get("team_id") or ""), user_id=str(data.get("user_id") or "")
        )

    async def send_message(
        self, channel: str, text: str, blocks: Sequence[Mapping[str, Any]] = ()
    ) -> str:
        """Post into ``channel``; returns the message ``ts``."""
        kwargs: dict[str, Any] = {
            "channel": channel,
            "text": text[:TEXT_LIMIT],
            "unfurl_links": False,
            "unfurl_media": False,
        }
        if blocks:
            kwargs["blocks"] = list(blocks)
        data = await self._call("chat_postMessage", **kwargs)
        return str(data.get("ts") or "")

    async def update_message(
        self, channel: str, ts: str, text: str, blocks: Sequence[Mapping[str, Any]] = ()
    ) -> None:
        """Replace a message's text and buttons (empty ``blocks`` removes them)."""
        await self._call(
            "chat_update", channel=channel, ts=ts, text=text[:TEXT_LIMIT], blocks=list(blocks)
        )

    async def send_ephemeral(self, channel: str, user: str, text: str) -> None:
        await self._call("chat_postEphemeral", channel=channel, user=user, text=text[:TEXT_LIMIT])

    async def respond(self, response_url: str, text: str) -> None:
        """An ephemeral reply to a slash command, usable from any channel."""
        if urlparse(response_url).hostname != "hooks.slack.com":
            raise SlackError("untrusted response URL")
        try:
            response = await self._http.post(
                response_url, json={"response_type": "ephemeral", "text": text[:TEXT_LIMIT]}
            )
        except httpx.HTTPError as exc:
            raise SlackError(f"Slack response failed: {type(exc).__name__}") from exc
        if response.status_code >= 400:
            raise SlackError(f"Slack response returned {response.status_code}")

    async def upload_file(self, channel: str, path: Path, title: str, comment: str = "") -> None:
        await self._call(
            "files_upload_v2",
            channel=channel,
            file=str(path),
            filename=path.name,
            title=title or path.name,
            initial_comment=comment or None,
        )

    async def download(self, url: str, max_bytes: int) -> bytes:
        """A shared file's bytes; raises when it exceeds ``max_bytes``."""
        parts = urlparse(url)
        host = parts.hostname or ""
        if parts.scheme != "https" or not host.endswith(_TRUSTED_FILE_HOSTS):
            raise SlackError("refusing to send the bot token to a non-Slack host")
        headers = {"Authorization": f"Bearer {self._token}"}
        try:
            async with self._http.stream("GET", url, headers=headers) as response:
                if response.status_code >= 300:
                    raise SlackError(f"Slack file download returned {response.status_code}")
                if response.headers.get("content-type", "").startswith("text/html"):
                    # A sign-in page: the token lacks files:read.
                    raise SlackError("Slack file download was refused (is files:read granted?)")
                if int(response.headers.get("content-length") or 0) > max_bytes:
                    raise FileTooLarge("file is larger than the allowed size")
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data += chunk
                    if len(data) > max_bytes:
                        raise FileTooLarge("file is larger than the allowed size")
        except httpx.HTTPError as exc:
            raise SlackError(f"Slack file download failed: {type(exc).__name__}") from exc
        return bytes(data)


# ---- Socket Mode -----------------------------------------------------------

OnEnvelope = Callable[[str, Mapping[str, Any]], object]


class SlackSocket:
    """The outbound Socket Mode connection; hands every envelope to ``on_envelope``."""

    def __init__(self, app_token: str, web: AsyncWebClient, on_envelope: OnEnvelope) -> None:
        self._client = SocketModeClient(app_token=app_token, web_client=web)
        self._on_envelope = on_envelope
        self._client.socket_mode_request_listeners.append(self._handle)

    async def _handle(self, client: AsyncBaseSocketModeClient, request: SocketModeRequest) -> None:
        # Slack redelivers anything not acknowledged within three seconds.
        await client.send_socket_mode_response(SocketModeResponse(envelope_id=request.envelope_id))
        if request.type not in (ENVELOPE_EVENTS, ENVELOPE_COMMAND, ENVELOPE_INTERACTIVE):
            return
        try:
            self._on_envelope(request.type, request.payload)
        except Exception:
            logger.error("Slack envelope handling failed", exc_info=True)

    async def start(self) -> None:
        try:
            await self._client.connect()  # type: ignore[no-untyped-call]
        except SlackApiError as exc:
            code = str(exc.response.get("error") or "unknown")
            error = SlackAuthError if code in _AUTH_FAILURES else SlackError
            raise error(f"Slack Socket Mode connection failed: {code}") from exc

    async def aclose(self) -> None:
        await self._client.close()  # type: ignore[no-untyped-call]
