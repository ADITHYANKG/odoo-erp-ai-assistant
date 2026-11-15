"""The agent's HTTP surface, served by FastAPI on the bot's asyncio loop.

Routes:

- ``GET /oauth/callback`` — the OAuth redirect target. Odoo's authorize
  endpoint 303-redirects the user's browser here with ``?code=...&state=...``;
  we complete the login (token exchange + whoami + binding save) and show a
  small page telling the user to return to Telegram.
- ``GET /oauth/link-status?t=...`` — polled by that page so it can say
  "Connected" once the code is typed in the chat. Answers only
  ``pending``/``connected``/``expired`` for the page's own random token.
- ``GET /health`` — liveness probe for future container orchestration.
- ``GET /static/avatar.png`` — the bot's avatar image, for chat app settings
  that want a public HTTPS image URL (Google Chat's "Avatar URL").
- ``GET|POST /webhook/whatsapp`` — WhatsApp Cloud API webhook: the Meta
  verification handshake works as soon as ``WHATSAPP_VERIFY_TOKEN`` is set;
  signed event POSTs are handed to the WhatsApp channel.
- ``POST /webhook/googlechat`` — Google Chat events, accepted only with a
  Google-signed ID token for this app; handed to the Google Chat channel.
- ``POST /webhook/slack`` — Slack events, interactivity and slash commands in
  HTTP mode, accepted only with a valid signing-secret signature; closed in
  Socket Mode, where Slack never calls us.

The server is embedded: uvicorn runs as a task inside the existing event loop
next to python-telegram-bot's polling — no separate process, no signals.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
from collections.abc import Awaitable, Callable, Generator, Mapping
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
)

from ont_agent.auth import LinkCode, LoginError, complete_login, link_status
from ont_agent.channels import is_google_chat, is_slack, is_whatsapp
from ont_agent.config import Settings
from ont_agent.log import get_logger
from ont_agent.slack import (
    ENVELOPE_COMMAND,
    ENVELOPE_EVENTS,
    ENVELOPE_INTERACTIVE,
)
from ont_agent.slack import (
    verify_signature as verify_slack_signature,
)
from ont_agent.store import BindingStore
from ont_agent.whatsapp import verify_signature

logger = get_logger(__name__)

OnBound = Callable[[LinkCode], Awaitable[None]]
#: Receives a verified WhatsApp webhook body; schedules the work and returns.
OnWhatsApp = Callable[[Mapping[str, Any]], object]
#: Receives a verified Google Chat event body; schedules the work and returns.
OnGoogleChat = Callable[[Mapping[str, Any]], object]
#: True when an ``Authorization`` header is a valid Google Chat token for this app.
VerifyGoogleChat = Callable[[str | None], Awaitable[bool]]
#: Receives a verified Slack envelope (its Socket Mode type name and payload).
OnSlack = Callable[[str, Mapping[str, Any]], object]

GOOGLE_CHAT_URL = "https://chat.google.com/"

#: The bot's avatar, served publicly so chat apps (Google Chat's "Avatar URL")
#: can point at ${AGENT_PUBLIC_URL}/static/avatar.png instead of a third-party host.
AVATAR_PATH = Path(__file__).parent / "static" / "avatar.png"

_PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{title}</title>{head_extra}</head>
<body style="font-family:sans-serif;max-width:30rem;margin:4rem auto;text-align:center">
<h2 id="title">{title}</h2><p id="message">{message}</p>{code}{action}{script}</body></html>"""

_CODE = (
    '<p id="code" style="font-size:2.2rem;font-weight:bold;letter-spacing:.4rem;'
    'font-family:monospace">{code}</p>'
)

#: Polls /oauth/link-status (relative, so it works behind any path prefix)
#: and swaps the page once the chat has confirmed or the login is gone.
#: Gives up after the code's lifetime so a forgotten tab stops polling.
_POLL = """<script>
(function () {{
  var token = {token}, tries = 0;
  var done = {{
    connected: ["Connected",
      "You're all set. Go back to the chat and ask away. You can close this tab."],
    expired: ["Login not completed",
      "This login expired or was replaced. Start again from the chat."]
  }};
  function show(status) {{
    document.title = done[status][0];
    document.getElementById("title").textContent = done[status][0];
    document.getElementById("message").textContent = done[status][1];
    var code = document.getElementById("code");
    if (code) code.remove();
  }}
  function poll() {{
    if (++tries > 220) return;
    fetch("link-status?t=" + encodeURIComponent(token), {{cache: "no-store"}})
      .then(function (r) {{ return r.ok ? r.json() : {{}}; }})
      .then(function (body) {{
        if (done[body.status]) show(body.status); else setTimeout(poll, 3000);
      }})
      .catch(function () {{ setTimeout(poll, 3000); }});
  }}
  setTimeout(poll, 3000);
}})();
</script>"""

_BUTTON = (
    '<p><a href="{url}" style="display:inline-block;padding:.7rem 1.4rem;'
    "background:{color};color:#fff;text-decoration:none;border-radius:.5rem;"
    'font-weight:bold">Return to {app}</a></p>'
)


def _page(
    title: str,
    message: str,
    status: int = 200,
    telegram_url: str | None = None,
    whatsapp_url: str | None = None,
    googlechat_url: str | None = None,
    slack_url: str | None = None,
    code: str | None = None,
    status_token: str = "",
) -> HTMLResponse:
    """Render the callback page; with a chat deep link it gains a return
    button. Without a code to copy it also auto-redirects, so the browser
    offers its native "Open Telegram?" prompt."""
    head_extra = ""
    action = ""
    target = telegram_url or whatsapp_url or googlechat_url or slack_url
    if target:
        safe_url = escape(target, quote=True)
        if code is None:
            head_extra = f'<meta http-equiv="refresh" content="2;url={safe_url}">'
        app, color = (
            ("Telegram", "#2AABEE")
            if telegram_url
            else ("WhatsApp", "#25D366")
            if whatsapp_url
            else ("Google Chat", "#1A73E8")
            if googlechat_url
            else ("Slack", "#4A154B")
        )
        action = _BUTTON.format(url=safe_url, color=color, app=app)
    html = _PAGE.format(
        title=escape(title),
        message=escape(message),
        head_extra=head_extra,
        code=_CODE.format(code=escape(code)) if code else "",
        action=action,
        # json.dumps quotes the token for JS; token_urlsafe never contains "<".
        script=_POLL.format(token=json.dumps(status_token)) if code and status_token else "",
    )
    return HTMLResponse(content=html, status_code=status)


def build_app(
    store: BindingStore,
    settings: Settings,
    on_bound: OnBound | None = None,
    bot_username: str | None = None,
    on_whatsapp: OnWhatsApp | None = None,
    whatsapp_number: str = "",
    on_googlechat: OnGoogleChat | None = None,
    verify_googlechat: VerifyGoogleChat | None = None,
    on_slack: OnSlack | None = None,
    slack_url: str | None = None,
) -> FastAPI:
    telegram_url = f"https://t.me/{bot_username}" if bot_username else None
    # An explicit WHATSAPP_BUSINESS_NUMBER wins; otherwise the caller may have
    # looked the number up from Meta.
    number = settings.whatsapp_business_number or whatsapp_number
    whatsapp_url = f"https://wa.me/{number}" if number else None
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    login_attempts: dict[bytes, asyncio.Task[LinkCode]] = {}
    login_attempts_lock = asyncio.Lock()

    async def complete_once(state: str, code: str) -> LinkCode:
        key = hashlib.sha256(state.encode()).digest()
        async with login_attempts_lock:
            task = login_attempts.get(key)
            if task is None:

                async def complete_and_notify() -> LinkCode:
                    binding = await complete_login(store, settings, state, code)
                    if on_bound is not None:
                        try:
                            await on_bound(binding)
                        except Exception:
                            logger.error("on_bound notification failed", exc_info=True)
                    return binding

                task = asyncio.create_task(complete_and_notify())
                login_attempts[key] = task

        try:
            bound = await asyncio.shield(task)
        except Exception:
            async with login_attempts_lock:
                if login_attempts.get(key) is task:
                    login_attempts.pop(key)
            raise

        async with login_attempts_lock:
            while len(login_attempts) > 256:
                login_attempts.pop(next(iter(login_attempts)))
        return bound

    # Query params are read from the raw request, never declared as typed
    # FastAPI parameters: declared-required params would answer missing input
    # with 422 JSON instead of our 400 HTML pages.

    @app.get("/oauth/callback")
    async def oauth_callback(request: Request) -> HTMLResponse:
        error = request.query_params.get("error")
        if error:
            description = request.query_params.get("error_description", "")
            logger.warning("Authorization denied: %s %s", error, description)
            return _page(
                "Login not completed",
                "Odoo did not authorize this login. Return to Telegram and try /start again.",
                status=400,
            )

        code = request.query_params.get("code", "")
        state = request.query_params.get("state", "")
        if not code or not state:
            return _page("Invalid request", "Missing code or state.", status=400)

        try:
            bound = await complete_once(state, code)
        except LoginError as exc:
            logger.warning("Login completion failed: %s", exc)
            return _page("Login not completed", str(exc), status=400)
        except Exception:
            logger.error("Unexpected error completing login", exc_info=True)
            return _page(
                "Something went wrong",
                "The connection could not be completed. Try /start again in the chat.",
                status=500,
            )

        on_whatsapp_chat = is_whatsapp(bound.chat_id)
        on_google_chat = is_google_chat(bound.chat_id)
        on_slack_chat = is_slack(bound.chat_id)
        channel = (
            "Slack"
            if on_slack_chat
            else "Google Chat"
            if on_google_chat
            else "WhatsApp"
            if on_whatsapp_chat
            else "Telegram"
        )
        return _page(
            "Almost connected",
            f"Signed in as {bound.odoo_name}. To finish, send this code in your "
            f"{channel} chat with the assistant. It expires in 10 minutes. If you did "
            "not start this login from your own chat, close this page and do not share "
            "the code with anyone.",
            telegram_url=None
            if on_whatsapp_chat or on_google_chat or on_slack_chat
            else telegram_url,
            whatsapp_url=whatsapp_url if on_whatsapp_chat else None,
            googlechat_url=GOOGLE_CHAT_URL if on_google_chat else None,
            slack_url=slack_url if on_slack_chat else None,
            code=bound.code,
            status_token=bound.status_token,
        )

    @app.get("/oauth/link-status")
    async def oauth_link_status(request: Request) -> JSONResponse:
        status = await link_status(store, request.query_params.get("t", ""))
        return JSONResponse({"status": status}, headers={"Cache-Control": "no-store"})

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/static/avatar.png")
    async def avatar() -> FileResponse:
        return FileResponse(
            AVATAR_PATH, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"}
        )

    @app.get("/webhook/whatsapp")
    async def whatsapp_verify(request: Request) -> PlainTextResponse:
        # Meta's subscription handshake (the hub.* names contain dots, so
        # they could never be typed parameters anyway). An empty configured
        # token means "webhook disabled" and must never match an empty
        # hub.verify_token.
        mode = request.query_params.get("hub.mode", "")
        token = request.query_params.get("hub.verify_token", "")
        challenge = request.query_params.get("hub.challenge", "")
        if settings.whatsapp_verify_token and (
            mode == "subscribe" and token == settings.whatsapp_verify_token
        ):
            logger.info("WhatsApp webhook verification handshake accepted")
            return PlainTextResponse(challenge)
        logger.warning("WhatsApp webhook verification rejected (mode=%s)", mode)
        return PlainTextResponse("Forbidden", status_code=403)

    @app.post("/webhook/whatsapp")
    async def whatsapp_event(request: Request) -> Response:
        body = await request.body()
        # The signature is the only proof a POST came from Meta; with no app
        # secret configured nothing can be verified, so nothing is accepted.
        if not verify_signature(
            settings.whatsapp_app_secret, body, request.headers.get("X-Hub-Signature-256")
        ):
            logger.warning("WhatsApp webhook POST refused: bad or missing signature")
            return Response(status_code=403)
        if on_whatsapp is None:
            logger.info("WhatsApp webhook event received (%d bytes) — channel off", len(body))
            return Response(status_code=200)
        try:
            payload = json.loads(body)
        except ValueError:
            return Response(status_code=400)
        if not isinstance(payload, dict):
            return Response(status_code=400)
        try:
            on_whatsapp(payload)
        except Exception:
            # Still 200: Meta retries failures for hours, and replaying a
            # payload that crashed us once only crashes us again.
            logger.error("WhatsApp event handling failed", exc_info=True)
        return Response(status_code=200)

    @app.post("/webhook/googlechat")
    async def googlechat_event(request: Request) -> Response:
        # The bearer token is the only proof a POST came from Google Chat for
        # this app; with nothing to verify against, nothing is accepted.
        if verify_googlechat is None or on_googlechat is None:
            return Response(status_code=401)
        if not await verify_googlechat(request.headers.get("Authorization")):
            logger.warning("Google Chat POST refused: bad or missing token")
            return Response(status_code=401)
        try:
            payload = json.loads(await request.body())
        except ValueError:
            return Response(status_code=400)
        if not isinstance(payload, dict):
            return Response(status_code=400)
        try:
            on_googlechat(payload)
        except Exception:
            # Still acknowledged: Chat retries failures, and replaying a
            # payload that crashed us once only crashes us again.
            logger.error("Google Chat event handling failed", exc_info=True)
        # An empty object acknowledges the event; the answer follows via the API.
        return JSONResponse({})

    @app.post("/webhook/slack")
    async def slack_event(request: Request) -> Response:
        # The signature is the only proof a POST came from Slack for this app;
        # without a signing secret (Socket Mode) nothing can be verified.
        body = await request.body()
        if on_slack is None or not verify_slack_signature(
            settings.slack_signing_secret,
            body,
            request.headers.get("X-Slack-Request-Timestamp"),
            request.headers.get("X-Slack-Signature"),
        ):
            logger.warning("Slack POST refused: bad or missing signature")
            return Response(status_code=401)

        payload: dict[str, Any]
        try:
            if "json" in request.headers.get("content-type", ""):
                decoded = json.loads(body)
                if not isinstance(decoded, dict):
                    return Response(status_code=400)
                if decoded.get("type") == "url_verification":
                    return JSONResponse({"challenge": str(decoded.get("challenge") or "")})
                envelope, payload = ENVELOPE_EVENTS, decoded
            else:
                form = parse_qs(body.decode(), keep_blank_values=True)
                if "payload" in form:
                    decoded = json.loads(form["payload"][0])
                    if not isinstance(decoded, dict):
                        return Response(status_code=400)
                    envelope, payload = ENVELOPE_INTERACTIVE, decoded
                elif "command" in form:
                    envelope = ENVELOPE_COMMAND
                    payload = {key: values[0] for key, values in form.items()}
                else:
                    return Response(status_code=400)
        except ValueError:
            return Response(status_code=400)
        try:
            on_slack(envelope, payload)
        except Exception:
            # Still acknowledged: Slack retries failures, and replaying a
            # payload that crashed us once only crashes us again.
            logger.error("Slack event handling failed", exc_info=True)
        return Response(status_code=200)

    return app


class _EmbeddedServer(uvicorn.Server):
    """uvicorn server that leaves signal handlers alone.

    Stock ``Server.serve()`` captures SIGINT/SIGTERM and restores the
    pre-uvicorn handlers on exit — which would clobber python-telegram-bot's
    own signal handling in the shared event loop.
    """

    @contextlib.contextmanager
    def capture_signals(self) -> Generator[None]:
        yield


@dataclass
class ServerHandle:
    """A running embedded HTTP server; ``stop()`` shuts it down cleanly."""

    server: uvicorn.Server
    task: asyncio.Task[None]

    async def stop(self) -> None:
        self.server.should_exit = True
        try:
            await asyncio.wait_for(self.task, timeout=10)
        except TimeoutError:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task


async def start_server(app: FastAPI, host: str, port: int) -> ServerHandle:
    """Serve ``app`` on the current event loop; returns once it is listening.

    Deliberately independent of AGENT_PUBLIC_URL: the public URL is what
    browsers (and the OAuth redirect) use — often a tunnel — while this is
    where the process actually listens (localhost, behind that tunnel).
    """
    config = uvicorn.Config(app, host=host, port=port, log_config=None, access_log=False)
    server = _EmbeddedServer(config)
    task = asyncio.create_task(server.serve(), name="http-server")
    while not server.started:
        # A taken port makes uvicorn's startup sys.exit(1) inside the task;
        # without this check the wait would hang forever.
        if task.done():
            exc = task.exception()
            raise RuntimeError(f"HTTP server failed to start on {host}:{port}") from exc
        await asyncio.sleep(0.02)
    logger.info("HTTP server listening on %s:%d", host, port)
    return ServerHandle(server=server, task=task)
