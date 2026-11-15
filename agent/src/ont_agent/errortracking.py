"""Error tracking: Sentry, or any Sentry-compatible server such as GlitchTip.

``init_error_tracking(settings)`` does nothing unless ``SENTRY_DSN`` is set;
the disabled path never imports the SDK. Errors only: tracing already lives
in Langfuse, so performance monitoring stays off.

What becomes an event: every ``logger.error(..., exc_info=True)`` the bot
already writes when a turn, brief, watcher, upload or voice note fails, and
any exception python-telegram-bot logs from a handler. Nothing in the
handlers has to call Sentry directly.

The bot carries employee data, so privacy is opt-out of everything:

- no request bodies (webhook payloads are users' messages), no local
  variables (frames hold tokens and Odoo records), no default PII;
- every string in an event or breadcrumb passes through
  :func:`ont_agent.redaction.redact_secrets` and has e-mail addresses masked;
- breadcrumbs come only from WARNING and above, not the INFO chatter that
  names chats and users.
"""

from __future__ import annotations

import logging
import os
import re
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from ont_agent.config import Settings
from ont_agent.log import get_logger
from ont_agent.redaction import redact_secrets

logger = get_logger(__name__)

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
EMAIL_MARKER = "[email]"


def scrub_text(text: str) -> str:
    """Remove credentials and e-mail addresses from one string."""
    cleaned, _ = redact_secrets(text)
    return _EMAIL.sub(EMAIL_MARKER, cleaned)


def scrub(value: Any) -> Any:
    """Recursively scrub every string inside an event structure."""
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, dict):
        return {key: scrub(item) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub(item) for item in value]
    if isinstance(value, tuple):
        return tuple(scrub(item) for item in value)
    return value


def _before_send(event: dict[str, Any], _hint: dict[str, Any]) -> dict[str, Any]:
    scrubbed: dict[str, Any] = scrub(event)
    return scrubbed


def _before_breadcrumb(crumb: dict[str, Any], _hint: dict[str, Any]) -> dict[str, Any]:
    scrubbed: dict[str, Any] = scrub(crumb)
    return scrubbed


def _release() -> str:
    """``SENTRY_RELEASE`` (e.g. a git sha from CI), else the package version."""
    explicit = os.environ.get("SENTRY_RELEASE")
    if explicit:
        return explicit
    try:
        return f"ont-agent@{version('ont-agent')}"
    except PackageNotFoundError:
        return "ont-agent@unknown"


def init_error_tracking(settings: Settings) -> bool:
    """Start the Sentry SDK when a DSN is configured; never raises.

    :return: True when error reporting is active.
    """
    if not settings.sentry_enabled:
        logger.info("Error tracking disabled (SENTRY_DSN not set)")
        return False
    try:
        import sentry_sdk
        from sentry_sdk.integrations.logging import LoggingIntegration

        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            environment=settings.sentry_environment,
            release=_release(),
            # Every category spelled out: an omitted one defaults to ON.
            data_collection={
                "user_info": False,
                "cookies": {"mode": "off"},
                "http_headers": {"request": {"mode": "off"}},
                "http_bodies": [],
                "url_query_params": {"mode": "off"},
                "graphql": {"document": False, "variables": False},
                "gen_ai": {"inputs": False, "outputs": False},
                "database_query_data": False,
                "queues": False,
                "stack_frame_variables": False,
            },
            max_request_body_size="never",
            traces_sample_rate=None,
            max_breadcrumbs=30,
            integrations=[
                LoggingIntegration(level=logging.WARNING, event_level=logging.ERROR),
            ],
            before_send=_before_send,  # type: ignore[arg-type]
            before_breadcrumb=_before_breadcrumb,
        )
    except Exception:
        # A bad DSN must not stop the bot answering.
        logger.warning("Error tracking could not start", exc_info=True)
        return False
    logger.info("Error tracking enabled (environment=%s)", settings.sentry_environment)
    return True
