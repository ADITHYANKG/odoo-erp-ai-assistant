"""Logging setup for the agent service.

Diagnostics go through :mod:`logging`; the user-facing report output of CLI
entry points stays on ``print``. Configure once at process start via
:func:`setup_logging`; every module gets its logger with :func:`get_logger`.

Environment:
    LOG_LEVEL   DEBUG | INFO | WARNING | ERROR  (default INFO)

Log files (AGENT_LOG_FILE, AGENT_BLOCKED_LOG_FILE) come from Settings and are
attached later with :func:`add_file_logs`.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

#: Logger for messages dropped by the Telegram allowlist.
BLOCKED_LOGGER = "ont_agent.blocked"
_FILE_MAX_BYTES = 5_000_000
_FILE_BACKUPS = 5

# Third-party loggers that flood DEBUG/INFO with per-request noise. They stay
# at WARNING unless the whole process is explicitly running at DEBUG.
_NOISY_LOGGERS = ("httpx", "httpcore", "mcp", "urllib3")

_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once; later calls are no-ops.

    :param level: overrides the ``LOG_LEVEL`` environment variable (tests).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    if resolved not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        resolved = "INFO"

    handler = logging.StreamHandler(sys.stderr)
    handler.set_name("ont_agent")  # identifiable among third-party root handlers
    handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(resolved)
    root.addHandler(handler)

    if resolved != "DEBUG":
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True


def add_file_logs(log_file: str = "", blocked_log_file: str = "") -> None:
    """Also write logs to rotating files; an empty path skips that file.

    ``log_file`` gets everything; ``blocked_log_file`` only allowlist drops.
    """
    if log_file:
        logging.getLogger().addHandler(_file_handler(log_file, "ont_agent.file"))
    if blocked_log_file:
        logging.getLogger(BLOCKED_LOGGER).addHandler(
            _file_handler(blocked_log_file, "ont_agent.blocked_file")
        )


def _file_handler(path: str, name: str) -> logging.Handler:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        path, maxBytes=_FILE_MAX_BYTES, backupCount=_FILE_BACKUPS, encoding="utf-8"
    )
    handler.set_name(name)
    handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))
    return handler


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger (``ont_agent.<module>``)."""
    return logging.getLogger(name)
