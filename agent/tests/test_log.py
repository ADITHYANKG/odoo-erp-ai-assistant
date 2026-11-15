"""Unit tests for ont_agent.log."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from ont_agent import log


@pytest.fixture(autouse=True)
def _reset_logging(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give every test a pristine logging world."""
    monkeypatch.setattr(log, "_CONFIGURED", False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    root = logging.getLogger()
    old_handlers, old_level = root.handlers[:], root.level
    root.handlers = []
    for name in log._NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.NOTSET)
    yield
    root.handlers, root.level = old_handlers, old_level


def test_defaults_to_info() -> None:
    log.setup_logging()
    assert logging.getLogger().level == logging.INFO


def test_respects_log_level_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "debug")  # case-insensitive
    log.setup_logging()
    assert logging.getLogger().level == logging.DEBUG


def test_invalid_level_falls_back_to_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "LOUD")
    log.setup_logging()
    assert logging.getLogger().level == logging.INFO


def _our_handlers() -> list[logging.Handler]:
    # pytest's logging plugin attaches its own handlers to the root logger,
    # so count only the one setup_logging names.
    return [h for h in logging.getLogger().handlers if h.get_name() == "ont_agent"]


def test_idempotent_no_duplicate_handlers() -> None:
    log.setup_logging()
    log.setup_logging()
    log.setup_logging()
    assert len(_our_handlers()) == 1


def test_noisy_libraries_quieted_at_info() -> None:
    log.setup_logging()
    assert logging.getLogger("httpx").level == logging.WARNING


def test_noisy_libraries_open_at_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    log.setup_logging()
    assert logging.getLogger("httpx").level == logging.NOTSET


def test_get_logger_namespacing() -> None:
    assert log.get_logger("ont_agent.x").name == "ont_agent.x"


def test_file_logs_split_blocked_messages_into_their_own_file(tmp_path) -> None:
    main_file, blocked_file = tmp_path / "logs" / "agent.log", tmp_path / "blocked.log"
    blocked = logging.getLogger(log.BLOCKED_LOGGER)
    old_blocked_handlers = blocked.handlers[:]
    log.setup_logging()
    try:
        log.add_file_logs(str(main_file), str(blocked_file))
        logging.getLogger("ont_agent.bot").warning("ordinary line")
        blocked.warning("Blocked message from Telegram id 42")
    finally:
        for handler in logging.getLogger().handlers + blocked.handlers:
            if handler.get_name() in ("ont_agent.file", "ont_agent.blocked_file"):
                handler.close()
        blocked.handlers = old_blocked_handlers

    main_text, blocked_text = main_file.read_text("utf-8"), blocked_file.read_text("utf-8")
    assert "ordinary line" in main_text and "id 42" in main_text
    assert "id 42" in blocked_text and "ordinary line" not in blocked_text


def test_file_logs_skip_empty_paths() -> None:
    log.add_file_logs("", "")
    assert not [h for h in logging.getLogger().handlers if h.get_name() == "ont_agent.file"]
