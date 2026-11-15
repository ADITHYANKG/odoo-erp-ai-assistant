"""Unit tests for ont_agent.config."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ont_agent.config import DEFAULT_OPENROUTER_MODEL, Settings, load_settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's real environment out of every test."""
    monkeypatch.delenv("MCP_URL", raising=False)
    monkeypatch.delenv("MCP_BEARER_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)


#: Every load_settings() call needs this now, and almost nothing in this
#: file is about it — so the helper supplies one unless the test writes its
#: own. The test that checks the requirement itself bypasses the helper.
_TEST_DSN = "postgresql://ont_agent:ont_agent@127.0.0.1:5566/ont_agent"


def _env_file(tmp_path: Path, content: str) -> Path:
    path = tmp_path / ".env"
    if "AGENT_DB_URL" not in content:
        content = f"AGENT_DB_URL={_TEST_DSN}\n" + content
    path.write_text(content, encoding="utf-8")
    return path


def test_reads_values_from_env_file(tmp_path: Path) -> None:
    env = _env_file(tmp_path, "MCP_URL=http://example:9999/mcp\nMCP_BEARER_KEY=abc123\n")
    settings = load_settings(env)
    assert settings.mcp_url == "http://example:9999/mcp"
    assert settings.mcp_bearer_key == "abc123"
    # Browser-facing Odoo base derives from the MCP URL when not set.
    assert settings.odoo_public_url == "http://example:9999"


def test_url_defaults_to_localhost_and_strips_trailing_slash(tmp_path: Path) -> None:
    env = _env_file(tmp_path, "MCP_BEARER_KEY=abc123\n")
    assert load_settings(env).mcp_url == "http://localhost:8069/mcp"

    env2 = _env_file(tmp_path, "MCP_URL=http://host/mcp/\nMCP_BEARER_KEY=abc123\n")
    assert load_settings(env2).mcp_url == "http://host/mcp"


def test_process_env_overrides_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = _env_file(tmp_path, "MCP_URL=http://from-file/mcp\nMCP_BEARER_KEY=file-key\n")
    monkeypatch.setenv("MCP_BEARER_KEY", "env-key")
    settings = load_settings(env)
    assert settings.mcp_bearer_key == "env-key"  # env wins
    assert settings.mcp_url == "http://from-file/mcp"  # file still supplies the rest


def test_missing_key_exits_with_actionable_message(tmp_path: Path) -> None:
    env = _env_file(tmp_path, "MCP_URL=http://host/mcp\n")
    with pytest.raises(SystemExit) as excinfo:
        load_settings(env)
    assert "MCP_BEARER_KEY" in str(excinfo.value)
    assert "4_create_mcp_key" in str(excinfo.value)  # tells the user how to fix it


def test_whitespace_only_key_treated_as_missing(tmp_path: Path) -> None:
    env = _env_file(tmp_path, "MCP_BEARER_KEY=   \n")
    with pytest.raises(SystemExit):
        load_settings(env)


def test_openrouter_fields_load_with_model_default(tmp_path: Path) -> None:
    env = _env_file(tmp_path, "MCP_BEARER_KEY=abc\nOPENROUTER_API_KEY=sk-or-x\n")
    settings = load_settings(env)
    assert settings.openrouter_api_key == "sk-or-x"
    assert settings.openrouter_model == DEFAULT_OPENROUTER_MODEL


def test_fallback_models_parse_from_comma_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_FALLBACK_MODELS", raising=False)
    env = _env_file(
        tmp_path,
        "MCP_BEARER_KEY=abc\n"
        "OPENROUTER_FALLBACK_MODELS= deepseek/deepseek-v4-flash-0731 , b/model ,\n",
    )
    assert load_settings(env).openrouter_fallback_models == [
        "deepseek/deepseek-v4-flash-0731",
        "b/model",
    ]

    env2 = _env_file(tmp_path, "MCP_BEARER_KEY=abc\n")
    assert load_settings(env2).openrouter_fallback_models == []


def test_openrouter_model_is_runtime_config(tmp_path: Path) -> None:
    env = _env_file(
        tmp_path,
        "MCP_BEARER_KEY=abc\nOPENROUTER_API_KEY=sk-or-x\nOPENROUTER_MODEL=meta/some-llm\n",
    )
    assert load_settings(env).openrouter_model == "meta/some-llm"


def test_llm_key_optional_for_tool_only_entry_points(tmp_path: Path) -> None:
    # Phase-1 lister must keep working without any OpenRouter configuration.
    env = _env_file(tmp_path, "MCP_BEARER_KEY=abc\n")
    settings = load_settings(env)  # require_llm defaults to False
    assert settings.openrouter_api_key == ""


def test_require_llm_exits_when_key_missing(tmp_path: Path) -> None:
    env = _env_file(tmp_path, "MCP_BEARER_KEY=abc\n")
    with pytest.raises(SystemExit) as excinfo:
        load_settings(env, require_llm=True)
    assert "OPENROUTER_API_KEY" in str(excinfo.value)


def test_require_telegram_reports_all_missing_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("OAUTH_CLIENT_ID", raising=False)
    env = _env_file(tmp_path, "MCP_BEARER_KEY=abc\n")
    with pytest.raises(SystemExit) as excinfo:
        load_settings(env, require_llm=True, require_telegram=True)
    message = str(excinfo.value)
    # One failure lists every missing variable, not just the first.
    assert "OPENROUTER_API_KEY" in message
    assert "TELEGRAM_BOT_TOKEN" in message
    assert "OAUTH_CLIENT_ID" in message


def test_telegram_settings_load_and_urls_derive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("OAUTH_CLIENT_ID", raising=False)
    monkeypatch.delenv("AGENT_PUBLIC_URL", raising=False)
    env = _env_file(
        tmp_path,
        "MCP_BEARER_KEY=abc\nOPENROUTER_API_KEY=sk\n"
        "TELEGRAM_BOT_TOKEN=123:tok\nOAUTH_CLIENT_ID=ont_x\n"
        "AGENT_PUBLIC_URL=http://cb:9999/\n",
    )
    settings = load_settings(env, require_llm=True, require_telegram=True)
    assert settings.telegram_bot_token == "123:tok"
    assert settings.oauth_client_id == "ont_x"
    assert settings.agent_public_url == "http://cb:9999"  # trailing slash stripped
    assert settings.redirect_uri == "http://cb:9999/oauth/callback"
    assert settings.token_url == "http://localhost:8069/mcp/oauth/token"
    # Bind address is independent of the public URL (which may be a tunnel).
    assert settings.callback_host == "localhost"
    assert settings.callback_port == 8181


def test_callback_bind_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CALLBACK_HOST", raising=False)
    monkeypatch.delenv("CALLBACK_PORT", raising=False)
    env = _env_file(
        tmp_path,
        "MCP_BEARER_KEY=abc\nCALLBACK_HOST=0.0.0.0\nCALLBACK_PORT=9000\n",
    )
    settings = load_settings(env)
    assert settings.callback_host == "0.0.0.0"
    assert settings.callback_port == 9000


def test_recursion_limit_default_and_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AGENT_RECURSION_LIMIT", raising=False)
    env = _env_file(tmp_path, "MCP_BEARER_KEY=abc\n")
    assert load_settings(env).recursion_limit == 16

    env2 = _env_file(tmp_path, "MCP_BEARER_KEY=abc\nAGENT_RECURSION_LIMIT=40\n")
    assert load_settings(env2).recursion_limit == 40


def test_token_streaming_defaults_on_and_can_be_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AGENT_TOKEN_STREAMING", raising=False)
    env = _env_file(tmp_path, "MCP_BEARER_KEY=abc\n")
    assert load_settings(env).token_streaming is True

    env2 = _env_file(tmp_path, "MCP_BEARER_KEY=abc\nAGENT_TOKEN_STREAMING=false\n")
    assert load_settings(env2).token_streaming is False


def test_invalid_ints_exit_with_aggregated_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Garbage numbers must land in the same clean exit as missing variables,
    # never a ValueError traceback.
    monkeypatch.delenv("CALLBACK_PORT", raising=False)
    monkeypatch.delenv("AGENT_RECURSION_LIMIT", raising=False)
    env = _env_file(
        tmp_path,
        "MCP_BEARER_KEY=abc\nCALLBACK_PORT=eight\nAGENT_RECURSION_LIMIT=lots\n",
    )
    with pytest.raises(SystemExit) as excinfo:
        load_settings(env)
    message = str(excinfo.value)
    assert "CALLBACK_PORT" in message
    assert "AGENT_RECURSION_LIMIT" in message


def test_daily_message_limit_default_and_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AGENT_DAILY_MESSAGE_LIMIT", raising=False)
    env = _env_file(tmp_path, "MCP_BEARER_KEY=abc\n")
    assert load_settings(env).daily_message_limit == 200

    env2 = _env_file(tmp_path, "MCP_BEARER_KEY=abc\nAGENT_DAILY_MESSAGE_LIMIT=5\n")
    assert load_settings(env2).daily_message_limit == 5


def test_daily_spend_limit_unset_is_none_set_is_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AGENT_DAILY_SPEND_LIMIT_USD", raising=False)
    env = _env_file(tmp_path, "MCP_BEARER_KEY=abc\n")
    assert load_settings(env).daily_spend_limit_usd is None

    env2 = _env_file(tmp_path, "MCP_BEARER_KEY=abc\nAGENT_DAILY_SPEND_LIMIT_USD=2.50\n")
    assert load_settings(env2).daily_spend_limit_usd == 2.50


def test_long_term_memory_settings_defaults_and_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in ("AGENT_LONG_TERM_MEMORY", "AGENT_MEMORY_EMBED_MODEL"):
        monkeypatch.delenv(key, raising=False)
    env = _env_file(tmp_path, "MCP_BEARER_KEY=abc\n")
    settings = load_settings(env)
    assert settings.long_term_memory is True
    assert settings.memory_embed_model == "openai/text-embedding-3-small"
    assert settings.memory_embed_dims == 1536
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"

    env2 = _env_file(
        tmp_path,
        "MCP_BEARER_KEY=abc\nAGENT_LONG_TERM_MEMORY=0\nAGENT_MEMORY_EMBED_MODEL=\n",
    )
    overridden = load_settings(env2)
    assert overridden.long_term_memory is False
    assert overridden.memory_embed_model == ""  # empty disables vector search


def test_embedding_dims_and_base_url_are_configurable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A different embedding model has a different vector width; both must
    # come from config or the index silently corrupts.
    for key in ("AGENT_MEMORY_EMBED_DIMS", "OPENROUTER_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    env = _env_file(
        tmp_path,
        "MCP_BEARER_KEY=abc\nAGENT_MEMORY_EMBED_DIMS=1024\n"
        "OPENROUTER_BASE_URL=http://proxy.internal/v1/\n",
    )
    settings = load_settings(env)
    assert settings.memory_embed_dims == 1024
    assert settings.openrouter_base_url == "http://proxy.internal/v1"  # slash stripped


def test_recall_cap_and_summarization_thresholds_are_configurable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in (
        "AGENT_MEMORY_MAX_RECALL",
        "AGENT_SUMMARY_TRIGGER_TOKENS",
        "AGENT_SUMMARY_KEEP_MESSAGES",
    ):
        monkeypatch.delenv(key, raising=False)
    defaults = load_settings(_env_file(tmp_path, "MCP_BEARER_KEY=abc\n"))
    assert defaults.memory_max_recall == 10
    assert defaults.summary_trigger_tokens == 60_000
    assert defaults.summary_keep_messages == 20

    env = _env_file(
        tmp_path,
        "MCP_BEARER_KEY=abc\nAGENT_MEMORY_MAX_RECALL=25\n"
        "AGENT_SUMMARY_TRIGGER_TOKENS=90000\nAGENT_SUMMARY_KEEP_MESSAGES=30\n",
    )
    tuned = load_settings(env)
    assert tuned.memory_max_recall == 25
    assert tuned.summary_trigger_tokens == 90_000
    assert tuned.summary_keep_messages == 30


def test_malformed_spend_limit_exits_with_actionable_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AGENT_DAILY_SPEND_LIMIT_USD", raising=False)
    env = _env_file(tmp_path, "MCP_BEARER_KEY=abc\nAGENT_DAILY_SPEND_LIMIT_USD=free\n")
    with pytest.raises(SystemExit) as excinfo:
        load_settings(env)
    assert "AGENT_DAILY_SPEND_LIMIT_USD" in str(excinfo.value)


def test_direct_settings_construction_ignores_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Settings is a plain pydantic model, NOT BaseSettings: constructing it
    # directly (as every test module does) must never read the developer's
    # real environment.
    monkeypatch.setenv("CALLBACK_PORT", "1")
    settings = Settings(mcp_url="http://x/mcp", mcp_bearer_key="k")
    assert settings.callback_port == 8181


def test_settings_is_frozen() -> None:
    settings = Settings(mcp_url="http://x/mcp", mcp_bearer_key="k")
    with pytest.raises(ValidationError):
        settings.mcp_url = "http://other/mcp"


# --- abuse controls --------------------------------------------------------

_ABUSE_KEYS = (
    "AGENT_GLOBAL_DAILY_SPEND_LIMIT_USD",
    "AGENT_BURST_LIMIT",
    "AGENT_BURST_WINDOW_SECONDS",
    "AGENT_ALLOWED_TELEGRAM_IDS",
    "AGENT_REQUIRED_ODOO_GROUP",
)


def _clear_abuse(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ABUSE_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_abuse_control_defaults_are_permissive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every new control defaults OFF except the burst guard, so upgrading
    # cannot silently lock an existing deployment out of its own bot.
    _clear_abuse(monkeypatch)
    settings = load_settings(_env_file(tmp_path, "MCP_BEARER_KEY=abc\n"))
    assert settings.global_daily_spend_limit_usd is None
    assert settings.allowed_telegram_ids == []
    assert settings.required_odoo_group == ""
    assert settings.burst_limit == 10
    assert settings.burst_window_seconds == 60


def test_abuse_controls_read_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_abuse(monkeypatch)
    env = _env_file(
        tmp_path,
        "MCP_BEARER_KEY=abc\n"
        "AGENT_GLOBAL_DAILY_SPEND_LIMIT_USD=25.00\n"
        "AGENT_BURST_LIMIT=3\n"
        "AGENT_BURST_WINDOW_SECONDS=30\n"
        "AGENT_ALLOWED_TELEGRAM_IDS=123, 456 ,789\n"
        'AGENT_REQUIRED_ODOO_GROUP="AI Assistant Users"\n',
    )
    settings = load_settings(env)
    assert settings.global_daily_spend_limit_usd == 25.00
    assert settings.burst_limit == 3
    assert settings.burst_window_seconds == 30
    # Whitespace around ids is tolerated; a pasted list should just work.
    assert settings.allowed_telegram_ids == [123, 456, 789]
    assert settings.required_odoo_group == "AI Assistant Users"


def test_malformed_telegram_id_list_exits_with_actionable_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Must FAIL rather than silently allow everyone: a typo in an allowlist
    # that quietly disables the allowlist is the worst possible outcome.
    _clear_abuse(monkeypatch)
    env = _env_file(tmp_path, "MCP_BEARER_KEY=abc\nAGENT_ALLOWED_TELEGRAM_IDS=123,@subin\n")
    with pytest.raises(SystemExit) as excinfo:
        load_settings(env)
    assert "AGENT_ALLOWED_TELEGRAM_IDS" in str(excinfo.value)


def test_log_files_resolve_under_the_agent_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AGENT_LOG_FILE", raising=False)
    monkeypatch.delenv("AGENT_BLOCKED_LOG_FILE", raising=False)
    assert load_settings(_env_file(tmp_path, "MCP_BEARER_KEY=abc\n")).log_file == ""

    env = _env_file(
        tmp_path,
        "MCP_BEARER_KEY=abc\nAGENT_LOG_FILE=logs/agent.log\nAGENT_BLOCKED_LOG_FILE=logs/blocked.log\n",
    )
    settings = load_settings(env)
    assert Path(settings.log_file).is_absolute()
    repo_root = Path(__file__).resolve().parents[1]
    assert Path(settings.log_file) == repo_root / "logs" / "agent.log"
    assert Path(settings.blocked_log_file).name == "blocked.log"


# --- proactive settings ----------------------------------------------------

_PROACTIVE_KEYS = (
    "AGENT_PROACTIVE",
    "AGENT_BRIEF_HOUR",
    "AGENT_BRIEF_MINUTE",
    "AGENT_BRIEF_CATCHUP_HOURS",
    "AGENT_DEFAULT_TIMEZONE",
    "AGENT_WATCHER_INTERVAL_MINUTES",
    "AGENT_MAX_WATCHERS_PER_USER",
    "AGENT_TICK_SECONDS",
)


def test_proactive_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _PROACTIVE_KEYS:
        monkeypatch.delenv(key, raising=False)
    settings = load_settings(_env_file(tmp_path, "MCP_BEARER_KEY=abc\n"))
    assert settings.proactive is True
    assert (settings.brief_hour, settings.brief_minute) == (8, 0)
    assert settings.brief_catchup_hours == 4
    assert settings.default_timezone == "Asia/Kolkata"
    assert settings.watcher_interval_minutes == 480  # 8 hours
    assert settings.max_watchers_per_user == 5
    assert settings.tick_seconds == 300


def test_proactive_settings_read_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _PROACTIVE_KEYS:
        monkeypatch.delenv(key, raising=False)
    env = _env_file(
        tmp_path,
        "MCP_BEARER_KEY=abc\n"
        "AGENT_PROACTIVE=0\n"
        "AGENT_BRIEF_HOUR=6\n"
        "AGENT_BRIEF_MINUTE=30\n"
        "AGENT_BRIEF_CATCHUP_HOURS=0\n"
        "AGENT_DEFAULT_TIMEZONE=Europe/London\n"
        "AGENT_WATCHER_INTERVAL_MINUTES=15\n"
        "AGENT_MAX_WATCHERS_PER_USER=2\n"
        "AGENT_TICK_SECONDS=60\n",
    )
    settings = load_settings(env)
    assert settings.proactive is False
    assert (settings.brief_hour, settings.brief_minute) == (6, 30)
    assert settings.brief_catchup_hours == 0
    assert settings.default_timezone == "Europe/London"
    assert settings.watcher_interval_minutes == 15
    assert settings.max_watchers_per_user == 2
    assert settings.tick_seconds == 60


def test_too_many_fallback_models_fails_at_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fourth fallback made EVERY question fail with "something went
    wrong" (live, 2026-09-01): OpenRouter rejects the whole request with
    "'models' array must have 3 items or fewer". This module promises to die
    at startup rather than midway through a conversation, so catch it here.
    """
    monkeypatch.delenv("OPENROUTER_FALLBACK_MODELS", raising=False)
    env = _env_file(
        tmp_path,
        "MCP_BEARER_KEY=abc\nOPENROUTER_FALLBACK_MODELS=a/one,b/two,c/three,d/four\n",
    )
    with pytest.raises(SystemExit) as excinfo:
        load_settings(env)

    message = str(excinfo.value)
    assert "OPENROUTER_FALLBACK_MODELS" in message
    assert "at most 3" in message
    assert "Remove 1" in message  # actionable, not just "invalid"


def test_three_fallback_models_are_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Three is the documented maximum, and the primary does not count.
    monkeypatch.delenv("OPENROUTER_FALLBACK_MODELS", raising=False)
    env = _env_file(
        tmp_path,
        "MCP_BEARER_KEY=abc\nOPENROUTER_FALLBACK_MODELS=a/one,b/two,c/three\n",
    )
    assert load_settings(env).openrouter_fallback_models == ["a/one", "b/two", "c/three"]


def test_a_malformed_token_key_stops_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bad key is a config error, and this module promises to die at
    startup rather than midway through a conversation."""
    monkeypatch.delenv("AGENT_TOKEN_KEY", raising=False)
    env = _env_file(tmp_path, "MCP_BEARER_KEY=abc\nAGENT_TOKEN_KEY=not-a-key\n")
    with pytest.raises(SystemExit) as excinfo:
        load_settings(env)

    message = str(excinfo.value)
    assert "AGENT_TOKEN_KEY" in message
    assert "Fernet.generate_key" in message  # tells you how to fix it


def test_a_valid_token_key_is_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cryptography.fernet import Fernet

    monkeypatch.delenv("AGENT_TOKEN_KEY", raising=False)
    key = Fernet.generate_key().decode()
    env = _env_file(tmp_path, f"MCP_BEARER_KEY=abc\nAGENT_TOKEN_KEY={key}\n")
    assert load_settings(env).token_key == key


def test_no_token_key_is_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Encryption is opt-in so it never blocks a first run or the test suite.
    monkeypatch.delenv("AGENT_TOKEN_KEY", raising=False)
    assert load_settings(_env_file(tmp_path, "MCP_BEARER_KEY=abc\n")).token_key == ""


# --- uploaded images and PDF files ------------------------------------------


def test_upload_settings_default_sensibly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("AGENT_UPLOADS", "AGENT_UPLOAD_MAX_MB", "AGENT_UPLOAD_FALLBACK_MODELS"):
        monkeypatch.delenv(name, raising=False)
    settings = load_settings(_env_file(tmp_path, "MCP_BEARER_KEY=abc\n"))

    assert settings.uploads is True
    assert settings.upload_max_mb == 5
    # Empty on purpose: a file must never be routed to a model that cannot
    # read it, and no default model name is safe to assume.
    assert settings.upload_fallback_models == []
    assert settings.upload_pdf_engine == ""


def test_upload_settings_can_be_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("AGENT_UPLOADS", "AGENT_UPLOAD_MAX_MB", "AGENT_UPLOAD_FALLBACK_MODELS"):
        monkeypatch.delenv(name, raising=False)
    env = _env_file(
        tmp_path,
        "MCP_BEARER_KEY=abc\n"
        "AGENT_UPLOADS=0\n"
        "AGENT_UPLOAD_MAX_MB=12\n"
        "AGENT_UPLOAD_FALLBACK_MODELS=a/vision, b/vision\n"
        "AGENT_UPLOAD_PDF_ENGINE=mistral-ocr\n",
    )
    settings = load_settings(env)

    assert settings.uploads is False
    assert settings.upload_max_mb == 12
    assert settings.upload_fallback_models == ["a/vision", "b/vision"]
    assert settings.upload_pdf_engine == "mistral-ocr"


def test_too_many_upload_fallbacks_stop_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same three-model ceiling as the text list: OpenRouter rejects the
    whole request, so a fourth entry would break every upload rather than
    degrade it."""
    monkeypatch.delenv("AGENT_UPLOAD_FALLBACK_MODELS", raising=False)
    env = _env_file(
        tmp_path,
        "MCP_BEARER_KEY=abc\nAGENT_UPLOAD_FALLBACK_MODELS=a/1,b/2,c/3,d/4\n",
    )
    with pytest.raises(SystemExit) as excinfo:
        load_settings(env)

    message = str(excinfo.value)
    assert "AGENT_UPLOAD_FALLBACK_MODELS" in message
    assert "at most 3" in message


# --- Postgres for conversation memory and preferences ---------------------


def test_a_missing_database_url_stops_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """There is no SQLite fallback any more, so a missing DSN has to fail
    here rather than at the first query — and the message must say how to
    fix it, because "connection refused" three seconds later would not."""
    monkeypatch.delenv("AGENT_DB_URL", raising=False)
    path = tmp_path / ".env"
    path.write_text("MCP_BEARER_KEY=abc\n", encoding="utf-8")  # bypass the helper

    with pytest.raises(SystemExit) as excinfo:
        load_settings(path)

    message = str(excinfo.value)
    assert "AGENT_DB_URL" in message
    assert "docker compose" in message


def test_agent_db_url_is_read_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AGENT_DB_URL", raising=False)
    dsn = "postgresql://ont_agent:ont_agent@localhost:5565/ont_agent"
    env = _env_file(tmp_path, f"MCP_BEARER_KEY=abc\nAGENT_DB_URL={dsn}\n")

    assert load_settings(env).agent_db_url == dsn


def test_token_refresh_margin_defaults_wide_enough_for_a_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A token is checked once at the start of a turn and then used for every
    tool call in it, and nothing recovers from a 401 partway through — so the
    margin has to outlast a realistic turn, not a median one."""
    monkeypatch.delenv("AGENT_TOKEN_REFRESH_MARGIN_SECONDS", raising=False)
    settings = load_settings(_env_file(tmp_path, "MCP_BEARER_KEY=abc\n"))

    assert settings.token_refresh_margin_seconds == 300


def test_token_refresh_margin_is_configurable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AGENT_TOKEN_REFRESH_MARGIN_SECONDS", raising=False)
    env = _env_file(tmp_path, "MCP_BEARER_KEY=abc\nAGENT_TOKEN_REFRESH_MARGIN_SECONDS=120\n")

    assert load_settings(env).token_refresh_margin_seconds == 120


def test_retention_windows_default_sensibly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in ("AGENT_CONVERSATION_RETENTION_DAYS", "AGENT_USAGE_RETENTION_DAYS"):
        monkeypatch.delenv(key, raising=False)
    settings = load_settings(_env_file(tmp_path, "MCP_BEARER_KEY=abc\n"))

    # 30 days matches the OAuth server's refresh_token_ttl_days: a user idle
    # that long has a dead login anyway, so retention can never discard a
    # conversation a working login could still have used.
    assert settings.conversation_retention_days == 30
    assert settings.usage_retention_days == 90


def test_retention_windows_are_configurable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in ("AGENT_CONVERSATION_RETENTION_DAYS", "AGENT_USAGE_RETENTION_DAYS"):
        monkeypatch.delenv(key, raising=False)
    env = _env_file(
        tmp_path,
        "MCP_BEARER_KEY=abc\nAGENT_CONVERSATION_RETENTION_DAYS=7\nAGENT_USAGE_RETENTION_DAYS=0\n",
    )
    settings = load_settings(env)

    assert settings.conversation_retention_days == 7
    assert settings.usage_retention_days == 0  # 0 disables that half


def test_thread_rotation_threshold_defaults_and_configures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AGENT_THREAD_MAX_CHECKPOINTS", raising=False)
    settings = load_settings(_env_file(tmp_path, "MCP_BEARER_KEY=abc\n"))
    # Retention only reaches IDLE threads, so an active user needs this or
    # their one thread grows forever. ~33 turns, ~10 MB.
    assert settings.thread_max_checkpoints == 100

    off = load_settings(
        _env_file(tmp_path, "MCP_BEARER_KEY=abc\nAGENT_THREAD_MAX_CHECKPOINTS=0\n")
    )
    assert off.thread_max_checkpoints == 0  # disables rotation


def test_pool_sizes_and_group_notice_window_default_and_configure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in (
        "AGENT_DB_POOL_MIN_SIZE",
        "AGENT_DB_POOL_MAX_SIZE",
        "AGENT_STORE_POOL_MAX_SIZE",
        "AGENT_GROUP_NOTICE_TTL_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = load_settings(_env_file(tmp_path, "MCP_BEARER_KEY=abc\n"))

    assert settings.db_pool_min_size == 1
    assert settings.db_pool_max_size == 8
    # Smaller on purpose: short queries, where LangGraph does the heavy
    # checkpoint writes.
    assert settings.store_pool_max_size == 4
    assert settings.group_notice_ttl_seconds == 3600

    tuned = load_settings(
        _env_file(
            tmp_path,
            "MCP_BEARER_KEY=abc\n"
            "AGENT_DB_POOL_MIN_SIZE=2\n"
            "AGENT_DB_POOL_MAX_SIZE=16\n"
            "AGENT_STORE_POOL_MAX_SIZE=6\n"
            "AGENT_GROUP_NOTICE_TTL_SECONDS=60\n",
        )
    )
    assert tuned.db_pool_min_size == 2
    assert tuned.db_pool_max_size == 16
    assert tuned.store_pool_max_size == 6
    assert tuned.group_notice_ttl_seconds == 60


def test_quiet_hours_default_and_configure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in ("AGENT_QUIET_HOURS_START", "AGENT_QUIET_HOURS_END"):
        monkeypatch.delenv(key, raising=False)
    settings = load_settings(_env_file(tmp_path, "MCP_BEARER_KEY=abc\n"))

    # 21:00-08:00 local, wrapping midnight.
    assert settings.quiet_hours_start == 21
    assert settings.quiet_hours_end == 8

    off = load_settings(
        _env_file(
            tmp_path,
            "MCP_BEARER_KEY=abc\nAGENT_QUIET_HOURS_START=0\nAGENT_QUIET_HOURS_END=0\n",
        )
    )
    # Equal values disable it, so an existing deployment is unchanged.
    assert off.quiet_hours_start == off.quiet_hours_end


def test_prompt_guard_defaults_and_configures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in ("AGENT_PROMPT_GUARD", "AGENT_PROMPT_LEAK_NGRAM"):
        monkeypatch.delenv(key, raising=False)
    settings = load_settings(_env_file(tmp_path, "MCP_BEARER_KEY=abc\n"))

    assert settings.prompt_guard is True  # on by default
    assert settings.prompt_leak_ngram == 12

    tuned = load_settings(
        _env_file(
            tmp_path,
            "MCP_BEARER_KEY=abc\nAGENT_PROMPT_GUARD=false\nAGENT_PROMPT_LEAK_NGRAM=20\n",
        )
    )
    assert tuned.prompt_guard is False
    assert tuned.prompt_leak_ngram == 20


def test_secret_scan_defaults_on_and_can_be_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AGENT_SECRET_SCAN", raising=False)
    settings = load_settings(_env_file(tmp_path, "MCP_BEARER_KEY=abc\n"))
    assert settings.secret_scan is True

    off = load_settings(_env_file(tmp_path, "MCP_BEARER_KEY=abc\nAGENT_SECRET_SCAN=false\n"))
    assert off.secret_scan is False


def test_latency_metrics_default_and_configure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in (
        "AGENT_LLM_METRICS",
        "AGENT_LLM_STATS_FETCH",
        "AGENT_LLM_STATS_DELAY_MS",
        "AGENT_LLM_STATS_ATTEMPTS",
        "AGENT_LLM_METRICS_QUEUE_MAX",
        "AGENT_LLM_METRICS_RETENTION_DAYS",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = load_settings(_env_file(tmp_path, "MCP_BEARER_KEY=abc\n"))

    # On by default: the worker is off the critical path and the endpoint it
    # calls returns metadata rather than inference, so there is no cost to
    # opt into and nothing an operator has to remember to switch on.
    assert settings.llm_metrics is True
    assert settings.llm_stats_fetch is True
    assert settings.llm_stats_delay_ms == 1500
    assert settings.llm_stats_attempts == 5
    assert settings.llm_metrics_queue_max == 500
    assert settings.llm_metrics_retention_days == 90

    tuned = load_settings(
        _env_file(
            tmp_path,
            "MCP_BEARER_KEY=abc\n"
            "AGENT_LLM_METRICS=false\n"
            "AGENT_LLM_STATS_FETCH=false\n"
            "AGENT_LLM_STATS_DELAY_MS=250\n"
            "AGENT_LLM_STATS_ATTEMPTS=5\n"
            "AGENT_LLM_METRICS_QUEUE_MAX=50\n"
            "AGENT_LLM_METRICS_RETENTION_DAYS=14\n",
        )
    )
    assert tuned.llm_metrics is False
    assert tuned.llm_stats_fetch is False
    assert tuned.llm_stats_delay_ms == 250
    assert tuned.llm_stats_attempts == 5
    assert tuned.llm_metrics_queue_max == 50
    assert tuned.llm_metrics_retention_days == 14


_WHATSAPP_KEYS = (
    "WHATSAPP_ACCESS_TOKEN",
    "WHATSAPP_PHONE_NUMBER_ID",
    "WHATSAPP_APP_SECRET",
    "WHATSAPP_VERIFY_TOKEN",
    "WHATSAPP_BUSINESS_NUMBER",
    "WHATSAPP_API_VERSION",
    "AGENT_ALLOWED_WHATSAPP_NUMBERS",
)

_WHATSAPP_ENV = (
    "MCP_BEARER_KEY=abc\n"
    "WHATSAPP_ACCESS_TOKEN=tok\n"
    "WHATSAPP_PHONE_NUMBER_ID=555\n"
    "WHATSAPP_APP_SECRET=sec\n"
    "WHATSAPP_VERIFY_TOKEN=vt\n"
)


def test_whatsapp_is_off_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _WHATSAPP_KEYS:
        monkeypatch.delenv(key, raising=False)
    settings = load_settings(_env_file(tmp_path, "MCP_BEARER_KEY=abc\n"))
    assert settings.whatsapp_enabled is False
    assert settings.allowed_whatsapp_numbers == []


def test_whatsapp_loads_when_fully_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in _WHATSAPP_KEYS:
        monkeypatch.delenv(key, raising=False)
    env = _env_file(
        tmp_path,
        _WHATSAPP_ENV
        + "WHATSAPP_BUSINESS_NUMBER=+1 555-123-4567\n"
        + "AGENT_ALLOWED_WHATSAPP_NUMBERS=+91 98765 43210, 14155550123\n",
    )
    settings = load_settings(env)
    assert settings.whatsapp_enabled is True
    assert settings.whatsapp_business_number == "15551234567"
    assert settings.allowed_whatsapp_numbers == ["919876543210", "14155550123"]
    assert (
        settings.whatsapp_graph_url
        == f"https://graph.facebook.com/{settings.whatsapp_api_version}"
    )


def test_partial_whatsapp_config_stops_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in _WHATSAPP_KEYS:
        monkeypatch.delenv(key, raising=False)
    env = _env_file(tmp_path, "MCP_BEARER_KEY=abc\nWHATSAPP_ACCESS_TOKEN=tok\n")
    with pytest.raises(SystemExit) as excinfo:
        load_settings(env)
    message = str(excinfo.value)
    assert "WHATSAPP_PHONE_NUMBER_ID" in message
    assert "WHATSAPP_APP_SECRET" in message
    assert "WHATSAPP_VERIFY_TOKEN" in message


def test_verify_token_alone_is_not_a_partial_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The handshake-only mode that existed before the channel did.
    for key in _WHATSAPP_KEYS:
        monkeypatch.delenv(key, raising=False)
    settings = load_settings(_env_file(tmp_path, "MCP_BEARER_KEY=abc\nWHATSAPP_VERIFY_TOKEN=vt\n"))
    assert settings.whatsapp_verify_token == "vt"
    assert settings.whatsapp_enabled is False


def test_malformed_whatsapp_allowlist_stops_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Same rule as the Telegram list: a typo must not silently allow everyone.
    for key in _WHATSAPP_KEYS:
        monkeypatch.delenv(key, raising=False)
    env = _env_file(tmp_path, "MCP_BEARER_KEY=abc\nAGENT_ALLOWED_WHATSAPP_NUMBERS=9198,abc\n")
    with pytest.raises(SystemExit) as excinfo:
        load_settings(env)
    assert "AGENT_ALLOWED_WHATSAPP_NUMBERS" in str(excinfo.value)
