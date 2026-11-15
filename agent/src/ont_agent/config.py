"""Environment-backed configuration for the agent service.

Loads ``agent/.env`` (see ``.env.example``) and fails fast with an actionable
message when a required value is missing or malformed — a misconfigured agent
should die at startup, not midway through a conversation.

``Settings`` is a frozen pydantic model (typed, validated, immutable).
Deliberately NOT ``pydantic-settings``' ``BaseSettings``: that would read the
developer's real environment on every direct ``Settings(...)`` construction,
which the tests (and test isolation generally) must never do. The env merge
lives in ``load_settings`` only.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import NoReturn

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, ValidationError

from ont_agent.log import get_logger

_AGENT_DIR = Path(__file__).resolve().parents[2]  # .../agent


def _agent_path(raw: str) -> str:
    """Resolve ``raw`` against agent/ so it does not depend on the cwd; empty stays empty."""
    return str(_AGENT_DIR / raw) if raw else ""


DEFAULT_OPENROUTER_MODEL = "google/gemini-3.7-flash"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: Meta Graph API version for WhatsApp Cloud API calls.
DEFAULT_WHATSAPP_API_VERSION = "v23.0"
DEFAULT_AGENT_PUBLIC_URL = "http://localhost:8181"
DEFAULT_MEMORY_EMBED_MODEL = "openai/text-embedding-3-small"
#: Vector width of DEFAULT_MEMORY_EMBED_MODEL; change both together.
DEFAULT_MEMORY_EMBED_DIMS = 1536
#: How many stored preferences are injected into one prompt.
DEFAULT_MEMORY_MAX_RECALL = 10
#: Summarization: compress a thread once it passes this many tokens, keeping
#: the most recent N messages verbatim. MUST stay far above one busy turn —
#: at 3_000 the summarizer fired MID-TURN and ate the user's own question.
DEFAULT_SUMMARY_TRIGGER_TOKENS = 60_000
DEFAULT_SUMMARY_KEEP_MESSAGES = 20
#: Burst guard: messages one Telegram sender may send per window before
#: being throttled. Bounds a flood in SECONDS; the daily caps below bound
#: the day. Both are needed — a daily cap alone lets 200 paid calls through
#: in one minute.
DEFAULT_BURST_LIMIT = 10
DEFAULT_BURST_WINDOW_SECONDS = 60
#: Ceiling on one chat turn (answer or approval resume). Every channel handles
#: a chat's messages one at a time, so a turn stuck on a network call that
#: never returns would otherwise block that chat for good. Generous on
#: purpose: a turn may make several model calls of up to 90 s each.
DEFAULT_TURN_TIMEOUT_SECONDS = 300
#: Proactive work (morning brief, watcher checks). The brief lands at
#: BRIEF_HOUR:BRIEF_MINUTE in each USER's own timezone, so one ticker serves
#: people in different regions.
DEFAULT_BRIEF_HOUR = 8
DEFAULT_BRIEF_MINUTE = 0
#: A brief missed because the machine was asleep is still worth sending a few
#: hours late; one missed by two days is noise. Zero disables catch-up.
DEFAULT_BRIEF_CATCHUP_HOURS = 4
#: Used when the Odoo user has no timezone set (whoami returns "").
DEFAULT_TIMEZONE = "Asia/Kolkata"
#: Watchers are the cost driver of the whole proactive layer: every check is
#: one model call, so the bill scales with watchers x users x checks per day.
#: Fifty users at five watchers each would be 250 calls an hour at a 60-minute
#: interval — tens of dollars a day, and the global spend cap would simply
#: stop the bot mid-morning. Eight hours (three checks a day) keeps standing
#: conditions useful at a fraction of that. The trade: a condition that
#: becomes true and false again inside one window is never seen, so anything
#: needing tighter latency wants a shorter interval AND a raised spend cap.
DEFAULT_WATCHER_INTERVAL_MINUTES = 480
DEFAULT_MAX_WATCHERS_PER_USER = 5
#: Local hours between which the bot sends nothing unprompted.
#:
#: Briefs already wait for brief_hour in the USER's timezone, but
#: watchers were pure interval arithmetic with no timezone at all - at an
#: 8-hour interval one of the three daily checks lands in their night by
#: construction, and a 3am alert is the fastest way to get muted.
#:
#: The window WRAPS midnight (21 -> 8). Equal values disable it, the same
#: convention as 0 disabling retention and rotation.
DEFAULT_QUIET_HOURS_START = 21
DEFAULT_QUIET_HOURS_END = 8
#: Refuse messages that ask for the agent's own configuration, and
#: replace any reply that reproduces it. Off switch for a client whose
#: vocabulary trips the patterns.
DEFAULT_PROMPT_GUARD = True
#: Consecutive words of the system prompt that must appear in a reply
#: before it counts as a leak.
#:
#: The number IS the trade-off. Too low and an honest answer trips it -
#: the prompt spells out what 'overdue' means, and a reply explaining
#: that filter could echo the wording. Too high and partial
#: reproduction slips through. Configurable so a false positive in the
#: field is a .env change, not a code change.
DEFAULT_PROMPT_LEAK_NGRAM = 12
#: Redact credentials out of tool results and replies. The switch exists
#: for a client whose data trips the one non-vendor-prefixed pattern;
#: there is no threshold, because the rules are structural.
DEFAULT_SECRET_SCAN = True
#: How often the ticker looks for due work. Finer than any schedule it
#: serves — it decides what is due, it is not itself the schedule.
DEFAULT_TICK_SECONDS = 300
#: Speech to text for Telegram voice notes. Served by OpenRouter's
#: /audio/transcriptions endpoint, so it obeys the OpenRouter-only rule.
DEFAULT_TRANSCRIBE_MODEL = "openai/gpt-transcribe"
#: Refresh an access token this many seconds BEFORE the server expires it.
#: Must exceed a realistic worst-case turn, not a median one: a token is
#: checked once at the start of a turn and then used for every tool call in
#: it, and nothing recovers from a 401 partway through. With a 900s
#: server-side TTL this leaves 600s of use per token — one extra HTTP round
#: trip per user per ten minutes, and no model cost.
DEFAULT_TOKEN_REFRESH_MARGIN_SECONDS = 300
#: Voice notes longer than this are refused before download — bounds both
#: cost and the memory held while transcribing.
DEFAULT_VOICE_MAX_SECONDS = 120
#: Uploaded images and PDF files larger than this are refused before
#: download. Telegram's getFile caps downloads at 20 MB regardless, and a
#: PDF's cost scales with its page count, so the useful limit is well under
#: that ceiling.
DEFAULT_UPLOAD_MAX_MB = 5
#: Audio clips sent in Slack are refused above this size, before download.
DEFAULT_SLACK_VOICE_MAX_MB = 2
DEFAULT_SLACK_COMMAND = "/odoo"
#: Semantic router settings (Aurelio Semantic Router + FastEmbed local ONNX encoder)
DEFAULT_SEMANTIC_ROUTER = False
DEFAULT_SEMANTIC_ROUTER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SEMANTIC_ROUTER_THRESHOLD = 0.55
#: How long the action log keeps what the bot did. Far longer than the
#: evidence cache: an audit trail answering "what did this thing do in
#: March?" is only useful if March is still there.
DEFAULT_ACTION_LOG_DAYS = 90
#: Delete a conversation thread this many days after its last message.
#:
#: Thirty because that is when the LOGIN dies anyway: the OAuth server's
#: refresh_token_ttl_days is 30, so a user idle that long must /start again
#: regardless. Aligning them means retention can never discard a
#: conversation a working login could still have used. 0 disables it.
DEFAULT_CONVERSATION_RETENTION_DAYS = 30
#: Delete usage counters older than this. Every read is for a SINGLE day, so
#: history is never consulted — but the weekly digest reads the user's LOCAL
#: day, which can sit either side of UTC, so this must never be tiny.
DEFAULT_USAGE_RETENTION_DAYS = 90
#: Start a fresh conversation thread once the current one holds this many
#: checkpoints, carrying the state forward (see rotation.py).
#:
#: Retention only deletes threads that go IDLE, so an active user's thread
#: grows forever — roughly 60 checkpoints a day at ~100 KB each. A hundred
#: is about 33 turns and caps a live thread near 10 MB. Rotating costs no
#: model call, so a low value is cheap. 0 disables it.
DEFAULT_THREAD_MAX_CHECKPOINTS = 100
#: How long a group chat is remembered as already told the bot is
#: private-only, so it is not re-notified on every message.
#:
#: A window rather than a size cap because it SELF-EMPTIES: after an
#: hour with no group traffic the map is empty, where a bounded LRU
#: would hold its entries for ever. A group re-added later gets one
#: more notice, which is the harmless direction.
DEFAULT_GROUP_NOTICE_TTL_SECONDS = 3600
#: Connection-pool sizes. Two pools exist: LangGraph's (checkpoints and
#: long-term memory, the heavy writer) and BindingStore's own, which
#: does short queries and is capped smaller.
DEFAULT_DB_POOL_MIN_SIZE = 1
DEFAULT_DB_POOL_MAX_SIZE = 8
DEFAULT_STORE_POOL_MAX_SIZE = 4
#: Per-call latency recording (metrics.py). On by default: it is one row per
#: model call, written by a background worker, and it is the only standing
#: evidence about how slow the configured model actually is.
DEFAULT_LLM_METRICS = True
#: Whether to ask OpenRouter for its OWN timings via GET /generation. Off
#: leaves total_ms (our wall clock) and drops ttft_ms — the agent does not
#: stream tokens, so time to first token is NOT measurable on this side.
#: The endpoint returns metadata, not inference: it costs no tokens.
DEFAULT_LLM_STATS_FETCH = True
#: How long to wait before the first /generation attempt. The row is not
#: readable the instant a completion returns; asking immediately 404s.
DEFAULT_LLM_STATS_DELAY_MS = 1500
#: Attempts before giving up and writing the row without OpenRouter's
#: timings. Each attempt waits LONGER than the last (1.5s, 3s, 4.5s...), so
#: five of them cover about 22 seconds. Measured 2026-09-10: a 63-token
#: answer's row was ready inside 1.5s, a 5130-token one was not ready after
#: three flat attempts — and long answers are the slow ones worth measuring.
DEFAULT_LLM_STATS_ATTEMPTS = 5
#: Backpressure bound on samples waiting to be written. A measurement queue
#: must never grow without limit while OpenRouter is unreachable — dropping
#: samples is the correct failure here, and it is logged.
DEFAULT_LLM_METRICS_QUEUE_MAX = 500
#: Delete latency rows older than this. Matches usage retention; unlike
#: usage_daily this is one row per CALL, so it is the larger of the two.
DEFAULT_LLM_METRICS_RETENTION_DAYS = 90
#: OpenRouter refuses a request whose "models" array holds more than three
#: entries ("'models' array must have 3 items or fewer"). The primary model
#: does NOT count toward it — verified against the live API 2026-09-01, after
#: a fourth fallback made EVERY question fail with "something went wrong".
MAX_FALLBACK_MODELS = 3

logger = get_logger(__name__)


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    mcp_url: str
    mcp_bearer_key: str
    openrouter_api_key: str = ""
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL
    #: Server-side failover: OpenRouter tries these in order when the primary
    #: model's providers are down, rate-limited, or moderation-blocked.
    openrouter_fallback_models: list[str] = []
    telegram_bot_token: str = ""
    oauth_client_id: str = ""
    agent_public_url: str = DEFAULT_AGENT_PUBLIC_URL
    odoo_public_url: str = ""
    callback_host: str = "localhost"
    callback_port: int = 8181
    #: Meta webhook verification token; empty = WhatsApp webhook disabled.
    whatsapp_verify_token: str = ""
    #: WhatsApp Cloud API credentials; the channel runs only when all four
    #: of token, phone number id, app secret and verify token are set.
    whatsapp_access_token: str = ""
    whatsapp_phone_number_id: str = ""
    #: Signs every webhook POST (X-Hub-Signature-256); unsigned calls are refused.
    whatsapp_app_secret: str = ""
    #: Digits of the business number, for the wa.me return link after login.
    whatsapp_business_number: str = ""
    whatsapp_api_version: str = DEFAULT_WHATSAPP_API_VERSION
    #: Sender numbers (digits, country code included) allowed to talk to the
    #: bot; empty = anyone. Telegram's allowlist is separate.
    allowed_whatsapp_numbers: list[str] = []
    #: Google Cloud project NUMBER of the Chat app; names the service account
    #: that signs every event Google sends us. Empty = Google Chat disabled.
    googlechat_project_number: str = ""
    #: Service-account JSON key the app uses to call the Chat API (chat.bot scope).
    googlechat_credentials_file: str = ""
    #: The HTTP endpoint URL configured in the Chat API console; events carry it
    #: as the token audience. Empty = AGENT_PUBLIC_URL + /webhook/googlechat.
    googlechat_endpoint_url: str = ""
    #: Workspace emails allowed to talk to the bot (lowercase); empty = anyone
    #: the Chat app is shared with.
    allowed_googlechat_emails: list[str] = []
    #: Slack bot token (xoxb-); empty = Slack disabled.
    slack_bot_token: str = ""
    #: Socket Mode app-level token (xapp-); needed only in socket mode.
    slack_app_token: str = ""
    #: Verifies every HTTP-mode POST; needed only in http mode.
    slack_signing_secret: str = ""
    #: "socket" (the agent connects out) or "http" (Slack posts to /webhook/slack).
    slack_mode: str = "socket"
    #: The slash command as registered in the manifest; used in help text.
    slack_command: str = DEFAULT_SLACK_COMMAND
    #: Member ids (U...) allowed to use the bot; empty = anyone in an allowed workspace.
    allowed_slack_users: list[str] = []
    #: Extra workspace ids (T...) whose members may use the bot; the bot's own
    #: workspace is always allowed.
    allowed_slack_teams: list[str] = []
    #: True = a person who has not logged in gets no reply at all, except to `login`.
    slack_silent_until_login: bool = True
    slack_voice_max_mb: int = DEFAULT_SLACK_VOICE_MAX_MB
    #: Langfuse observability; empty keys = tracing disabled entirely.
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "http://localhost:3000"
    #: Sentry/GlitchTip error tracking; empty DSN = error reporting off.
    sentry_dsn: str = ""
    sentry_environment: str = "development"
    #: PostgreSQL connection string — REQUIRED. Everything the agent
    #: persists lives here: bindings and credentials, usage counters, the
    #: action log, watchers, conversation checkpoints and preferences.
    #: There is no SQLite fallback; see agent/docker-compose.yaml.
    agent_db_url: str = ""
    #: Max LangGraph steps per question — the runaway-loop ceiling.
    recursion_limit: int = 16
    #: Abuse backstop: max new questions one ODOO USER may ask per day.
    #: Keyed by login, not chat, so the same person in two chats shares one
    #: budget rather than getting two.
    daily_message_limit: int = 200
    #: Real-dollar cap (OpenRouter's actual billed cost) per Odoo user per
    #: day; ``None`` = disabled.
    daily_spend_limit_usd: float | None = None
    #: Real-dollar cap across ALL users per day; ``None`` = disabled. The
    #: per-user cap bounds one person, this bounds the bill: total exposure
    #: is otherwise per-user cap x however many people have bound a chat.
    global_daily_spend_limit_usd: float | None = None
    #: Telegram user ids allowed to talk to the bot at all; empty = anyone.
    #: Blocks strangers BEFORE the Odoo login page, unlike the group check.
    allowed_telegram_ids: list[int] = []
    #: Odoo group required to complete a binding, matched against the full
    #: name whoami reports; empty = any internal Odoo user may connect.
    required_odoo_group: str = ""
    #: Rotating log files; empty = terminal only. Relative paths are under agent/.
    log_file: str = ""
    blocked_log_file: str = ""
    #: Per-sender burst guard (see the DEFAULT_* comments).
    burst_limit: int = DEFAULT_BURST_LIMIT
    burst_window_seconds: int = DEFAULT_BURST_WINDOW_SECONDS
    #: Wall-clock limit on one chat turn; 0 disables it.
    turn_timeout_seconds: int = DEFAULT_TURN_TIMEOUT_SECONDS
    #: Master switch for everything the bot does unprompted.
    proactive: bool = True
    brief_hour: int = DEFAULT_BRIEF_HOUR
    brief_minute: int = DEFAULT_BRIEF_MINUTE
    brief_catchup_hours: int = DEFAULT_BRIEF_CATCHUP_HOURS
    default_timezone: str = DEFAULT_TIMEZONE
    watcher_interval_minutes: int = DEFAULT_WATCHER_INTERVAL_MINUTES
    max_watchers_per_user: int = DEFAULT_MAX_WATCHERS_PER_USER
    #: Local hours in which nothing unprompted is sent (see the
    #: DEFAULT_* comments). Equal values disable it.
    quiet_hours_start: int = DEFAULT_QUIET_HOURS_START
    quiet_hours_end: int = DEFAULT_QUIET_HOURS_END
    #: System-prompt extraction guardrail (see the DEFAULT_* comments).
    prompt_guard: bool = DEFAULT_PROMPT_GUARD
    prompt_leak_ngram: int = DEFAULT_PROMPT_LEAK_NGRAM
    #: Credential redaction (see the DEFAULT_* comment).
    secret_scan: bool = DEFAULT_SECRET_SCAN
    tick_seconds: int = DEFAULT_TICK_SECONDS
    #: Voice questions; False makes the bot decline them politely.
    #: Turn record names in answers into links back into Odoo. Needs
    #: odoo_public_url, which is rewired on every tunnel restart.
    token_streaming: bool = True
    deep_links: bool = True
    #: Sender name in the email approval preview; keep equal to the Odoo
    #: system parameter ont_mcp_mail.sender_name so the preview matches.
    mail_sender_name: str = "AI Assistant"
    voice: bool = True
    transcribe_model: str = DEFAULT_TRANSCRIBE_MODEL
    voice_max_seconds: int = DEFAULT_VOICE_MAX_SECONDS
    #: Seconds before server-side expiry at which a token is refreshed.
    token_refresh_margin_seconds: int = DEFAULT_TOKEN_REFRESH_MARGIN_SECONDS
    #: Reading uploaded images and PDF files; False declines them politely.
    uploads: bool = True
    upload_max_mb: int = DEFAULT_UPLOAD_MAX_MB
    #: Fallbacks used ONLY on a turn carrying a file. Deliberately separate
    #: from openrouter_fallback_models and empty by default: most models
    #: cannot read an image or a PDF, and one that cannot would answer from
    #: the caption alone — a confident answer about a document it never
    #: saw, which is worse than the request failing. Set this only to
    #: models verified to accept image and file input.
    upload_fallback_models: list[str] = []
    #: OpenRouter's file-parser engine for PDF files. Empty uses their
    #: default; other engines are charged per thousand pages.
    upload_pdf_engine: str = ""
    #: Fernet key encrypting the OAuth tokens in agent.db. Empty leaves them
    #: in plaintext (the pre-2026-09-02 behaviour), with a startup warning.
    token_key: str = ""
    #: Semantic routing for fast greeting / out-of-scope decisions
    semantic_router: bool = DEFAULT_SEMANTIC_ROUTER
    semantic_router_model: str = DEFAULT_SEMANTIC_ROUTER_MODEL
    semantic_router_threshold: float = DEFAULT_SEMANTIC_ROUTER_THRESHOLD
    #: Weekly "what I did" digest, and how long its source log is kept.
    digest: bool = True
    action_log_days: int = DEFAULT_ACTION_LOG_DAYS
    conversation_retention_days: int = DEFAULT_CONVERSATION_RETENTION_DAYS
    usage_retention_days: int = DEFAULT_USAGE_RETENTION_DAYS
    #: Checkpoints a thread may hold before it is rotated (see rotation.py).
    thread_max_checkpoints: int = DEFAULT_THREAD_MAX_CHECKPOINTS
    #: Anti-spam window for the private-chat-only notice.
    group_notice_ttl_seconds: int = DEFAULT_GROUP_NOTICE_TTL_SECONDS
    #: Connection-pool sizes (see the DEFAULT_* comments).
    db_pool_min_size: int = DEFAULT_DB_POOL_MIN_SIZE
    db_pool_max_size: int = DEFAULT_DB_POOL_MAX_SIZE
    store_pool_max_size: int = DEFAULT_STORE_POOL_MAX_SIZE
    #: Long-term per-user memory (preferences that survive /new and /logout).
    long_term_memory: bool = True
    #: OpenRouter embedding model for memory search; empty = no vector search.
    memory_embed_model: str = DEFAULT_MEMORY_EMBED_MODEL
    #: Vector width of that model — MUST match it (text-embedding-3-small is
    #: 1536, qwen3-embedding-0.6b is 1024); a mismatch corrupts the index.
    memory_embed_dims: int = DEFAULT_MEMORY_EMBED_DIMS
    #: OpenAI-compatible base URL every OpenRouter call goes through.
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL
    #: How many stored preferences are injected into one prompt.
    memory_max_recall: int = DEFAULT_MEMORY_MAX_RECALL
    #: Per-call latency recording, and how it is gathered (see metrics.py).
    llm_metrics: bool = DEFAULT_LLM_METRICS
    llm_stats_fetch: bool = DEFAULT_LLM_STATS_FETCH
    llm_stats_delay_ms: int = DEFAULT_LLM_STATS_DELAY_MS
    llm_stats_attempts: int = DEFAULT_LLM_STATS_ATTEMPTS
    llm_metrics_queue_max: int = DEFAULT_LLM_METRICS_QUEUE_MAX
    llm_metrics_retention_days: int = DEFAULT_LLM_METRICS_RETENTION_DAYS
    #: Conversation summarization thresholds (see the DEFAULT_* comments).
    summary_trigger_tokens: int = DEFAULT_SUMMARY_TRIGGER_TOKENS
    summary_keep_messages: int = DEFAULT_SUMMARY_KEEP_MESSAGES

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def sentry_enabled(self) -> bool:
        return bool(self.sentry_dsn)

    @property
    def whatsapp_enabled(self) -> bool:
        return bool(
            self.whatsapp_access_token
            and self.whatsapp_phone_number_id
            and self.whatsapp_app_secret
            and self.whatsapp_verify_token
        )

    @property
    def whatsapp_graph_url(self) -> str:
        return f"https://graph.facebook.com/{self.whatsapp_api_version}"

    @property
    def googlechat_enabled(self) -> bool:
        return bool(self.googlechat_project_number and self.googlechat_credentials_file)

    @property
    def googlechat_audience(self) -> str:
        return self.googlechat_endpoint_url or f"{self.agent_public_url}/webhook/googlechat"

    @property
    def googlechat_sender(self) -> str:
        """Email of the per-project service account Google signs events as."""
        return (
            f"service-{self.googlechat_project_number}@gcp-sa-gsuiteaddons.iam.gserviceaccount.com"
        )

    @property
    def slack_enabled(self) -> bool:
        return bool(self.slack_bot_token)

    @property
    def redirect_uri(self) -> str:
        return f"{self.agent_public_url}/oauth/callback"

    @property
    def authorize_url(self) -> str:
        return f"{self.odoo_public_url}/mcp/oauth/authorize"

    # Server-to-server calls use the MCP host: the browser-facing tunnel URL changes.
    @property
    def token_url(self) -> str:
        return f"{_derive_odoo_public_url(self.mcp_url)}/mcp/oauth/token"

    @property
    def revoke_url(self) -> str:
        return f"{_derive_odoo_public_url(self.mcp_url)}/mcp/oauth/revoke"


def _derive_odoo_public_url(mcp_url: str) -> str:
    return mcp_url.removesuffix("/mcp")


def _check_service_account_file(path: str) -> list[str]:
    """Problems with the Google service-account key file; empty when usable."""
    if not path:
        return ["GOOGLE_CHAT_CREDENTIALS_FILE is missing (service-account JSON key)"]
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError:
        return [f"GOOGLE_CHAT_CREDENTIALS_FILE {path!r} cannot be read"]
    except ValueError:
        return [f"GOOGLE_CHAT_CREDENTIALS_FILE {path!r} is not valid JSON"]
    missing = [
        key
        for key in ("client_email", "private_key", "token_uri")
        if not isinstance(data, dict) or not data.get(key)
    ]
    if missing:
        return ["GOOGLE_CHAT_CREDENTIALS_FILE lacks " + ", ".join(missing)]
    return []


def _check_slack(bot_token: str, app_token: str, signing_secret: str, mode: str) -> list[str]:
    """Problems with the SLACK_* settings; each message names the setting to fix."""
    if mode not in ("socket", "http"):
        return [f"SLACK_MODE must be 'socket' or 'http', got {mode!r}"]
    problems: list[str] = []
    if not bot_token:
        problems.append("SLACK_BOT_TOKEN is missing (the xoxb- token from Install App)")
    elif bot_token.startswith("xoxp-"):
        problems.append(
            "SLACK_BOT_TOKEN is a user token (xoxp-); use the Bot User OAuth Token (xoxb-)"
        )
    elif not bot_token.startswith("xoxb-"):
        problems.append("SLACK_BOT_TOKEN must be a bot token starting with xoxb-")
    if mode == "socket":
        if not app_token:
            problems.append("SLACK_APP_TOKEN is missing (the xapp- token with connections:write)")
        elif not app_token.startswith("xapp-"):
            problems.append("SLACK_APP_TOKEN must be an app-level token starting with xapp-")
    elif not signing_secret:
        problems.append("SLACK_SIGNING_SECRET is missing (Basic Information, needed in http mode)")
    return problems


def _slack_ids(name: str, raw: str, prefixes: str, problems: list[str]) -> list[str]:
    """Comma-separated Slack ids (``U...``, ``T...``), uppercased."""
    ids: list[str] = []
    for item in raw.split(","):
        value = item.strip().upper()
        if not value:
            continue
        if not re.fullmatch(rf"[{prefixes}][A-Z0-9]{{2,}}", value):
            problems.append(f"{name} must be comma-separated Slack ids, got {item.strip()!r}")
            continue
        ids.append(value)
    return ids


def load_settings(
    env_file: Path | None = None,
    *,
    require_llm: bool = False,
    require_telegram: bool = False,
) -> Settings:
    """Build Settings from ``env_file`` (default: agent/.env) plus process env.

    File first, process environment on top — so a container (or a test) can
    override without touching the file.

    :param require_llm: a missing ``OPENROUTER_API_KEY`` is fatal.
    :param require_telegram: missing Telegram/OAuth login configuration is
        fatal; every missing variable is reported in one message.
    """
    values: dict[str, str | None] = {
        **dotenv_values(env_file or _AGENT_DIR / ".env"),
        **os.environ,
    }

    def get(key: str) -> str:
        return (values.get(key) or "").strip()

    problems: list[str] = []

    def get_int(key: str, default: int) -> int:
        raw = get(key)
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            problems.append(f"{key} must be an integer, got {raw!r}")
            return default

    def get_bool(key: str, default: bool) -> bool:
        raw = get(key).lower()
        if not raw:
            return default
        return raw not in ("0", "false", "no", "off")

    def get_int_list(key: str) -> list[int]:
        items: list[int] = []
        for raw in get(key).split(","):
            item = raw.strip()
            if not item:
                continue
            try:
                items.append(int(item))
            except ValueError:
                problems.append(f"{key} must be comma-separated integers, got {item!r}")
        return items

    def get_optional_float(key: str) -> float | None:
        raw = get(key)
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            problems.append(f"{key} must be a number, got {raw!r}")
            return None

    def get_float(key: str, default: float) -> float:
        val = get_optional_float(key)
        return default if val is None else val

    if not get("AGENT_DB_URL"):
        problems.append(
            "AGENT_DB_URL is missing. Everything the agent persists lives in "
            "Postgres now — there is no SQLite fallback. Start it with: "
            "docker compose -f agent/docker-compose.yaml up -d"
        )
    if not get("MCP_BEARER_KEY"):
        problems.append(
            "MCP_BEARER_KEY is missing "
            "(mint one: python ../odoo-setup/4_create_mcp_key.py)"
        )
    if require_llm and not get("OPENROUTER_API_KEY"):
        problems.append("OPENROUTER_API_KEY is missing (get a key at https://openrouter.ai/keys)")
    if require_telegram:
        if not get("TELEGRAM_BOT_TOKEN"):
            problems.append("TELEGRAM_BOT_TOKEN is missing (from @BotFather)")
        if not get("OAUTH_CLIENT_ID"):
            problems.append(
                "OAUTH_CLIENT_ID is missing (run: ./venv/Scripts/python "
                "experiments/5_create_agent_oauth_client.py)"
            )

    mcp_url = (get("MCP_URL") or "http://localhost:8069/mcp").rstrip("/")
    callback_port = get_int("CALLBACK_PORT", 8181)
    recursion_limit = get_int("AGENT_RECURSION_LIMIT", 16)
    daily_message_limit = get_int("AGENT_DAILY_MESSAGE_LIMIT", 200)
    daily_spend_limit_usd = get_optional_float("AGENT_DAILY_SPEND_LIMIT_USD")
    global_daily_spend_limit_usd = get_optional_float("AGENT_GLOBAL_DAILY_SPEND_LIMIT_USD")
    allowed_telegram_ids = get_int_list("AGENT_ALLOWED_TELEGRAM_IDS")
    allowed_whatsapp_numbers: list[str] = []
    for raw_number in get("AGENT_ALLOWED_WHATSAPP_NUMBERS").split(","):
        digits = re.sub(r"[\s+\-]", "", raw_number)
        if not digits:
            continue
        if not digits.isdigit() or len(digits) > 15:
            problems.append(
                "AGENT_ALLOWED_WHATSAPP_NUMBERS must be comma-separated phone numbers "
                f"with country code, got {raw_number.strip()!r}"
            )
            continue
        allowed_whatsapp_numbers.append(digits)
    whatsapp_core = {
        "WHATSAPP_ACCESS_TOKEN": get("WHATSAPP_ACCESS_TOKEN"),
        "WHATSAPP_PHONE_NUMBER_ID": get("WHATSAPP_PHONE_NUMBER_ID"),
        "WHATSAPP_APP_SECRET": get("WHATSAPP_APP_SECRET"),
    }
    if any(whatsapp_core.values()):
        # Half-configured is a typo, not a choice: say what is missing now
        # rather than serving a webhook that silently ignores everyone.
        missing = [name for name, value in whatsapp_core.items() if not value]
        if not get("WHATSAPP_VERIFY_TOKEN"):
            missing.append("WHATSAPP_VERIFY_TOKEN")
        if missing:
            problems.append("WhatsApp is partly configured; missing " + ", ".join(missing))
    googlechat_project_number = get("GOOGLE_CHAT_PROJECT_NUMBER")
    googlechat_credentials_file = _agent_path(get("GOOGLE_CHAT_CREDENTIALS_FILE"))
    if googlechat_project_number or googlechat_credentials_file:
        if not googlechat_project_number.isdigit():
            problems.append(
                "GOOGLE_CHAT_PROJECT_NUMBER must be the numeric Google Cloud project "
                f"number, got {googlechat_project_number!r}"
            )
        problems.extend(_check_service_account_file(googlechat_credentials_file))
    allowed_googlechat_emails = [
        item.strip().lower()
        for item in get("AGENT_ALLOWED_GOOGLE_CHAT_EMAILS").split(",")
        if item.strip()
    ]
    slack_bot_token = get("SLACK_BOT_TOKEN")
    slack_app_token = get("SLACK_APP_TOKEN")
    slack_signing_secret = get("SLACK_SIGNING_SECRET")
    slack_mode = (get("SLACK_MODE") or "socket").lower()
    if slack_bot_token or slack_app_token or slack_signing_secret:
        problems.extend(
            _check_slack(slack_bot_token, slack_app_token, slack_signing_secret, slack_mode)
        )
    allowed_slack_users = _slack_ids(
        "AGENT_ALLOWED_SLACK_USERS", get("AGENT_ALLOWED_SLACK_USERS"), "UW", problems
    )
    allowed_slack_teams = _slack_ids(
        "AGENT_ALLOWED_SLACK_TEAMS", get("AGENT_ALLOWED_SLACK_TEAMS"), "TE", problems
    )
    slack_command = get("SLACK_COMMAND") or DEFAULT_SLACK_COMMAND
    if not slack_command.startswith("/"):
        slack_command = "/" + slack_command
    slack_voice_max_mb = get_int("AGENT_SLACK_VOICE_MAX_MB", DEFAULT_SLACK_VOICE_MAX_MB)
    burst_limit = get_int("AGENT_BURST_LIMIT", DEFAULT_BURST_LIMIT)
    burst_window_seconds = get_int("AGENT_BURST_WINDOW_SECONDS", DEFAULT_BURST_WINDOW_SECONDS)
    turn_timeout_seconds = get_int("AGENT_TURN_TIMEOUT_SECONDS", DEFAULT_TURN_TIMEOUT_SECONDS)
    brief_hour = get_int("AGENT_BRIEF_HOUR", DEFAULT_BRIEF_HOUR)
    brief_minute = get_int("AGENT_BRIEF_MINUTE", DEFAULT_BRIEF_MINUTE)
    brief_catchup_hours = get_int("AGENT_BRIEF_CATCHUP_HOURS", DEFAULT_BRIEF_CATCHUP_HOURS)
    watcher_interval_minutes = get_int(
        "AGENT_WATCHER_INTERVAL_MINUTES", DEFAULT_WATCHER_INTERVAL_MINUTES
    )
    max_watchers_per_user = get_int("AGENT_MAX_WATCHERS_PER_USER", DEFAULT_MAX_WATCHERS_PER_USER)
    quiet_hours_start = get_int("AGENT_QUIET_HOURS_START", DEFAULT_QUIET_HOURS_START)
    quiet_hours_end = get_int("AGENT_QUIET_HOURS_END", DEFAULT_QUIET_HOURS_END)
    prompt_leak_ngram = get_int("AGENT_PROMPT_LEAK_NGRAM", DEFAULT_PROMPT_LEAK_NGRAM)
    tick_seconds = get_int("AGENT_TICK_SECONDS", DEFAULT_TICK_SECONDS)
    voice_max_seconds = get_int("AGENT_VOICE_MAX_SECONDS", DEFAULT_VOICE_MAX_SECONDS)
    upload_max_mb = get_int("AGENT_UPLOAD_MAX_MB", DEFAULT_UPLOAD_MAX_MB)
    token_refresh_margin_seconds = get_int(
        "AGENT_TOKEN_REFRESH_MARGIN_SECONDS", DEFAULT_TOKEN_REFRESH_MARGIN_SECONDS
    )
    action_log_days = get_int("AGENT_ACTION_LOG_DAYS", DEFAULT_ACTION_LOG_DAYS)
    conversation_retention_days = get_int(
        "AGENT_CONVERSATION_RETENTION_DAYS", DEFAULT_CONVERSATION_RETENTION_DAYS
    )
    usage_retention_days = get_int("AGENT_USAGE_RETENTION_DAYS", DEFAULT_USAGE_RETENTION_DAYS)
    llm_metrics_retention_days = get_int(
        "AGENT_LLM_METRICS_RETENTION_DAYS", DEFAULT_LLM_METRICS_RETENTION_DAYS
    )
    thread_max_checkpoints = get_int(
        "AGENT_THREAD_MAX_CHECKPOINTS", DEFAULT_THREAD_MAX_CHECKPOINTS
    )
    group_notice_ttl_seconds = get_int(
        "AGENT_GROUP_NOTICE_TTL_SECONDS", DEFAULT_GROUP_NOTICE_TTL_SECONDS
    )
    token_key = get("AGENT_TOKEN_KEY")
    if token_key:
        # A malformed key is unambiguously a config error, so it stops the
        # bot here rather than surfacing on somebody's first question.
        from ont_agent.crypto import TokenCipher

        try:
            TokenCipher(token_key)
        except ValueError as exc:
            problems.append(str(exc))
    fallback_models = [
        item.strip() for item in get("OPENROUTER_FALLBACK_MODELS").split(",") if item.strip()
    ]
    upload_fallback_models = [
        item.strip() for item in get("AGENT_UPLOAD_FALLBACK_MODELS").split(",") if item.strip()
    ]
    if len(upload_fallback_models) > MAX_FALLBACK_MODELS:
        problems.append(
            f"AGENT_UPLOAD_FALLBACK_MODELS lists {len(upload_fallback_models)} models; "
            f"OpenRouter accepts at most {MAX_FALLBACK_MODELS}. Remove "
            f"{len(upload_fallback_models) - MAX_FALLBACK_MODELS} of them."
        )
    if len(fallback_models) > MAX_FALLBACK_MODELS:
        # Caught here rather than on every question: OpenRouter rejects the
        # whole request, so a fourth entry breaks answering entirely.
        problems.append(
            f"OPENROUTER_FALLBACK_MODELS lists {len(fallback_models)} models; OpenRouter "
            f"accepts at most {MAX_FALLBACK_MODELS}. Remove "
            f"{len(fallback_models) - MAX_FALLBACK_MODELS} of them."
        )

    def fail(details: list[str]) -> NoReturn:
        logger.error("Configuration errors: %s", "; ".join(details))
        listing = "\n".join(f"  - {d}" for d in details)
        sys.exit(f"Configuration errors in {_AGENT_DIR / '.env'}:\n{listing}")

    if problems:
        fail(problems)

    try:
        settings = Settings(
            mcp_url=mcp_url,
            mcp_bearer_key=get("MCP_BEARER_KEY"),
            openrouter_api_key=get("OPENROUTER_API_KEY"),
            openrouter_model=get("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL,
            openrouter_fallback_models=fallback_models,
            telegram_bot_token=get("TELEGRAM_BOT_TOKEN"),
            oauth_client_id=get("OAUTH_CLIENT_ID"),
            agent_public_url=(get("AGENT_PUBLIC_URL") or DEFAULT_AGENT_PUBLIC_URL).rstrip("/"),
            odoo_public_url=(get("ODOO_PUBLIC_URL") or _derive_odoo_public_url(mcp_url)).rstrip(
                "/"
            ),
            callback_host=get("CALLBACK_HOST") or "localhost",
            callback_port=callback_port,
            whatsapp_verify_token=get("WHATSAPP_VERIFY_TOKEN"),
            whatsapp_access_token=whatsapp_core["WHATSAPP_ACCESS_TOKEN"],
            whatsapp_phone_number_id=whatsapp_core["WHATSAPP_PHONE_NUMBER_ID"],
            whatsapp_app_secret=whatsapp_core["WHATSAPP_APP_SECRET"],
            whatsapp_business_number=re.sub(r"\D", "", get("WHATSAPP_BUSINESS_NUMBER")),
            whatsapp_api_version=get("WHATSAPP_API_VERSION") or DEFAULT_WHATSAPP_API_VERSION,
            allowed_whatsapp_numbers=allowed_whatsapp_numbers,
            googlechat_project_number=googlechat_project_number,
            googlechat_credentials_file=googlechat_credentials_file,
            googlechat_endpoint_url=get("GOOGLE_CHAT_ENDPOINT_URL"),
            allowed_googlechat_emails=allowed_googlechat_emails,
            slack_bot_token=slack_bot_token,
            slack_app_token=slack_app_token,
            slack_signing_secret=slack_signing_secret,
            slack_mode=slack_mode,
            slack_command=slack_command,
            allowed_slack_users=allowed_slack_users,
            allowed_slack_teams=allowed_slack_teams,
            slack_silent_until_login=get_bool("AGENT_SLACK_SILENT_UNTIL_LOGIN", True),
            slack_voice_max_mb=slack_voice_max_mb,
            langfuse_public_key=get("LANGFUSE_PUBLIC_KEY"),
            langfuse_secret_key=get("LANGFUSE_SECRET_KEY"),
            langfuse_base_url=(get("LANGFUSE_BASE_URL") or "http://localhost:3000").rstrip("/"),
            sentry_dsn=get("SENTRY_DSN"),
            sentry_environment=get("SENTRY_ENVIRONMENT") or "development",
            agent_db_url=get("AGENT_DB_URL"),
            recursion_limit=recursion_limit,
            daily_message_limit=daily_message_limit,
            daily_spend_limit_usd=daily_spend_limit_usd,
            global_daily_spend_limit_usd=global_daily_spend_limit_usd,
            allowed_telegram_ids=allowed_telegram_ids,
            required_odoo_group=get("AGENT_REQUIRED_ODOO_GROUP"),
            log_file=_agent_path(get("AGENT_LOG_FILE")),
            blocked_log_file=_agent_path(get("AGENT_BLOCKED_LOG_FILE")),
            burst_limit=burst_limit,
            burst_window_seconds=burst_window_seconds,
            turn_timeout_seconds=turn_timeout_seconds,
            proactive=get_bool("AGENT_PROACTIVE", True),
            brief_hour=brief_hour,
            brief_minute=brief_minute,
            brief_catchup_hours=brief_catchup_hours,
            default_timezone=get("AGENT_DEFAULT_TIMEZONE") or DEFAULT_TIMEZONE,
            watcher_interval_minutes=watcher_interval_minutes,
            max_watchers_per_user=max_watchers_per_user,
            quiet_hours_start=quiet_hours_start,
            quiet_hours_end=quiet_hours_end,
            prompt_guard=get_bool("AGENT_PROMPT_GUARD", DEFAULT_PROMPT_GUARD),
            prompt_leak_ngram=prompt_leak_ngram,
            secret_scan=get_bool("AGENT_SECRET_SCAN", DEFAULT_SECRET_SCAN),
            tick_seconds=tick_seconds,
            token_streaming=get_bool("AGENT_TOKEN_STREAMING", True),
            deep_links=get_bool("AGENT_DEEP_LINKS", True),
            mail_sender_name=get("AGENT_MAIL_SENDER_NAME") or "AI Assistant",
            voice=get_bool("AGENT_VOICE", True),
            transcribe_model=get("AGENT_TRANSCRIBE_MODEL") or DEFAULT_TRANSCRIBE_MODEL,
            voice_max_seconds=voice_max_seconds,
            token_refresh_margin_seconds=token_refresh_margin_seconds,
            uploads=get_bool("AGENT_UPLOADS", True),
            upload_max_mb=upload_max_mb,
            upload_fallback_models=upload_fallback_models,
            upload_pdf_engine=get("AGENT_UPLOAD_PDF_ENGINE"),
            token_key=token_key,
            semantic_router=get_bool("AGENT_SEMANTIC_ROUTER", DEFAULT_SEMANTIC_ROUTER),
            semantic_router_model=get("AGENT_SEMANTIC_ROUTER_MODEL")
            or DEFAULT_SEMANTIC_ROUTER_MODEL,
            semantic_router_threshold=get_float(
                "AGENT_SEMANTIC_ROUTER_THRESHOLD", DEFAULT_SEMANTIC_ROUTER_THRESHOLD
            ),
            digest=get_bool("AGENT_DIGEST", True),
            action_log_days=action_log_days,
            conversation_retention_days=conversation_retention_days,
            usage_retention_days=usage_retention_days,
            llm_metrics=get_bool("AGENT_LLM_METRICS", DEFAULT_LLM_METRICS),
            llm_stats_fetch=get_bool("AGENT_LLM_STATS_FETCH", DEFAULT_LLM_STATS_FETCH),
            llm_stats_delay_ms=get_int("AGENT_LLM_STATS_DELAY_MS", DEFAULT_LLM_STATS_DELAY_MS),
            llm_stats_attempts=get_int("AGENT_LLM_STATS_ATTEMPTS", DEFAULT_LLM_STATS_ATTEMPTS),
            llm_metrics_queue_max=get_int(
                "AGENT_LLM_METRICS_QUEUE_MAX", DEFAULT_LLM_METRICS_QUEUE_MAX
            ),
            llm_metrics_retention_days=llm_metrics_retention_days,
            thread_max_checkpoints=thread_max_checkpoints,
            group_notice_ttl_seconds=group_notice_ttl_seconds,
            db_pool_min_size=get_int("AGENT_DB_POOL_MIN_SIZE", DEFAULT_DB_POOL_MIN_SIZE),
            db_pool_max_size=get_int("AGENT_DB_POOL_MAX_SIZE", DEFAULT_DB_POOL_MAX_SIZE),
            store_pool_max_size=get_int("AGENT_STORE_POOL_MAX_SIZE", DEFAULT_STORE_POOL_MAX_SIZE),
            long_term_memory=get_bool("AGENT_LONG_TERM_MEMORY", True),
            memory_embed_model=(
                get("AGENT_MEMORY_EMBED_MODEL")
                if "AGENT_MEMORY_EMBED_MODEL" in values
                else DEFAULT_MEMORY_EMBED_MODEL
            ),
            memory_embed_dims=get_int("AGENT_MEMORY_EMBED_DIMS", DEFAULT_MEMORY_EMBED_DIMS),
            openrouter_base_url=(get("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL).rstrip(
                "/"
            ),
            memory_max_recall=get_int("AGENT_MEMORY_MAX_RECALL", DEFAULT_MEMORY_MAX_RECALL),
            summary_trigger_tokens=get_int(
                "AGENT_SUMMARY_TRIGGER_TOKENS", DEFAULT_SUMMARY_TRIGGER_TOKENS
            ),
            summary_keep_messages=get_int(
                "AGENT_SUMMARY_KEEP_MESSAGES", DEFAULT_SUMMARY_KEEP_MESSAGES
            ),
        )
    except ValidationError as exc:  # defensive net: values above are pre-coerced
        fail(
            [
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()
            ]
        )

    logger.debug(
        "Settings loaded: MCP_URL=%s, model=%s, odoo_public=%s",
        settings.mcp_url,
        settings.openrouter_model,
        settings.odoo_public_url,
    )
    return settings
