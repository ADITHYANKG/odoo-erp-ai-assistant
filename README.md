# odoo-erp-ai-assistant

An AI assistant over Odoo 17 ERP. Users chat with it on Telegram and WhatsApp (Slack and Google Chat adapters are also included). It answers questions about live ERP data, searches policy documents, and proposes writes and emails that the user must approve.

The agent is built on LangGraph with models from OpenRouter. It reaches Odoo only through an MCP (Model Context Protocol) endpoint, and every chat user logs in to Odoo through OAuth 2.1 + PKCE, so the bot only sees what that user is allowed to see.

## Layout

| Folder | What it is |
|---|---|
| [`agent/`](agent/) | The chatbot (Python, uv). Start with [`agent/README.md`](agent/README.md). |
| [`odoo-addons/`](odoo-addons/) | Odoo 17 modules that provide the MCP server side. |
| [`odoo-setup/`](odoo-setup/) | Scripts that set up Odoo for the agent (MCP key, OAuth client, base URL, write scope). |
| [`docs/`](docs/) | How the two sides fit together. The agent's half is in [`agent/docs/ODOO_CONTRACT.md`](agent/docs/ODOO_CONTRACT.md). |

### Odoo modules

| Module | Role |
|---|---|
| `muk_mcp` | `/mcp` endpoint (Streamable HTTP) and the core tools, by MuK IT (LGPL-3) |
| `muk_mcp_access` | Allowlist of the models the bot can read and write |
| `ont_mcp_oauth` | OAuth 2.1 + PKCE login per chat user, JWT bearer validation |
| `ont_mcp_mail` | `send_email` tool with a recipient allowlist |
| `ont_mcp_rag` | `search_documents` over Odoo Documents (embeddings) |
| `ont_mcp_fieldguard` | Drops unknown fields from reads instead of failing the call |
| `ont_mcp_trace` | Chatter note on every record the bot writes |

## Quick start

You need Git, Docker, Python 3.13+ and [uv](https://docs.astral.sh/uv/), plus an [OpenRouter](https://openrouter.ai) API key and a Telegram bot token from @BotFather.

### 1. Start Odoo

```bash
docker compose up -d
```

This starts Odoo 17 Community and Postgres, creates a database called `odoo` and installs the MCP modules from `odoo-addons/`. The first start takes a few minutes. Then open <http://localhost:8069> and log in as `admin` / `admin` (change the password).

Settings you can override in a `.env` file next to `docker-compose.yml`: `ODOO_PORT` (default `8069`), `ODOO_DB` (`odoo`), `ODOO_DB_PASSWORD` (`odoo`) and `ODOO_MODULES` (the modules to install).

> **Community vs Enterprise:** `ont_mcp_rag` (searching policy documents) needs Odoo Enterprise's `documents` module, so it isn't installed by default. Everything else works on Community. With an Enterprise licence, use your own Enterprise image and add `ont_mcp_rag` to `ODOO_MODULES`.

To use an Odoo you already run instead, add `odoo-addons/` to its `addons_path` and install the same modules.

### 2. Connect the agent to Odoo

```bash
cd agent && uv sync && cd ..
cp odoo-setup/.env.example odoo-setup/.env      # Odoo login for the scripts

uv run --project agent python odoo-setup/4_create_mcp_key.py            # prints MCP_BEARER_KEY
uv run --project agent python odoo-setup/5_create_agent_oauth_client.py # prints OAUTH_CLIENT_ID
uv run --project agent python odoo-setup/6_enable_agent_writes.py       # optional: lets the bot write
```

`3_set_public_base_url.py` is only needed when Odoo is reached through a public URL (a tunnel or a server).

### 3. Start the bot

```bash
cd agent
cp .env.example .env    # fill in TELEGRAM_BOT_TOKEN, OPENROUTER_API_KEY, MCP_BEARER_KEY, OAUTH_CLIENT_ID
docker compose up -d db # the bot's own Postgres
uv run ont-agent-bot
```

Message your bot on Telegram and send `/start` to log in to Odoo. See [`agent/README.md`](agent/README.md) for everything else.

## Optional: tracing and error tracking

The bot runs without either of these. Each one switches on only when its settings in `agent/.env` are filled in.

| Tool | What it gives you | Settings | Setup guide |
|---|---|---|---|
| [Langfuse](https://langfuse.com) | A trace of every conversation turn (model calls, tool calls, cost, latency), plus 👍/👎 feedback as scores | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | [agent/deploy/langfuse/](agent/deploy/langfuse/README.md) |
| [Sentry](https://sentry.io) | An issue with a traceback for every error the bot logs | `SENTRY_DSN`, `SENTRY_ENVIRONMENT`, `SENTRY_RELEASE` | [agent/deploy/sentry/](agent/deploy/sentry/README.md) |

Both can be self-hosted on the same server as the bot, or used as hosted services (Langfuse Cloud, sentry.io). Personal data is stripped before anything is sent to Sentry; see the Sentry guide for details.

## Licence

The Odoo modules are LGPL-3, as stated in each `__manifest__.py`.
