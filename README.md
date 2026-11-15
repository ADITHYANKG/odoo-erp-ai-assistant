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

1. Add `odoo-addons/` to your Odoo 17 `addons_path` and install the `ont_mcp_*` modules (their dependencies, including `muk_mcp`, install automatically).
2. Run the setup scripts in `odoo-setup/` (in number order 3 → 6) against your Odoo, and copy the printed values into `agent/.env`.
3. In `agent/`: `cp .env.example .env`, fill it in, then `uv sync` and `uv run python -m ont_agent` (see `agent/README.md` for details).

## Optional: tracing and error tracking

The bot runs without either of these. Each one switches on only when its settings in `agent/.env` are filled in.

| Tool | What it gives you | Settings | Setup guide |
|---|---|---|---|
| [Langfuse](https://langfuse.com) | A trace of every conversation turn (model calls, tool calls, cost, latency), plus 👍/👎 feedback as scores | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | [agent/deploy/langfuse/](agent/deploy/langfuse/README.md) |
| [Sentry](https://sentry.io) | An issue with a traceback for every error the bot logs | `SENTRY_DSN`, `SENTRY_ENVIRONMENT`, `SENTRY_RELEASE` | [agent/deploy/sentry/](agent/deploy/sentry/README.md) |

Both can be self-hosted on the same server as the bot, or used as hosted services (Langfuse Cloud, sentry.io). Personal data is stripped before anything is sent to Sentry; see the Sentry guide for details.

## Licence

The Odoo modules are LGPL-3, as stated in each `__manifest__.py`.
