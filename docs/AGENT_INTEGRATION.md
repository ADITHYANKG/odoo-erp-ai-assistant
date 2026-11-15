# Agent integration: what the chatbot depends on in this repo

The Acme chatbot (`ont-agent`, Telegram + WhatsApp) lives in the sibling repo `agent/`. It has no code of its own in this repo: it reads and writes Odoo through the MCP endpoint and logs users in through OAuth. Odoo is the **provider** of that interface, so changes here can break the bot even though no agent code is touched. The consumer side is documented in `docs/ODOO_CONTRACT.md`. Keep the two docs in sync.

## Modules the chatbot relies on

| Module | Bot feature it powers | What breaks the bot |
|---|---|---|
| `muk_mcp` | The `/mcp` endpoint and the 20 core tools (search, read, create, write, …) | Renaming tools or changing their arguments; changing the bearer-key format |
| `muk_mcp_access` | The allowlist of models (and read/write per model) the bot can reach | A model missing from the allowlist: the bot answers "no access" |
| `ont_mcp_oauth` | Per-user login (OAuth 2.1 + PKCE), JWT issuer/audience, scopes | Changing issuer/audience, scopes, `web.base.url`, or the client's redirect URIs |
| `ont_mcp_mail` | The `send_email` tool and its recipient allowlist | Renaming the tool or tightening the allowlist |
| `ont_mcp_rag` | The `search_documents` tool over Odoo Documents (policies) | Moving documents out of indexed folders; removing `attachment_indexation` |
| `ont_mcp_fieldguard` | Unknown fields are dropped from reads instead of erroring | Uninstalling it: the LLM's guessed fields then fail whole calls |
| `ont_mcp_trace` | Chatter note on records the bot changes | Nothing functional, but you lose the audit trail |

The business modules (`ont_crm`, `ont_timesheet`, `ONT_FINMET`, `custom business modules`, …) don't depend on MCP. The bot reaches their **data** only if their models are on the `muk_mcp_access` allowlist.

## Setup scripts used by the agent (`experiments/`)

| Script | Produces |
|---|---|
| `3_set_public_base_url.py` | `web.base.url`, which becomes the agent's `ODOO_PUBLIC_URL` and the OAuth issuer |
| `4_create_mcp_key.py` | The agent's `MCP_BEARER_KEY` |
| `5_create_agent_oauth_client.py` | The agent's `OAUTH_CLIENT_ID`. The redirect URI is `<AGENT_PUBLIC_URL>/oauth/callback` |
| `6_enable_agent_writes.py` | Raises the OAuth client's scope to `mcp:write` |
| `10_tighten_mcp_allowlist.py` | Security-reviewed allowlist (no writes to users, employees, company, or bank details) |
| `11_seed_injection_probe.py` | A canary record for the bot's prompt-injection tests |

## Checklist for Odoo changes

When a change adds or renames a model or field, seeds data, or touches any `muk_mcp*` / `ont_mcp_*` module, ask:

1. **Should the bot see this data?** If yes, add the model to the `muk_mcp_access` allowlist and choose read-only or write. Sensitive models stay read-only or off.
2. **Did a tool's name or arguments change?** If yes, update `agent/src/ont_agent/tools.py` (and the prompts in `agent.py`) in a matching PR.
3. **Did OAuth or the URLs change?** If yes, update the agent's `.env` / deployment variables.
4. **Is it demo or seed data?** Bot answers are evaluated against the seeded data, so tell the agent side.
