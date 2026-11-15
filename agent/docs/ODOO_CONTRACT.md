# Integration contract: ont-agent ↔ Odoo

This repo (`agent/`) is the chatbot. It reaches Odoo only through MCP over HTTP; the provider side lives in the sibling repo `odoo-addons/`, which has the full details in `docs/AGENT_INTEGRATION.md`. If you change anything in this file, update that one as well.

## What the agent needs from Odoo

| Odoo module (`odoo-addons/`) | What the agent relies on |
|---|---|
| `muk_mcp` | `/mcp` endpoint (Streamable HTTP) and the 20 core tools |
| `muk_mcp_access` | Model allowlist: decides which models the bot can read and write |
| `ont_mcp_oauth` | OAuth 2.1 + PKCE login for each Telegram/WhatsApp user, plus JWT bearer validation |
| `ont_mcp_mail` | `send_email` tool and its recipient allowlist |
| `ont_mcp_rag` | `search_documents` tool (policy documents in Odoo Documents) |
| `ont_mcp_fieldguard` | Drops unknown fields from reads, so the call doesn't fail |
| `ont_mcp_trace` | Chatter note on every record the bot writes |

## Configuration that links the two

| Agent env var | Produced on the Odoo side by |
|---|---|
| `MCP_URL` | Odoo base URL + `/mcp` |
| `MCP_BEARER_KEY` | `../odoo-setup/4_create_mcp_key.py` |
| `OAUTH_CLIENT_ID` | `../odoo-setup/5_create_agent_oauth_client.py` (the redirect URI must be `AGENT_PUBLIC_URL/oauth/callback`) |
| `ODOO_PUBLIC_URL` | Odoo `web.base.url`, set by `../odoo-setup/3_set_public_base_url.py` |
| `REQUIRED_ODOO_GROUP` | An Odoo group XML id that must exist |
| — (write scope) | `../odoo-setup/6_enable_agent_writes.py` raises the client scope to `mcp:write` |

## Changes that affect both repos

- Renaming, removing, or changing the arguments of an MCP tool. Check `src/ont_agent/tools.py` and the prompts in `src/ont_agent/agent.py`.
- Adding a new model the bot should query. Add it to the `muk_mcp_access` allowlist in Odoo; the agent itself needs no change.
- Changing the OAuth scopes, token lifetime, or JWT issuer/audience in `ont_mcp_oauth`. Check `src/ont_agent/oauth_flow.py` and `auth.py`.
- Adding a new custom tool module (`ont_mcp_*`). If needed, update the agent's tool filtering or routing (`router.py`).
