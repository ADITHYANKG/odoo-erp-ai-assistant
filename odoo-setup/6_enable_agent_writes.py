"""Phase 5: enable write access for the Telegram agent (idempotent).

Two server-side switches must BOTH be on before the agent can write:
  1. The agent's OAuth client scope ceiling -> "write", so the authorization
     server may grant `mcp:write` (the client still only gets what the user
     approves on the consent screen).
  2. `allow_write = True` on the muk_mcp_access allowlist entry for the
     models the agent may modify (res.partner only, for now). Token scope and
     the per-model flag are independent gates - a write needs both.

IMPORTANT: refresh tokens can never WIDEN scope, so anyone already logged in
through Telegram keeps read-only tokens until they /logout and /start again.

Usage:
    ./venv/Scripts/python experiments/6_enable_agent_writes.py
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

ODOO_URL = os.getenv("ODOO_URL", "http://localhost:8069")
ODOO_DB = os.getenv("ODOO_DB")
ODOO_USERNAME = os.getenv("ODOO_USERNAME")
ODOO_PASSWORD = os.getenv("ODOO_PASSWORD")

JSONRPC_ENDPOINT = f"{ODOO_URL}/jsonrpc"

CLIENT_NAME = "ont-agent (Telegram)"
WRITABLE_MODELS = ["res.partner"]


def jsonrpc(service, method, args):
    """Call an Odoo JSON-RPC service method and return the result."""
    payload = {
        "jsonrpc": "2.0",
        "method": "call",
        "params": {"service": service, "method": method, "args": args},
        "id": 1,
    }
    response = requests.post(JSONRPC_ENDPOINT, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    if data.get("error"):
        raise RuntimeError(data["error"])
    return data["result"]


def authenticate():
    uid = jsonrpc("common", "login", [ODOO_DB, ODOO_USERNAME, ODOO_PASSWORD])
    if not uid:
        raise RuntimeError("Authentication failed - check credentials in .env")
    return uid


def execute_kw(uid, model, method, args, kwargs=None):
    return jsonrpc(
        "object",
        "execute_kw",
        [ODOO_DB, uid, ODOO_PASSWORD, model, method, args, kwargs or {}],
    )


def raise_client_ceiling(uid):
    clients = execute_kw(
        uid, "ont_mcp_oauth.client", "search_read",
        [[["name", "=", CLIENT_NAME]]],
        {"fields": ["id", "scope"], "limit": 1},
    )
    if not clients:
        raise SystemExit(
            f"OAuth client '{CLIENT_NAME}' not found - run "
            "experiments/5_create_agent_oauth_client.py first."
        )
    client = clients[0]
    if client["scope"] == "write":
        print(f"OAuth client '{CLIENT_NAME}': scope ceiling already 'write'")
    else:
        execute_kw(
            uid, "ont_mcp_oauth.client", "write",
            [[client["id"]], {"scope": "write"}],
        )
        print(f"OAuth client '{CLIENT_NAME}': scope ceiling read -> write")


def allow_model_writes(uid, model_name):
    entries = execute_kw(
        uid, "muk_mcp_access.model", "search_read",
        [[["model_name", "=", model_name]]],
        {"fields": ["id", "allow_read", "allow_write"], "limit": 1,
         "context": {"active_test": False}},
    )
    if entries:
        entry = entries[0]
        if entry["allow_write"]:
            print(f"Allowlist '{model_name}': allow_write already True")
        else:
            execute_kw(
                uid, "muk_mcp_access.model", "write",
                [[entry["id"]], {"allow_write": True}],
            )
            print(f"Allowlist '{model_name}': allow_write False -> True")
    else:
        model_ids = execute_kw(
            uid, "ir.model", "search", [[["model", "=", model_name]]], {"limit": 1}
        )
        if not model_ids:
            raise SystemExit(f"Model '{model_name}' not found in ir.model")
        execute_kw(
            uid, "muk_mcp_access.model", "create",
            [{"model_id": model_ids[0], "allow_read": True, "allow_write": True}],
        )
        print(f"Allowlist '{model_name}': entry created (read + write)")


def main():
    uid = authenticate()
    raise_client_ceiling(uid)
    for model_name in WRITABLE_MODELS:
        allow_model_writes(uid, model_name)
    print()
    print("Done. Reminder: existing Telegram logins keep READ-ONLY tokens")
    print("(refresh can never widen scope) - users must /logout then /start")
    print("again to approve the new write permission on the consent screen.")


if __name__ == "__main__":
    main()
