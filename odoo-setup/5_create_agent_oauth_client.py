"""Register (idempotently) the Telegram agent's OAuth client in Odoo.

Creates an ont_mcp_oauth.client the agent uses for the per-user login flow:
public client (PKCE, no secret), read-only scope ceiling, refresh allowed.
Re-running updates the redirect URI instead of duplicating the client.

Usage:
    uv run --project agent python odoo-setup/5_create_agent_oauth_client.py \
        [--public-url http://localhost:8181]

Copy the printed client_id into agent/.env as OAUTH_CLIENT_ID.
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--public-url",
        default="http://localhost:8181",
        help="Base URL of the agent's callback server (AGENT_PUBLIC_URL).",
    )
    args = parser.parse_args()
    redirect_uri = args.public_url.rstrip("/") + "/oauth/callback"

    uid = authenticate()
    existing = execute_kw(
        uid, "ont_mcp_oauth.client", "search_read",
        [[["name", "=", CLIENT_NAME]]],
        {"fields": ["id", "client_id", "redirect_uris"], "limit": 1},
    )

    if existing:
        record = existing[0]
        execute_kw(
            uid, "ont_mcp_oauth.client", "write",
            [[record["id"]], {"redirect_uris": redirect_uri}],
        )
        client_id = record["client_id"]
        print(f"Updated existing client #{record['id']} '{CLIENT_NAME}'")
    else:
        new_id = execute_kw(
            uid, "ont_mcp_oauth.client", "create",
            [{
                "name": CLIENT_NAME,
                "client_type": "public",
                "scope": "read",
                "allow_refresh": True,
                "redirect_uris": redirect_uri,
            }],
        )
        client_id = execute_kw(
            uid, "ont_mcp_oauth.client", "read", [[new_id], ["client_id"]]
        )[0]["client_id"]
        print(f"Created client #{new_id} '{CLIENT_NAME}'")

    print(f"  redirect URI: {redirect_uri}")
    print(f"  scope ceiling: read (PKCE public client, refresh enabled)")
    print()
    print("Copy into agent/.env:")
    print(f"  OAUTH_CLIENT_ID={client_id}")


if __name__ == "__main__":
    main()
