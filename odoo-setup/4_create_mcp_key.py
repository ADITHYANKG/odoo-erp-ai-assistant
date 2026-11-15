"""Mint a muk_mcp bearer key for the agent service via JSON-RPC.

The plaintext is returned exactly once — copy it into agent/.env as
MCP_BEARER_KEY. The key belongs to the JSON-RPC user from .env (admin for
now); per-executive OAuth tokens replace this in the Telegram phase.

Usage:
    ./venv/Scripts/python experiments/4_create_mcp_key.py [--name NAME] [--scope read|write]
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
    parser.add_argument("--name", default="ont-agent (phase 1)")
    parser.add_argument(
        "--scope",
        choices=["read", "write"],
        default="read",
        help="Key scope; phase 1 only reads, so 'read' is the safe default.",
    )
    args = parser.parse_args()

    uid = authenticate()
    key = execute_kw(
        uid, "muk_mcp.key", "generate_playground_key", [args.name, args.scope]
    )

    print(f"Created MCP key #{key['id']} '{key['name']}'")
    print(f"  scope:      {key['scope']}")
    print(f"  rate limit: {key['rate_limit']}/min")
    print(f"  prefix:     {key['key_prefix']}")
    print()
    print("One-time plaintext (copy into agent/.env as MCP_BEARER_KEY):")
    print()
    print(f"  {key['plaintext']}")


if __name__ == "__main__":
    main()
