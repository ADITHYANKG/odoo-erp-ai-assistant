"""Point Odoo's public base URL at a new tunnel (cloudflared / ngrok) and pin it.

Quick tunnels hand out a fresh random hostname on every restart, and Odoo's
`web.base.url` drives the OAuth issuer, the token audience and every
/.well-known document. Two things bite you each time:

  1. If the URL is FROZEN, Odoo keeps serving the previous tunnel's hostname,
     so AI clients are pointed at a dead address.
  2. If it is NOT frozen, Odoo rewrites it from the request host on every
     Settings-group login -- and behind a TLS-terminating tunnel that often
     lands as http://, which public AI clients refuse.

This sets the value, freezes it so nothing rewrites it, and restarts the Odoo
container so the HTTP workers drop their cached copy. That last step is the
subtle one: changing the parameter from outside the server leaves running
workers serving the OLD value, which makes the database and the actual
endpoint disagree.

Usage:
    uv run --project agent python odoo-setup/3_set_public_base_url.py https://new-name.trycloudflare.com
    uv run --project agent python odoo-setup/3_set_public_base_url.py --unfreeze
    uv run --project agent python odoo-setup/3_set_public_base_url.py --show
"""
import os
import subprocess
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

ODOO_URL = os.getenv("ODOO_URL", "http://localhost:8069")
ODOO_DB = os.getenv("ODOO_DB")
ODOO_USERNAME = os.getenv("ODOO_USERNAME")
ODOO_PASSWORD = os.getenv("ODOO_PASSWORD")

JSONRPC_ENDPOINT = f"{ODOO_URL}/jsonrpc"
CONTAINER = "main-odoo-1"
FREEZE_KEY = "web.base.url.freeze"
URL_KEY = "web.base.url"


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


def read_params(uid):
    ids = execute_kw(uid, "ir.config_parameter", "search",
                     [[["key", "in", [URL_KEY, FREEZE_KEY]]]])
    rows = execute_kw(uid, "ir.config_parameter", "read", [ids, ["key", "value"]])
    return {row["key"]: row["value"] for row in rows}


def set_param(uid, key, value):
    ids = execute_kw(uid, "ir.config_parameter", "search", [[["key", "=", key]]])
    if ids:
        execute_kw(uid, "ir.config_parameter", "write", [ids, {"value": value}])
    else:
        execute_kw(uid, "ir.config_parameter", "create", [{"key": key, "value": value}])


def unset_param(uid, key):
    ids = execute_kw(uid, "ir.config_parameter", "search", [[["key", "=", key]]])
    if ids:
        execute_kw(uid, "ir.config_parameter", "unlink", [ids])


def restart_odoo():
    print(f"Restarting {CONTAINER} so workers reload the parameter ...")
    subprocess.run(["docker", "restart", CONTAINER], check=True,
                   stdout=subprocess.DEVNULL)


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1

    uid = authenticate()

    if args[0] == "--show":
        for key, value in sorted(read_params(uid).items()):
            print(f"{key} = {value}")
        return 0

    if args[0] == "--unfreeze":
        unset_param(uid, FREEZE_KEY)
        restart_odoo()
        print("Unfrozen. Odoo will rewrite web.base.url on the next admin login.")
        return 0

    url = args[0].rstrip("/")
    if not url.startswith("https://"):
        print(f"Refusing {url!r}: public AI clients require an HTTPS endpoint.")
        return 1

    set_param(uid, URL_KEY, url)
    set_param(uid, FREEZE_KEY, "True")
    restart_odoo()
    print(f"{URL_KEY} = {url}  (frozen)")
    print("Reload Settings -> MCP -> Connect AI; the HTTPS warning should be gone.")
    print(f"Verify: {url}/.well-known/oauth-authorization-server")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
