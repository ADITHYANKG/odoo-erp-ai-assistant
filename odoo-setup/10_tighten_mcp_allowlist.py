"""Tighten the muk_mcp_access allowlist after a security review.

The allowlist (19 entries) was populated incidentally by the demo seeders
and never reviewed. This closes the surfaces an attacker or a prompt
injection could reach even with a legitimately approved write:

  res.users        write -> off   group membership = privilege escalation
  hr.employee      write -> off   personal / salary data
  res.company      write -> off   company + mail configuration
  mail.mail        archived       create_records on it would send mail to
                                  ANY address, bypassing send_email's
                                  recipient allowlist; read exposes every
                                  sent email body
  res.partner.bank archived       vendor bank-account changes are the
                                  classic payment-fraud vector; IBANs
  confirm.stock.sms archived      stray transient wizard = an SMS surface

Left alone (legitimate business writes, each still behind MCP scope +
human approval): res.partner, crm.lead, sale.order(.line), purchase.order,
stock.picking, account.move.

Archiving is reversible: untick Active on the entry, or set active=True.
send_email is NOT affected by archiving mail.mail — that tool creates the
record with sudo() inside ont_mcp_mail and never consults the allowlist.

Idempotent: already-correct entries are reported and left untouched.

Usage:
    uv run --project agent python odoo-setup/10_tighten_mcp_allowlist.py
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

ACCESS_MODEL = "muk_mcp_access.model"

# Models that keep read access but must never be written through MCP.
REVOKE_WRITE = ["res.users", "hr.employee", "res.company"]
# Models that should not be reachable through MCP at all.
ARCHIVE = ["mail.mail", "res.partner.bank", "confirm.stock.sms"]


def jsonrpc(service, method, args):
    """Call an Odoo JSON-RPC service method and return the result."""
    payload = {
        "jsonrpc": "2.0",
        "method": "call",
        "params": {"service": service, "method": method, "args": args},
        "id": 1,
    }
    response = requests.post(JSONRPC_ENDPOINT, json=payload, timeout=60)
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


def snapshot(uid):
    """Every allowlist entry, archived ones included."""
    return execute_kw(
        uid,
        ACCESS_MODEL,
        "search_read",
        [[]],
        {
            "fields": ["id", "model_name", "allow_read", "allow_write", "active"],
            "context": {"active_test": False},
            "order": "model_name",
        },
    )


def show(title, rows):
    print(f"\n{title}")
    print(f"  {'model':22} {'read':>5} {'write':>6} {'active':>7}")
    for row in rows:
        print(
            f"  {row['model_name']:22} {str(row['allow_read']):>5} "
            f"{str(row['allow_write']):>6} {str(row['active']):>7}"
        )


def main():
    uid = authenticate()
    before = snapshot(uid)
    by_name = {row["model_name"]: row for row in before}
    show("BEFORE (entries this script targets):",
         [r for r in before if r["model_name"] in REVOKE_WRITE + ARCHIVE])

    changed = 0
    for model_name in REVOKE_WRITE:
        entry = by_name.get(model_name)
        if not entry:
            print(f"\n- {model_name}: not in the allowlist, nothing to do")
            continue
        if not entry["allow_write"]:
            print(f"\n- {model_name}: write already off")
            continue
        execute_kw(uid, ACCESS_MODEL, "write", [[entry["id"]], {"allow_write": False}])
        print(f"\n- {model_name}: write revoked")
        changed += 1

    for model_name in ARCHIVE:
        entry = by_name.get(model_name)
        if not entry:
            print(f"\n- {model_name}: not in the allowlist, nothing to do")
            continue
        if not entry["active"]:
            print(f"\n- {model_name}: already archived")
            continue
        execute_kw(uid, ACCESS_MODEL, "write", [[entry["id"]], {"active": False}])
        print(f"\n- {model_name}: archived (no longer reachable via MCP)")
        changed += 1

    after = snapshot(uid)
    show("AFTER (same entries):",
         [r for r in after if r["model_name"] in REVOKE_WRITE + ARCHIVE])
    active = [r for r in after if r["active"]]
    print(f"\n{changed} change(s). Allowlist now exposes {len(active)} model(s) via MCP:")
    print("  " + ", ".join(sorted(r["model_name"] for r in active)))
    writable = sorted(r["model_name"] for r in active if r["allow_write"])
    print(f"  writable ({len(writable)}): " + ", ".join(writable))


if __name__ == "__main__":
    main()
