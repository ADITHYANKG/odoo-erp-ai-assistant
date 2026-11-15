"""Watch Settings -> MCP -> Notifications fill up and get delivered.

Rows only appear for an OPEN, initialized MCP session when an MCP tool is
created, edited or deleted. The Telegram bot never keeps a session open, so
this script plays a long-lived client (like Claude Desktop) by hand:

1. opens a session (initialize + notifications/initialized),
2. waits while you edit any tool in Odoo -> a row appears, Delivered = off,
3. polls GET /mcp (SSE) -> prints the event, Odoo flips Delivered = on,
4. closes the session (DELETE) unless --keep.

Usage (from the repository root):
    uv run python scripts/16_mcp_notifications_probe.py [--keep]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx
from dotenv import dotenv_values

ENV = dotenv_values(Path(__file__).resolve().parent.parent / ".env")
MCP_URL = ENV.get("MCP_URL") or "http://127.0.0.1:8069/mcp"
PROTOCOL = "2025-11-25"  # session-based revision; the stateless one has no notifications


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="leave the session open")
    args = parser.parse_args()

    key = ENV.get("MCP_BEARER_KEY")
    if not key:
        raise SystemExit("MCP_BEARER_KEY missing from .env")
    headers = {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": PROTOCOL,
    }

    with httpx.Client(timeout=30, headers=headers) as client:
        init = client.post(MCP_URL, json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL,
                "capabilities": {},
                "clientInfo": {"name": "notifications-probe", "version": "0"},
            },
        })
        init.raise_for_status()
        sid = init.headers.get("Mcp-Session-Id")
        if not sid:
            raise SystemExit(f"No session id returned: {init.text}")
        client.headers["Mcp-Session-Id"] = sid
        client.post(MCP_URL, json={
            "jsonrpc": "2.0", "method": "notifications/initialized",
        }).raise_for_status()
        print(f"Session open: {sid}")

        try:
            while True:
                answer = input(
                    "\nEdit/save any tool in Odoo (MCP -> Tools), check the Notifications tab "
                    "(Delivered = off),\nthen press Enter to poll, or type q to quit: "
                )
                if answer.strip().lower() == "q":
                    break
                stream = client.get(MCP_URL, headers={"Accept": "text/event-stream"})
                stream.raise_for_status()
                events = [line for line in stream.text.splitlines() if line.startswith("data:")]
                if not events:
                    print("No pending notifications.")
                for line in events:
                    print("Received:", line[5:].strip())
        finally:
            if not args.keep:
                client.delete(MCP_URL)
                print("Session closed.")


if __name__ == "__main__":
    main()
