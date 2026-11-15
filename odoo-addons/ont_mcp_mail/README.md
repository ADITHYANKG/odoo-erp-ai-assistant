# MCP Mail (Ontash)

One Model Context Protocol (MCP) tool — **`send_email`** — added on top of
`muk_mcp` via the documented extension path (`_inherit = 'muk_mcp.mixin'` +
`@mcp_tool`; no MuK code modified).

## What it does

Sends an email through Odoo's configured outgoing mail server:

- `to` (comma-separated), `subject`, `body` (plain text; escaped and rendered
  as paragraphs — model-supplied HTML is never trusted).
- Optional `record_model` + `record_id` (+ `report_ref`): renders that
  record's PDF report server-side (default: the model's main `qweb-pdf`
  report) and attaches it — "email the quotation" in one call, without the
  client ever touching the binary.
- Optional `export`: generates a complete CSV/XLSX attachment from an Odoo
  model, fields, and domain or record IDs. Record rules and field access are
  enforced as the authenticated user, and file bytes never enter model context.
- The mail is created in Odoo's queue (visible under Settings → Technical →
  Email), stamped with the authenticated user's identity, and sent
  immediately; SMTP failures surface as a clean tool error.
- Set the system parameter `ont_mcp_mail.sender_email` to the SMTP account or
  a verified sender alias to use one stable From address for assistant mail.

## Security

- **Write category** — requires the `mcp:write` token scope (per-user gated
  by `ont_mcp_oauth`'s "MCP: Allow Write" group when installed), and any
  category-based confirmation layer (the Telegram agent's Approve tap)
  applies unchanged.
- Recipient validation and record access checks (including the
  `muk_mcp_access` allowlist, consulted with the **read** category — attaching
  only reads the record) run as the real user; only the final mail-queue
  write uses a narrow `sudo()`, because the queue model is admin-only.

## Tests

`tests/test_send_email.py` — validation, captured sending via
`mock_mail_gateway` (never a real SMTP call in tests), attachment rendering
with a `qweb-text` report (no wkhtmltopdf dependency), and read-scope denial.

```
docker exec odoo19-ent-2 sh -c "/opt/odoo/odoo-bin -c /opt/odoo/odoo.conf -d ont_test --http-port=8072 -i ont_mcp_mail --test-enable --test-tags /ont_mcp_mail --stop-after-init"
```
