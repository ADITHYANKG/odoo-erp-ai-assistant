# MCP Trace (Ontash)

After every successful **write-category** Model Context Protocol (MCP) tool
call, posts an internal note on the affected record's chatter:

> 🔧 via MCP: **update_records** by Administrator — fields: city, phone

So the record's right-side log always shows what the agent did, next to
Odoo's own field-tracking entries — a human-readable audit trail on top of
the technical log in Settings → Technical → MCP → Logs.

## Behavior

- Traced: `create_records`, `update_records`, `call_method` (method name
  shown). Reads stay silent by design.
- Not traced: `post_message` (its content IS the chatter entry), `send_email`
  (posts its own linked message when a record is attached), `delete_records`
  (the record's chatter is gone with it — deletions stay in the MCP log).
- Notes are `mail.mt_note` (internal): followers receive no email from agent
  activity. Author = the authenticated MCP user.
- Mass writes are capped at 10 noted records per call (rest logged only).
- A posting failure can never break or roll back the tool call
  (savepoint + suppress, per the muk_ai_chatter precedent).

Implementation: one `muk_mcp.tool._execute` inherit
(`models/mcp_tool.py`) — MuK code untouched; chains with `ont_mcp_oauth`'s
call_method gate via normal model MRO.

## Tests

```
docker exec odoo19-ent-2 sh -c "/opt/odoo/odoo-bin -c /opt/odoo/odoo.conf -d ont_test --http-port=8072 -i ont_mcp_trace --test-enable --test-tags /ont_mcp_trace --stop-after-init"
```
