# MCP OAuth (Ontash) — `ont_mcp_oauth`

OAuth 2.1 authorization server on top of the free `muk_mcp` MCP server, so remote
AI clients (ChatGPT.com Custom MCP connectors, Claude.ai Custom Connectors,
custom agents) can sign in to Odoo via the standard authorization-code + PKCE
flow and receive short-lived, audience-bound JWT access tokens.

Clean-room implementation of the public OAuth RFCs — an Ontash-owned equivalent
of the paid `muk_mcp_oauth`. **Two scopes:** `mcp:read`, `mcp:write` (mapping 1:1
onto muk_mcp's existing read/write tool enforcement). Bearer API keys from
`muk_mcp` keep working unchanged for local clients.

Target: **Odoo 17 on-premise and odoo.sh. Not for Odoo Online.**

## What it adds

| Endpoint | RFC | Purpose |
|---|---|---|
| `/.well-known/oauth-protected-resource[/mcp]` | 9728 | Protected Resource Metadata |
| `/.well-known/oauth-authorization-server[/mcp]` | 8414 | Authorization Server Metadata |
| `/mcp/oauth/jwks.json` | 7517 | RS256 public keys |
| `/mcp/oauth/register` | 7591 | Dynamic Client Registration (default **OFF**) |
| `/mcp/oauth/authorize` | 6749 + PKCE S256 | authorization code + per-scope consent screen |
| `/mcp/oauth/token` | 6749 | `authorization_code` + rotating `refresh_token` |
| `/mcp/oauth/revoke` | 7009 | idempotent revocation |

Access tokens are RS256 JWTs, `aud`-bound to `<base>/mcp` (RFC 8707), default TTL
900 s. Refresh tokens rotate; replaying a consumed refresh outside the grace
window (default 5 s) revokes the whole token family. Reusing an authorization
code revokes the family too. Every OAuth event is written to muk_mcp's audit log
(secrets never logged — only client-id/jti prefixes).

## Setup

1. Ensure `web.base.url` is your real **HTTPS** origin, and freeze it:
   Settings → Technical → System Parameters → `web.base.url` = `https://odoo.example`,
   and set `web.base.url.freeze` = `True`. **This is security-critical, not
   cosmetic.** Odoo rewrites `web.base.url` from the request's Host header on
   every successful login by a Settings-group user unless the freeze parameter
   is set (`base/models/res_users.py`). The token issuer, the `aud` binding and
   all `/.well-known` metadata derive from it, so an unfrozen value means one
   admin login from another hostname invalidates every live token and
   republishes the OAuth endpoints on that host. The MCP OAuth settings block
   warns while it is unfrozen and offers a one-click **Freeze Base URL**.
2. **Single-database routing is mandatory, not optional**: set `db_name` (or a
   `dbfilter`) in `odoo.conf`. The root `/.well-known/*` documents are served
   without a `?db=` selector, so the moment a second database exists on the
   cluster (even a test copy), every cookieless discovery request fails with
   "No database is selected" and no external client can connect. Prefer
   `db_name` over `dbfilter` on dev machines — the test runner's `-d` overrides
   `db_name` per process, whereas `dbfilter` takes precedence and breaks it.
3. Create clients in **Settings → MCP → OAuth Clients** (or enable Dynamic Client
   Registration in **Settings → General Settings → MCP OAuth (Ontash)** to let
   clients self-register). For a manual client, set the exact redirect URI(s):
   - Claude.ai: `https://claude.ai/api/mcp/auth_callback`
   - ChatGPT.com: the callback shown in its connector UI
4. Tune TTLs and the DCR toggle in the same settings section.

Users can review and revoke their own AI connections under **Preferences → AI
Connections** (no administrator needed); administrators see the same panel on
any user under Settings → Users. A record rule scopes the list to the user's own
grants, and revoking kills that client's live token families immediately.

For a confidential client, use **Regenerate Secret** on the client form — the
secret is displayed once and stored only as a PBKDF2 hash. A confidential client
with no secret is flagged in the form and cannot obtain a token.

## Manual end-to-end test (ChatGPT / Claude.ai)

Public AI clients call your server from the internet, so a local `localhost`
instance is not reachable. Use an HTTPS tunnel:

1. Start a tunnel: `cloudflared tunnel --url http://localhost:8160`
   (or `ngrok http 8160`). Note the `https://...` URL.
2. Set `web.base.url` to that HTTPS URL and freeze it (step 1 above).
3. Enable Dynamic Client Registration (or pre-create a client with the
   connector's redirect URI).
4. **ChatGPT.com**: Settings → Apps → enable Developer mode → Create app →
   Custom MCP → Server URL `https://<tunnel>/mcp` → Auth OAuth 2.1
   (auto-discovered) → Connect → sign in to Odoo → approve → Refresh tools.
5. **Claude.ai**: Settings → Connectors → Custom Connectors → Add → Remote MCP
   URL `https://<tunnel>/mcp` → Auth OAuth (auto-discovered) → sign in → approve.
6. Verify: approve **read-only** (uncheck write) and confirm a write tool is
   refused; then revoke from Preferences → AI Connections and confirm the
   connector must re-authorize.

## Automated tests

Run against a throwaway database (never a production DB), on a free HTTP port:

```
docker exec odoo19-ent-2 sh -c \
  "/opt/odoo/odoo-bin -c /opt/odoo/odoo.conf -d <testdb> --http-port=8072 \
   -u ont_mcp_oauth --test-enable --test-tags /ont_mcp_oauth --stop-after-init"
```

Coverage: PKCE/redirect-URI/JWT unit tests (incl. `alg:none`/HS256 rejection),
full authorization-code flow, code-replay family revocation, refresh
rotation/grace/replay, `/revoke` semantics, JWT verification at `/mcp` (bad
sig/aud/iss/exp/revoked-jti), read-scope-blocks-write, DCR on/off, metadata
shape, and security-negative cases (open-redirect attempts, CRLF in state,
forged/cross-user consent, no-secrets-in-logs). The muk_mcp bearer-key path is
regression-tested to still work.

## Security notes

- PKCE S256 mandatory for all clients; `plain` rejected.
- RS256-only JWTs; local JWKS only; forged `jti` rejected by the token registry.
- Codes/refresh tokens stored only as SHA-256 hashes; client secrets PBKDF2;
  signing private key readable by `base.group_system` only.
- Exact-match redirect URIs; error pages (not redirects) until the URI is
  validated; redirects re-serialized (no header injection).
- Consent POST is CSRF-protected and reloads every parameter server-side from a
  single-use pending-request record — hidden form fields are never trusted.
- Per-IP rate limits on token/register/revoke/authorize; per-client on `/mcp`.
- Dynamic Client Registration is **off by default**; unused DCR clients are
  auto-reaped after 24 h.
- Token/revoke read parameters from the request **body only** (never the query
  string), so codes, PKCE verifiers and tokens can't leak into access logs.
- Token scope fails **closed**: a JWT without an explicit `mcp:write` claim is
  treated as read-only.
- **Per-user method-call gate.** The `call_method` MCP tool (invoke any public
  method — the widest write surface) additionally requires membership in
  **"MCP: Allow Method Calls"** (`ont_mcp_oauth.group_mcp_method_call`;
  administrators via implication). Enforced server-side in a `muk_mcp.tool`
  inherit (`models/mcp_tool.py`) inside the audited execution path, so denied
  attempts appear in the MCP log with status `denied`.
- **Per-user write ceiling.** Only members of the security group
  **"MCP: Allow Write"** (`ont_mcp_oauth.group_mcp_write`; administrators have
  it via group implication) may ever be granted `mcp:write`. The effective
  grant is requested ∩ client ceiling ∩ user ceiling
  (`res.users._mcp_scope_ceiling()`), enforced at three points: the authorize
  GET (non-members never see the write checkbox, and a remembered write
  consent cannot skip past it), the consent POST (a crafted `scope_write=on`
  is discarded), and `_issue_tokens` — the mint choke point — so losing the
  group narrows the very next refreshed token to read-only.
- A replayed authorization code revokes exactly the token family minted from
  that code (tracked via a stored code-hash linkage), not a user+client guess.
- **Internal users only.** Portal/share users are refused at both the consent
  screen and at `/mcp`, matching the boundary muk's bearer keys get implicitly
  from their `base.group_user` ACL.
- JWT verification at `/mcp` is rate-limited per IP (default 300/min, config
  parameter `ont_mcp_oauth.jwt_attempts_per_minute`) so unauthenticated
  signature checking and `jti` probing cannot be used as a CPU amplifier.
- Dynamically registered clients get only the scope they asked for, and only
  the grants they asked for — a client that does not request `refresh_token` is
  never issued one.
- Consent requests are consumed by a single atomic `DELETE ... RETURNING`, so a
  double-submitted consent form cannot mint two authorization codes from one
  user interaction.
- TTL settings are bounded (access token 60 s–24 h, refresh 1–365 days, refresh
  grace 0–60 s) and clamped again on read, so a value hand-edited into System
  Parameters cannot create an effectively immortal token or disable
  refresh-rotation replay detection.
- JWT segments must be strict base64url; lenient decoding (which would give one
  token many equivalent spellings) is rejected.
- Retired signing keys remain resolvable only until their stamped `Valid Until`,
  then are deleted, bounding how long a leaked key can be abused.
- The JWKS endpoint skips a key it cannot parse rather than failing, so one bad
  row cannot take OAuth discovery offline.
- Offboarding revokes: deactivating a user, or demoting them out of internal
  access, immediately revokes their live tokens, and the token endpoint refuses
  to mint for an ineligible account — so re-enabling a user never silently
  resurrects the connectors they had authorised.
- MCP sessions are bound to `user_id` and every lookup re-checks it, so one
  user cannot resume, read or terminate another's session (regression-tested).

### Signing-key rotation vs. revocation

Manage keys under **Settings → MCP → OAuth Signing Keys**.

**Rotate Key** is a graceful hand-over, not a cutover. The outgoing key is
retired but stays resolvable and stays published in the JWKS, so access tokens
signed just before rotation keep working for the rest of their lifetime. Its
`Valid Until` is stamped at rotation from the access-token TTL then in force
(plus one hour of leeway, capped at 24 h); after that it is deleted
automatically. Lowering the TTL later does not retroactively shorten an
already-stamped window.

**Revoke Now** is the incident-response path. If a private key may have
**leaked**, rotating alone is not enough — until the row is gone, an attacker
holding the key can still mint tokens that verify. Rotate first, then Revoke
Now on the retired key: that invalidates every access token it signed,
immediately. The active key cannot be revoked (the server would have nothing
left to sign with).

Retirement is **one-way**: a retired key cannot be reactivated (it may have been
retired because it leaked), and keys can only be deleted through **Revoke Now**,
which refuses to remove the active key.

### Operational note: the model allowlist does not cover `read_resource`

`muk_mcp_access` enforces its model allowlist inside `_resolve_model()`. Three
built-in tools never resolve a model and are therefore **not** restricted by it:
`system_info`, `list_modules`/`list_languages`, and — the one that matters —
**`read_resource`**, which serves `odoo://attachment/<id>` and
`odoo://record/<model>/<id>/<field>` for *any* model, gated only by the user's
own Odoo ACL.

So allowlisting `res.partner` alone does **not** confine an MCP client to
partner data: a read-scoped token can still pull any attachment or binary field
the user can read. Treat the allowlist as a query-surface control, not a data
boundary, and scope MCP users' Odoo permissions accordingly.

### Operational note: read-only scope depends on tool categories

A `mcp:read` grant is enforced by muk_mcp, which refuses any tool whose
`category` is not `read`. All built-in mutating tools (`call_method`,
`create_records`, `update_records`, `delete_records`, `post_message`) are
correctly categorised. But the category of database-record tools is editable in
the UI: **a custom tool that writes while categorised `read` silently gives
every read-only OAuth token write access.** Review the category whenever you add
an MCP tool.

### Residual risk: client impersonation under DCR

With Dynamic Client Registration enabled, anyone who can reach the endpoint may
register a client with an arbitrary display name (e.g. "Claude.ai") and their own
redirect URI. The consent screen therefore always shows the **redirect URI**
alongside the name — teach users to check it. Keep DCR off unless you need it,
and prefer pre-registering clients in Settings → MCP → OAuth Clients.

### Interoperability

- **Client authentication:** `none` (public + PKCE), `client_secret_post`, and
  `client_secret_basic` (RFC 6749 §2.3.1) are all supported; Basic credentials
  are percent-decoded per the RFC, and a failed header auth answers 401 with a
  `WWW-Authenticate: Basic` challenge. Dynamic Client Registration follows the
  RFC 7591 default: a registration that *omits* `token_endpoint_auth_method` is
  confidential (`client_secret_basic`) and receives a secret — public clients
  must ask for `none` explicitly (ChatGPT, Claude.ai and mcp-remote all do).
- **Browser clients:** discovery, register, token and revoke carry `cors='*'`
  so browser-based clients (MCP Inspector) work; `/mcp/oauth/authorize` is
  top-level navigation and needs none.
- **Discovery:** RFC 8414 paths plus `/.well-known/openid-configuration`
  aliases for OpenID-Connect-style probers; `/mcp/` (trailing slash) 308s to
  `/mcp` so pasted URLs still bootstrap.
- **Throttling:** every 429 carries `Retry-After: 60`, and a rate-limited JWT
  on `/mcp` answers **429**, never 401 — a 401 would make clients "fix" their
  token by re-running OAuth, turning throttling into a re-auth storm.
- **Lossy networks:** replaying a just-consumed refresh token *inside* the
  grace window (default 30 s) returns a fresh token pair instead of
  `invalid_grant` — the Auth0/Okta "reuse interval" pattern — so a client that
  lost the response to a timeout self-heals instead of forcing the user to
  re-authorize. Replay *after* the window still revokes the whole family.
- **Clocks:** all expiry/consumption comparisons use the application clock on
  both sides, so app-host vs database-host clock skew cannot misclassify a
  benign retry as a replay or kill authorization codes in flight.

### Deployment note: odoo.sh staging

`data/neutralize.sql` runs automatically on every staging/dev copy of
production: it deletes the signing keys, all token families, and pending
codes/requests, and forces Dynamic Client Registration off. Without it, a
staging copy would keep the production private keys (so production access
tokens verify on staging until expiry) and — because `web.base.url` is frozen —
serve discovery metadata pointing clients at the production endpoints. Do not
delete this file.

### Deployment note: rate limiting behind a proxy

Per-IP rate limits use the request's remote address. Behind a reverse proxy
(odoo.sh, an HTTPS tunnel, or nginx), set `proxy_mode = True` in `odoo.conf` so
Odoo honours `X-Forwarded-For`; otherwise every client shares the proxy's IP and
the limits throttle everyone together.
