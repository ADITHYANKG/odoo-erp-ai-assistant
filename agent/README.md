# ont-agent

Acme AI-over-ERP agent: lets company executives query their Odoo data through
chat platforms (Telegram first). A standalone *client* of the Odoo stack — it
talks to muk_mcp's `/mcp` endpoint (Model Context Protocol over Streamable
HTTP) and calls its large language model through **OpenRouter only**.

Managed with [uv](https://docs.astral.sh/uv/); dependencies live in
`pyproject.toml` / `uv.lock`, the environment in `.venv/` (auto-created).

## Setup

```bash
cp .env.example .env
# mint a bearer key (from the sibling odoo repo) and paste it into .env:
python ../odoo-setup/4_create_mcp_key.py
```

Every tunable is an environment variable — model, embedding model and its
vector width, provider base URL, database paths, recursion and summarization
thresholds, and all of the limits below. `.env.example` documents each one
with its default; nothing is hardcoded in a module.

## Phase 1 — prove the MCP connection

```bash
uv run python -m ont_agent.mcp_client     # or: uv run ont-agent-list-tools
```

Lists the 22 tools exposed by the Odoo MCP server (20 from muk_mcp, plus
`send_email` from ont_mcp_mail and `search_documents` from ont_mcp_rag) and makes one authenticated `whoami` call. Set `LOG_LEVEL=DEBUG` for verbose diagnostics (stderr).

## Phase 2 — ask a real question

```bash
uv run python -m ont_agent.chat "How many contacts do we have?"
# or: uv run ont-agent-chat "..."
```

A LangGraph agent picks among the read-only Odoo tools (write tools are
withheld client-side on top of the read-scope key), answers grounded in tool
results, and logs every tool call with timings. The model comes from
OpenRouter only — set `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` in `.env`.
Access-denied data (the muk_mcp_access allowlist) is reported plainly; runaway
tool loops degrade into a plain-language answer at the recursion limit.

## Phase 3 — Telegram, as yourself

```bash
uv run ont-agent-bot
```

Long-polls Telegram and runs an embedded FastAPI app (uvicorn on the same
event loop) at `AGENT_PUBLIC_URL` (default `http://localhost:8181`) serving
`/oauth/callback`, `/health`, and a dormant `/webhook/whatsapp` (Meta's
verification handshake activates once `WHATSAPP_VERIFY_TOKEN` is set). In
Telegram:
`/start` → tap **Log in to Odoo** → sign in with YOUR Odoo account → approve
the consent screen → the bot greets you by name. From then on every question
runs under *your* Odoo user: your record rules, your audit trail. `/logout`
revokes the tokens.

Interactive answers use Telegram's native animated drafts in private chats
while the model generates them. Drafts are plain text and update about once
per second; the completed response is persisted with full formatting, buttons,
freshness information and attachments. Unsupported chats fall back to an
editable placeholder. Set `AGENT_TOKEN_STREAMING=false` to restore one-shot
delivery. Proactive messages remain one-shot.

Setup once: `odoo-setup/5_create_agent_oauth_client.py` registers the OAuth
client (→ `OAUTH_CLIENT_ID`), and `TELEGRAM_BOT_TOKEN` comes from @BotFather.

Production notes: put `AGENT_PUBLIC_URL` behind a public HTTPS host so phones
can complete the login. The OAuth tokens in `data/agent.db` are encrypted at
rest — set `AGENT_TOKEN_KEY` (see `.env.example`); without it they are stored
in plaintext and the bot says so at startup.

**Logs.** The service writes diagnostics to **stderr** and never opens a log
file itself — that is what makes it drop cleanly into a container or systemd
unit, which capture stderr for you. Whoever launches the process therefore
decides where it lands. Locally we redirect to `logs/bot.log`
(gitignored via `*.log`):

```powershell
Start-Process uv -ArgumentList 'run','ont-agent-bot' `
  -WorkingDirectory . -RedirectStandardError .\logs\bot.log -WindowStyle Hidden
```

Run it in a terminal instead and the same output goes to the console. Set
`LOG_LEVEL=DEBUG` for verbose diagnostics (it also unmutes the noisy
`httpx`/`mcp` loggers). Nothing rotates the file yet — truncate it if it
grows, or let a supervisor handle rotation once the service is containerised.

## Phase 4 — conversation memory

Each Telegram chat is a persistent conversation thread: follow-up questions
("and how many of those are companies?") resolve against the previous turns.
Implementation: a LangGraph `AsyncSqliteSaver` checkpointer (`data/memory.db`,
override with `AGENT_MEMORY_DB`), thread id = chat id. Old turns are
summarized automatically once history grows past a token threshold, so memory
never grows without bound. The same checkpointer is the foundation for phase
5's human-confirmation interrupts.

Commands: `/new` (or `/reset`) forgets the conversation but keeps the login;
`/logout` revokes tokens **and** wipes the conversation, so the next person
binding the chat inherits nothing. Memory survives bot restarts (it is on
disk); the one-shot `ont-agent-chat` CLI stays stateless by design.

This layer is deliberately *short-term* and chat-scoped. Preferences that
should outlive a conversation live in a separate store — see phase 7.

## Phase 5 — write actions behind human confirmation

The bot can now create, update, and delete records and post chatter messages —
but a write NEVER runs on the model's say-so alone. LangGraph's
`HumanInTheLoopMiddleware` pauses the graph at an interrupt *before* the tool
executes; Telegram shows the exact action with ✅ Approve / ❌ Cancel buttons
(plus ✏️ Edit for email proposals: your next message revises the draft and a
corrected proposal comes back for approval), and only an Approve tap resumes
the thread and runs the tool. Rejection tells the model not to retry. Typing a
new message while a confirmation is open auto-cancels the pending action. The
proposal message is preserved after the decision (verdict appended, result
sent separately), so the chat keeps a readable audit trail.

**The model is only offered tools the user can actually use.** The bot reads
the *granted* OAuth scope from the token response, so a read-only user never
sees `send_email`, `create_records`, `update_records`, `delete_records` or
`post_message` at all — it says up front that it cannot, instead of proposing
an action and asking you to approve something guaranteed to fail. The scope is
re-read on every token refresh, so granting or revoking write in Odoo takes
effect within one token lifetime without a `/logout`.

`call_method` (business actions: confirm orders, post invoices, approve
leave) is narrower still: offered only to members of the Odoo security group
**"MCP: Allow Method Calls"** (administrators via implication) — learned from
`whoami` at login and enforced server-side by an `ont_mcp_oauth` gate either
way. Every use still requires the Approve tap. Users must `/logout` + `/start`
after a group change for the bot to re-learn their eligibility.

**When an approved action is refused**, the frozen proposal reads
**🚫 Blocked by Odoo** — not "Approved" (which reported the button press, not
the outcome) and not "Cancelled" (which means *you* declined; the two must stay
distinguishable in a scrollback). The reply is then *only* the refusal and its
reason. It deliberately does not answer the underlying question instead: on a
blocked exfiltration attempt that consolation answer is precisely the data the
server just refused to send (live incident 2026-08-31 — an approved
`send_email` to an outside address was refused, and the bot printed all 19
overdue invoices into the chat).

Server-side prerequisites (both required, `odoo-setup/6_enable_agent_writes.py`
sets them): the OAuth client's scope ceiling raised to write, and
`allow_write` on the muk_mcp_access allowlist entry per writable model
(`res.partner` only for now). Users must `/logout` + `/start` once to approve
the new write permission on the consent screen — refresh can never widen scope.

## Phase 6 — charts and PDF reports

Ask for "a bar chart of contacts per country" or "a PDF report of our
companies" and the file arrives in the chat: two **local tools** —
`render_chart` (Plotly → PNG photo, styled on the data-viz reference
palette; ten chart types: bar, horizontal_bar, line, area, scatter, pie,
donut, treemap, funnel, waterfall) and `create_pdf_report` (fpdf2 →
document) — run inside the agent
after it retrieves the data over MCP, and the bot sends the rendered files
along with a short text reply. Same grounding rule as text answers: chart and
table data come only from tool results. Rendering never triggers the
Approve/Cancel gate (it changes nothing in Odoo); temp files are deleted
after sending.

PNG export runs through Kaleido, which drives a local Chrome/Chromium —
auto-detected if installed, or fetch a private copy once with
`uv run plotly_get_chrome`. (Deployment note: the future agent container
image must include Chromium.)

**Email delivery**: the `send_email` MCP tool (our `ont_mcp_mail` Odoo
module) sends mail through Odoo's configured SMTP server — "send quotation
S00004 to x@y.com" attaches the record's PDF server-side. Write-category:
requires write scope and always pauses for the Approve tap; the mail is
logged in Odoo's email queue under the user's identity. (Emailing
agent-rendered charts/PDFs is a later increment.)

## Phase 7 — long-term memory (preferences that outlive the conversation)

Phase 4's checkpointer is *short-term*: `/new` wipes it, `/logout` wipes it,
and summarization compresses old turns away. A second, independent layer keeps
working preferences — "always show me amounts in lakhs", "I prefer PDF
reports" — keyed by **Odoo login** rather than chat thread, so they survive all
three.

Implementation: a LangGraph `AsyncSqliteStore` in its own `data/facts.db`
(`AGENT_FACTS_DB`), with LangMem's manage/search tools bound to the user's
namespace, and semantic recall through **OpenRouter embeddings**
(`AGENT_MEMORY_EMBED_MODEL`, default `openai/text-embedding-3-small`; its
vector width must match `AGENT_MEMORY_EMBED_DIMS` or the index silently
corrupts). Leave the embedding model empty to run the store without vector
search; `AGENT_LONG_TERM_MEMORY=0` disables the layer entirely.

Namespace labels escape the login injectively — LangGraph rejects periods, so
a raw email address would have failed *every* write, and a naive replacement
would collapse `john.doe` and `john_doe` into one namespace, which is a
cross-user leak. The identity boundary matters because a single chat can change
who it is: `/logout` then `/start` as someone else reads a different namespace.

`/forget` erases the calling user's preferences **and** the current
conversation — clearing only the store let the model quote a just-deleted
preference back from its own earlier reply. `/logout` deliberately does *not*
clear preferences: they belong to the Odoo user, not the chat.

Preferences shape *how* answers are presented (format, currency, detail level),
never *what* the data says — every business number still comes from a fresh
tool call. Inspect what is stored with
`experiments/12_inspect_agent_memory.py`.

## Phase 8 — safety and abuse controls

Layers, outermost first. Each is independent; none relies on the model
behaving.

**Who may talk to the bot at all.** `AGENT_ALLOWED_TELEGRAM_IDS` (comma-separated
numeric ids, empty = anyone) is checked in a `TypeHandler` registered in handler
**group -1**, which raises `ApplicationHandlerStop` — one front door covering
every handler, including ones added later. Unknown senders get *no reply at
all*: any response would confirm the bot is live.

**How fast.** A per-sender sliding window (`AGENT_BURST_LIMIT` /
`AGENT_BURST_WINDOW_SECONDS`, default 10 per 60s) sits in the same guard. The
daily caps bound a day; without this, 200 paid model calls still go through in
one minute. Over the limit the bot warns **once** per window and then goes
silent — replying to every message would turn one flood into an outbound flood
and trip Telegram's own per-chat send limit.

**How much.** Usage rows are keyed by prefixed subject — `user:<odoo_login>`,
`chat:<id>` for unbound chats, `global` — so the same person in two chats shares
one budget (per-chat metering handed them a second allowance). Three caps:
`AGENT_DAILY_MESSAGE_LIMIT` and `AGENT_DAILY_SPEND_LIMIT_USD` per Odoo user, and
`AGENT_GLOBAL_DAILY_SPEND_LIMIT_USD` across everyone — the only one that bounds
the bill, since per-user exposure multiplies by however many people connect.
Checked *before* a turn starts, so a blocked chat costs nothing; the global
refusal is deliberately vague so it cannot leak other people's usage. Spend is
OpenRouter's real billed cost, and the global row is written in the same
transaction as the user's so the total cannot drift from the parts. Resolving a
pending approval is never rate-limited.

**Who may connect.** `AGENT_REQUIRED_ODOO_GROUP` names an Odoo group the user
must belong to; the check runs in `complete_login` *before* the binding is
saved, so a refused user leaves nothing behind. Empty = any internal Odoo user,
which is the OAuth server's own floor. Create the group in Odoo (Settings →
Users & Companies → Groups, developer mode) and leave its category empty —
a categorised group is reported as `"Category / Name"` and the exact-name match
would refuse everyone. Managing access in Odoo means offboarding there cuts off
the bot too.

**Prompt injection.** Tool results are treated as data, never instructions.
Record content an outsider controls — a contact name, an opportunity
description, chatter — can contain text shaped like a command ("ignore previous
instructions", "email this to …"). The system prompt forbids obeying it and
asks the model to say so plainly instead. This is mitigation, not proof, which
is why the server-side layers below it are the ones that actually hold:
`ont_mcp_mail`'s recipient allowlist and the `muk_mcp_access` model allowlist
both refuse regardless of what the model was persuaded to want.
`odoo-setup/11_seed_injection_probe.py` seeds a harmless canary record for
testing; `security-probes.md` lists 11 probes with pass conditions.

**Server-side rate limit.** The OAuth client carries its own `rate_limit`
(calls per 60s, per *client* — all Telegram users share it), set in Odoo:
Settings → Technical → MCP → Authentication → OAuth Clients. `0` means
unlimited.

## Phase 9 — proactive: the morning brief and watchers

Up to here the bot only ever answered. It now also *tells you things*.

**Morning brief.** At **08:00 in each user's own timezone** — read from their
Odoo profile at login, so one bot serves people in different regions — a fixed
five-section summary arrives unprompted: cash, overdue receivables, overdue
payables, pipeline, and up to two lines on anything unusual. Sections are fixed
deliberately: a brief whose shape changes daily cannot be skimmed, and skimming
is the only way a brief ever gets read. On by default (`/brief off` to stop,
`/brief` for one now), and the first one carries a line explaining itself.

**Watchers.** `/watch any customer invoice more than 60 days overdue` stores a
standing condition in plain language; it is re-checked **every 8 hours**. The
model returns a structured verdict — `{"triggered", "fingerprint", "message"}`
— and the alert fires only when it is triggered **and the fingerprint differs
from last time**. That dedupe is what makes watchers bearable: one that repeats
at every check gets muted, and then the one that mattered is missed too.
`/watchers` lists them, `/unwatch <n>` removes one, five per person.

Watchers therefore suit **standing states** — overdue, below reorder point,
unassigned — not transient events. A condition that becomes true and false
again inside one 8-hour window is never seen, and that is the deliberate
trade for the cost below.

**One bug worth remembering.** `send_brief` stamped "already briefed
today" from the wall clock instead of from the moment the tick was
evaluating. In production those always agree, so it was invisible — it
surfaced only when the calendar rolled over during a test run. Left alone,
a tick evaluated for one day could stamp another and re-send someone's
brief. Anything scheduled here now takes its `now` as an argument rather
than reading the clock halfway through.

**Scheduling is state-driven, not cron.** A ticker runs every few minutes and
asks *"what is due?"*, answering from stored timestamps rather than firing at a
fixed moment. That is idempotent, survives restarts, and gives catch-up free: a
laptop asleep at 08:00 and woken at 11:00 still delivers the brief, while one
woken at 14:00 correctly stays quiet (`AGENT_BRIEF_CATCHUP_HOURS`). A cron job
set for 08:00 on a sleeping machine simply never fires — the worst failure a
reminder can have. The tick itself costs nothing: two small SQLite reads, no
model call, no Odoo call. Only genuinely-due work spends anything.

Three rules govern everything proactive, and each exists for a specific
failure:

- **Read-only** — no human is present to approve a write.
- **Stateless** — sharing the conversation thread would let an 08:00 brief
  silently auto-cancel an approval left open overnight.
- **Spend-capped, not message-capped** — the daily message limit exists to stop
  a person hammering the bot, and nobody is hammering anything at 8am; but both
  *spend* ceilings apply, because unattended jobs are exactly how a bill gets
  away from you.

**Cost in practice — watchers are the spend driver, not the tick.** A brief is
roughly 6 cents; each watcher check is 2–4 cents. The tick itself is free. So
the bill is `watchers x users x checks per day`, and it is the only figure here
that scales with headcount:

| Users | Watchers each | Interval | Calls/day | Rough $/day |
|-------|---------------|----------|-----------|-------------|
| 2 | 5 | 60 min | 240 | $5–10 |
| 50 | 5 | 60 min | 6,000 | $120–240 |
| 50 | 5 | **480 min** | 750 | **$15–30** |

At a 60-minute interval a real team would burn through
`AGENT_GLOBAL_DAILY_SPEND_LIMIT_USD` before lunch — the cap doing its job, but
as an outage. **`AGENT_WATCHER_INTERVAL_MINUTES` was therefore raised to 480
(8 hours) on 2026-09-02.** Shortening it multiplies the bill by the same
factor, so raise the spend caps in the same change or the bot simply stops
part-way through the day.

## Phase 10 — voice questions

An executive in a car cannot type. Send a Telegram voice note and the bot
replies *"🎤 I heard: …"* then answers it.

Transcription goes through OpenRouter's **`/audio/transcriptions`** endpoint —
*not* `/chat/completions`, which rejects a transcription model outright with
"cannot be used with the chat/completions endpoint". Telegram records OGG/Opus
and OpenRouter accepts it **as-is**, verified end to end against the live API,
so there is no ffmpeg transcoding step and none is needed. The response returns
`usage.cost`, so voice spend lands in the same ledger as everything else — about
$0.0045 per minute of audio.

The transcript is echoed rather than acted on silently: transcription is never
perfect, and seeing the words is how someone catches a misheard customer name
before trusting the answer built from it. Over-long notes are refused *before*
download (`AGENT_VOICE_MAX_SECONDS`), the daily budget is checked before any
transcription happens, and a failure degrades to "I couldn't make that out"
rather than an exception — the person is driving. `AGENT_VOICE=0` declines
voice notes politely.

## Phase 11 — links, follow-ups and freshness

Three small things that make an answer feel connected to Odoo rather than
floating above it. All three ride one **`Evidence`** object collected as a
side effect of the tool calls the model was already making — record names
with their Odoo URLs, and rows seen per model. No extra lookups.

**Deep links.** Invoice numbers, customers and orders in an answer become
tappable, opening that exact record: `…/odoo/account.move/26`. Linking runs
*after* HTML escaping and inside the inline renderer, so model output still
cannot inject markup and links compose with bold. One constraint worth
knowing: Telegram renders no link inside a monospace block, and tables are
monospace blocks — so records that appear only in a table get a compact
"🔗 Open in Odoo" line beneath instead. Without that fallback the feature
would do nothing on precisely the answers where record numbers cluster.

**Follow-up buttons.** *📊 Show as chart · 👥 By customer · 📧 Email me
this* — one tap instead of retyping. Each button runs a canned follow-up on
the **same conversation thread**, so "show that as a chart" resolves against
the answer just given with no re-fetching. Which buttons appear is decided
in code from the answer's shape, but gated on whether the turn actually
fetched anything: "what can you help me with?" mentions "customer invoices"
in passing, and that alone was enough to offer a customer breakdown under an
answer holding no data at all. 📧 is withheld from read-only users, since
offering a button the server will refuse is the failure fixed in phase 5.

**Freshness stamps.** A faint line under any data-backed answer: *"As of
17:42 · 19 invoices"*, in the user's own timezone. It silently answers "is
this current?" before anyone thinks to ask — which matters most for a number
someone scrolls back to next week. Counts come from rows the tools really
returned, so a pure `search_count` answer gets a timestamp and no count
rather than a dishonest "0 records".

## Phase 12 — the action log, `/audit` and the weekly digest

Writes used to leave three traces — a Langfuse trace, an Odoo chatter note
from `ont_mcp_trace`, and a line in stderr — and the bot could query none of
them. There was no way to answer *"what has this thing been doing?"*

Every decided write is now recorded with **what happened**, not what was
intended:

| Outcome | Meaning |
|---|---|
| `done` | Approved and executed |
| `blocked` | Approved, then refused by Odoo (recipient allowlist, permissions) |
| `cancelled` | The user tapped Cancel |
| `revised` | The user tapped ✏️ Edit and reworded it |

The distinction between `done` and `blocked` is the point: a log that
records intent rather than outcome is worse than none. The capture is
complete *because every write tool is confirm-gated* — nothing reaches Odoo
by another route, which is the property the whole audit rests on.

**`/audit`** gives administrators a cross-user view (`/audit 30` for a wider
window), capped with the remainder **stated** rather than silently dropped.
"Administrator" means the exact group `Role / Administrator`: twenty-one
Odoo groups are named "<App> / Administrator", and matching loosely would
hand this view to a sales manager. The flag is read at login, so an existing
session needs `/logout` + `/start` before `/audit` works.

**The weekly digest** arrives Monday morning in each user's timezone, via
the same ticker as the brief: the actions taken in their name, plus one line
of activity context. A week with no activity sends nothing; a week with
questions but no changes says so explicitly — that reassurance only means
something because the busy weeks read differently. Retention is 90 days
(`AGENT_ACTION_LOG_DAYS`), far longer than anything else the bot keeps,
because an audit trail that cannot reach back is not one.

## Phase 13 — coloured approval buttons

The approval tap is the last gate before the bot writes to Odoo or sends
mail, and it is usually made on a phone, quickly. Three identical grey
buttons made "send this" and "don't" distinguishable only by reading.

A plain traffic light now: **✅ Approve green, ❌ Cancel red, ✏️ Edit blue**.
Green means go and red means stop, and it never changes meaning between
prompts — which is the point, since a signal that inverts is worse than no
signal.

An action that **cannot be taken back** — `delete_records` and `send_email`,
the two with no compensating action — is flagged as **⚠️ Approve** rather
than by recolouring. Turning Approve red would put two red buttons side by
side on exactly the prompts where telling them apart matters most; a warning
word is unambiguous where a second red is not. A test asserts Approve and
Cancel are never the same colour, across every action type.

`style` is additive: a client too old to understand it renders an ordinary
button and everything still works. Callback data is unchanged, so approval
messages sent before the deploy still function when tapped — also pinned by
a test, since that would have been a silent breakage.

There is **no opacity or shade control**: the API offers three named styles
and the client decides how they render. Inline keyboards (what the approval
prompt uses) already render softer than reply keyboards.

## Phase 14 - uploaded images and PDF files

Send a photo or a PDF and ask about it. The Odoo tools stay available in the
same turn, which is the point: *"here is a vendor bill - is it already in our
system, and does the amount match?"* is one question, not two.

**Nothing parses files here.** OpenRouter reads PDFs server-side and the
model reads images directly, so there is no `pypdf`, no `openpyxl`, and no
new dependency - `langchain-openrouter` already converts LangChain data
content blocks into OpenRouter's own shapes. Verified live on 2026-09-02: a
generated invoice image and a generated PDF were both read back correctly,
and each turn's real cost arrived through the usual callback, so uploads join
the same spend ledger as everything else.

**Uploads use a separate fallback list.** `AGENT_UPLOAD_FALLBACK_MODELS` is
empty by default and never inherits the text fallbacks. Measured the same
day: one configured text fallback accepts text only, and another takes
images but not files. Routing a file to either would not fail - it would
answer from the caption alone, confident about a document it never saw.
Primary model or nothing is the safer default.

**The file does not stay in the conversation.** Once answered, the upload is
replaced on the thread by a one-line description (`bill.pdf (PDF, 1.2 MB)`).
Left in place, the encoded payload would be written to `memory.db` and
re-sent to the model on every later question in that chat - a single 4 MB
upload quietly taxing the whole conversation, in both bill and database size.
A test asserts the payload is gone and the question remains.

**Refusals are cheap and specific.** Feature off, unsupported type, oversize,
over budget - all decided from what Telegram *declares*, before a byte is
downloaded, the same order the voice path uses. An unsupported file is
refused **by name** ("that looks like a spreadsheet"), because "I can't read
that" teaches nobody where the boundary is.

Spreadsheets are deliberately excluded. They are not a vision problem: they
need local parsing and a row-limit strategy, since a 10,000-row sheet will
not fit in context.

Settings: `AGENT_UPLOADS`, `AGENT_UPLOAD_MAX_MB` (default 5),
`AGENT_UPLOAD_FALLBACK_MODELS`, `AGENT_UPLOAD_PDF_ENGINE`.

## Phase 15 - OAuth tokens encrypted at rest

`agent.db` holds live Odoo access and refresh tokens. Anyone with the file
can act as those users until the tokens expire - acceptable on a laptop,
not on a host handed to a customer, and the first thing a security review
finds.

The two token columns are encrypted with Fernet (`crypto.py`,
`AGENT_TOKEN_KEY`), authenticated so a wrong key fails loudly rather than
decrypting to plausible garbage. Values carry an `enc:v1:` prefix, which is
what keeps *"written before encryption"* distinguishable from *"cannot be
decrypted"* - the whole failure design rests on telling those apart, and
inferring it from Fernet's own format would conflate them.

**Migration needs nothing.** A plaintext row still reads and re-encrypts on
its next save, which happens on every token refresh.

Three failure modes, each with a deliberate answer:

| Situation | Behaviour |
|---|---|
| Malformed key | **Startup refuses**, naming the variable and the generator command |
| Key rotated | Bot runs; the log states the key changed; only affected users re-run `/start` |
| Key **removed** | Same - and this one nearly escaped |

That last row was found while verifying the other two. Without it,
`get_binding` handed the ciphertext to Odoo as a bearer token and got a 401
that reads as a random auth failure rather than the config change made
minutes earlier. A remembered key fingerprint also distinguishes "key
removed" from "never set", which are otherwise identical on disk.

**Scope, stated honestly:** a key stored beside the database protects
backups, snapshots and copies. It does not protect against an attacker
already on the running host, who can read the key as easily as the data.
`memory.db` and `facts.db` remain unencrypted by decision (2026-09-02) -
see CLAUDE.md for the SQLCipher and disk-encryption routes that were
investigated and declined.

## Phase 16 - document search: answering from company policy

*"What is the harassment policy?"* is now answered from the company's own
PDF, with a citation, instead of from the model's general knowledge. Built
as a separate Odoo module, `custom_addons/ont_mcp_rag`, which has its own
README with the full design; the summary that matters here:

- **Setup is one checkbox.** Settings -> Technical -> AI Document Search,
  tick a folder. A scheduled action indexes new uploads every 10 minutes,
  so adding a document needs no further action.
- **Retrieval is hybrid** - pgvector cosine plus PostgreSQL full-text,
  fused by reciprocal rank - because similarity misses exact terms and
  keywords miss paraphrase.
- **The similarity gate is deliberately loose (0.30).** Measured over 14
  answerable and 6 unanswerable questions, the two populations separate by
  0.022 - lowest genuine 0.379, highest absent 0.357. A threshold in that
  band would fit the sample and break on the next phrasing, so the gate
  only removes obvious nonsense and a SYSTEM_PROMPT rule makes the model
  judge relevance, which it can do and no scalar can.
- **`query` is declared REQUIRED in the tool schema.** Left to inference,
  the model opened with `search_documents({})`, was told there was no
  search text, and retried - three calls where one does, enough on slower
  questions to exhaust the recursion limit and degrade the answer. Fixing
  it took citations from 7/10 to 10/10.

Measured live against the HR policy (`experiments/13`):

```
search_documents called : 11/11   <-- the headline number
expected section cited  : 10/10
absent-policy control   : declined honestly, naming the folder searched
```

**The tool-call rate is the headline** because the worst failure here is
invisible to every other measure: the model never searches, answers from
general knowledge, and produces something that reads perfectly while not
being this company's policy.

Two follow-on fixes came from watching it run:

- **A policy answer is not a dataset.** It is grounded (a tool ran) and its
  section numbers read as figures, so both existing gates offered *Show as
  chart* under the sick-leave policy. `DOCUMENT_TOOLS` plus a
  `from_documents` flag withhold chart and breakdown when every tool used
  was a document search. Email stays - "email me the leave policy" is
  reasonable. A mixed turn keeps its data buttons.
- **Stray LaTeX never reaches the reader** (`latex.py`). The model emits it
  unbidden whenever an answer compares numbers, and a travel-policy answer
  arrived reading `Premium Economy for segments $\ge$ 6 hours`. Symbols are
  CONVERTED, not deleted: "at least six hours" and "six hours" say
  different things, and on a reimbursement rule silently dropping the
  operator is worse than showing markup. Wired into `render_markdown`, so
  briefs, watcher alerts and digests are covered too.

## Phase 17 - conversation memory and preferences on Postgres

The agent kept three SQLite files beside the code. Two of them now live in
their own Postgres container (`docker-compose.yaml`).

**The reason is operational, not performance.** At this scale SQLite reads
in microseconds against a 26-second model call. But when this ships to a
client, a Postgres database lands inside their existing backup regime,
while a SQLite file in a Docker volume is something a person has to
remember. Secondarily, SQLite is single-writer, so more than one agent
container is impossible by construction.

**Its own container, not Odoo's server.** The agent is deliberately a
standalone service; a client may run Odoo Online or odoo.sh where creating
a database on their server is not an option. Keeping it separate also
means `docker compose down` on the Odoo stack cannot take conversation
memory with it.

**`agent.db` deliberately did not move.** The test suite constructs
`InMemorySaver` and never instantiates the SQLite checkpointer or store, so
moving those two cost ~10 lines in one function and zero test changes.
`agent.db` is the opposite: 110 call sites construct
`BindingStore(tmp_path / "t.db")`, and moving it means `uv run pytest`
requires a running Postgres. That is worth paying once, at
containerisation, not twice.

`AGENT_DB_URL` empty keeps the SQLite files, so the switch stays reversible
and a machine without Docker still works.

**Two Windows-only traps, both diagnosed the hard way:**

- psycopg's async mode **refuses the default ProactorEventLoop** - the pool
  never opens and the bot dies at startup. `bot.py:main` sets
  `WindowsSelectorEventLoopPolicy` on win32 only; the Linux container this
  ships in has no such problem.
- The DSN must use **`127.0.0.1`, not `localhost`**. Windows resolves
  localhost to `::1` first, Docker publishes the port on IPv4 only, and the
  connection hangs until the pool times out with no useful error.

Verified live: checkpoints write and read back, and `adelete_thread`
genuinely wipes a thread - which is what `/new` and `/logout` depend on.

## Phase 18 - all state on Postgres, SQLite retired

`agent.db` was the last SQLite file, and the one holding OAuth credentials
and the audit trail. It now lives in the same Postgres as conversation
memory and preferences, and there is no SQLite fallback left anywhere.

**`uv run pytest` needs Docker from here on.** That is the whole reason
this was deferred once: 110 test call sites constructed
`BindingStore(tmp_path / "t.db")`, and `tmp_path` gave every test a private
database for free. `tests/conftest.py` rebuilds that with a Postgres
**schema** per test - cheap enough to do ~130 times a run, and a test that
opens the same DSN twice (the key-rotation ones do) still lands on the same
tables. If the container is not up the suite skips with one sentence naming
the compose command, rather than 130 connection errors. Suite time went
39s to ~61s.

**`chat_id` is BIGINT, and that one is not cosmetic.** Telegram supergroup
ids sit below -2^31. SQLite's INTEGER is 64-bit so this never surfaced;
Postgres INTEGER is 32-bit and would overflow - failing for *some* users
only, which is the worst way to fail. Verified live against the real
database with -1002147483649.

**Deleting the migration code found a real bug.** The `ALTER TABLE` block
and the `usage_daily` rekey existed only to upgrade SQLite files created in
August, so they went. The suite immediately failed with `column "is_admin"
does not exist` - because `is_admin` and `last_digest_week` had never been
added to `_SCHEMA` at all. They existed ONLY as migrations, so any
genuinely fresh database had been missing them since the day they were
introduced. Nobody had created one until now.

Three smaller dialect changes worth knowing, each found by a failing test:
- `ON CONFLICT ... DO UPDATE SET message_count = message_count + 1` is
  **ambiguous** in Postgres; the target table has to be named.
- `cursor.lastrowid` does not exist - `INSERT ... RETURNING id`.
- `INSERT OR REPLACE` has no equivalent; spelled out as `ON CONFLICT`.

## Phase 19 - token refresh that survives slow turns and Odoo restarts

Three fixes in `auth.py:ensure_access`, all found by reviewing the codebase
for holes rather than by anything breaking.

**A token could expire mid-turn with nothing to recover it.** The refresh
margin was 60 seconds, chosen before we knew how long a turn runs. With a
900s server TTL that meant a turn could START with 60 seconds of validity -
while `AGENT_RECURSION_LIMIT` is 64 and the primary model measures ~26s to
first token. There is no 401 handling anywhere, so the MCP call simply
failed and the answer degraded silently. The margin is now
`AGENT_TOKEN_REFRESH_MARGIN_SECONDS`, default **300**, clamped to half the
token's life so a short TTL cannot be swallowed whole.

**Any OAuth error destroyed the login.** `ensure_access` deleted the
binding on every `OAuthError`. Only `invalid_grant` means the token family
is actually dead; `server_error` or a 5xx through the tunnel is transient,
and deleting for those silently unbound every user whose token needed
refreshing during an Odoo restart. Transient failures now raise and keep
the login, so the next question just works once the server is back.

**Concurrent refresh is now serialised per chat.** Refresh tokens rotate,
so redeeming one invalidates it, and the proactive ticker is a separate
asyncio task that can reach a chat at the same moment a handler does.
`ont_mcp_oauth` forgives this inside a 30-second grace window - but that
window is a system parameter a client can set to 0, and the agent should
not depend on how someone else's Odoo is configured. The lock re-reads the
binding after acquiring, which is the whole point: a waiter captured its
copy before waiting and would otherwise refresh with a spent token.

Verified against the live server, not just fixtures: two concurrent
`ensure_access` calls made **one** refresh and returned the same token, the
stored expiry moved from 840s to 600s ahead, and a simulated `server_error`
left the binding and its refresh token intact.

**What this does not fix:** a pathological 64-step turn can still outrun a
900s token entirely. Nothing short of mid-turn re-authentication would, and
tools carry a bearer token bound at graph-build time, so a 401 partway
through cannot be retried without rebuilding the graph.

## Phase 20 - retention, and the open transaction it uncovered

Two tables grew without bound. Conversation checkpoints were the real
problem: each is a full snapshot of a chat's state, one chat's history
reached **98 MB** before the Postgres move, and an active user generates
roughly 60 a day. `/new` and `/logout` clean up after an attentive user;
nothing cleaned up after one who simply stopped.

A daily pass (`housekeeping.py`) now deletes threads idle for 30 days and
usage counters older than 90. Thirty because that is when the login dies
anyway - the OAuth server's `refresh_token_ttl_days` is 30 - so retention
can never discard a conversation a working login could still have used.

**Whole threads only.** `adelete_thread` is the one prune API LangGraph
implements for Postgres; `aprune` raises `NotImplementedError` and its
docstring warns that dropping ancestor checkpoints can make delta channels
reconstruct as **empty**, corrupting the surviving state rather than merely
losing history. Hand-rolled SQL with a silent failure mode was not worth it.

**Scheduled by state, not cron** - the same pattern as the briefs, stamped
in `meta`, so it is idempotent and catches up after downtime. Stamped
*before* the work, so a pass that dies does not retry every five minutes
for the rest of the day.

### The bug the tests found

Writing the tests, `AsyncPostgresSaver.setup()` hung. `pg_stat_activity`
showed `CREATE INDEX CONCURRENTLY` blocked on a virtual transaction id -
and the blocker was **the bot's own connection, sitting `idle in
transaction`**, its last query the ticker's watcher read.

In psycopg's default transaction mode a plain `SELECT` opens a transaction
that nothing closes; `BindingStore` only committed on writes, so between
them the connection parked an open transaction indefinitely. That does two
things quietly: it blocks `CREATE INDEX CONCURRENTLY`, and it stops VACUUM
reclaiming dead tuples - which would have made this entire retention pass
delete rows Postgres could never reclaim.

The store now connects with `autocommit=True`. The existing `commit()`
calls become harmless no-ops, and `record_usage` asks for a transaction by
name because its two INSERTs must land together. Confirmed live: zero
`idle in transaction` connections after the restart, where there had been
one. (Phase 22 moved the store onto a pool; autocommit now applies to every
pooled connection, and the count is still zero.)

## Phase 21 - thread rotation (2026-09-04)

Retention above only reaches threads that go IDLE, which leaves out exactly
the user it was written for: someone who chats daily is never idle, so
their single thread grew without bound at roughly 60 checkpoints a day.

Rotation caps it. Past `AGENT_THREAD_MAX_CHECKPOINTS` (100, about 33 turns,
~10 MB) the chat moves to a fresh thread `<chat>-<epoch>` and the retired
one is deleted. The user is told nothing, because nothing is lost.

**Why nothing is lost.** `create_agent`'s state is `messages` plus
`jump_to` (ephemeral) and `structured_response` (unused here), and
`SummarizationMiddleware` writes only `messages`. So `messages` IS the
durable conversation, and the summarizer already bounds it. The agent asks
for the latest state and never lists checkpoint history - there is no
`alist` and no time travel in the codebase - so what rotation drops has no
reader.

**How the new thread is seeded, and how it is not.** The carried messages
are prepended to the turn's own input and the thread comes into existence
through a normal invocation. The obvious alternative, `aupdate_state` on
the new thread, would raise `InvalidUpdateError("Ambiguous update, specify
as_node")`: with no saved checkpoint LangGraph resolves the node only when
the input channel is itself a node (`pregel/main.py:2375`), and a compiled
agent's is `__start__`, which is not.

**Two phases.** `plan_rotation` decides and writes nothing; `commit()` runs
once the turn has returned - including a turn that ended in a pending
approval, whose decision `resume_pending` must find on the new thread.
Before commit the epoch is unchanged and the old thread intact, so every
crash point is safe: the next turn simply rotates again, and the orphaned
new thread ages out through retention. The old thread is deleted only after
its content is provably carried.

**The epoch lives in `meta`,** keyed `thread_epoch:<chat_id>`, not in a
`bindings` column. `Binding` is a frozen dataclass rebuilt field by field
in `auth.py:_refresh_binding`, where five fields are already carried by
hand; forgetting a sixth would reset the epoch on the next token refresh
and resurrect a thread id that had been deleted. Epochs are monotonic and
survive `/logout`, so a reused id can never collide with a thread whose
delete failed. Epoch 0 keeps the bare chat id, so threads that existed
before this change were not orphaned by the deploy.

**Rotation never happens while an approval is open** - the interrupt lives
in that thread's checkpoint and cannot be carried - and it is planned after
`answer_question`'s auto-cancel, so a cancellation is carried rather than
stranded. `clear_memory` and `resume_pending` resolve the thread through
`current_thread`; without that, `/new` would delete `<chat>` and report a
cleared conversation while the real one carried on.

Failing rotation costs disk, so nothing about it raises: an unreadable
checkpoint count returns 0, a broken snapshot returns "no rotation", and
`commit` swallows and logs. Breaking the user's actual question to save
space would be the wrong trade.

## Phase 22 - a pool for the store, and an evicting notice map (2026-09-04)

The last two findings of the 2026-09-04 review, both recorded then as
"notes, not work". Re-checked before implementing, one of them was not.

### The shared connection was a crash, not a slow leak

`BindingStore` opened a single `AsyncConnection`, shared by three tasks on
one event loop: the Telegram handlers, the proactive ticker and the embedded
callback server. Reads are safe - psycopg serialises statements and buffers
results - but a **transaction is connection scoped**, and `record_usage`
opens one explicitly so its two INSERTs land together.

Measured directly rather than reasoned about, and the result was worse than
the note claimed. It does not lose data quietly; it raises:

- a concurrent store method that commits (`set_meta`, `save_binding`,
  `log_action`, almost all of them) →
  `ProgrammingError("Explicit commit() forbidden within a Transaction context")`
- a concurrent write that does not commit →
  `OutOfOrderTransactionNesting`

So the window was small - two INSERTs against a ticker that runs every 300
seconds - but anything landing in it crashed that request outright.

Each method now borrows a connection from a pool for the length of its own
statements, which makes the collision impossible by construction rather than
improbable. `_purge_expired` takes its caller's connection so a public call
is still one acquisition. The pool is the store's **own**, not LangGraph's:
that one is built with `row_factory=dict_row` because LangGraph reads
columns by name, while all 46 store reads are positional - sharing it would
have meant rewriting every read as well as every acquisition. Sizes come
from `AGENT_STORE_POOL_MAX_SIZE` (4, short queries) and
`AGENT_DB_POOL_MIN_SIZE`/`MAX_SIZE` for LangGraph's (the heavy writer),
which were hardcoded literals until now.

The `db` property is gone; `connection()` replaces it, and is also the seam
tests use to age or forge rows. The suite got *faster* - about 160s to 97s -
since work no longer queues behind one connection.

### `_TOLD_ABOUT_PRIVATE` now sheds entries

The map of groups already told this bot is private-only was a `set[int]`
that nothing ever cleared. Small in practice - the allowlist check runs
first, so only an allowlisted sender can add to it - but an unbounded
collection with no eviction all the same.

It is now `dict[int, float]` pruned on every call, which is the rule
`BurstLimiter` already documents for the same reason. A time window
(`AGENT_GROUP_NOTICE_TTL_SECONDS`, 3600) rather than a size cap because it
**self-empties**: after the TTL with no group traffic nothing is retained,
where an LRU would hold N entries for ever. A group re-added later gets one
more notice, which is the harmless direction.

`_REFRESH_LOCKS` in `auth.py` is the same shape and was deliberately left
alone: one lock per chat this process has served, bounded by user count, and
the locks have to persist to do their job.

## Phase 23 - quiet hours (2026-09-04)

An overposting review asked whether the bot defaults to silence when it is
unsure. Mostly it does — `parse_verdict` treats any unparseable reply as
"not triggered" so a confused model can never manufacture an alert, and a
watcher fires only when the fingerprint has actually changed. But one path
ignored the clock entirely.

`brief_due` and the Monday branch of `digest_due` both use
`zone_for(binding, settings)` and wait for `brief_hour`. `watcher_cutoff` is
pure interval arithmetic with **no timezone at all**, so at
`AGENT_WATCHER_INTERVAL_MINUTES=480` one of the three daily checks lands in
the user's night by construction. `digest_due` had a smaller version of it:
its catch-up branch returns due for any later day in a week with no digest
yet, regardless of hour, so a digest missed on Monday could arrive at 3am
Tuesday.

`in_quiet_hours(binding, now, settings)` is one pure predicate beside the
other scheduling functions, reusing `zone_for`'s existing fallback chain
(user tz -> `AGENT_DEFAULT_TIMEZONE` -> `datetime.UTC`). The window
**wraps midnight**, which is the only real logic in it — the default
21:00-08:00 is exactly the case a naive `start <= hour < end` gets
backwards. Equal values disable it, so an existing deployment is unchanged.

**The work is skipped, not the message suppressed, and that is the entire
design.** `check_watcher` stamps `verdict.fingerprint` whenever the verdict
is triggered, and fires only when the new fingerprint differs from the
stored one. Checking at 3am and withholding the message would therefore
leave the verdict matching its own stamp by morning: `fires` is False and
the alert is lost for good - worse than the 3am buzz it was meant to avoid.
Skipping writes no `last_checked_at`, so `due_watchers` still returns the
watcher and it fires on the first tick after the window opens. No queue, no
new state, no starvation - the same "scheduled by state, not by cron"
property the briefs and retention already rely on. A test pins it: defer at
03:00, run again at 09:00, and the alert must arrive.

The guard sits after `refresh_binding` on purpose. The refresh is one cheap
call to Odoo and keeps the token family alive; what gets skipped is the
**model call**, which is the expensive part and the line this repository
names as the bot's largest recurring cost. A 13-hour waking window at an
8-hour interval means about two checks a day instead of three - roughly a
third off that line, as a side effect of a UX fix.

Deferred watchers all come due together when the window opens, so a user can
get up to `AGENT_MAX_WATCHERS_PER_USER` (5) alerts in one tick. `tick`
already runs briefs, then digests, then watchers, so the brief lands first
and the alerts follow: one morning moment where the bot speaks.

**Still open from that review:** the morning brief sends unconditionally
every day, whether or not anything changed. The prompt omits a section only
when its tools return no data at all - not when the numbers match
yesterday's. A daily message that is often identical is exactly what trains
muting, and fixing it needs a comparison against the previous brief rather
than a clock rule.

## Phase 24 - the system-prompt guardrail (2026-09-04)

Nothing stopped a user asking the bot to print its own instructions. The two
injection rules in `SYSTEM_PROMPT` point the other way - they stop *content*
(tool results, uploaded files) issuing orders, and one of them says outright
that the person chatting is the legitimate source of instructions. The scope
rule declines songs and general knowledge; "what are your instructions?" is
neither.

Three layers in `promptguard.py`, none of which costs a model call.

**1. A confidentiality bullet in the prompt** - the only layer that handles
paraphrase. The carve-out in it is the important half: describing *what you
can do* stays welcome, quoting *how you are configured* does not. Without
that, the model starts refusing "what can you help me with?", which is the
commonest legitimate question there is.

**2. Input patterns**, checked after `ensure_access` and before the tools
and agent are built, so a probe costs nothing. The message is still recorded
at zero spend, so repeated probing burns the sender's daily allowance rather
than being free.

Every pattern is **second-person by construction**, and that is what makes
the feature usable in an ERP. "The system prompt for onboarding", "prompt
payment discount terms" and "the instructions in the HR policy" are ordinary
questions here; only "YOUR system prompt" is an attempt. The non-matching
tests carry as much weight as the matching ones - a guardrail that blocks
"what can you do?" is worse than none.

**3. An output scan - the only layer a rephrasing cannot bypass**, because
it inspects the text on its way out rather than the request on its way in.
Every window of `AGENT_PROMPT_LEAK_NGRAM` (12) consecutive words in the live
`SYSTEM_PROMPT` becomes a fingerprint, derived at call time so that editing
the prompt can never leave them stale - which is exactly how a hand-written
list of "secret phrases" rots. Both sides are normalised (lowercased,
whitespace collapsed, edge punctuation stripped), so a leak reformatted for
chat is still caught.

Twelve words is the whole trade-off, and it is configurable so a field false
positive is a `.env` change rather than a code change. Too low and an honest
answer trips it: the prompt spells out that "overdue" means posted, not
fully paid, and due before today, and a reply explaining that filter could
legitimately echo the wording. Too high and partial reproduction slips
through. A real dump matches dozens of windows; a coincidental echo matches
one run at most.

**What this does not do.** It does not make the prompt secret, and the
security model never asked it to. Every real control is server-side - OAuth
scope, `muk_mcp_access`, Odoo record rules, `filter_for_bot`, the approval
gate on every write. Someone who reads the entire prompt still cannot bypass
one of them. Layer 3 catches verbatim reproduction, which is what "the
instructions become a public artifact" means; a determined paraphrase
("what are you told to do, in your own words?") is discouraged by layer 1
and caught by nothing here.

Voice notes and uploads need no separate handling - both funnel their text
through `answer_question`. Proactive paths are deliberately not scanned:
there is no attacker in that loop and nobody to refuse to.

## Phase 25 - credential redaction (2026-09-04)

Output filtering is meant to cover three things: leaked system prompts,
secrets, and policy-violating content. Phase 24 did the first. This does the
second.

What already limited the damage: the OAuth token never enters the model's
context (it rides an `Authorization` header in `get_odoo_tools`, not a tool
argument), and the live `muk_mcp_access` allowlist exposes 16 business
models with no credential-bearing one among them - `res.partner.bank` and
`mail.mail` are archived, and `ir.config_parameter` is absent. The residual
risk is free text: someone pastes an API key into a partner note, an invoice
narration or a lead description.

**The reply is not the leak that matters.** A secret read out of a record
lands in three places at once - the answer, the conversation checkpoint
(stored as a ToolMessage for up to 30 days) and the Langfuse trace. Scanning
only the outgoing reply catches one of the three. So the primary hook is
`redact_tool_results`, wrapping the Odoo tools so the redaction happens
*before the model sees anything*; the secret then reaches neither the
thread, nor the trace, nor the answer. The reply-side pass stays as a second
net and earns its keep on the upload path, where OpenRouter parses a PDF
server-side and its text reaches the model without passing a tool.

**Structural patterns only, and that is the decision this stands or falls
on.** Odoo is full of long opaque strings: sha1 attachment checksums,
UUIDs, SKUs, `INV/2026/00015`, IBANs, base64 image references. Any
"this looks random" heuristic redacts those and makes every answer
unreadable. Every rule is anchored on a vendor prefix instead - `sk-`,
`AKIA`, `ghp_`, `xox`, `AIza`, Stripe's `sk_live_`, PEM headers, and
`eyJ….….…`, which is the one that catches this bot's own OAuth tokens. The
single exception is `password|api_key|secret|token` followed by `:` or `=`,
where that punctuation is the entire safety margin: "Password reset
requested" does not match, `password=hunter2xyz` does. It is flagged in the
code as the first thing to drop if a client's data ever trips it.

It **redacts rather than replaces**, unlike the prompt guard: the span
becomes `[redacted: AWS access key id]` and the rest of the answer survives,
because a note with one stray key still has a legitimate answer in it. Only
the KIND is ever logged - writing the value into `bot.log` would move the
secret rather than remove it, and a test asserts it.

Two smaller things worth keeping: the tool wrapper returns the **original
result object** when nothing is found, so the overwhelmingly common path
involves no shape conversion and no behaviour change; and it reads
`args_schema` with `getattr`, because it wraps every Odoo tool where
`_wrap_links` only ever sees the handful in `LINKABLE_TOOLS`.

The module is `redaction.py`, not `secrets.py` - `auth.py` does
`import secrets` for `token_urlsafe`, and a same-named module inside the
package is a shadowing trap waiting for the first relative import.

**Still not done: output moderation for policy-violating content**, the
third item. It needs a classifier call on every reply, on a bot already
paying ~26s to first token, to police authorised employees using their own
ERP. The prompt's scope rule plus the action log is the proportionate
answer, and the cost is not.

## Phase 26 - latency recorded per model call (2026-09-10)

Every number this project had about model speed was a one-off.
`google/gemini-3.7-flash` measured **25.9s** to first token on 2026-09-01
and **3.3s** on a live request nine days later, and nothing wrote either
down. Model choice, the watcher interval and the fallback list all turn on
latency, and all three were being argued from benchmarks somebody had to
remember to run.

`metrics.py` records **one row per model call** in `llm_calls`: the model
that answered, the provider that served it, timings, tokens and cost. Read
it back with

```bash
uv run ont-agent-latency --days 7     # add --kind chat | brief | watcher
```

which prints calls, median and p90 time to first token, median total and
spend per model, slowest first. Medians rather than means — one 90-second
outlier swamps a mean, and the question is what a typical answer costs in
waiting.

### Time to first token cannot be measured on this side

The agent streams graph state, not tokens, so wrapping a call yields its
TOTAL duration and nothing finer. The real figure is OpenRouter's
`GET /generation` (`latency`), fetched afterwards by generation id. That
endpoint returns metadata rather than inference, so **it costs no tokens**.

Two things about that endpoint were verified live rather than assumed:

- **`generation_time` is the whole request at OpenRouter's edge**, so
  decode time is `generation_time - latency`, and `total_ms -
  generation_ms` is everything that is *not* the model — our network hop
  and client overhead.

  This was got wrong first, and the way it was got wrong is worth keeping.
  OpenRouter's OpenAPI spec illustrates the fields with `latency: 1250`
  against `generation_time: 1200` — a total smaller than its own prefix,
  which looks like proof that `generation_time` excludes the wait. Those
  are hand-written placeholders, not a captured request. The first four
  real calls settled it: `latency + generation_time` came to 4376-12138ms
  while our own wall clock around the entire HTTP call was 3350-6951ms, and
  a provider cannot spend longer than the client waited. `generation_time`
  alone sat just under our clock every time, the remaining 464-1494ms being
  ours. **Prefer one live request over an example value in a spec.**
- **The lookup has to back off.** A 63-token answer's row was readable
  inside 1.5s; a 5130-token one still 404ed after three flat attempts and
  read fine a moment later. Long answers are exactly the slow ones a
  latency table exists to catch, so a flat delay would have lost the
  readings that matter most. Five attempts at growing waits cover ~22s.

### The generation id only exists in the callback

`ChatOpenRouter` does not override `_combine_llm_outputs`, and the base
implementation returns `{}` — so the `LLMResult` handed back by `ainvoke`
has an empty `llm_output`. Probed live, it comes back as `[]`. Callbacks
receive `flattened_output` instead, built per result as
`LLMResult(generations=..., llm_output=res.llm_output)`, which keeps it.
Reading the id from the return value would have yielded nothing, silently,
for ever. `extract_call` reads both `llm_output` and `response_metadata`,
since live traffic showed each carrying the id.

### It can never make an answer slower

Samples go to a **bounded** `asyncio.Queue` and a background worker does the
fetching and the writing, well after the reply has gone out. Three
consequences, each deliberate:

- **A full queue drops samples and logs a running count.** While OpenRouter
  is unreachable the worker stalls on retries while calls keep finishing;
  an unbounded queue would turn a measurement feature into a memory leak.
  Losing samples is the right failure for a measurement table.
- **A failed lookup still stores the row**, with our own wall clock and a
  NULL time to first token. The report counts those separately and says
  "time to first token known for N", because a median over three readings
  out of two hundred calls is a different claim from one over all of them.
- **The handler is an `AsyncCallbackHandler`**, not the sync
  `BaseCallbackHandler` that `SpendAccumulator` uses. LangChain dispatches a
  sync handler from an async run through `run_in_executor` — on a worker
  thread, where touching an `asyncio.Queue` is undefined. Spend gets away
  with it because it only adds a float; enqueuing does not. Chat models also
  fire `on_chat_model_start`, never `on_llm_start`.

Retention is 90 days in the same daily pass as everything else. Unlike
`usage_daily` this is a row per **call** — several per question — so it is
the larger of the two tables by a wide margin.

## Known gaps

**The morning brief goes out whether or not anything changed.** Found in the
overposting review of 2026-09-04, alongside the quiet-hours hole that Phase
23 closed. The brief prompt omits a section only when its tools return no
data at all - never when the numbers are identical to yesterday's. A daily
message that is often the same as the last one is exactly what trains a
reader to mute it, which undoes the point of sending it. The fix is a
comparison against the previous brief, not another clock rule, so it was
left out of Phase 23 rather than bolted on.

**The primary model is slow.** `google/gemini-3.7-flash` measured 25.9
seconds to first token, about ten times every alternative benchmarked, and
every answer pays it. See the table below. An earlier probe the same day
gave 5.5s for the same model, so re-measure before concluding it is the
model rather than an OpenRouter provider route.

**`OPENROUTER_FALLBACK_MODELS` takes at most three entries.** A fourth makes
OpenRouter reject every request, which surfaced as "Sorry — something went
wrong" on every question. `config.py` now refuses to start on a bad value
rather than letting it reach users, since this module's whole promise is to
fail at startup rather than midway through a conversation.

## Bot API 10 modernisation

`python-telegram-bot` 22.8 already targets **Bot API 10.0**, so both of
these are available today with no upgrade.

**Streamed drafts.** `send_message_draft` shows an animated, ephemeral
preview while an answer is being generated. The bot reuses one nonzero draft
ID throughout a turn so Telegram animates each update, then sends the complete
message normally to persist it. Native drafts are **private-chat only** and
ephemeral, so an editable placeholder remains the compatibility fallback.

**Coloured buttons.** `InlineKeyboardButton(style=…)` accepts `PRIMARY`,
`SUCCESS` and `DANGER`, plus `icon_custom_emoji_id`. The approval prompt is
the obvious candidate — Approve green, Cancel red — since it is the last
gate before the bot writes to the ERP or sends mail, and it is usually
tapped quickly on a phone. Older clients ignore `style` and render a normal
button, so it degrades safely.

### Why token streaming was not the win it looked like

Streaming only helps if the model emits text progressively. Measured on
2026-09-01, medians of three runs on the same prompt:

| Model | First text | Total | Chunks / 100 chars | $/1M out |
|---|---|---|---|---|
| `anthropic/claude-haiku-4.5` | 1.10s | 2.91s | 9.2 | $5.00 |
| `google/gemini-3.5-flash-lite` | 1.22s | 1.94s | 0.9 | $2.50 |
| `deepseek/deepseek-v4-flash` | 2.74s | 4.36s | 4.5 | $0.16 |
| `google/gemini-3.7-flash` *(configured)* | **25.90s** | 27.43s | 0.9 | $3.75 |

Gemini Flash buffers: it produces nothing, then dumps the answer in a couple
of chunks. Streaming it would change "wait, then see the answer" into
"wait, then see the answer arrive in two flashes".

Two things follow. First, streaming is a **model decision, not a UI one** —
and model choice must be judged on tool-calling accuracy against
`csuite-probes.md`, not prose latency, because choosing the right Odoo tool
with the right filters is the actual job. Second, and more urgently, **the
configured model's 26-second latency is a live problem independent of
streaming** — about ten times every alternative measured. An earlier probe
the same day gave 5.5s for the same model, so re-measure before concluding
it is the model rather than a bad provider route.

## Observability — Langfuse tracing + 👍/👎 evaluation

With `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` set in `.env` (from the
self-hosted Langfuse at `LANGFUSE_BASE_URL`, default http://localhost:3000 →
project → Settings → API Keys), every Telegram turn becomes a full trace:
the large language model (LLM) generations with token counts and cost, every
MCP tool call with inputs/outputs, and timings. Trace names:
`<channel>-ask` (questions) and `<channel>-resume` (Approve/Cancel/Edit
decisions), where the channel is `telegram`, `whatsapp`, `googlechat` or
`slack`; `user` = the Odoo login, `session` = the chat, tag = the channel.

**Cost tracking:** generation costs are the *actual amounts OpenRouter
charged* for each request — OpenRouter returns the billed cost in every
response and the handler ingests it directly (`cost_details`), so offers and
price changes are always reflected. No price table to maintain; the
OpenRouter dashboard remains the billing source of truth.

**Evaluation loop:** every finished answer carries 👍/👎 buttons; a tap
writes a `user-feedback` score (1/0) onto that exact trace and retires the
buttons. Approval prompts keep their Approve/Edit/Cancel buttons instead —
feedback applies to answers, not proposals.

Leave the keys empty to disable observability entirely — the bot then never
touches the Langfuse SDK and behaves exactly as before. Tracing failures
(server down, bad keys) are never fatal; the bot answers regardless.
Implementation: `src/ont_agent/tracing.py` (the only module that imports
`langfuse`), wired per-turn in `bot.py` via the LangChain callback handler.

On the server, Langfuse runs as its own compose project in `/opt/langfuse`
and the bot reaches it at `http://172.17.0.1:3000`; setup and operation are in
[deploy/langfuse/README.md](deploy/langfuse/README.md).

## Evaluation

Live signals landing as scores on Langfuse traces:

1. **User feedback** — 👍/👎 buttons (`user-feedback`) under every answer.
2. **Decision scores** — every Approve/Cancel/Edit tap (`action-decision`,
   scored onto the trace that proposed the action: falling approval rate =
   worsening proposals).
3. Traces carry the question as input and the answer as output, so UI
   LLM-as-judge evaluators (Langfuse → Evaluators, connection via
   OpenRouter) map variables cleanly when configured.

Offline/manual evaluation is done by hand against the seeded data
(`experiments/8_seed_business_workflow.py` prints the expected answers).

**Document search has a scripted evaluation**, because its worst failure is
invisible to every other measure — the model answering a policy question
from general knowledge instead of searching, producing something that reads
perfectly while not being this company's policy:

```bash
agent/.venv/Scripts/python.exe experiments/13_probe_policy_questions.py
```

Eleven questions with the section each must cite, including one about a
policy that deliberately does not exist. It reports whether the tool was
called, whether the right section came back, and whether the answer cites
it. Run it after any change to the chunker, the retrieval gate or the
routing rules in `SYSTEM_PROMPT`.

## Quality gates (all must pass)

```bash
uv run ruff format .   # formatting
uv run ruff check .    # lint
uv run mypy            # strict type checking
uv run pytest          # unit tests — 643 as of 2026-09-10
```

Document search lives in an Odoo module, so its 28 tests run there instead
(never against the live database):

```bash
docker exec odoo19-ent-2 sh -c "/opt/odoo/odoo-bin -c /opt/odoo/odoo.conf   -d ont_test --http-port=8072 -u ont_mcp_rag   --test-enable --test-tags /ont_mcp_rag --stop-after-init"
```

## Roadmap

| Phase | Deliverable |
|---|---|
| 1 ✅ | MCP client: list tools + authenticated call |
| 2 ✅ | LangGraph agent loop + OpenRouter model — first real question over `res.partner` |
| 3 ✅ | Telegram bridge + per-user OAuth binding (chat ID → Odoo user) |
| 4 ✅ | Per-user conversation memory (SQLite checkpointer, `/new`, auto-summarization) |
| 5 ✅ | Write actions behind human confirmation (interrupts + Approve/Cancel buttons) |
| 6 ✅ | Charts + PDF reports in Telegram, plus `send_email` |
| 7 ✅ | Long-term per-user memory (preferences surviving `/new` and `/logout`) |
| 8 ✅ | Safety + abuse controls (access gates, burst/daily/global limits, injection hardening) |
| 9 ✅ | Proactive: morning brief in each user's timezone + plain-language watchers |
| 10 ✅ | Voice questions (Telegram voice notes → OpenRouter transcription) |
| 11 ✅ | Odoo deep links, follow-up buttons and freshness stamps, on one shared `Evidence` object |
| 12 ✅ | Action log, `/audit` for administrators, and the weekly "what I did" digest |
| 13 🔜 | Show your working — a button revealing the records and queries behind a number. The evidence is already collected; it needs persisting per message. |
| 14 🔜 | Share to a colleague — forward an answer over Telegram with attribution, to bound users only. |
| 15 🔜 | Pinned questions (`/pin cash` → `/cash`, registered in that user's own "/" menu). |
| 16 🔜 | Undo on approved writes — hold the action, show an Undo button, then execute. Holding sidesteps rollback entirely: sent email cannot be unsent and deletes cannot be restored, so an undo that "reverses" would work for two actions out of six. |
| 17 🔜 | "Since yesterday" — a delta rather than another snapshot, from a daily metrics table collected with no model call. |
| 18 🔜 | Approval inbox (`/needs`). Blocked on data, not code: Odoo's `approvals` module is uninstalled, so the only real queue today is 9 draft quotations. |
| 19 ✅ | Coloured approval buttons — green go, red stop, ⚠️ label for actions that cannot be undone |
| 20 ✅ | Uploaded images and PDF files — read alongside live Odoo data, stripped off the thread once answered |
| 21 ✅ | OAuth tokens encrypted at rest (Fernet, `AGENT_TOKEN_KEY`); malformed key stops startup, changed or removed key logs loudly |
| 22 ✅ | Document search — company policies answered from Odoo Documents with citations (`ont_mcp_rag`), hybrid retrieval, 11/11 tool-call rate |
| 23 ✅ | All state on the agent's own Postgres (`AGENT_DB_URL`, required) — SQLite retired entirely; tests run in a schema each |
| 24 ✅ | Private chats only — the bot refuses and leaves any group, with a second gate in `complete_login` |
| 25 ✅ | Token refresh survives slow turns and Odoo restarts — wider margin, transient errors keep the login, per-chat lock |
| 26 ✅ | Daily retention for idle conversations and old usage rows — plus the idle-in-transaction bug it uncovered |
| 27 ✅ | Thread rotation — an ACTIVE user's thread is bounded too, carried across losslessly and committed only after the turn returns |
| 28 ✅ | `BindingStore` on a connection pool — a shared connection made a concurrent write during `record_usage` raise, measured, not theorised |
| 29 ✅ | Quiet hours — nothing unprompted between 21:00 and 08:00 in the user's own timezone; watchers are deferred, never swallowed |
| 30 ✅ | System-prompt guardrail — a confidentiality rule, input patterns and an n-gram output scan, none costing a model call |
| 31 ✅ | Credential redaction — vendor-prefix patterns strip keys out of tool results BEFORE the model sees them, so nothing reaches the thread or the trace |
| — | **On hold: streamed drafts** (`send_message_draft`) and the `date_time` entity. Both available in the installed library; deferred 2026-09-01. See "Deferred — Bot API 10 modernisation" for the measurements and reasoning. Member tags are not applicable — groups only. |
| — | **Open gaps:** primary model latency, and the morning brief sending unconditionally whether or not anything changed. See "Known gaps". |
| — | **Also outstanding: uptime.** The bot is a manually launched process — no container, no restart policy, nothing polling `/health`, and quick tunnels that die on any network blip. Containerise with a restart policy, move to a named Cloudflare tunnel, then point a probe at `/health`. |
