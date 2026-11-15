# Error tracking with Sentry (optional)

The bot can report every error it logs to Sentry: failed turns, morning briefs, watchers, uploads, voice notes and crashes in chat handlers. Each one becomes an issue with its traceback. When `SENTRY_DSN` is empty, error tracking is off and the Sentry SDK is never imported.

You can use any of these. The bot only needs a DSN.

| Option | Good for |
|---|---|
| [sentry.io](https://sentry.io) (hosted) | Quickest start; the free plan is enough for one bot |
| [Self-hosted Sentry](https://develop.sentry.dev/self-hosted/) | Keeping error data on your own server (needs about 16 GB RAM) |
| [GlitchTip](https://glitchtip.com) | A much lighter self-hosted server that accepts the same SDK |

## What is sent, and what isn't

The bot handles employee and ERP data, so the SDK is set up to send as little as possible (see `src/ont_agent/errortracking.py`):

- **Not sent:** request bodies (webhook payloads are users' messages), local variables in stack frames, cookies, headers, user info, or LLM inputs and outputs.
- **Scrubbed:** every string in an event or breadcrumb has credentials (API keys, tokens) and e-mail addresses removed before it leaves the bot.
- **Breadcrumbs:** only WARNING and above, so the INFO lines that name chats and users stay out.
- **Errors only:** performance tracing is off, because tracing already lives in Langfuse.

## Self-hosting Sentry next to the bot

These steps follow Sentry's official self-hosted repo. Check [its docs](https://develop.sentry.dev/self-hosted/) for the current requirements and version.

```bash
sudo mkdir -p /opt/sentry && sudo chown "$USER" /opt/sentry
git clone https://github.com/getsentry/self-hosted.git /opt/sentry
cd /opt/sentry
git checkout <latest release tag>     # e.g. from https://github.com/getsentry/self-hosted/releases
./install.sh                          # asks you to create the first admin user
docker compose up -d
```

Sentry's web UI listens on port `9000`. Keep it off the public internet: bind it to `127.0.0.1` (or firewall the port) and open it through an SSH tunnel:

```bash
ssh -L 9000:localhost:9000 <user>@<server>
# then http://localhost:9000
```

## Connecting the bot

1. In Sentry, create a project (platform: **Python**) and copy its DSN from **Settings → Projects → \<project\> → Client Keys (DSN)**.
2. If the bot runs in Docker on the same host as a self-hosted Sentry, change the DSN's host to the Docker bridge address so the container can reach it, for example:
   ```
   http://<public-key>@172.17.0.1:9000/<project-id>
   ```
   Make sure Sentry also listens on that address (not only `127.0.0.1`), or put both stacks on a shared Docker network.
3. Set these in the bot's `.env`:

   | Variable | Value |
   |---|---|
   | `SENTRY_DSN` | The DSN from step 1 or 2 |
   | `SENTRY_ENVIRONMENT` | `production`, `staging`, … (default `development`); separates servers in the Sentry UI |
   | `SENTRY_RELEASE` | Optional, e.g. the git commit from CI; defaults to the package version |

   `SENTRY_RELEASE` is read from the process environment only. With Docker Compose that's fine (`env_file: .env` exports it); when running with `uv run` locally, export it in the shell.

4. Recreate the bot (`docker compose up -d --force-recreate bot`). Its log should show:
   ```
   Error tracking enabled (environment=production)
   ```
   If the DSN is wrong you'll see `Error tracking could not start` instead. The bot keeps running either way.

## Checking that it works

Errors only reach Sentry when something actually fails. A quick way to test is to stop Odoo for a moment and ask the bot a question: the failed turn should appear as a new issue within a few seconds.
