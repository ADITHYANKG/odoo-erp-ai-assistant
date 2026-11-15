# Self-hosted Langfuse (server)

Langfuse runs as its own compose project next to the bot, on the same host.
Nothing in the bot's compose file changes: the bot reaches Langfuse through
the Docker bridge address, so any redeploy of the bot keeps tracing.

| Thing | Value |
|---|---|
| Directory | `/opt/langfuse` (official `docker-compose.yml` + the files here) |
| UI / API | `127.0.0.1:3000` on the host, `172.17.0.1:3000` from containers |
| Bot setting | `LANGFUSE_BASE_URL="http://172.17.0.1:3000"` in the bot's `.env` |
| Project | org `Acme`, project `odoo-erp-ai-assistant` (created on first boot) |

All other ports (Postgres, ClickHouse, Redis, MinIO, worker) are bound to
`127.0.0.1` only. Signup is disabled; the only login is the init admin.

## First install

```bash
sudo mkdir -p /opt/langfuse && sudo chown "$USER" /opt/langfuse && cd /opt/langfuse
curl -fsSL https://raw.githubusercontent.com/langfuse/langfuse/main/docker-compose.yml -o docker-compose.yml

# The official file publishes the UI and MinIO on all interfaces — bind them to localhost
sed -i -e 's#^\(\s*\)- 3000:3000#\1- 127.0.0.1:3000:3000#' \
       -e 's#^\(\s*\)- 9090:9000#\1- 127.0.0.1:9090:9000#' docker-compose.yml

# From this directory of the repo:
cp docker-compose.override.yml /opt/langfuse/
bash gen-env.sh <admin-email> /opt/langfuse    # prints the UI login and the bot's two keys

cd /opt/langfuse && docker compose up -d
curl -s http://localhost:3000/api/public/health
```

`gen-env.sh` refuses to overwrite an existing `.env`: `SALT` and
`ENCRYPTION_KEY` cannot change once Langfuse holds data, and the databases
keep the passwords they were first created with.

Then set the three `LANGFUSE_*` lines in the bot's `.env` (keys from
`LANGFUSE_INIT_PROJECT_PUBLIC_KEY` / `LANGFUSE_INIT_PROJECT_SECRET_KEY` in
`/opt/langfuse/.env`) and recreate the bot. Its log should show
`Langfuse tracing enabled (http://172.17.0.1:3000)`.

## Opening the UI

```bash
ssh -i <key>.pem -L 3000:localhost:3000 ubuntu@<server>
# then http://localhost:3000 — login is LANGFUSE_INIT_USER_* in /opt/langfuse/.env
```

## Operating

```bash
cd /opt/langfuse
docker compose ps
docker compose logs --tail 100 langfuse-web
docker compose pull && docker compose up -d   # upgrade within major v4
```

Never `docker compose down -v` — it deletes every trace and the project keys.
