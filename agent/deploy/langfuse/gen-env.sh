#!/usr/bin/env bash
# Generate /opt/langfuse/.env with fresh random secrets and a project key pair.
#
# Usage: ./gen-env.sh [admin-email] [target-dir]
# Refuses to overwrite an existing .env: SALT and ENCRYPTION_KEY cannot change
# once Langfuse has stored data, and the databases keep their first passwords.
set -euo pipefail

ADMIN_EMAIL="${1:-admin@example.com}"
TARGET_DIR="${2:-/opt/langfuse}"
ENV_FILE="$TARGET_DIR/.env"

if [[ -e "$ENV_FILE" ]]; then
    echo "Refusing to overwrite $ENV_FILE" >&2
    exit 1
fi

hex() { openssl rand -hex "$1"; }

PG_PW=$(hex 16)
MINIO_PW=$(hex 16)
ADMIN_PW=$(hex 12)
PK="pk-lf-$(cat /proc/sys/kernel/random/uuid)"
SK="sk-lf-$(cat /proc/sys/kernel/random/uuid)"

umask 077
cat > "$ENV_FILE" <<EOF
# Langfuse (self-hosted) for odoo-erp-ai-assistant. Never commit this file.
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=$(hex 32)
SALT=$(hex 32)
ENCRYPTION_KEY=$(hex 32)
TELEMETRY_ENABLED=false

POSTGRES_PASSWORD=$PG_PW
DATABASE_URL=postgresql://postgres:$PG_PW@postgres:5432/postgres
CLICKHOUSE_PASSWORD=$(hex 16)
REDIS_AUTH=$(hex 16)

# MinIO: the root password and the three S3 secrets must be identical.
MINIO_ROOT_PASSWORD=$MINIO_PW
LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY=$MINIO_PW
LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY=$MINIO_PW
LANGFUSE_S3_BATCH_EXPORT_SECRET_ACCESS_KEY=$MINIO_PW

# Headless init: org, project, API keys and admin user, created on first boot.
LANGFUSE_INIT_ORG_ID=acme
LANGFUSE_INIT_ORG_NAME=Acme
LANGFUSE_INIT_PROJECT_ID=odoo-erp-ai-assistant
LANGFUSE_INIT_PROJECT_NAME=odoo-erp-ai-assistant
LANGFUSE_INIT_PROJECT_PUBLIC_KEY=$PK
LANGFUSE_INIT_PROJECT_SECRET_KEY=$SK
LANGFUSE_INIT_USER_EMAIL=$ADMIN_EMAIL
LANGFUSE_INIT_USER_NAME=Admin
LANGFUSE_INIT_USER_PASSWORD=$ADMIN_PW
EOF

echo "Wrote $ENV_FILE"
echo
echo "UI login:  $ADMIN_EMAIL / $ADMIN_PW"
echo
echo "Put these in the agent's .env:"
echo "LANGFUSE_PUBLIC_KEY=\"$PK\""
echo "LANGFUSE_SECRET_KEY=\"$SK\""
