# Google Chat Chatbot Setup Guide

This guide documents the reusable setup pattern for integrating an existing chatbot backend with Google Chat.

## 1. Create a Google Cloud project

Create or select a dedicated Google Cloud project for the chatbot.

Get the **Project Number** from:

Google Cloud Console → IAM & Admin → Settings → Project number

Example:

```env
GOOGLE_CHAT_PROJECT_NUMBER=466728399479
```

Do not confuse the **Project ID** with the **Project Number**.

## 2. Enable the Google Chat Application Programming Interface (API)

Go to:

APIs & Services → Library → Google Chat API → Enable

Then open:

Google Chat API → Configuration

## 3. Configure the Google Chat app

Under **Application info**, configure:

- App name
- Avatar URL
- Description

Example:

```text
App name: ERP AI Assistant
Avatar URL: https://i.imgur.com/xxxxxx.png
Description: Acme AI assistant
```

The avatar should be a publicly accessible Hypertext Transfer Protocol Secure (HTTPS) image URL.

Enable:

```text
Enable interactive features
```

For a normal existing backend chatbot, keep the Google Workspace add-on option disabled unless specifically needed.

Enable:

```text
Join spaces and group conversations
```

This allows users to direct-message the bot, add it to spaces, and mention it in group conversations.

## 4. Configure the public webhook

Under:

Connection settings → HTTP endpoint URL

use the chatbot's public endpoint.

Example:

```text
https://bot.example.com:8443/webhook/googlechat
```

Recommended environment variables:

```env
AGENT_PUBLIC_URL=https://bot.example.com:8443
GOOGLE_CHAT_ENDPOINT_URL=https://bot.example.com:8443/webhook/googlechat
```

The endpoint must be publicly reachable, use HTTPS, have a valid Transport Layer Security (TLS) certificate, accept POST requests, and return responses quickly.

Basic reachability test:

```bash
curl -i https://bot.example.com:8443/webhook/googlechat
```

A response such as:

```text
HTTP/2 405
allow: POST
```

is fine for a GET request because it confirms the route exists and only accepts POST.

## 5. Configure triggers

Use:

```text
Use a common HTTP endpoint URL for all triggers
```

and enter the same webhook URL.

The same FastAPI route can receive events such as Message, Added to space, Removed from space, Commands, and App Home events if enabled.

For a basic chatbot, App Home is not required.

## 6. Create a service account if the bot needs authenticated Google API calls

Create one under:

IAM & Admin → Service Accounts → Create Service Account

Example:

```text
erpbot@your-project.iam.gserviceaccount.com
```

If the chatbot needs to call Google Chat APIs itself, create a JavaScript Object Notation (JSON) key:

Service Account → Keys → Add Key → Create new key → JSON

Store the downloaded file securely.

Example host path:

```text
/opt/agent/secrets/googlechat-sa.json
```

Example environment variable:

```env
GOOGLE_CHAT_CREDENTIALS_FILE=secrets/googlechat-sa.json
```

Do not commit the credential file to Git.

Recommended `.gitignore`:

```gitignore
secrets/
*.json
```

## 7. Understand the two service accounts

### Your application service account

Used when your chatbot calls Google APIs.

This is the account associated with the downloaded JSON key.

### Google-managed Chat service account

Google creates another identity similar to:

```text
service-PROJECT_NUMBER@gcp-sa-gsuiteaddons.iam.gserviceaccount.com
```

This represents Google Chat when Google sends requests to your webhook.

Do not confuse these two identities.

## 8. Docker credential setup

Example Docker Compose mount:

```yaml
volumes:
  - ./secrets:/app/secrets:ro
```

This means a host file such as:

```text
/opt/agent/secrets/googlechat-sa.json
```

appears inside the container as:

```text
/app/secrets/googlechat-sa.json
```

The `:ro` mount keeps it read-only inside the container.

## 9. Fix file permissions

Check the container user:

```bash
docker compose --profile bot exec bot id
```

Example:

```text
uid=10001(agent) gid=10001(agent)
```

Make the credential readable by the container group while keeping it private:

```bash
sudo chown ubuntu:10001 /opt/agent/secrets/googlechat-sa.json
chmod 640 /opt/agent/secrets/googlechat-sa.json
```

Validate from inside Docker:

```bash
docker compose --profile bot exec bot python -m json.tool   /app/secrets/googlechat-sa.json   > /dev/null && echo "Credential OK"
```

## 10. Verify configuration inside Docker

Check environment variables:

```bash
docker compose --profile bot exec bot printenv | grep -E 'GOOGLE_CHAT|AGENT_PUBLIC_URL'
```

Check the credential path:

```bash
docker compose --profile bot exec -T bot python - <<'PY'
import os
from pathlib import Path

p = Path(os.environ["GOOGLE_CHAT_CREDENTIALS_FILE"])

print("Configured:", p)
print("Resolved:", p.resolve())
print("Exists:", p.exists())
print("Readable:", os.access(p, os.R_OK))
PY
```

Expected:

```text
Exists: True
Readable: True
```

## 11. Configure visibility

During testing:

```text
Make this Chat app available to specific people and groups
```

Add tester accounts, for example:

```text
you@example.com
```

After testing, make the app available to all permitted users in the Google Workspace organization if required.

Remember there may be two access-control layers:

```text
Google Chat visibility
+
AGENT_ALLOWED_GOOGLE_CHAT_EMAILS
```

If the backend uses `AGENT_ALLOWED_GOOGLE_CHAT_EMAILS`, update or remove that restriction when making the bot organization-wide.

## 12. OAuth 2.0 credentials: when needed

OAuth 2.0 credentials are not required for the basic message webhook flow.

Use a service account when:

```text
The chatbot itself performs an action as the app.
```

Use OAuth 2.0 when:

```text
The chatbot performs an action on behalf of a signed-in Google user.
```

OAuth 2.0 setup may require an OAuth consent screen, OAuth client ID, OAuth client secret, and appropriate scopes.

## 13. Verify incoming Google requests

For production, do not trust every POST request sent to the webhook.

Google sends authenticated requests. The backend should verify the Google-signed bearer token and reject invalid requests.

Conceptually:

```text
POST /webhook/googlechat
Authorization: Bearer <Google-signed-token>
```

The backend should validate token authenticity, the expected audience, and that the request belongs to the configured Chat app.

Invalid requests should normally be rejected with:

```text
401 Unauthorized
```

## 14. Test the Google Chat app

After saving the Google Chat configuration:

```text
Google Chat
→ New chat
→ Search the configured app name
→ Open the app
→ Send "Hello"
```

Watch backend logs:

```bash
cd /opt/odoo-erp-ai-assistant
docker compose --profile bot logs -f bot
```

You should see a request similar to:

```text
POST /webhook/googlechat
```

## 15. Final architecture

```text
Google Chat user
      ↓
Google Chat App
      ↓
HTTPS POST
      ↓
Public webhook
      ↓
Caddy / reverse proxy
      ↓
FastAPI / Uvicorn
      ↓
Chatbot / Agent
      ↓
Large Language Model (LLM) + tools
      ↓
Response
      ↓
Google Chat
```

## 16. Reusable checklist

1. Create a dedicated Google Cloud project.
2. Get the Project Number.
3. Enable Google Chat API.
4. Configure app name, avatar, and description.
5. Enable interactive features.
6. Enable joining spaces and group conversations.
7. Configure the public HTTPS webhook.
8. Configure a common HTTP endpoint for triggers.
9. Configure app visibility.
10. Enable error logging.
11. Create a service account if outbound Google API calls are required.
12. Download and securely store the JSON key.
13. Configure `GOOGLE_CHAT_PROJECT_NUMBER`.
14. Configure `GOOGLE_CHAT_CREDENTIALS_FILE`.
15. Configure `AGENT_PUBLIC_URL`.
16. Configure `GOOGLE_CHAT_ENDPOINT_URL`.
17. Mount the credential read-only into Docker.
18. Verify permissions.
19. Verify incoming Google-signed requests.
20. Test with one user first.
21. Expand visibility to the organization when ready.

## Minimum required configuration

```text
Google Cloud project
+
Google Chat API
+
Chat app configuration
+
Public HTTPS webhook
+
Secure backend request verification
```

Add a service account only when the chatbot needs authenticated outbound Google API calls.
