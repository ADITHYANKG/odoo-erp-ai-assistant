<#
.SYNOPSIS
    Bring the fully LOCAL stack back up: Odoo + agent DB + bot, behind two
    fresh Cloudflare quick tunnels.

.DESCRIPTION
    Use after a laptop restart or sleep. Quick-tunnel hostnames change every
    time, and four places must agree on them, so this rewires all of them:

      1. Odoo containers (../odoo) and the agent Postgres (this repo)
      2. Two quick tunnels: Odoo (8069) and the bot's callback server (8181)
      3. Odoo web.base.url  -> Odoo tunnel   (also restarts the Odoo container)
      4. OAuth client redirect URI -> <bot tunnel>/oauth/callback
      5. This repo's .env: ODOO_PUBLIC_URL and AGENT_PUBLIC_URL
      6. Restart the bot (uv run ont-agent-bot), logging to logs\bot.log
      7. Verify both tunnels end to end

    The server deployment has its own Odoo and bot and is not touched. The
    OAuth client updated here is the one in the LOCAL Odoo database, found by
    OAUTH_CLIENT_ID in .env.

.EXAMPLE
    .\scripts\restart_local_stack.ps1
    .\scripts\restart_local_stack.ps1 -Follow      # then tail the bot log (Ctrl+C stops tailing, not the bot)
    .\scripts\restart_local_stack.ps1 -OdooRepo D:\code\odoo
#>
param(
    [string]$OdooRepo = (Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) "odoo"),
    [string]$OdooDb = "odoo",
    [string]$OdooDbContainer = "main-db-1",
    [switch]$Follow
)

$ErrorActionPreference = "Stop"
$agentRepo = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $agentRepo ".env"
$scratch = Join-Path $env:TEMP "local-stack-restart"
$odooPython = Join-Path $OdooRepo "venv\Scripts\python.exe"

function Step($n, $text) { Write-Host "`n[$n/7] $text" -ForegroundColor Cyan }
function Fail($text) { Write-Host "ERROR: $text" -ForegroundColor Red; exit 1 }

function Wait-Http($url, $seconds) {
    for ($i = 0; $i -lt $seconds; $i++) {
        try {
            $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5
            if ($r.StatusCode -lt 500) { return $r }
        } catch {}
        Start-Sleep 1
    }
    return $null
}

function Get-EnvValue($key) {
    $line = Get-Content $envFile | Where-Object { $_ -match "^$key=" } | Select-Object -First 1
    if (-not $line) { return $null }
    return ($line -replace "^$key=", "").Trim().Trim('"')
}

function Start-QuickTunnel($name, $target) {
    $log = Join-Path $scratch "tunnel_$name.log"
    Start-Process cloudflared -ArgumentList 'tunnel', '--url', $target `
        -RedirectStandardError $log -WindowStyle Hidden
    Write-Host "  waiting for $name tunnel URL" -NoNewline
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep 1
        Write-Host "." -NoNewline
        if (Test-Path $log) {
            $m = Select-String -Path $log -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' | Select-Object -First 1
            if ($m) { Write-Host ""; return $m.Matches[0].Value }
        }
    }
    Write-Host ""
    Fail "No URL for the $name tunnel. See $log"
}

# --- Preflight --------------------------------------------------------------
if (-not (Test-Path $envFile)) { Fail "$envFile not found (copy .env.example first)." }
if (-not (Test-Path (Join-Path $OdooRepo "docker-compose.yml"))) { Fail "Odoo repo not found at $OdooRepo (pass -OdooRepo)." }
if (-not (Test-Path $odooPython)) { Fail "Odoo helper venv missing: $odooPython" }
if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) { Fail "cloudflared is not on PATH." }
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Fail "uv is not on PATH." }
$clientId = Get-EnvValue "OAUTH_CLIENT_ID"
if (-not $clientId) { Fail "OAUTH_CLIENT_ID is empty in .env." }
if ($clientId -notmatch '^[A-Za-z0-9_-]+$') { Fail "OAUTH_CLIENT_ID has unexpected characters." }
New-Item -ItemType Directory -Force $scratch | Out-Null
New-Item -ItemType Directory -Force (Join-Path $agentRepo "logs") | Out-Null

# --- 1. Docker + containers ---------------------------------------------------
Step 1 "Starting Docker containers..."
$ver = & docker info --format "{{.ServerVersion}}" 2>$null
if (-not $ver) { Fail "Docker engine is not running. Start Docker Desktop and re-run." }
Write-Host "  Docker v$ver" -ForegroundColor Green

& docker compose -f (Join-Path $OdooRepo "docker-compose.yml") up -d
if ($LASTEXITCODE -ne 0) { Fail "Odoo stack failed to start." }
& docker compose -f (Join-Path $agentRepo "docker-compose.yaml") up -d db
if ($LASTEXITCODE -ne 0) { Fail "Agent database failed to start." }

# A containerised bot (compose profile "bot") would hold port 8181.
$botContainer = & docker ps -q --filter "name=^ont-agent-bot$"
if ($botContainer) {
    Write-Host "  stopping the containerised bot (local run uses uv)"
    & docker stop ont-agent-bot | Out-Null
}

if (-not (Wait-Http "http://127.0.0.1:8069/web/login" 90)) { Fail "Odoo did not answer on :8069 within 90s." }
Write-Host "  Odoo is up" -ForegroundColor Green

# --- 2. Tunnels ---------------------------------------------------------------
Step 2 "Starting Cloudflare quick tunnels..."
# Only our quick tunnels; a cloudflared Windows service (named tunnel) is left alone.
Get-CimInstance Win32_Process -Filter "Name='cloudflared.exe'" |
    Where-Object { $_.CommandLine -match 'tunnel\s+--url' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Remove-Item (Join-Path $scratch "tunnel_*.log") -Force -ErrorAction SilentlyContinue

# 127.0.0.1, not localhost: Windows resolves localhost to ::1 first.
$odooTunnel = Start-QuickTunnel "odoo" "http://127.0.0.1:8069"
$agentTunnel = Start-QuickTunnel "agent" "http://127.0.0.1:8181"
Write-Host "  Odoo  : $odooTunnel" -ForegroundColor Green
Write-Host "  Agent : $agentTunnel" -ForegroundColor Green

# --- 3. web.base.url ----------------------------------------------------------
Step 3 "Setting Odoo web.base.url (restarts the Odoo container)..."
& $odooPython (Join-Path $OdooRepo "experiments\3_set_public_base_url.py") $odooTunnel
if ($LASTEXITCODE -ne 0) { Fail "Could not set web.base.url (check ODOO_* in $OdooRepo\.env)." }
if (-not (Wait-Http "http://127.0.0.1:8069/web/login" 90)) { Fail "Odoo did not come back after the restart." }

# --- 4. OAuth redirect URI ----------------------------------------------------
Step 4 "Updating OAuth client redirect URI..."
$redirect = "$agentTunnel/oauth/callback"
$sql = "UPDATE ont_mcp_oauth_client SET redirect_uris = '$redirect' WHERE client_id = '$clientId';"
$out = & docker exec $OdooDbContainer psql -U odoo -d $OdooDb -t -c $sql
if ($LASTEXITCODE -ne 0) { Fail "psql failed updating the OAuth client." }
if ("$out" -notmatch 'UPDATE 1') { Fail "OAuth client $clientId not found in the local Odoo DB ($out). Create it with 5_create_agent_oauth_client.py." }
Write-Host "  $clientId -> $redirect" -ForegroundColor Green

# --- 5. .env ------------------------------------------------------------------
Step 5 "Updating .env..."
Copy-Item $envFile (Join-Path $scratch ".env.backup") -Force
$updates = [ordered]@{ ODOO_PUBLIC_URL = $odooTunnel; AGENT_PUBLIC_URL = $agentTunnel }
$lines = [System.Collections.Generic.List[string]](Get-Content $envFile)
foreach ($key in $updates.Keys) {
    $idx = -1
    for ($i = 0; $i -lt $lines.Count; $i++) { if ($lines[$i] -match "^$key=") { $idx = $i; break } }
    if ($idx -ge 0) { $lines[$idx] = "$key=$($updates[$key])" } else { $lines.Add("$key=$($updates[$key])") }
    Write-Host "  $key=$($updates[$key])"
}
# LF endings and no BOM, as the file already uses (a BOM would corrupt the first key).
[IO.File]::WriteAllText($envFile, (($lines -join "`n") + "`n"), (New-Object Text.UTF8Encoding($false)))
Write-Host "  backup: $scratch\.env.backup" -ForegroundColor DarkGray

# --- 6. Bot -------------------------------------------------------------------
Step 6 "Restarting the bot..."
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -match 'ont-agent-bot' -and $_.ProcessId -ne $PID } |
    ForEach-Object { Write-Host "  stopping PID $($_.ProcessId)"; Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep 2
$holder = Get-NetTCPConnection -LocalPort 8181 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($holder) { Fail "Port 8181 is still held by PID $($holder.OwningProcess)." }

$botLog = Join-Path $agentRepo "logs\bot.log"
Start-Process uv -ArgumentList 'run', 'ont-agent-bot' -WorkingDirectory $agentRepo `
    -RedirectStandardError $botLog -WindowStyle Hidden
if (-not (Wait-Http "http://127.0.0.1:8181/health" 60)) { Fail "Bot did not answer /health within 60s. See $botLog" }
Write-Host "  bot is up (log: $botLog)" -ForegroundColor Green

# --- 7. Verify ----------------------------------------------------------------
Step 7 "Verifying through the tunnels (DNS can take a few seconds)..."
$okOdoo = Wait-Http "$odooTunnel/web/login" 45
$okAgent = Wait-Http "$agentTunnel/health" 45

Write-Host "`n--- Status ---" -ForegroundColor Yellow
Write-Host ("Odoo  {0,-55} {1}" -f $odooTunnel, $(if ($okOdoo) { "OK" } else { "FAIL" }))
Write-Host ("Agent {0,-55} {1}" -f $agentTunnel, $(if ($okAgent) { "OK" } else { "FAIL" }))
Write-Host "MCP   http://127.0.0.1:8069/mcp (local, no tunnel)"
Write-Host "`nIn Telegram: /start -> Log in to Odoo." -ForegroundColor Yellow
Write-Host "If WhatsApp / Google Chat / Slack HTTP are enabled, re-point their webhooks at $agentTunnel." -ForegroundColor Yellow

# AGENT_LOG_FILE is the full rotating log; bot.log only has raw stderr.
$agentLog = Get-EnvValue "AGENT_LOG_FILE"
$liveLog = if ($agentLog) { Join-Path $agentRepo $agentLog } else { $botLog }
Write-Host "`nLive log: Get-Content `"$liveLog`" -Wait -Tail 50" -ForegroundColor DarkGray
if ($Follow) {
    Write-Host "Following the bot log - Ctrl+C stops tailing; the bot keeps running.`n" -ForegroundColor Yellow
    Get-Content $liveLog -Wait -Tail 50
}
if (-not ($okOdoo -and $okAgent)) { exit 1 }
