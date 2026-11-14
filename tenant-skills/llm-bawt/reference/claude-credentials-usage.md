# Claude credentials + subscription usage endpoint (/v1/usage)

> Part of the llm-bawt skill. Read when usage or claude-code inference "stopped working", before touching credential refresh, or when wiring a new provider into /v1/usage.

Canonical, multi-provider plan-usage (Claude `/usage`-style limits) for the UI.
`?provider=claude` (one provider) · `?bot_id=<slug>` (resolve provider) · no
filter (all registered). Only **Claude** is implemented; z.ai/codex are
stubs. Code: `src/llm_bawt/service/usage/`. **Full config + troubleshooting:
`~/dev/llm-bawt/docs/usage-endpoint.md`.**

**Claude credential model (TASK-635, live 2026-07-21): ONE app-owned login
feeds EVERYTHING — usage AND claude-code bridge inference.**

- **One bundle:** a full-scope `claudeAiOauth` bundle (`user:inference` +
  `user:profile` + …) at `~/.config/llm-bawt/claude-usage/.credentials.json`
  (host; env `CLAUDE_CREDENTIALS_PATH`, legacy `CLAUDE_USAGE_CREDENTIALS_PATH`
  honored; `CLAUDE_USAGE_CREDENTIALS_MODE=owned`). Minted by the single
  **`claude` provider adapter** wizard — Reconnect button in the usage popup,
  or Models admin → Provider accounts (API:
  `/v1/providers/claude/login/start|complete`, PTY-driven
  `claude auth login --claudeai` in an isolated tmp dir). The manual
  `HOME=/tmp/x claude login` dance is obsolete.
- **The app is the SOLE refresher** (refresh tokens rotate on use — single-use):
  refresh is serialized behind a lock, plus a proactive lifespan loop refreshes
  at expiry−20min (`usage/claude_oauth.py`), so the bundle never goes stale
  even when idle.
- **Everything else is a read-only consumer.** The usage adapter reads
  in-process. The claude-code bridge resolves tokens (per turn, no restart
  needed after a re-login): ro-mounted bundle at
  `/home/bridge/.claude-broker/.credentials.json` → broker endpoint
  `GET /v1/providers/claude/token` (`?force=true` after a 401; optional
  `BRIDGE_CLAUDE_TOKEN_SECRET`/`X-Bridge-Token` guard) → legacy
  `~/.claude/.credentials.json` (read-only) → env `CLAUDE_CODE_OAUTH_TOKEN`
  (long-lived setup-token, `user:inference` only — fine for inference, 403s on
  usage).
- **NEVER re-add a second refresher** (bridge-side or otherwise): two writers
  racing one rotate-on-use chain = `invalid_grant`, the exact failure TASK-635
  removed. The old `claude-sub`/`claude-usage` dual adapters are gone.

**If usage or claude-code inference "stopped working":**
`curl -s 'localhost:8642/v1/usage?provider=claude' | jq '{status,error}'` and
`curl -s 'localhost:8642/v1/providers/claude/token' | jq .state`.
`stale`/`unauthorized` → the refresh chain died upstream or the bundle lacks
scopes → **one-click Reconnect** in the usage popup (or Models admin →
Provider accounts → Claude); fixes usage and bots together, no restarts.
`rate_limited` → 429, cache backs off. Single failure domain is by design:
a dead credential takes usage AND claude-code bots once the access token
lapses — acceptable because recovery is the 15-second UI reconnect.

## Summary formerly in the parent SKILL.md (TASK-1640)

**Claude credentials (TASK-635): ONE app-owned OAuth bundle feeds usage AND claude-code inference; the app is the SOLE refresher — NEVER add a second refresher** (rotate-on-use refresh tokens; two writers = `invalid_grant`). Broken usage/inference → one-click Reconnect in the usage popup. Diagnostics + full model: `reference/claude-credentials-usage.md`.
