# API authentication (`/v1/*`)

Every llm-bawt HTTP request passes one ASGI middleware
(`src/llm_bawt/api_auth/`). It names the caller (a *principal*) and, in
`enforce`, refuses requests that name nobody. Route-specific checks (service
bearer, bridge gate, turn capability) still apply on top. The MCP server
(:8001) has its own gate; see [MCP_SERVER.md](MCP_SERVER.md#authentication).

## Mode

The global runtime setting `api_auth_mode` is `off`, `log` or `enforce`.

| Mode | Missing credential | Bad credential |
|---|---|---|
| `off` | admitted | admitted |
| `log` | admitted and counted | refused (401) |
| `enforce` | refused: 401, or WebSocket close 4401 | refused (401) |

- The setting is read through a 5 s cache. `LLM_BAWT_API_AUTH_MODE` is only the
  default when the store has no row.
- A failed read never changes the effective mode. Before the first successful
  read the mode is *unresolved*: protected routes answer 503
  `auth_mode_unresolved`.
- `LLM_BAWT_API_AUTH_MODE_FORCE=off|log|enforce` overrides everything, for an
  operator running without the database.
- **Rollback:** write `log` (or `off`) to `api_auth_mode`; it takes effect
  within the cache window, without a restart.

## Credentials and principals

Checked in this order; the first that matches names the caller.

| Credential | Principal | Sent by |
|---|---|---|
| `X-LLM-Bawt-Task-Turn-Context` (signed turn capability) | `bot:<id>` | agent bridges, for a bot's turn |
| `X-LLM-Bawt-Key: <principal>.<key>` | `key:<principal>` | BawtHub web and voice, bridges, the app's own client, houses, host scripts |
| `Authorization: Bearer <service token>` | `service:bawthub-service` | BawtHub service calls |
| `X-Bridge-Token` | `bridge:bridge-<gate>` | bridge-gated routes (for example the git-credential broker) |

A malformed, invalid, revoked or repeated `X-LLM-Bawt-Key` is refused in every
mode except `off`; a turn capability is identity only.

### Per-caller keys

A key is derived, not stored: HMAC-SHA256 of a master-key subkey
(`llm-bawt-api-key/v1`) over the principal name. Principal names match
`[a-z0-9][a-z0-9-]{0,62}`.

- **Give a container a key:** add `principal=/dir[:uid[:gid]]` to
  `LLM_BAWT_API_KEY_EXPORTS` on the app. The app writes `<dir>/api-key` (0600,
  or 0640 with a gid). The consumer mounts that directory read-only (not a
  single-file bind) and sets `LLM_BAWT_API_KEY_FILE=<dir>/api-key`;
  `llm_bawt.api_auth.client.add_auth_header()` does the rest.
- **A caller outside the hub** (another host) gets a copy of its derived key
  in a file only its user can read. Prefer a route-limited principal.
- **Route-limited principals** are listed in code
  (`ROUTE_LIMITED_PRINCIPALS` in `middleware.py`) so a setting can't widen
  them. Such a key is refused on any other route. Add the entry and have it
  live *before* handing out the key, or the key works everywhere.
- **Revoke one principal:** add its name to the `api_auth_denylist` runtime
  setting. **Rotate every key:** bump the derivation label version.

Nothing in this path logs a key. Don't print one to check it; compare its
length or use it in a request.

## Unauthenticated routes

Only `/health`, the Google Home OAuth/fulfillment endpoints
(`/integrations/google-home/{authorize,token,fulfillment}`), `/webhook/nextcloud`
(each with its own verification) and, when Prometheus metrics are enabled,
`GET /metrics` (read-only, bounded telemetry). Everything else needs a
credential under `enforce`.

## Observing callers

`GET /v1/admin/auth-observations` (service bearer) returns the mode and
recent caller counts: rows keyed `[kind, who…, method, route]` with count and
first/last time, capped at 2,000 rows and 48 h. Anonymous callers appear as
`["anon", <ip>, <user-agent>, …]`; the counts are recorded before refusal, so
in `enforce` an `anon` row is a caller now getting 401. The MCP server's view
is under `mcp`. Counts are per process and reset when the app restarts; the
app's `http_access` log lines carry the status code for older history.

## Rolling out to a new installation

1. Set `api_auth_mode=log` and give every caller its key.
2. Watch `auth-observations` for `anon` rows across a window that covers
   low-frequency callers (scheduled jobs, notifiers), and key each one.
3. When the window is clean, write `enforce`. Confirm an anonymous request to
   a protected route returns 401 and the UI, bridges, houses, CLI and voice
   still work.
