# Provider Usage Endpoint

`GET /v1/usage` exposes provider-pluggable subscription windows and prepaid API
balances. A `kind=balance` snapshot is not a subscription quota. The route is in
[src/llm_bawt/service/routes/usage.py](../src/llm_bawt/service/routes/usage.py)
and the adapter registry is in
[src/llm_bawt/service/usage/](../src/llm_bawt/service/usage).

## API

| Request | Meaning |
|---|---|
| `GET /v1/usage` | All registered providers |
| `GET /v1/usage?provider=claude` | One provider |
| `GET /v1/usage?bot_id=<slug>` | Resolve via the bot's bound catalog endpoint |
| `GET /v1/usage?access_path=<key>` | Resolve the frozen route of a completed turn |
| `...&force=true` | Bypass cache |
| `PUT /v1/usage/{provider}/balance` `{"amount_usd": 157}` | Set "$X left as of now" (anchor providers only) |
| `DELETE /v1/usage/{provider}/balance` | Clear the anchor |

A `bot_id`/`access_path` query echoes the resolved route as `access_path_key`.
Balance writes return the provider's report: `404` for an unknown provider,
`409` for a provider that reports its own balance or has none, `400` unless
`amount_usd` is a positive finite number below 10^9.

## Registered providers

The registry currently includes:

- `claude` (`anthropic-oauth` subscription windows)
- `anthropic_api` (`anthropic-api` prepaid balance, anchor mode)
- `zai`
- `openai_chatgpt`
- `xai` (`xai-chat`/`xai-responses` prepaid balance, anchor mode)
- `openai_api` (`openai-api`/`openai-responses` prepaid balance, anchor mode)
- `openrouter` (`openrouter-anthropic` prepaid balance, anchor mode: its credit
  endpoint needs a management key, and the connected key is an inference key)
- `moonshot` (`moonshot-anthropic` prepaid balance from Moonshot's balance API;
  registered only when a Moonshot OpenPlatform key is configured)

Support level differs by adapter. The canonical response carries `kind`:
`subscription_windows` uses `limits[]`, while `balance` uses `balance` (and
`spend` where reported) with `balance_supported=true`. Every report, including
error and no-balance reports, also carries `manual_balance_supported`: true
only when the balance write routes accept an anchor (anchor mode). Unrecognized
paths are unavailable; a model name alone never selects a subscription
credential. Local engines and the custom `openai-compatible` path have no
known billing account and stay unavailable.

## Prepaid API balances (TASK-1461)

Implemented in
[src/llm_bawt/service/usage/balance.py](../src/llm_bawt/service/usage/balance.py).
An adapter declares `balance_mode`:

- `anchor`: the user enters the remaining credit. A connected pay-per-token API
  key without a balance API it can read is one data-only `ApiKeyBalanceAdapter`
  subclass (provider connection id, access paths); its report is `ok` once that
  key is connected. `api_balances` holds one row
  per provider (`anchor_usd`, server-stamped `anchor_at`, `spent_usd`,
  `unpriced_turns`). `remaining_usd = anchor_usd - spent_usd` is read from that
  row on every request; it is never cached and never computed by scanning
  turn logs.
- `provider`: the provider's balance API is authoritative; nothing counts down
  locally. Its `balance` has `source=provider_reported`, and `amount_usd`,
  `used_usd` and `as_of` are `null` (never invented).

`spent_usd` moves only inside the transaction that ends a turn, by the change in
that turn's `api_balance_ledger` row (`turn_id` primary key). Every terminal
writer charges: `TurnLogStore.update_turn`, Stop/recovery `settle_open_turn`
and the stale-turn reap (all through `record_terminal_effects`), and the
expired-delivery settle (`charge_settled_turns`), so a checkpointed cost on a
turn that never finalizes normally still counts. A replayed finalize charges
nothing; a revised cost charges the difference. Only turns created after
`anchor_at` whose usage carries a frozen `access_path` of that provider count.
Charges and anchor writes take one per-provider transaction lock, and setting
the anchor recounts the ledger after `anchor_at`, so a turn finishing while the
anchor commits is never missed. A counted turn without a valid
`total_cost_usd` is never charged a guess: it increments `unpriced_turns` and
the balance reads as unknown until it is set again. A key reconnected after the
anchor also voids the balance.

Every app-owned `turn_complete` (streaming finalize, non-streaming
completed/cancelled/error, Stop snapshot, stale reap, maintenance release)
carries `balance_update`, built by `usage.balance.balance_update_for_turn`
from the committed turn row:

- `null`: the turn's usage names no balance path, so no balance moved;
- anchor mode: `{provider, access_path_key, balance, updated_at}`, where
  `balance` is the committed counter (or `null` when unknown);
- provider mode, or an anchor read that failed: `{provider, access_path_key,
  balance: null, updated_at, refresh: true}`, so readers refetch the report once.

`updated_at` and `balance.updated_at` are fractional epochs that order updates.
Passive OpenClaw runs have no app turn and move no balance.

The turn-log scan (`backfill_ledger`) runs only on ledger creation, legacy
migration (the retired `anthropic_api_balance_anchor` runtime setting moves into
`api_balances` once) and `ApiBalanceStore.repair`. Runtime-setting writes to
`anthropic_api_balance_anchor` or `anthropic_api_monthly_budget_usd` return
`400`.

## Caching

Successful snapshots are cached for `LLM_BAWT_USAGE_CACHE_TTL` seconds
(default `120`; an adapter may set a shorter `report_ttl`, which Moonshot uses
as a 10 s debounce). On refresh errors or upstream `429`s, the endpoint returns
the last good cached snapshot when it has one. Anchored balances are overlaid
fresh on every read, including cached reports.

## Anthropic API spend

`anthropic_api` reports UTC calendar-day, 7-day and month-to-date USD spend.
Connect an `anthropic-admin` Usage reporting key separately from the
`anthropic-api` inference key. The Admin credential is encrypted by the provider
connection store and is never exposed by the inference token broker. With an
Admin key, the adapter reads Anthropic's paginated organization Cost Report
(Anthropic excludes Priority Tier costs from this endpoint). It does not request
the separate Usage Report because no usage breakdown is displayed; an unused
report cannot hold up spend. Without the Admin key, the adapter sums the
`api_balance_ledger` since the `anthropic-api` key was connected. That is an
**estimate limited to recorded app turns**, not the whole account. Neither
source debits the balance; the per-turn counter does. Failed report reads are
unavailable, never silently replaced with zero spend or a different account's
subscription quota.

## ChatGPT / Codex usage

`openai_chatgpt` fetches current subscription limits from the same dedicated
endpoint used by the official Codex client:

`GET https://chatgpt.com/backend-api/wham/usage`

This is a non-inference request; it does not select a model or create a hidden
chat turn. A `401` triggers one bounded force-refresh through the app-owned
OAuth bundle, then one retry. Real Codex `/responses` calls also expose quota
headers; the claude-code bridge saves those passively in Redis as a fallback.
If the dedicated endpoint fails and only an old passive snapshot is available,
the adapter returns `status=usage_stale` with `cached=true`. This means the
quota values are old, not that the credential is broken. Credential expiry and
refresh-chain health remain authoritative under `GET /v1/providers/health`.

## Claude credential model

Claude usage is separate from Claude inference.

- The bridge inference token (`CLAUDE_CODE_OAUTH_TOKEN`) is not sufficient for
  `/api/oauth/usage`.
- The usage adapter needs a `claude login` style OAuth bundle with
  `user:profile` scope.

Credential handling is implemented in
[src/llm_bawt/service/usage/claude_oauth.py](../src/llm_bawt/service/usage/claude_oauth.py).

### Modes

- `shared`: read-only reuse of an existing login bundle; no refresh/write
- `owned`: the app refreshes and rewrites its own dedicated bundle

In the main compose stack, the app is configured for `owned` mode and uses:

`/root/.config/llm-bawt/claude-usage/.credentials.json`

### TASK-635: this is now THE Claude credential (single login)

The owned bundle is no longer usage-only — it is the deployment's ONE Claude
credential. The `claude` provider adapter's wizard login mints a full-scope
bundle (`user:inference` + `user:profile` + …); the app is the sole refresher
(serialized + a proactive lifespan loop that refreshes at `expiresAt − 20min`,
so it never lapses even when idle). Consumers are read-only:

- the usage adapter (same process),
- the claude-code bridge, via a read-only compose mount
  (`CLAUDE_CREDENTIALS_PATH`) with `GET /v1/providers/claude/token` as its
  stale-file/force fallback (the app refreshes on demand; optional
  `BRIDGE_CLAUDE_TOKEN_SECRET` guards the endpoint via `X-Bridge-Token`).

The bridge never refreshes — the old dual-login (`claude-sub` + `claude-usage`)
and its refresh-rotation race are gone.

### TASK-845: recover rotation inside an active native Claude turn

TASK-636 moved the authoritative bundle to the encrypted provider connection
row; the app remains the sole refresher. TASK-671 fetches a fresh token when
starting a turn, but that token can be revoked while the SDK is executing tools.
Restarting/replaying the whole turn is unsafe once tools or text have run.

With the bridge's loopback server and app broker configured, native Claude SDK
requests now pass through `/claude-oauth/v1/messages` (also `count_tokens`).
This is a byte-preserving HTTP gateway, **not** a Responses API translator:
model IDs, beta headers, signed thinking and the SDK's tool loop remain native.

- Read the broker token for **each HTTP request**, including requests after tools.
- On an HTTP 401, ask the app for recovery using
  `force=true&rejected_token_sha256=<digest>`. Under its refresh lock, the app
  reuses a newer token if one already exists; only the current rejected token
  triggers another rotation. Concurrent rejected-A callers therefore reuse B.
- Retry that rejected HTTP request at most once, before any response is
  forwarded. Never retry a partial stream or restart the SDK turn. Repeated
  401s retain the existing terminal credential-error/reconnect contract.
- Broker outage does not silently use an old bootstrap bearer. The gateway
  reports unavailability. Deployments without an app broker, or with the
  loopback proxy explicitly disabled, retain the legacy direct path (and its
  startup-only recovery limitation).

Activation requires app reload **before** the explicitly approved bridge
reload, so conditional refresh is present before the gateway uses it. Source
changes alone do not update either already-running process. No dependency or
image rebuild is needed.

Verification: `test_claude_oauth_rotation.py`, `test_claude_oauth_gateway.py`,
and the opt-in `RUN_CLAUDE_OAUTH_SDK_PROBE=1` test in
`test_claude_oauth_gateway_sdk.py` (installed CLI, fake upstream, isolated HOME,
one local Read; no live provider traffic/credentials or bot sessions).

## Relevant environment variables

| Variable | Default | Purpose |
|---|---|---|
| `CLAUDE_CREDENTIALS_PATH` | unset | Preferred bundle-path override (TASK-635 name) |
| `CLAUDE_USAGE_CREDENTIALS_PATH` | `~/.config/llm-bawt/claude-usage-credentials.json` | Legacy bundle-path env (still honored) |
| `CLAUDE_USAGE_CREDENTIALS_MODE` | `shared` | `shared` or `owned` |
| `LLM_BAWT_USAGE_CACHE_TTL` | `120` | Cache TTL in seconds |
| `ZAI_API_KEY` | unset | Required for live z.ai usage |

## Quick check

```bash
curl -s 'http://localhost:8642/v1/usage?provider=claude' | jq
curl -s 'http://localhost:8642/v1/usage' | jq '.providers[] | {provider,status,available}'
```

## Common Claude failures

| Status | Meaning |
|---|---|
| `stale` | Shared credential exists but its access token expired |
| `usage_stale` | Cached quota values are shown because a live usage refresh failed; credential health is separate |
| `unauthorized` | Missing bundle or wrong scope |
| `rate_limited` | Upstream usage endpoint returned `429` |
| `error` | Network or unexpected upstream failure |
| `not_implemented` | Placeholder adapter rather than a live implementation |
