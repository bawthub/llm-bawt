# Backup maintenance model (TASK-1704 slice B) — DESIGN

Status: draft for DESIGN review. Builds on slice A (`service/maintenance_health.py`:
`classify_failure`, `MonitoredClient`, `MaintenanceHealth`). Replaces the dropped
credential-chain draft (`PROVIDER_CREDENTIAL_FAILOVER.md`, deleted, never committed).

## Goal

When a maintenance call's provider returns `spending_limit`, retry it on an explicitly
configured backup model on another provider, instead of failing until someone tops up.
Unset backup = today's behavior plus the slice A alert.

## Setting

- New global runtime setting `maintenance_model_fallback` in `setting_definitions.py`,
  next to `maintenance_model`. Default `None`. Global only (no per-bot/per-job variant).
- Value is a model alias from the model catalog, chosen by the user. No code default,
  no seed value, no model literal anywhere.
- Optional `maintenance_fallback_cooldown_seconds` (global setting, default declared in
  `setting_definitions.py` like other tunables, user-editable). A provider reset hint
  (`Retry-After` / reset header or body field) overrides it when present.

## Scope of calls

Every call that goes through `_get_background_client(job=…)` / `MonitoredClient`
(summaries, memory extraction, profile maintenance, recall, utility completions),
whether the job resolved a job-specific model (e.g. `profile_maintenance_model`) or
`maintenance_model`. Conversations with bots are out of scope.

**Explicit request models are excluded** (Loopy should-fix): when a caller names a model
(`request.model` on `/v1/llm/complete`, or a bot/job override passed by the caller), the
backup is never used; a `spending_limit` fails visibly with the slice A alert. The backup
only replaces models resolved from settings (job-specific setting or `maintenance_model`),
because the user consented to the backup's provider for settings-resolved maintenance,
not for a model someone picked by name.

## Behavior

1. Call primary. Slice A classifies the failure.
2. Only `spending_limit` triggers the backup. `rate_limit`, `auth`, timeouts,
   `unconfigured`, empty reply and everything else fail as today.
3. On `spending_limit` the primary's provider is marked exhausted (see State) and the
   same request is replayed once on the backup. Never more than one backup attempt.
4. While the provider is exhausted, calls go straight to the backup (no doomed primary
   call), until the cooldown expires.
5. After cooldown exactly one call (atomic claim) probes the primary. The probe is a
   real job's call and never costs that job: if it hits `spending_limit` again, the same
   call falls through to the backup in that one attempt, the provider is re-marked
   exhausted and the cooldown resets. Success clears exhausted state; other errors don't
   change exhausted state. Concurrent calls keep using the backup during the probe.

### Validation (fail visibly, never substitute)

At resolve time, the backup is rejected with a visible `unconfigured`/`incompatible`
failure (slice A alert) if: the alias isn't in the catalog; it's on the same provider as
the failing primary (same balance, pointless); or it can't serve the call (e.g. needs
tools/json/bounded query the backup client lacks). The primary error is then reported as
is. Backup is never chosen implicitly, and no other model is ever tried.

## State (durable, shared)

Redis key `maintenance:fallback:<provider>` = `{exhausted_since, until, reason_class,
hint_source, state}` with **no TTL**: the record does not expire when the cooldown
ends. `state` is `exhausted` → (cooldown passes, record stays) → only cleared by a
successful probe. Every call reads the record: before `until` it goes to the backup;
after `until` it must win the probe claim (`SET NX` on
`maintenance:fallback:<provider>:probe`, TTL = probe call timeout + margin, value = token) to touch the
primary, and every loser still goes to the backup. So no call reaches the primary after
cooldown except the one probe, however late the first post-cooldown call arrives (Loopy
B1). A crashed probe's claim expires and the next call re-claims.

Writes: the first exhaustion creates the record with `SET NX` (switched edge). After
that, only the current probe holder may change the record, fenced by a **probe token**
(Codex r1 cmvtitngc00xl0mld0w33cejg): `claim_probe` writes a fresh random token as the
probe key's value (`SET NX EX`) and returns it. Both edge scripts take
`(exhausted_since, token)` and act only if the record's `exhausted_since` matches **and**
`GET probe == token`; otherwise they are a no-op:
- re-arm (probe hit `spending_limit`): set new `until`, delete the probe key. The cooldown
  always resets (Loopy r2 cmvtiaov600x80mldsivzmelm).
- clear (probe succeeded): delete record and probe key, write the episode (Notifications).
A probe whose lease expired has lost its token (expired, or replaced by a newer
holder's, or deleted by a re-arm), so a late success or late spending_limit can never
delete or extend a cooldown it no longer owns; it only affects its own call's result.
Survives app restarts and is shared across workers.

Redis unavailable → stateless per-call mode: every call tries the primary, and a
`spending_limit` falls through to the backup in the same call (no "skip the primary"
optimization). One alert says exhaustion state is unavailable. Every switch is still
recorded per call (served model + `via_fallback`), so Redis down is never silent and
never switches without a record.

Keyed by provider, not model: a spending limit is per provider account.

## Notifications (transitions only)

The bell is **pull-based** (slice A): BawtHub polls `/v1/health/maintenance` and shows
every row whose status is not `ok` (`systemHealth.ts` maps non-ok children to a degraded
parent; `ok` rows never reach the bell). Nothing is pushed, so there is no send that can
fail and no outbox. Notices are non-ok rows rendered on every poll from durable Redis
state, one row name per provider, `maintenance_fallback:<provider>`:
- **switched** (record present): `degraded`, "<provider> spending limit since
  <exhausted_since>: maintenance on backup <alias>; next probe after <until>".
- **switched back** (no record, episode key present): `degraded`, "<provider> was on
  backup from <first_since> to <recovered_at> (<n> episode(s)); back on primary". The
  clear script writes the episode key `maintenance:fallback:<provider>:recovered` =
  `{first_since, recovered_at, episodes}` in the same atomic step that deletes the record,
  merging with an existing episode key (keeps the earliest `first_since`, increments
  `episodes`), with TTL `RECOVERED_ROW_SECONDS` (24h, renewed on each clear). The row stays
  non-ok for that window, then the key expires and the row disappears. The notice window
  is a named constant, documented as a deliberate "you missed this" period.
- **backup also failed**: loud, a slice A `down` row for the job naming both models and
  both error classes (type + status only, per slice A B2).
No per-call notifications. A record wins over an episode key for the same provider (one
row; a re-exhaustion shows as switched, and its later clear merges into the episode).

Why notices can't be lost (Codex r1 cmvtitngc00xm0mldzj0t1909, Loopy B3
cmvtiaov600x90mld616rhkvl): every switch creates a record with no TTL, and the only way
a record goes away is the clear script, which atomically writes the episode. So from the
first switch until 24h after the last recovery, Redis always holds a key that renders a
non-ok row. An episode that starts and ends between two polls is still shown (as the
episode row) for 24h, far longer than any poll interval; an app crash or restart between
edge and poll changes nothing because the row is re-derived from Redis on every poll.
Why it can't repeat or flap: one row name per provider, status and text are a pure
function of the two keys, edges are atomic and token-fenced (State), so polls with
unchanged state render byte-identical rows, and concurrent callers crossing an edge
produce one state, hence one row. Redis unavailable → one `maintenance_fallback_state`
`down` row instead (see State); it is not silent.

Accepted limit: the bell shows current state of the last 24h, not a full log. Several
episodes inside the window collapse into one row with a count and the overall span.

## Attribution

Each call records which model actually served it: `MaintenanceHealth.record_success/
record_failure` already take `model`/`provider`; the backup path passes the backup's,
plus `via_fallback=True`. Turn/usage logs for utility calls record the served model.

The served model is only known after the call (Loopy B2). Today
`_get_background_client` returns the primary alias up front and the utility and
summarization consumers report it. Change: `MonitoredClient` exposes
`last_served_model` / `last_served_provider` / `last_via_fallback`, set per call, and
those consumers (`utility_completion.py` response `model`, `history_summaries` /
summarization records, usage logs) read it after the call instead of the pre-resolved
alias. A test asserts the reported model equals the backup when the backup served.
The health snapshot shows "on backup since …" per job.

## UI (BawtHub)

Settings page: a model picker for `maintenance_model_fallback` next to the maintenance
model picker, catalog-driven, with a "None" option, plus the cooldown field. The picker
copy states plainly that while the backup is in use, maintenance data (memories,
summaries, profile facts) is sent to the backup model's provider; choosing it is the
consent, so the label must say so. Shows a
warning chip if the chosen backup shares the maintenance model's provider.

## Tests

- spending_limit → backup served, recorded with backup model, one "switched" event.
- rate_limit / auth / timeout → no backup, no state change.
- Unset backup → today's failure + alert; same-provider / unknown alias → visible failure.
- Exhausted state skips primary; survives a fresh process (Redis fixture).
- Cooldown: provider hint wins over setting; exactly one concurrent probe; probe success
  → "switched back"; probe spending_limit re-arms.
- Probe hits spending_limit → same call served by backup, provider re-marked, cooldown reset.
- Exhausted record outlives `until`: a first call long after cooldown plus N concurrent
  calls → exactly one primary call (the probe), the rest on backup.
- Crashed probe: claim TTL expires, next call re-probes.
- Poll rows: exhausted → one `degraded` row, stable across repeated polls and a fresh
  process; probe success → same row name, still `degraded` (episode) for the window, then
  gone; switch+recover between two polls → episode row visible; two episodes → one row,
  count 2; never an `ok` row; no duplicates.
- Fencing: probe lease expires mid-call, another call re-claims and re-arms; the late
  first probe's success or spending_limit changes nothing (record and `until` intact).
- Utility/summary responses and usage logs report the backup model when it served.
- Explicit `request.model` + spending_limit → no backup, visible failure.
- Concurrent calls crossing an edge → exactly one "switched" / one "switched back".
- Redis down → per-call fallthrough still works, one state-unavailable alert, every
  switch recorded.
- Backup also fails → loud alert with both classes, no secrets in reason/log.
- No model literal: guard test greps the new module.

## Out of scope

Credential chains, OAuth/sign-in (dropped by Nick). Bot conversation failover.

## Open (Nick)

Which model is the backup — configuration only, not a code dependency.
