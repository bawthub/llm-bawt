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
`maintenance:fallback:<provider>:probe`, TTL = probe call timeout + margin) to touch the
primary, and every loser still goes to the backup. So no call reaches the primary after
cooldown except the one probe, however late the first post-cooldown call arrives (Loopy
B1). A crashed probe's claim expires and the next call re-claims.

Writes: the first exhaustion creates the record with `SET NX` (switched edge). A probe
that hits `spending_limit` again re-arms the existing record with a Lua compare-and-set
(match on `exhausted_since`, set new `until`, release the probe claim), not with
`SET NX`, so the cooldown always resets (Loopy r2 cmvtiaov600x80mldsivzmelm). Survives app restarts
and is shared across workers.

Redis unavailable → stateless per-call mode: every call tries the primary, and a
`spending_limit` falls through to the backup in the same call (no "skip the primary"
optimization). One alert says exhaustion state is unavailable. Every switch is still
recorded per call (served model + `via_fallback`), so Redis down is never silent and
never switches without a record.

Keyed by provider, not model: a spending limit is per provider account.

## Notifications (transitions only)

Via the existing bell/notify path, one event per transition:
- **switched**: "<provider> spending limit: maintenance now on backup <alias>".
- **switched back**: probe succeeded, "<provider> back as maintenance model".
- **backup also failed**: loud — a slice A maintenance-failure warning for the job,
  naming both models and both error classes (type + status only, per slice A B2).
No per-call notifications.

Transition dedupe (Loopy B3): the state change itself is the dedupe. The exhausted
record is written with `SET NX` (switched) and deleted by a compare-and-delete Lua script
(switched back), so exactly one caller wins each edge. The winner, in the same atomic
step, appends the event to a Redis outbox list (`maintenance:fallback:outbox`, `LPUSH`
inside the same `MULTI`/Lua). A publisher drains the outbox and removes an entry only
after the bell send succeeded (`LMOVE` to a processing list, ack by `LREM`; entries
left in processing are retried by the running publisher, not only at startup: each
processing entry carries `claimed_at`; the publisher loop re-sends any entry older than
a visibility timeout, with backoff, until the bell send succeeds (Loopy r2
cmvtiaov600x90mld616rhkvl). A crash or send failure therefore re-sends
instead of losing the notice; delivery is at-least-once, and the bell row is keyed by
the event id so a re-send updates rather than duplicates.

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
- Outbox: publisher fails/crashes after the edge win → notice re-sent on restart, once
  in the bell.
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
