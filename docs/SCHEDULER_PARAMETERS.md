# Scheduled job parameters (TASK-1197, TASK-1198)

Maintenance job configuration remains in `scheduled_jobs`. The opt-in
`JobParameterRegistry` supplies validated models and JSON Schema metadata;
`IdleContextResetParameters` is the first registered model. This is not a new
scheduler or a replacement for legacy runtime settings.

## API and UI

`GET /v1/jobs` adds these fields to each job:

- `parameter_schema`: JSON Schema describing supported parameters, defaults,
  labels, bounds and help; null for an unmigrated job.
- `parameters`: validated effective values, including defaults; null for an
  unmigrated job or invalid configuration.
- `configuration_error`: validation error when a registered job's stored
  configuration is invalid. Invalid values never silently fall back at execution.
- `stored_parameters`: unvalidated registered fields exposed only when configuration
  is invalid, so individual repairs preserve the other stored values. These are
  not effective policy; unknown fields remain in storage, not in this editor.

`PATCH /v1/jobs/{job_id}` accepts any subset of:

```json
{
  "enabled": true,
  "interval_minutes": 5,
  "parameters": {
    "resident_tokens_gt": 100000,
    "idle_minutes_gt": 15,
    "protected_bot_ids": []
  }
}
```

Updates lock the job row and commit together. Parameters merge with stored values
and are validated before commit. Omitted parameters inherit their documented
defaults. Common patch fields reject explicit null; unknown fields,
booleans-as-integers and numeric strings are rejected. Legacy null optional
model/path/assignee/package values retain their inherited meaning. Cadence is
an integer from 1 through 525600 minutes. Enabling a paused
job or changing its interval schedules the next occurrence one interval after
save. Parameter-only changes do not alter the next-run timestamp. Disabling does
not cancel an in-flight run. Prompt schedules are excluded: their separate
owner-scoped API remains authoritative.

Settings → Automation exposes effective values and a schema-driven editor with
Save/Cancel, validation and save feedback. Invalid configuration never implies
repair: pausing or changing cadence submits only that common-setting change.
Only edited parameter fields are submitted, and default previews are never
silently saved. Unmigrated jobs expose only common cadence/enablement controls;
their global/job-specific settings are untouched.
Run history requests `include_result=true` and can display the policy snapshot.

## Idle Context Reset

| Parameter | Default | Accepted values | Meaning |
|---|---:|---|---|
| `resident_tokens_gt` | 100000 | integer 1–2000000 | Resident context must be strictly greater |
| `idle_minutes_gt` | 15 | integer 1–10080 | Completion must be strictly more than this many minutes ago |
| `protected_bot_ids` | [] | array of nonempty bot IDs | Additional exclusions; built-in protected bots cannot be removed |
| `continuity` | `configured` | `configured`, `inline+summaries`, `inline`, `summaries`, `none` | What the new session carries (TASK-1203) |

### Continuity (TASK-1203)

`configured` ("Use bot setting") keeps the established behavior: each reset
follows the bot's own `history_scope`. Any other value is one of the canonical
`history_scope` values (`setting_definitions.HISTORY_SCOPES`) and replaces the
bot's scope for this job's resets only; it never writes the bot's setting.

| Value | UI label | Seed | Pre-seed summary of outgoing thread |
|---|---|---|---|
| `inline+summaries` | Messages + summaries | recent messages + summaries | yes |
| `inline` | Messages only | recent messages | no (would starve the raw seed) |
| `summaries` | Summaries only | summaries | yes |
| `none` | Nothing (clean start) | empty | no; the background summarization job still archives it later |

The scope rides only the persisted `new_session_intent.scheduler_parameters`
snapshot. At `/new` preparation `NewSessionExecution.scheduled_history_scope`
reads it from the delivery's own durable metadata, never from request fields.
`prepare_agent` resolves one concrete scope per configured `/new` (the override,
else the bot setting, read once via `utils.history.effective_history_scope`) and
hands that same value to `_maybe_summarize_on_new`, `maybe_build_session_seed`,
`build_context_seed` and the session continuity receipt, so a concurrent
bot-setting edit cannot split them. Only `NewSessionLifecycle` stamps
`new_session_intent`: public `POST /v1/inter-bot-deliveries` rejects caller
metadata carrying that key (422). Ordinary chat `/new`, bot-delivery and API
resets never carry an override. Scheduled
resets already admit only `claude-code` bots, so Codex needs no separate rule.
Older persisted intents without `continuity` decode as `configured`.

The scheduler snapshots saved configuration while claiming the occurrence. The
same validated, immutable policy is used by candidate filtering, submission,
and durable claim-time rechecks. New intents persist this policy; older intents
without it retain historical defaults. An edit affects future sweeps, not an
already admitted reset. The stable job/session/measurement idempotency key is
unchanged: another sweep with edited settings attaches to the original delivery
and retains its original claim-time snapshot rather than performing another reset.
The occurrence claim commits `parameters` plus `interval_minutes` in the RUNNING
JobRun before execution begins. Failed sweeps retain that snapshot, even if the
job configuration changes during execution. Invalid configuration is recorded
as a `configuration_error` snapshot and fails without running the reset.

Busy checks, session/measurement fences, provider evidence and receipt validation
remain mandatory. Threshold configuration does not initialize missing accepted
session pointers: the existing-session rollout gap is a separate issue.

## Remaining maintenance jobs (TASK-1198)

The existing Automation editor consumes these registered flat schemas. Prompt
schedules retain their separate owner-scoped API and are not maintenance jobs.
This is a read-time migration, not a data rewrite: saved job JSON wins over
current shared defaults, then the historical code default. Reads, restart and
unrelated parameter patches do not materialize inherited defaults into storage.
Cadence and enablement remain the existing row fields; bootstrap never rewrites
an existing row. Blank model inherits the same global resolution and fallback
policy as before; choosing an explicit model disables fallback as before.

| Job | Editable controls and prior effective source |
| --- | --- |
| Profile maintenance | `entity_id`, `entity_type` from saved JSON (defaults user/user); `dry_run=false`; optional explicit `model`, otherwise shared profile/maintenance model. |
| Memory consolidation | `entity_id` from JSON (system); `dry_run=false`. Consolidation-only task flags remain fixed. |
| Memory decay | `entity_id` from JSON (system); `dry_run=false`. Decay-only task flags remain fixed. |
| History summarization | JSON user/system, heuristic fallback/true, chunk tokens/4000, model override; session gap, minimum messages, protected turns and trigger inherit `summarization_job` unless overridden. Fact importance and profile extraction inherit the shared extraction settings unless overridden. |
| Memory extraction | JSON user/system, batch size/50 (1–1000), model override; enabled, minimum importance and profile extraction inherit the existing shared extraction config unless overridden. |
| Media GC | JSON user/system, grace days/7, dry run/false. |
| Tool-result GC | JSON user/system, retention days/14, dry run/false. |
| Harness version check | JSON user/system, constraints path/installed pins, packages/installed list, assignee/none, priority/MEDIUM. Empty optional values retain those defaults. |
| Playground expiry | JSON user/system. Trash retention remains shared with the Playground UI. |

The history eligibility probe and worker receive the same session controls.
Both raw-message and post-summary fact extraction consume their own job's
importance/profile options. A local profile option also reaches the inner
profile helper; it never changes the shared config for manual extraction.
Post-summary extraction retains its independent model and lack of the raw
extraction enabled gate. Explicit zero trigger and nonpositive chunk limits
retain model-budget and no-split semantics respectively.

### Controls intentionally kept shared

Manual MCP extraction still uses extraction enablement/importance/profile
settings. Manual summarization and `/new` still use `summarization_job`.
Model aliases, endpoint credentials, fallback/spending policy, context limits,
embedding model/dimension and scheduler poll interval remain service-wide.
Maintenance decay half-life also controls chat retrieval scoring; consolidation
and pruning algorithms retain their existing numeric defaults (including the
consolidator's 0.75 threshold), without introducing ignored numeric parameters.
Profile attribute importance stays at the effective inner 0.7 threshold; the
unused similarly named config field is not newly activated. Playground trash
retention is also used to display deletion deadlines. None of these shared
controls is removed or duplicated as a new authoritative global setting.

Focused verification: `tests/test_job_parameters.py`, existing idle-reset and
lifecycle-intent/scheduler tests, plus frontend configuration/proxy tests. No live
job edits or provider resets are needed for these tests.
