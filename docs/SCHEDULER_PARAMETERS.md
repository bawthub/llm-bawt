# Scheduled job parameters (TASK-1197)

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
and are validated before commit. Omitted parameters receive model defaults;
explicit nulls, unknown fields, booleans-as-integers and numeric strings are
rejected. Cadence is an integer from 1 through 525600 minutes. Enabling a paused
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

## Extending in TASK-1198

For each remaining job, first trace its actual runtime settings and shared
consumers. Add a `JobParameters` model and register its job type only when its
execution path consumes those validated effective values. Initialize/migrate
from the existing effective settings with explicit precedence; never silently
replace operator choices with new defaults. Keep genuinely shared service/model
settings global. Do not remove old global controls while other consumers need
them. Until registered and integrated, legacy config JSON remains opaque to this
framework and existing execution paths continue unchanged.

Focused verification: `tests/test_job_parameters.py`, existing idle-reset and
lifecycle-intent/scheduler tests, plus frontend configuration/proxy tests. No live
job edits or provider resets are needed for these tests.
