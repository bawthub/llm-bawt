# Scheduled prompts — end-to-end design and implementation plan

Status: **Application implementation ready for review; live activation and PostgreSQL/live-inference verification outstanding.** Researched 2026-09-15 for the operator. Reuses TASK-166, TASK-167, TASK-168 and TASK-169 (the latter reopened at the operator's request). This supersedes their April one-shot-first specs. One-time, interval and cron schedules are all release scope.

## 1. Product contract

A schedule sends a saved prompt as its owning user to a selected bot through the ordinary chat pipeline. The management UI is the primary surface; the chat composer offers a prefilled shortcut. No browser tab needs to remain open. No OS crontab, shell-command scheduling, new queue service, alternate model invocation path, or automatic permission bypass.

Required/editable fields:

| Field | Contract/default |
|---|---|
| Name | Required, 1–120 trimmed characters; list-friendly label. |
| Description | Optional, at most 2,000 characters; explains purpose, not injected into the prompt. |
| Prompt | Required nonblank text, at most 100,000 characters; literal text, not a template language. |
| Target bot | Required concrete existing enabled bot; no wildcard. Enforce backend capability and user access. |
| Model | Explicit catalog model reference or **Bot default at run time**. Validate capability on save and immediately before dispatch. Never change the bot's saved default. Record the actually resolved model on every run. |
| Schedule type | `once`, `interval`, or `cron`. Exactly one timing configuration. |
| One-time date | Offset-aware execution instant; selected IANA timezone shown alongside local date/time. Must be in the future on create/reschedule. |
| Interval | Positive whole minutes (minimum 1), anchored to an explicit first-run instant. UI supports minutes/hours/days. Elapsed duration, not calendar time; every 24 hours is not the same as daily at 09:00 across DST. |
| Cron | Five-field Unix expression; timezone required. Friendly editor plus advanced text. Daily/weekly/monthly presets are cron, not special server scheduling logic. |
| Timezone | IANA identifier, initially browser timezone (for example, `America/New_York`), never a fixed ET/EST offset. Persist timezone on every schedule. Store execution instants in UTC. |
| Clear context | Default **on**: a fresh, non-active thread for each occurrence. Off: reuse this schedule's own thread across runs. Neither choice resets the user's active chat or deletes anything. Label: “Start each run with fresh conversation context.” |
| Memory | Separate advanced `augment_memory` and `extract_memory` booleans; default to ordinary-chat behavior. Clear conversation context is not memory deletion and does not promise removal of personality/profile information. |
| Enabled | Save enabled by default, with a prominent **Save paused** option. |
| Start/end bounds | Optional end instant for recurring schedules; explicit interval anchor and optional cron start bound. End is exclusive. No arbitrary 30-day lead-time limit. |
| Missed-run policy | `skip` or `run_latest`; default `run_latest` within a 1-hour start deadline. No unlimited catch-up flood. Configurable grace 60 seconds–7 days. |
| Context destination | V1 intentionally uses a dedicated automation thread only. No implicit delivery into whatever personal thread happens to be active. |

Schedule lifecycle: active, paused, completed (one-shot or end bound exhausted), cancelled (soft delete). Run state is separate: pending, queued, running, succeeded, failed, skipped, cancelled, unknown. A failed occurrence does not mean a recurring schedule stopped.

Management actions: create, edit, pause/resume, duplicate (opens unsaved draft), run now, cancel schedule, inspect run history, cancel a still-queued occurrence, open resulting conversation/turn. Run now creates a separate manual occurrence and does not consume a one-shot or move recurring cadence; label this clearly. It requires confirmation, works on paused schedules, rejects cancelled/completed schedules, and uses a client idempotency key so double-click/retry is one operation.

## 2. Research and library decision

### Timing: croniter 6.2.4

Use `croniter==6.2.4` (released 2026-07-10), maintained under Pallets Community Ecosystem. It solves expression parsing, ranges/steps/lists, next/previous occurrence calculation, timezone-aware iteration and strict calendar validation. Use Python `zoneinfo`, aware datetimes and bounded `max_years_between_matches` (8 years, covering leap-day schedules). Lock the dependency; do not write a cron parser.

Public V1 grammar is deliberately narrower than croniter's complete grammar: exactly five Unix fields; numeric/name lists, ranges, steps, `*`; Sunday 0 or 7; day-of-month/day-of-week use standard Unix **OR** semantics. Reject seconds/year fields, Quartz `?`, `L`, `W`, `#`, hashed/random forms, inline TZ directives and macro aliases. This keeps editor, description and backend semantics aligned. Backend validation and preview are authoritative; frontend interpretation is presentation only. Impossible dates must fail validation, not create a permanently broken job.

### Editor: react-cron-generator 2.6.0 + cronstrue

Use the existing React 19 stack with `react-cron-generator` in **`isUnix={true}`** mode, never its default Quartz mode. It supports React 19 without Ant Design. Use `cronstrue` for human descriptions, but verify expressions against the server preview before saving. Lazy-load the editor on the scheduling route/dialog. Scope styling to the scheduling feature and use existing slate/glass theme. Verify the chosen package's published typings, keyboard behavior and 375px layout before integrating; package README accessibility claims are not test evidence.

`react-js-cron` 6.0.2 is a viable editor but requires Ant Design >=6; adding a second design system solely for a cron picker is not justified. `cronstrue` 3.26.0 has no runtime dependencies and is also the chosen editor's existing dependency.

### Alternatives considered

- **APScheduler 3.11.3**: mature date/interval/cron triggers, misfire/coalescing controls and persistent stores. A good greenfield scheduler. Here it would introduce a second scheduler/job store alongside `JobScheduler`, `scheduled_jobs`, `job_runs` and the durable dispatcher. Its 3.x FAQ explicitly warns against sharing a persistent job store across processes because synchronization is absent. Its cron semantics also differ from Unix in important ways. Do not migrate maintenance scheduling just to add prompt schedules.
- **Celery Beat / Redis-backed JS queues**: require another worker/runtime and operational lifecycle despite an existing app-owned durable dispatch path. No demonstrated need.
- **Host cron / Kubernetes CronJobs**: wrong ownership and deployment surface for user-editable application rows. Kubernetes documentation is useful guidance on deadlines, overlap and idempotency, not a proposed dependency.

Primary sources:
- https://github.com/pallets-eco/croniter
- https://pypi.org/project/croniter/6.2.4/
- https://apscheduler.readthedocs.io/en/3.x/userguide.html
- https://apscheduler.readthedocs.io/en/3.x/faq.html#how-do-i-share-a-single-job-store-among-one-or-more-worker-processes
- https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html#daylight-saving-time-behavior
- https://github.com/sojinantony01/react-cron-generator
- https://github.com/bradymholt/cRonstrue
- https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/

### DST is a policy, not an implementation accident

An isolated executable probe against croniter 6.2.4 found:
- `30 2 * * *`, New York, 2027 spring transition: next occurrence is **03:00 EDT** on March 14, not 02:30 (which does not exist).
- `30 1 * * *`, New York, 2026 fall transition: returns **both** 01:30 EDT and 01:30 EST on November 1.
- Strict validation rejects `0 0 31 2 *`.

Product policy: calendar schedules **skip nonexistent wall-clock times** and fire **once at the first occurrence of a repeated wall-clock minute**. Wrap the library's candidates with a small policy adapter: confirm the local candidate actually matches the expression; reject `fold=1`; require strictly increasing UTC instants. Do not rewrite timezone or cron calculations. Test from before, inside and after both transition windows, including restart between folds and less-common non-hour transitions. Interval schedules remain elapsed UTC durations. Preview uses this exact adapter. Explain the policy beside timezone selection. For a user-entered one-time ambiguous local time, require an explicit offset/choice; never silently guess. Reject nonexistent one-time local times.

## 3. Architecture: reuse what exists

Current verified seams:
- `service/scheduler.py`: `JobScheduler`, `ScheduledJob`, `JobRun`, enum migration, maintenance loop. Currently serial maintenance processing, interval-only, no prompt job type, no atomic prompt claims.
- `inter_bot_delivery.py`: durable store already accepts `AuthorReference.user(...)`, request payload/model, metadata and stable idempotency keys; same-key submissions use a DB lock.
- `service/inter_bot_dispatcher.py`: per-target FIFO, PostgreSQL leadership, busy deferral, transport leases/recovery, normal chat/SSE execution, terminal-turn verification.
- `service/routes/sessions.py`: `activate=false` creates a born-archived thread.
- `service/routes/history_seed.py`: explicitly selected threads hydrate only their own history.
- `agent_context.py::rotate_delivery_session`: current reset policies rotate the active thread. **Do not use these policies for the schedule clear-context flag.**
- BawtHub already has `ModelSelect`, bot/model hooks, TanStack Query, unified event stream and `/api/chat/proxy/v1/*` forwarding. Reuse these.

Flow:

```
Schedule editor → BawtHub user-scoped proxy → /v1/prompt-schedules
    → existing scheduled_jobs + prompt-specific detail
    → atomically reserve due occurrence + advance schedule
    → durable occurrence/outbox receipt
    → existing delivery store (user authorship, prefer_steer=false)
    → existing per-target FIFO / normal chat pipeline / approvals
    → existing turn + messages + SSE
    → reconciled run status / management history / conversation link
```

Timing and execution remain separate. The scheduling producer must not await model inference or slow maintenance jobs. Extend the existing `JobScheduler` lifecycle with a lightweight prompt sweep owned/stopped by it (target 5-second cadence); keep maintenance cadence and processing intact. No separate deployed scheduler service. The prompt sweep can execute safely in multiple app processes through database claims; only the existing delivery dispatcher owns delivery leadership.

## 4. Persistence and atomicity

Keep `scheduled_jobs` and `job_runs` as the canonical job/run identities. Add `SEND_PROMPT` to `JobType` with the existing PostgreSQL enum migration convention. Generic maintenance execution must explicitly exclude prompt jobs; prompt-specific scheduling must never fall through to interval-only timestamp handling or activity gating.

Use typed companion SQLModel tables rather than adding many nullable prompt columns to every maintenance job:

- `prompt_schedules`: `job_id` primary/FK to scheduled_jobs; owner_user_id, name, description, prompt, requested_model nullable, schedule_type, timezone, once_at, interval_seconds, interval_anchor_at, cron_expression, starts_at, ends_at, clear_context, augment_memory, extract_memory, dedicated_session_id nullable, missed_policy, misfire_grace_seconds, lifecycle, revision, created_at, updated_at, cancelled_at. Target bot/enabled/next_run_at remain on ScheduledJob; mutate the pair transactionally. Explicit constraints enforce each schedule variant. Owner and due-time indexes.
- `prompt_occurrences`: `run_id` primary/FK to job_runs; job_id, occurrence_key unique per job, scheduled_for UTC, kind scheduled/manual, schedule_revision, immutable request/config snapshot, session_id, delivery_id unique nullable, state, attempt/enqueue diagnostics, created/updated timestamps. JobRun holds standard timestamps/duration/result/error; companion gives proper indexed audit/correlation rather than hiding all relationships in JSON.

Reserve under row locking (`FOR UPDATE SKIP LOCKED`): re-read current enabled/lifecycle/revision, calculate eligibility, check nonterminal occurrence, insert JobRun + occurrence snapshot, and advance `next_run_at` in the same transaction. Occurrence key is the scheduled UTC instant for automatic runs, or caller UUID for manual runs. Uniqueness is the final defense against duplicate ticks.

The occurrence is the durable outbox; do not create a second queue. After commit, enqueue to the existing delivery store using `schedule:{job_id}:occurrence:{occurrence_key}`. If the process dies after delivery enqueue but before saving delivery_id, resubmit the immutable snapshot with the same key and reconcile the same delivery. If it dies before enqueue, pending occurrence recovery picks it up. Never recompute prompt/model/target from an edited schedule during outbox recovery. If explicit model is unresolved/disabled at dispatch, fail clearly rather than silently substituting another.

No blanket exactly-once claim: at-least-once recovery plus one logical occurrence and existing transport deduplication. Ambiguous accepted inference is **unknown/needs inspection**, never automatically replayed as a fresh prompt. Preserve existing side-effect guards and recovery behavior; record actual evidence.

## 5. Execution semantics and limits

- Bot busy: enqueue a separate turn via `prefer_steer=false`; never inject a scheduled prompt into an in-flight conversation, never bypass concurrency or approvals.
- Same schedule overlap: maximum one nonterminal occurrence (pending/queued/running). While occupied, coalesce missed slots to at most one latest eligible occurrence after completion, or skip according to policy. Do not append an unbounded FIFO backlog.
- Start deadline applies while pending/queued as well as after downtime. Revalidate inside the claim transaction before reserving a turn; expired queued prompt deliveries become skipped/cancelled without invoking a model. Must not affect ordinary inter-bot callbacks.
- Intervals advance from the original anchor, never completion time. Cron advances from calendar occurrences. Evaluate the latest eligible slot arithmetically/library `get_prev`, not by iterating millions of missed minutes. Bound every search.
- Scheduler downtime: recurring `run_latest` emits at most one within grace; `skip` records missed-window summary and advances. One-shot missed beyond grace becomes terminal skipped, not silently enabled forever.
- One-shot becomes completed only when its automatic occurrence is terminal; an error remains visible on that occurrence. Manual test runs do not complete it.
- Pause/cancel linearize with claiming: block new occurrence creation and cancel provably unaccepted queued occurrence(s). Already executing work finishes, with explicit UI feedback; these controls are not abort buttons. Resume computes future cadence and does not replay paused time. Reject edits to execution-defining fields while an occurrence is nonterminal (409); name/description can change. Optimistic `revision` prevents stale-editor overwrites.
- End bound prevents automatic starts at/after that instant, including queued work; does not kill already executing turns. Bounds and grace apply atomically at dispatch, not only at schedule materialization.
- Defaults: one nonterminal run per schedule; one active turn per target through existing queue. Cap active schedules per user (initial server policy 100) and validate minimum interval 60s. Future adjustable quotas should follow existing runtime-setting patterns, not new env guesses.
- Completed/failed/skipped runs retained for 90 days through existing GC infrastructure with bounded batches; retain schedule itself and aggregate last-run/counters. FK behavior must preserve normal turn/chat history. Cancellation is soft; no UI purge in V1.

### Session and model behavior

Completed dedicated turns carry a bounded main-session receipt (TASK-1994): schedule/run identity, task references, UTC finish time, a 240-character prompt excerpt, a 1,000-character reply excerpt and source user/assistant message IDs. Up to five newest pending receipts append to the next canonical active SDK thread's outbound user message, shared by Claude Code and Codex, without changing its pinned system prompt or stored user text. Other/dedicated sessions and maintenance commands do not consume them. An immediate main-chat question reconciles already-finished runs without waiting for the five-second scheduler sweep.

`prompt_occurrences.main_notice_json` stores the receipt; `main_notice_turn_id` is stamped on first real model output, not dispatch acceptance. A pre-SDK failure therefore leaves it pending across retries or main-session rotation. A crash between model consumption and the acknowledgement can replay the same stable run ID; this is not a strict exactly-once transport. Excess receipts are deferred, not dropped; unconsumed receipts are excluded from run-history pruning. Old terminal runs are not backfilled. These short notices do not seed the main transcript into automation sessions or request another wake-up.

Create sessions lazily when an occurrence will be delivered. Use normal born-archived creation/storage primitives with deterministic occurrence/session identity; no active-session rotation. `clear_context=true`: one empty explicit thread per occurrence. `false`: one persistent explicit thread for this schedule (minted lazily) across runs. Persist chosen session before outbox enqueue so recovery cannot create a second thread. Show “automation” origin and schedule name in the thread title/metadata. Opening the transcript is user action; dispatch never changes active selection.

Use `session_policy=continue` on the delivery and an explicit owned `session_id`, not `reset_without_history`. That existing reset function archives the active personal thread; using it would violate the product contract. New empty explicit threads seed no unrelated messages/summaries; verify SDK resume keys are scoped to the explicit thread. Clear context does not edit bot-wide history_scope, model, or memories. Existing authorship must be `user`, not fake bot `scheduler` attribution. Sender namespace may identify the scheduler internally; visible author is the owner.

Verify model compatibility and bind resolution once per occurrence. Request-local overrides must not mutate the user's ordinary conversation model/session. For backends that cannot honor explicit thread isolation/model override, reject creation with a capability error rather than degrade silently. V1 targets the deployed Claude Code/proxy path; no claim of untested legacy backend support. No tests against Mira or other protected targets.

## 6. API and user scoping (TASK-167)

Canonical new route: `/v1/prompt-schedules` (the old `/v1/messages/schedule` existed only in task text, so no deployed compatibility route is necessary).

- `POST /preview`: validate timing and return normalized timing, next 5–10 UTC/local occurrences with offsets, timezone/DST warnings; no persistence or model calls.
- `POST /`: create with idempotency key and effective owner.
- `GET /`: cursor-paginated list, owner-scoped; optional bot/lifecycle/search filters, capped limit; includes next_run_at, last-run summary and counts without N+1.
- `GET /{id}`: complete editable definition, revision and summary.
- `PATCH /{id}`: partial update with required expected revision; validate merged definition and reject conflicting nonterminal work.
- `POST /{id}/pause`, `/resume`: idempotent lifecycle controls.
- `DELETE /{id}`: soft cancel, preserving runs.
- `POST /{id}/run-now`: returns occurrence receipt, not “completed”; required idempotency key.
- `GET /{id}/runs`: cursor-paginated immutable snapshot, timestamps, status, delivery/turn/message/session correlation, error/skip reason and actual model.
- `POST /{id}/runs/{run_id}/cancel`: only before transport acceptance; terminal cancellation idempotent; 409 if already running/accepted.

404 for missing or wrong-owner rows; 409 for stale revision or disallowed lifecycle/concurrency action; 422 with field-level errors for invalid cron/timezone/model/bot/range. Scope every nested action, including runs and preview capability checks. Internal API is LAN-trusted like existing routes; do not claim caller-supplied user_id is authentication. At the BawtHub boundary use the authenticated entity where present, otherwise the established selected-user dev behavior; a query/body override must not beat authenticated identity. Extend the existing chat proxy surface rather than inventing a new Traefik-owned /api prefix. UI query keys include user and filters; clear/replace caches on user switch.

## 7. UI (TASK-169)

Primary deep-linkable page: `/tools/schedules`, discoverable from the Tools launcher with a color-emoji tile. Reuse existing page shell, ModelSelect, bot catalog, slate inputs and floating surfaces. Do not add a top-level nav destination unless mirrored in the launcher per design rules.

- List/table on desktop, stacked cards at 375px: name, description, bot, model/default badge, human timing, timezone, next run (relative + exact), last result, active/paused state, context mode.
- Filters: bot, active/paused/completed/cancelled, search; obvious Create schedule button. Loading, empty, filtered-empty and unavailable/error states; retain current rows during background refresh.
- Editor sections: **What** (name/description/prompt/bot/model), **When** (Once / Every interval / Calendar & cron; IANA timezone; next-five preview), **Context** (clear-context checkbox plus exact explanation), **Advanced** (bounds, missed policy/deadline, memory).
- Disable save while preview is stale/invalid/loading, preserve draft on failure; abort stale preview requests. One-time local values must be resolved in the selected timezone, not accidentally via browser-local Date parsing. Server returns DST ambiguity/nonexistence errors with actionable choices.
- Details/run-history pane: immutable prompt/model snapshot, scheduled vs actual start, duration, status/error/skip reason, transport receipt, link to transcript/turn. Distinguish **Queued** from **Succeeded**. Never say a prompt was sent just because an API accepted it.
- Run now confirmation makes clear it is an extra execution. Cancel is separate from pause; show if already-running work continues. Duplicate opens a new unsaved paused-by-default draft without silently scheduling another billable job.
- Composer shortcut opens the same editor prefilled with draft prompt, current bot/model/user. Support both compact/mobile and desktop composer affordances. Original draft remains until server acceptance; recurring creation should not unexpectedly send an immediate chat message.
- Delivered-message badge uses trusted occurrence/turn correlation, not a prompt-text heuristic or invented message meta field. Existing messages table has authorship fields but no general scheduler meta column. Add a bounded history/session response enrichment keyed by delivery/trigger-message ID, and expose `{schedule_id, occurrence_id, scheduled_for}`. Preserve it in live SSE/history adapters. No per-message query loop.
- Subscribe to existing durable-delivery/turn events to invalidate matching owner-scoped schedule/run queries; add a small `prompt_schedule_changed` event for CRUD lifecycle changes. Re-fetch on reconnect/focus and use modest visible-page polling fallback. Do not create another SSE connection.
- Keyboard/focus and labels, 44px targets, no overflow at 375px; scope third-party cron editor styles. Check Unix editor presets against backend golden cases (Sunday, weekdays, day-of-month/day-of-week OR).

## 8. Implementation slices and verification

### TASK-166 — timing, persistence and occurrence reservation

1. Add pinned croniter dependency/lock; pure timing module + DST policy tests.
2. Add SEND_PROMPT enum and companion tables/indexes/constraints using existing schema conventions; idempotent upgrade of populated databases.
3. Implement preview/next/latest/anchor calculations, transactional due reservation, immutable snapshots and restartable outbox receipts.
4. Integrate lightweight prompt sweep with JobScheduler start/stop; protect maintenance behavior; no blocking inference in scheduler.
5. PostgreSQL integration tests for two concurrent claimers, duplicate keys, atomic advancement, downtime coalescing, active-run overlap, pause races and restart recovery. SQLite alone cannot verify advisory locks/FOR UPDATE.

### TASK-167 — validated scoped API

1. Pydantic tagged timing schemas and catalog/capability validation; ownership enforcement consistent with trusted LAN and authenticated UI boundary.
2. Preview, CRUD, revision conflicts, lifecycle controls, idempotent run-now and paged run history.
3. Extend existing BawtHub chat proxy for authenticated owner binding on this route family.
4. API tests covering wrong owner on every mutation/read, malformed/rate/bounds inputs, paused/ended schedules and merged PATCH validation. Verify OpenAPI and dev proxy route reachability.

### TASK-168 — durable delivery, session isolation and observability

1. Consume occurrence outbox via existing delivery enqueue with user authorship, stable idempotency key, explicit session/model and prefer_steer=false.
2. Use non-activating sessions; test clear-context on/off and SDK thread binding; never call active-thread reset.
3. Apply prompt-only deadline/pause/cancel gates at delivery claim; prevent overlap/backlog; preserve existing non-prompt callback semantics.
4. Reconcile terminal delivery/turn state into runs; crash/unknown handling; emit lifecycle invalidation and bounded history enrichment; run retention.
5. Fault injection at reserve/commit/enqueue/receipt/transport-acceptance/terminal boundaries; busy bot, retry, cancellation races, disabled model/bot, active personal chat unaffected. Existing delivery regression suite must pass.

### TASK-169 — management and composer UI + integrated acceptance

1. Add lazy-loaded Unix cron editor/descriptions and isolated theme adapter.
2. Owner-scoped API client/hooks, list/editor/details/actions/preview; use existing bot/model selection.
3. Tools launcher route and composer shortcut in both UI variants; avoid unrelated chat viewport changes already in flight.
4. SSE invalidation/history badge + reconnect fallback; deep links, errors, draft preservation and accessibility.
5. Client `tsc --noEmit`, server `pnpm run build:server` on echo, relevant UI tests, dev checks at 375px and desktop. End-to-end: one-time delivery; recurring interval twice; cron preview; pause/resume; manual-run cadence unchanged; edit/cancel conflicts; wrong user; refresh/restart recovery; context isolation and model override. Use only an authorized disposable test bot, never Mira. No production release/rebuild unless the operator asks.

Dependency order: 167 and 168 depend on 166; 169 depends on both 167 and 168. API and delivery slices can be developed independently after the common schema/contract is fixed.

## Implementation receipt — foundation slice (2026-09-15)

- Added `prompt_timing.py`: strict five-field Unix grammar, croniter-backed wall-clock iteration, explicit gap/fold policy, interval anchoring, inclusive start/exclusive end, bounded latest-slot recovery, preview and ambiguous local-time resolution.
- Added `prompt_schedule_models.py` and `prompt_schedule_store.py`: companion tables, constraints, unique occurrence and one-nonterminal-run indexes, transactional occurrence+JobRun reservation, immutable outbox snapshots and skip/coalescing behavior.
- Small representation refinement from the proposed schema above: timing is persisted together as a validated `PromptTiming` JSON object (`timing_json`), not duplicated into separate timing columns. Operational due pointers, lifecycle, owner and occurrence correlation remain typed/indexed columns. This avoids drift between API and storage timing validation; malformed DB rows must fail closed.
- Added `SEND_PROMPT` schema migration path, excluded prompts from generic maintenance dispatch and unscoped maintenance list/run/trigger routes. Added optional independent prompt sweep lifecycle, deliberately not wired in app startup until TASK-168 provides a delivery consumer.
- `croniter==6.2.4` pinned in pyproject. Local uv lock regenerated; this repository ignores `uv.lock`, so the reproducible tracked dependency is the exact pyproject pin.
- Hermetic tests use SQLite for persistence plus PostgreSQL DDL compilation; they do **not** prove PostgreSQL concurrent row-lock behavior. Live PostgreSQL concurrency/migration verification remains outstanding. No scheduler tables were migrated in the running service, no live prompt schedules created, no bot inference/restart/deploy/commit performed.
- API, actual outbox delivery/reconciliation, model/catalog validation, owner quota, lifecycle mutation controls, retention and management UI remain subsequent work. Foundation is not a claim that scheduled prompts now work end-to-end.

## Integrated implementation receipt (2026-09-15)

All four implementation slices are present in the shared working trees:
- Owner-scoped validated API, authoritative preview/local-time resolver, idempotent creation/run-now, revision-protected updates, pause/resume/cancellation and run history. UI proxy binds authenticated identity; internal API preserves the existing LAN trust model.
- Durable delivery consumer/startup wiring, immutable outbox recovery, prompt-only claim deadlines, non-steering execution, isolated non-active threads and canonical request-local model binding. Necessary history/seed instance isolation fixes prevent default-user/model or global-summary leakage.
- Management UI and Unix cron editor, safe Apply preset behavior (upstream widget otherwise rewrites advanced expressions), both composer entry points, scoped SSE refresh and trusted scheduled-message badges.
- Review fixes: generic delivery APIs cannot expose/cancel scheduled receipts; extraction opt-out persists atomic source provenance and summary extraction respects it; bot purge guards outstanding work and deletes companions child-first.
- Bounded hourly retention removes terminal runs older than 90 days, retaining each schedule's latest receipt and all unknown outcomes. Does not delete messages, sessions or transport receipts.

Final parent-run evidence: **157 backend tests passed** across timing/store/API/delivery/review fixes/retention/history/scheduler/streaming; **31 focused frontend/proxy/provenance tests passed**; echo client tsc and server build:server passed; focused Ruff and both diff checks passed. Subagent browser harness verified desktop/mobile and editor interactions against intercepted APIs, not live inference.

**Not live-complete:** read-only live `/v1/prompt-schedules?user=snark` returns 404 because app has not been reloaded with the new API/dependency. Dev container dependencies are separate from host pnpm installation; cron editor dependency availability requires the authorized dev dependency workflow. No production rebuild, service restart, bridge/Redis restart, live migration, schedule creation or real bot execution was performed. PostgreSQL advisory/row-lock concurrency and live SDK execution remain verification gaps, not claimed successes. All code is uncommitted. Production release requires the operator's explicit request.

## 9. Definition of done / current evidence

Done means a user creates, manages and observes scheduled prompts in BawtHub, actual turns arrive through normal chat, and restart/race/context/user-boundary tests pass. Saving a row, returning queued, syntax checking, or updating a task status is not end-to-end verification.

Current evidence for this refinement: repository seams inspected; old specs and TASK-169 cancellation history reviewed (no recorded superseding implementation); upstream library metadata/docs read; isolated croniter 6.2.4 timezone/DST/impossible-date probe executed. No feature implementation, dependency changes to either application, service restart, schedule creation, bot execution or production changes were performed by this design pass. Existing BawtHub chat viewport working-tree edits were left untouched.
