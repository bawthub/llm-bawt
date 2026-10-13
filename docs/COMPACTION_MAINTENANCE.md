# Explicit compaction maintenance isolation (TASK-1072)

Implementation companion to [COMPACTION_POLICY.md](COMPACTION_POLICY.md).
This work is separate from the reviewed TASK-1071 default-summary policy.
Implementation/review only: no commit, push, restart, deployment, or live-session
compaction was performed. This is not an activation or universal acceptance receipt.

## Supported entry points

- A text `/v1/chat/steer` submission against an explicit `/compact` turn is
  retained in the existing durable delivery outbox, not injected into the SDK.
  Both `now` and `next` user priorities become subsequent ordinary work.
- An actual `/compact` submitted through text steering against an ordinary turn
  is also queued as its own turn. Bridge send preflight therefore applies the
  TASK-1071 default to a bare command; custom instructions bypass that default.
- Unaccepted durable bot deliveries wait behind an explicit compact turn's
  publication barrier. Their original payload and stable recovery identities
  remain owned by the existing outbox, not a second queue implementation.
- The bridge rejects SDK injections into an explicit maintenance run, including
  the race where the app has already selected a steer. Definitive unaccepted
  maintenance errors allow existing fallback recovery. Accepted or ambiguous
  steers retain the SAME deterministic RPC; they must never become new turns.
- Classification is an actual leading `/compact` token, including whitespace and
  multiline suffixes, not ordinary prose mentioning the command.

## Acceptance, ordering, and recovery

The user route validates ownership through its existing checks and retains the
original user-message UUID, user author, source session, and exact received text.
The durable acceptance boundary is the outbox insert. The idempotency key combines
source turn and user-message UUID; retrying after source completion reattaches to
the same record rather than returning a stale-turn response that invites a fresh
send. The first stored payload wins even if a retry changes its text.

Eager history persistence uses that same UUID. A transient history error does not
undo acceptance: the dispatcher still wakes and normal dispatch repairs history.
A lifecycle notification exception is logged, not surfaced as a rejected durable
submission. The UI queued acknowledgment removes the optimistic interrupt anchor
while retaining its message identity; successor output uses normal turn events.
A delayed acknowledgment cannot regress an already-started/completed lifecycle.

Each unaccepted delivery captures a maintenance source in metadata at enqueue.
This read-only snapshot deliberately takes **no target advisory lock**: scheduled
prompt producers can already hold a job lock, whereas dispatch orders locks
`target -> job -> delivery`. Adding a target lock in enqueue would invert that
order. Claim rechecks the boundary under the existing target lock. Mutations to
an existing barrier also use that lock.

A terminal `turn_logs` row is insufficient: the app can wake the dispatcher before
publishing its ordinary `turn_complete`. Before claiming a successor, the
dispatcher publishes a canonical terminal snapshot and requires the Redis XADD
acknowledgment. Failure/None leaves work queued. Release records the matching
source ID; a new compaction discovered before claim establishes a new fence.

A crash after XADD but before the DB receipt can repeat the **same terminal
snapshot**, including turn/assistant-message IDs; it does not replay the work.
Different FIFO heads may also publish the same source snapshot. Existing frontend
assistant-ID deduplication is relied upon. This is at-least-once notification,
not a claim of exactly-once event delivery. Work execution inherits the existing
outbox's bounded recovery contract and deterministic bridge request IDs.

No-progress and failed-publication drains do not immediately wake themselves.
Real wake signals or the existing recovery sweep retry them, avoiding a hot loop.

## Verification (2026-10-13)

Expanded final backend run: **301 passed, 12 deselected**, two upstream WebSocket
deprecation warnings. Included maintenance queue/dispatch tests, dispatcher
supervision, scheduled prompts, policy/MCP tests, installed-CLI fake-provider
probes, ordinary steering/escalation/timeline, transport, send modes, sync
visibility, proxy context-window environment, thread binding, and reset-lineage
tests. Eleven integration tests and one explicit real-provider measurement test
were deselected. This supersedes the earlier 240-pass scoped run.

```sh
RUN_COMPACT_SDK_PROBE=1 \
COMPACT_PROBE_CLI=/app/.venv/lib/python3.12/site-packages/claude_agent_sdk/_bundled/claude \
.venv/bin/python -m pytest \
  tests/test_maintenance_queue.py tests/test_maintenance_dispatch.py \
  tests/test_inter_bot_dispatcher_supervision.py tests/test_prompt_delivery.py \
  tests/test_compact_policy.py tests/test_compact_delivery.py tests/test_compact_sdk.py \
  tests/test_claude_active_run_next_steer.py tests/test_user_steer_escalation_route.py \
  tests/test_steer_timeline.py tests/test_inter_bot_transport.py \
  tests/test_inter_bot_send_modes.py tests/test_inter_bot_sync_visibility.py \
  tests/test_agent_context.py tests/test_proxy_context_window_env.py \
  tests/test_thread_agent_keys.py tests/test_session_reset_lineage.py \
  -k 'not real_provider' -q
```

Tests exercise success/error/abort publication, unacknowledged publication retry,
crash before publication, between publish and receipt, and after receipt but before
claim (fresh queue object reads the durable receipt), FIFO head selection, new compaction races,
accepted/ambiguous recovery exemptions, original UUID/session/text preservation,
history failure, dispatcher publication-before-claim, lifecycle failure, and
progress-aware wake behavior. SQLite and stubbed collaborators are isolated;
these do **not** prove PostgreSQL advisory-lock concurrency or live end-to-end
exactly-once execution. An enqueue SQL-spy regression prevents reintroducing the
scheduler target-lock inversion; it is not a live deadlock test.

The archived-source test traverses real `rotate_delivery_session(continue)`,
dispatch request validation, and `_bind_agent_thread` with a synthetic archived
source row. It asserts the original durable thread and SDK resume key, original
frontend UUID, and no active-session lookup/creation. The enqueue SQL-spy also
asserts that original UUID in both stored row parameters and the dispatch payload.
Visible 409/422/503 rejection is tested before enqueue for unsupported backends,
invalid UUID/empty text, unavailable queue, and missing source session. These are
hermetic boundary tests, not live archival/concurrent message-upsert proof.

Model routing is untouched. There is no compact-specific override in this work;
real-provider TASK-1071 measurements selected Astra explicitly as probe input,
not as a feature default.

Frontend verification on echo: client `tsc --noEmit --pretty false` passed;
`maintenanceAck.test.ts`, `queuedSteers.test.ts`, and
`queuedSteers.dispatch.test.ts` passed **8 tests**. No browser/live compaction
activation test was run. Focused Python Ruff and `git diff --check` passed.

## Scope and unresolved verification

- Automatic CLI compaction is unchanged; this classifier marks explicit commands,
  not an automatic compaction occurring inside an ordinary inference turn.
- This adds durable routing to the existing **text steering** entry point and
  fences the **durable outbox**. It does not add a general acceptance queue to
  direct `/v1/chat/completions`, image/attachment sends, or the `/new` bypass.
  Existing UI image-steering rejection remains unchanged. Do not claim universal
  isolation/publication ordering for those other paths; reviewer scope assessment
  and further implementation are needed if they are included in acceptance.
- Existing cancellation/escalation and ordinary steering behavior are otherwise
  unchanged. New queued acknowledgments disable steer cancel/escalate capabilities.
- PostgreSQL lock/concurrency integration and deployed end-to-end behavior remain
  unverified. A read-only environment check on 2026-10-13 found PostgreSQL
  **client tools only** (host versions 17/18, no `postgres`/`initdb`/`pg_ctl`
  server binaries), no running isolated test PostgreSQL container, and no existing
  repository fixture for an isolated cluster. The only observed running database
  was the production `bawthub-postgres` container. Under the instruction not to
  mutate production schema/services or test on shared live work, the requested
  real-store concurrency test was **not run**. A separately provisioned isolated
  PostgreSQL instance is prerequisite to verifying enqueue/claim/advisory-lock
  races. SQLite and SQL-spy tests remain bounded evidence, not a substitute.
  No protected bot or live transcript was used.
- TASK-1071 meaningful summary-bloat/latency improvement remains **unproven**.
  No additional paid probe was run; its old temporary raw measurement artifact
  is unavailable. This queue work does not close that performance acceptance gap.
