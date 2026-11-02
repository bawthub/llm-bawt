# Exact-commit batch gate: inert contracts and claim (TASK-1865)

## Implemented in this slice

`ops/gate_plan.py` defines a closed `llm-bawt.ops.gate-plan/v1` value:

```json
{"schema":"llm-bawt.ops.gate-plan/v1","repository":"llm-bawt","source_sha":"0123456789abcdef0123456789abcdef01234567","target":"test-all"}
```

Only `llm-bawt` is supported. BawtHub and other repositories fail closed;
BawtHub full offline parity and dispatch remain future work. The fixed
`test-all` workload corresponds to `scripts/test_suite.py offline`: unit,
private PostgreSQL, and local integration with failure-on-skip enforcement.
This is a workload identity, **not** an executable command or authoritative
host `make gate` receipt. There are no caller paths, argv, environment,
patches, target nodes, ref resolution or caller-selected attempt numbers.
The decoder requires every field, an exact lowercase 40-hex commit SHA,
and no additional fields. Constructing a plan neither verifies that commit
exists nor grants permission to execute it.

`ops/gate_receipt.py` provides an internal direct-callback serialization and
framing seam. No production caller attaches it yet. When a trusted internal
job stores this callback, the existing app-owned ops receipt outbox delivers
its terminal ledger result. No new scheduler, thread, table, worker branch,
Docker action, endpoint, tool argument or catalog row is introduced.

The existing sender validates the gate plan against the job's stored args,
checks the callback bot/user against the job caller, requires a session,
and refuses an approved job's direct gate callback. Approved jobs must use
their existing approval continuation instead, never both paths. A future
admission adapter must derive all destination fields from signed caller
context, not accept a destination argument. Stored callback data is not an
authorization capability.

Gate wake framing is registered as `ops.batch_gate_receipt`, resolved at
send time with a seed fallback. Existing quiesce framing is unchanged.
All five terminal ledger states can wake a new session-bound turn through
the existing durable delivery, using `ops-receipt:<job_id>` and enqueue then
stamp. A crash between these steps reuses the durable delivery after restart.
An active job produces no callback. The callback is a **wake**, not runner
retirement, a verified gate pass, completed recipient execution, or SHIP
permission. It does not retry, publish or push anything. The existing outbox
must be running to enqueue; the durable delivery dispatcher must be available
to execute the recipient's new turn.

## Dormant atomic authorization/claim

`ops_operations.execution_tag` is a nullable, DB-owned exact tag slug. It is
included in operation API serialization, recorded revisions and snapshot
content hashes. The normal OpsStore additive-column migration adds it as NULL;
it does not populate/enable any grant, seed an operation, change approval
policy or reinterpret legacy nongate execution. No live migration or source
activation is part of this change. Deployment must use the existing explicit
source/service activation process; an import alone never starts a gate.

`ops/gate_claim.py` exposes an **internal, unused** snapshot builder and
`OpsStore.claim_gate_job(job_id, turn_token=...)`. No route, tool, reconciler or
production admission path calls it. Tests create synthetic ledger rows only.
The generic claim and dispatcher refuse the gate snapshot marker, so a gate
cannot accidentally execute the placeholder operation's Docker command.
The gate-only snapshot has no executable spec and is not a regular ops
approval snapshot. It is content integrity data, not an approval capability.

A future trusted adapter must derive the opaque turn token from authenticated
request context, never tool/body identity/tag arguments. The existing signed
turn-capability verifier binds the stored caller bot, user, turn and snapshot
session; the token itself is never persisted. There is no `required_tag`
argument. This internal synchronous seam currently requires a valid original
turn token at claim time (including its existing TTL). A future durable
admission adapter must preserve authenticated provenance without treating a
raw body or integrity hash as proof; that integration is not supplied here.

The PostgreSQL claim transaction locks the current operation and job, uses
`bot_tags.catalog.lock_bot`, then locks the caller profile, exact tag catalog
row and assignment. Unknown/NULL/missing tags, missing profiles and private or
inactive callers deny. It revalidates the strict stored GatePlan, exact
operation revision/script hash, whole operation including execution_tag and
snapshot content hash. A changed operation committed before claim denies,
including a raw grant edit without a version bump. Existing tag revocation
waits on the same advisory lock; assignment/profile row locks also fence their
deletion. Revocation or operation edits waiting behind a winning transaction
are prospective: they do not cancel the claimed invocation. A denial writes
nothing, a winning QUEUED-to-DISPATCHING CAS appends only the existing ops
transition, and rollback leaves no claim or transition.

An additional transaction-scoped repo+SHA lock serializes even different
operation rows. This seam is deliberately **first-attempt-only**: any prior
nonqueued or dispatched job with that exact plan prevents another claim,
regardless of terminal state. There is no bypass/attempt argument and no
retirement implementation. LOST/timed_out without a dispatch timestamp also
deny a second job. Legitimate retries remain unavailable until positive
retirement and active-attempt/publication fencing are implemented. Existing
operation concurrency caps are respected, but no runner or process-worker
slot is allocated because nothing is dispatched.

Focused disposable-PG tests observe real lock waits and commit both orderings
of tag revocation and operation grant edits, competing same-job claims and
cross-operation/cross-caller repo+SHA claims, plus snapshot/caller/plan drift,
all terminal-state refusals, idempotency, rollback and no dispatch/callback.
No live bots, model calls or production database are involved.

## Indexed prior-attempt history

The first-attempt fence now uses a single indexed existence query with `LIMIT 1`,
not a Python scan/JSON decode of every nonqueued ops job. `ops_jobs.gate_plan_key`
is an internal, bounded identity derived from a strictly decoded GatePlan by
`OpsStore.create_job`; callers cannot supply it. Nongate/invalid args get NULL.
The claim verifies its own key against the strict stored plan. It still acquires
the repo+SHA advisory lock **before** querying history, preserving the existing
cross-operation race ordering and first-attempt-only semantics.

A partial index contains only non-NULL keys whose row is nonqueued or has a
dispatch timestamp. The predicate deliberately retains never-dispatched queued
cancellations as permanent blockers; LOST/timeouts remain blockers too. No retry,
retirement, admission, callback or executor path is enabled by this optimization.

The ordinary additive OpsStore migration adds the nullable column and derives
keys for **all** preexisting rows, including queued rows, in 128-row keyset pages
within the schema transaction. Strict decoding handles old JSON order, whitespace
and escaped strings exactly as the old scan did. Malformed/ordinary args stay
NULL, and args/snapshots/state/events are not rewritten. Backfill runs only when
the column is added, not on every startup or claim. On PostgreSQL the ADD COLUMN
table lock lasts through backfill/index creation; failure rolls back together.
Activation therefore includes a one-time blocking schema transaction and must be
sequenced by the service owner; this source milestone performs **no live migration**.
A mixed old/new gate writer is unsupported: an old writer lacks this derived key;
gates remain dormant until the whole trusted path is deployed. Normal nongate
writers and dispatch behavior are unchanged.

Disposable-PG regressions retain the existing real claim/revocation lock races,
prove old terminal rows still deny after migration, and inspect `EXPLAIN ANALYZE`
on the actual lookup with 3,000 unrelated poisoned rows and 1,000 same-plan queued
rows: the partial attempt index is used without a ledger sequential scan. Migration
interruption/rollback and key drift are covered separately. This is bounded-query
evidence, not runner execution, SHIP proof or an operational throughput benchmark.

## Dormant immutable request admission

`ops/gate_admission.py:create_gate_job` is a create-only **internal, unused**
adapter over the existing ops ledger. Trusted code must select the operation
slug from configuration and obtain the opaque turn capability from authenticated
request context; neither is a new tool/body identity or grant argument. The
closed GatePlan still refuses BawtHub, arbitrary commands, paths and attempts.

The canonical `llm-bawt.ops.gate-request/v1` payload binds the operation and plan
to the capability's bot, user, turn, session and trigger message. The capability
and issue time are never persisted. A valid reissued capability for that same
turn can read the existing request. Another identity/workload with the same
bounded idempotency key conflicts before returning a row. Identical replay is a
read, even after a catalog edit or terminal result; it never rebuilds the stored
snapshot or launches/retries anything. Concurrent creation uses the existing
unique ledger key and snapshot check, not a second admission table or scheduler.

New requests require an enabled, nondeleted, nonquiescing operation with a
configured DB-owned grant. Creation does **not** check/authorize execution: the
separate gate-only claim still rechecks the current catalog, caller restrictions
and tag assignment atomically. Revocation after creation still denies claim.
Persisted provenance is not a durable authorization token; the existing original
capability TTL still applies to a separate claim. No production caller, claim,
callback attachment, runner launch or host proof publication is enabled here.
Generic dispatch still refuses these queued snapshots. Focused disposable-PG
coverage proves concurrent same-key replay/conflict, generic exclusion and
revocation between creation and claim, without a runner or live identities.

## Independent exact-commit source staging

`ops/gate_source.py:staged_gate_source` is an unused context-managed staging
helper for the future fixed runner. Only trusted code supplies the repository
root and an owned, non-group/other-writable staging parent outside that root.
It creates a new private directory, never a linked worktree or shared index.
Dirty/staged/untracked source files and canonical Git config, hooks, credentials,
remotes, alternates and parent history are not copied. Arbitrary **committed**
secrets cannot be scrubbed while preserving exact source identity.

Fixed system Git plumbing reads and hash-verifies the requested original commit,
its complete tree closure and raw blobs under a default-deny environment, with
replacement objects, lazy fetch, global/system config and hooks disabled. No
repository code, checkout filter or template executes. Direct blob writes avoid
`git archive`'s `export-ignore`/`export-subst` behavior and checkout EOL conversion.
Unsupported symlinks, gitlinks, special files, non-UTF-8/unsafe/duplicate paths
refuse before yielding. Explicit bounds refuse rather than truncate: 100,000
files/tree entries, 512MiB file bytes, 64MiB per object and total tree metadata,
64 tree levels and 4,095-byte relative paths.

The fresh `.git` contains only newly written original commit/tree/blob objects,
fixed minimal config, detached HEAD, a shallow boundary and a private index.
Before yielding, the helper independently checks every exported path, raw blob
hash and executable mode, exact file/directory inventory and private HEAD/tree.
The source repository's files, index, refs and metadata remain unchanged. The
context cleans its private stage on refusal, consumer failure and normal exit.

The staging owner must keep the directory frozen while it is validated and
handed to the sandbox; adversarial concurrent staging writers are unsupported.
A future launcher must arrange unprivileged access and private scratch/mounts;
this helper alone neither runs nor isolates repository code. Its result is an
**initial source identity**, not an execution grant, toolchain/image attestation,
trusted gate result or host receipt. No production path calls it or dispatches a
runner. Focused scratch-Git tests cover metadata isolation, exact older commits,
dirty exclusion, hostile filters/environment/replacements, link/path/size
refusals, post-export tampering and partial-stage cleanup.

## Unused fixed socketless launcher

`ops/gate_runner.py:run_gate` is an **internal, unused** launcher over the
existing worker's `DockerAPI.request`/`attach_stream`, never its generic
`execute`. No route, catalog entry, dispatcher, worker branch or production
caller invokes it. Source-only publication does not authorize even a scratch
Docker run. Its focused tests use fake Docker; actual image compatibility and
end-to-end execution remain unproven.

The trusted caller must provide the still-owned/frozen `GateSource`, a ledger
job ID, private scratch parent, and **separately trusted, DB-frozen** SHA256 of a
bounded environment-evidence JSON document. There is no environment fallback,
image tag, pull, build or caller-selected workload. The closed
`llm-bawt.ops.gate-environment/v1` evidence has exactly these string fields:

- `schema`, `repository`, `source_sha`, `tree_sha`, `workload`, `platform` bind
  the exact staged commit/tree to `llm-bawt`, `test-all/offline`, `linux/amd64`.
- `runner_image` / `postgres_image` are immutable repository digest references;
  `runner_image_id` / `postgres_image_id` are immutable local image config IDs.
  PG must match the committed private-PG fixture's fixed image/protocol.
- `toolchain_sha256`, `dependency_closure_sha256`, `asset_closure_sha256` identify
  trusted packaging reports for the image's actual tool binaries, complete
  offline Python/build dependency closure and runtime asset closure for this
  source and fixed workload. These are **not** declared version comparisons or
  the finite asset manifest. Reports must cover `/opt/gate/uv-cache`, tokenizer
  data at `/opt/gate/tiktoken`, `/etc/mime.types`, shell/Git/Node/uv/Python and
  the workload's build tools, with no runtime downloads or host caches.

The evidence producer, report verification/approval and live configuration are
**not implemented**. The dormant trusted-snapshot adapter below freezes a selected
DB configuration at admission, but no production dispatch consumes it. Checking an
independently approved digest binds trusted packaging claims; it does not prove
those claims, authenticate arbitrary submitted JSON, or authorize execution.
No configured evidence or approved runnable image is supplied by this slice.
The existing finite manifest cannot pass this schema and is not upgraded into
an attestation. Missing/incompatible/unpinned evidence refuses before Docker.
Local-only image inspection verifies both IDs/digests and platform, rejecting
unexpected image environment defaults and inherited writable volumes.

The source handoff changes only private directory/Git-metadata readability;
its host parent stays 0700 and tracked bytes/modes stay unchanged. The runner
gets just RO binds for source, fixed PG binding and the per-job Unix-socket
directory. UID 65532 copies source/private Git metadata and the image's uv cache
into bounded private tmpfs, then runs exactly `uv run --offline --isolated
--frozen --extra dev --extra service --extra mcp --extra search --extra memory
python -m scripts.test_suite offline`. Fixed worker/budget settings match the
preserved CI values (2 workers; 3s unit / 7s real-process), not inherited host
environment. The PG sidecar runs as UID 999 with private disposable tmpfs data
and only that job's RW Unix-socket directory. Both have network-none, RO root,
no capabilities, no-new-privileges, memory/CPU/PID bounds, no stdin/restart and
no daemon logs. No Docker socket, credentials, shared source/cache, authoritative
receipt root, network, host port, dependency installer or repository code reaches
the socket-owning process. "Socketless" excludes control sockets, not the private
PG socket or the runner's local integration loopback sockets.

Lifetime is bounded by a process-local 1800s OS deadline, with separate 60s
cleanup; the helper refuses a non-main thread or a preexisting deadline rather
than silently replacing another owner's timer. Future worker integration must
explicitly arrange this custody. Runner output is attached before start, capped
at 1MiB without truncation, returned only as **untrusted bytes**, and never echoed.
A zero exit/output alone is not a verified gate receipt or SHIP permission.

Names are deterministic per job/role; a unique per-invocation label establishes
cleanup custody, not retry authority. Existing containers/scratch are never
adopted or removed. A lost create/start response is uncertain, never retried.
Cleanup removes only matching owned identities and verifies ID/name absence;
unknown/failed removal refuses success and retains private scratch for later
investigation. Even positive immediate cleanup is **not** the missing durable
retirement/publication fence. Process death can still leave labeled containers;
reconciler adoption/recovery and active-attempt host publication remain future
integration work. No new scheduler, ledger, retry or callback path is created.

## Dormant configured environment and durable caller provenance

`ops/gate_trust.py:create_trusted_gate_job` is an additional **unused** admission
adapter. It accepts the same fixed plan, trusted operation slug, authenticated
turn capability and bounded idempotency key, but no environment/digest, caller,
tag, destination, path, image or command argument. It reads exactly one existing
`runtime_settings` global (`scope_id='*'`) value at the internally derived key
`ops_gate_environment:llm-bawt:<exact source SHA>`. There is no settings cache,
bot/request override, environment fallback, bootstrap, default or auto-seed.
The config envelope is closed and bounded to 16KiB:

```json
{"schema":"llm-bawt.ops.gate-environment-config/v1","evidence_json":"<exact UTF-8 JSON bytes described above>","evidence_sha256":"<independently approved raw-byte SHA256>"}
```

Only trusted configuration writers may establish this value after actual
packaging-report review. This code creates no configuration or evidence producer;
a digest stored beside evidence is a DB trust assertion, not independent proof
of packaging claims. Missing, duplicate, oversized, malformed, source-incompatible
or unpinned values refuse. Admission uses the same pure environment decoder as
the launcher, which still additionally checks the independently staged tree.
Before the existing `create_job` inserts, its transaction fence rereads the
selected config with a shared row lock. A committed change refuses rather than
silently rebinding; a waiting edit cannot change the captured evidence before
insert commits. A later edit is prospective: **future claim integration must
revalidate live configuration together with current grant/catalog state**.

The existing invocation snapshot stores the exact evidence bytes/digest and
full immutable request, including all signed caller/session/trigger identifiers.
A deterministic HMAC-SHA256 covers the entire snapshot, operation, request,
idempotency key and evidence under the existing persistent server root with
versioned domain `llm-bawt.ops.gate-request-provenance/v1`. It stores neither the
expiring turn capability nor its issue time. Same-turn token reissue produces
the same snapshot, so concurrent insertion uses the existing ledger unique key
and payload/snapshot conflict handling. Identical replay verifies the stored
proof and is read-only, even after configuration removal or terminalization;
legacy unsigned records are never upgraded or repaired in place.

`read_trusted_gate_request` verifies that HMAC, snapshot hash, closed workload
and immutable ledger columns and returns the frozen evidence/provenance without
reopening the original expiring capability. That establishes historical
**authenticity/integrity only**, never permission to execute. Ordinary hashes
cannot replace the MAC; key rotation/unavailability and drift refuse. Derived
key material stays server-side and must not enter the socketless runner.
There is no new table, column, migration, scheduler or production caller. Both
generic dispatch and the existing strict gate-only claim refuse these enriched
snapshots; the old token-only admission/claim path remains unchanged and dormant.
No callback, retry, container, receipt publication or SHIP path is enabled.

Focused tests are authored for frozen/replayed/expired-token provenance,
configuration refusal/drift, key failure/rotation, ordinary-rehash tampering,
ledger binding, deterministic concurrent inserts and an observed PostgreSQL
config-edit lock wait. They require the separately scheduled supported test
slot; authored tests alone are not a passing/runtime receipt.

## Dormant trusted-request claim with fresh authority

`ops/gate_trusted_claim.py:claim_trusted_gate_job` is an additional unused,
PostgreSQL-only internal adapter. It accepts only a stored job ID and store,
not a caller, token, tag, plan, environment or command override. The stored
server MAC authenticates its original caller without reopening an expired turn
capability; that identity is then checked against **current** locked DB authority.
No route, store entry point, scheduler, executor or runner uses this adapter.

The legacy claim shares its unchanged operation/job locks, current-grant check
and first-attempt transition through small extracted private helpers. Both
ordinary and legacy token-only claims still refuse enriched snapshots. The new
adapter checks the entire authenticated original operation/caller/plan/limits
against the freshly locked operation using canonical equality, then locks the
current profile, grant catalog and assignment and refuses private/inactive bots.
It rereads the one exact-source global config under `FOR SHARE`, requires its
exact evidence bytes/digest to equal the frozen request and retains those locks
through the existing QUEUED-to-DISPATCHING transaction/event. Lock order is
operation -> job -> bot advisory -> profile -> tag catalog -> assignment ->
config -> repo+SHA advisory. Committed prior changes deny; later writers wait
and are prospective. There is no pre-lock authority/config cache.

The same indexed history query and operation slot cap remain authoritative,
including across the legacy and trusted paths. Any prior nonqueued attempt,
even terminal without a dispatch timestamp, still blocks a new key; this slice
has no retirement proof or attempt-two bypass. Rollback leaves neither the
transition nor its event. The claim itself launches nothing and acquires no
worker slot or callback: missing runtime qualification, dispatch/recovery,
retirement and trusted receipt publication remain explicit future work.

New regressions cover fresh config/catalog/grant denials, durable authentication,
original-token expiry, observed before/after PostgreSQL writer lock waits,
rollback, concurrent claims and shared first-attempt/slot fences. They must run
under an actual supported verification slot; authored tests alone prove nothing.
The SQL suite manifest also registers the previous trust-admission tests and
these new claim tests for the integrator's offline gate (no gate rerun here).

## Dormant one-shot dispatch custody and read-only recovery observation

`ops/gate_custody.py` adds an unused identity-reservation context and recovery
observer, not an executor or production dispatcher. It requires an authenticated
already-DISPATCHING request from `claim_trusted_gate_job`; generic `claim_job`
is not interchangeable and already populates incompatible host metadata. No
current service, reconciler, worker or route calls the new module.

`reserve_gate_launch` takes only internal store/job/private receipt-root inputs.
It authenticates before creating local state, takes the existing per-job
`submission.lock` nonblockingly, then locks and rechecks the fresh ledger row.
The existing `host_unit_name` stores `gate-custody:<intent SHA256>` (a custody
identity, not acceptance of a running container), and `status_file_path` points
to `gate-intent.json` (not a terminal receipt). No table/schema is added and no
lifecycle state/event changes. The immutable intent binds job ID, exact snapshot
and request digests, claim timestamp, fixed workload, environment digest, both
deterministic role names and a random owner nonce. It is published via exclusive
temp/hard-link/fsync before transaction commit. Only successful commit yields
the identity, with the submission lock held until the consumer scope ends.

A committed DB digest marker refuses a second reservation even after journal
loss. A published orphan after rollback or interrupted commit also refuses;
it is never silently adopted, repaired or removed. Preexisting ordinary worker
files, partial intent/temp files and hostile links/special files refuse. Private
receipt directories must be owned 0700, regular single-link files owned 0600;
the trusted root is never a runner mount. Concurrent adversarial writers with
that trusted identity/storage custody are outside the same frozen-storage
assumption as source staging. No expiring capability, signing key, caller data
or raw environment evidence is written into the intent.

The fixed launcher accepts an optional internal `owner_nonce` for its existing
ownership labels; old inert helper uses still generate a nonce. That argument is
an identity label only, **not** execution permission, a retry token or proof the
caller owns a live reservation. A local flock cannot fence a process that copies
the nonce then outlives its context. Future production integration must arrange
worker capacity, supported pinned runtime, execution/terminal fencing and process
custody before consuming it. This module does not wire a launcher call or enable
even a scratch container. Generic service recovery is still incompatible with
gate snapshots and must not be used as gate recovery.

`observe_gate_launch` opens only existing storage under the same nonblocking
lock. It authenticates the stored request and exact DB-bound intent before two
GET-only Docker inspections. Both role names, immutable ID shape and complete
job/source/environment/role/nonce labels must match. It reports only busy or
per-role missing/created/running/exited observations. Missing/invalid storage,
ledger/intent drift, a foreign owner or any uncertain daemon response refuses.
It never starts, adopts, removes, retries, writes metadata or imports a result.
Even two 404s or an exited runner with exit zero are **not** positive durable
retirement, a terminal receipt, gate pass, host proof or SHIP permission.

Authored regressions cover concurrent reservation, interrupted consumers and
dead lock holders, rollback-orphan and commit-acknowledgment uncertainty,
nonterminal/terminal journal loss, fresh-row denial, exact nonce/intent binding,
unsafe storage, GET-only recovery and no lifecycle writes. Focused execution
requires Gavel's actual supported test custody; authored tests alone are not a
passing receipt. Runtime packaging qualification, production dispatch/worker
capacity, durable retirement/attempt-two fencing, receipt/callback attachment
and active-attempt trusted host publication remain unfinished. No packaging
infrastructure, scheduler, policy, config, catalog or activation change is added.

## Dormant committed admission capacity and pre-start ledger guard

`ops/gate_start.py` is a separate unused context after `reserve_gate_launch`'s
consumer scope ends. It does not call the runner, create a container, dispatch,
mark accepted/running/terminal, attach a callback, or publish a gate result.
It uses existing nullable ledger fields; no schema/catalog/config change is
introduced and generic claims remain unchanged.

Under the same nonblocking per-job submission flock, phase one locks current
operation -> shared worker-slot advisory lock -> fresh job row. It authenticates
the existing trusted claim and exact nonce-bound custody intent, refuses closed
or previously reserved jobs, unavailable operations and process-spawn holds,
and conservatively shares `live_worker_slots`/`worker_concurrency` with ordinary
process jobs. The gate is **not** mislabeled as a process executor to change
`needs_worker_slot` semantics. The strictest admitted/current/holder operation
cap still applies. `worker_slot_at` records capacity; `log_file_path` points to
private `gate-start.json`, not runner output or an ordinary worker log.

The start marker binds the existing custody digest, job, owner nonce, claim
and slot timestamps. Exclusive owned-0600 creation and file/directory fsync
precede the capacity transaction's commit. No confirmed commit means no start
boundary is yielded. A partial/orphan marker, committed ledger reservation,
occupied journal, interrupted consumer, missing storage or uncertain commit
never authorizes another reservation, even after terminalization. No repair,
cleanup, automatic slot release or retry path is added.

Phase two re-locks **only** the fresh job and compares the exact retained intent,
nonce, timestamp and start marker before exposing the short consumer boundary.
A preceding terminal transition refuses. A concurrent terminal writer waits
only while that PostgreSQL transaction/session survives; phase two acquires no
operation/slot lock after the job lock. The submission flock stays held through
both phases and the consumer scope. Authorization remains the earlier trusted
claim's linearization, with prospective revocation; this context is not another
authority grant or production dispatch entry point.

**This is admission reservation plus a bounded ledger guard, not active-lifetime
capacity or a complete launch permit.** If the DB session dies during the
consumer scope, its row lock disappears while caller code may continue. A
copied nonce/boundary cannot be revoked by the local flock. A `lost`/`timed_out`
row also stops counting in the existing shared budget even if a future runner
survives. Durable one-shot evidence prevents reissuing this identity but does
not retire that runner or prevent another job from taking the freed slot.
Positive retirement/lifetime and lost-session fencing are prerequisites to
production launch integration. Keep the workload outside this short DB scope;
no claim of unconditional terminal exclusion, retirement or safe live dispatch
is made here.

Authored targeted regressions cover ordinary-process/shared-budget interaction,
observed operation/slot/job lock waits, terminal-before/after-boundary ordering,
spawn holds, operation caps, commit uncertainty, rollback orphans, interrupted
consumers, storage loss and changed marker/nonce/ledger evidence. Functional
execution still requires the supported test target under actual host custody;
source review and authored tests alone are not passing execution receipts.
The earlier custody and claim receipts remain independent and reusable.

## Dormant positive retirement of reserved gate capacity

`ops_jobs.gate_retired_at` (nullable, additive) closes the freed-slot gap
above for the shared and per-operation budgets. A gate row with a gate-custody
marker and `worker_slot_at` counts in `live_worker_slots` and `slot_holders` in
**any** state until that stamp exists; active rows count once, and ordinary
terminal non-gate rows stay excluded. No such rows exist while dispatch is
unwired, so current admission is unchanged.

`gate_retire.retire_gate_slot` (no production caller) holds the submission
flock, requires a terminal row with a durable start reservation, then lists
**all** containers carrying the exact job+nonce labels (GET-only, `all=1`;
labels are immutable, so renames cannot hide one) and re-inspects each by
immutable ID. Every match must be `exited`, not running/restarting, with
explicit `RestartPolicy.Name == "no"`; zero matches is positive absence. It
then CAS-stamps the row once under a fresh row lock. Any other state, missing
field, 404 after listing, daemon error or changed ledger refuses with a fixed
code and the slot stays held; an unreadable intent therefore keeps the slot
until an operator acts. No removal, retry, result, callback or launch.

Limits: this is not a DB-loss or copied-identity fence. A late creator that
outlived its session with a copied nonce, or a manual restart of a retained
exited container, is not prevented. Retirement is not permission for a second
attempt or SHIP.

## Before any trusted dispatch wiring

The recorded security design r2 approval is
`cmvwjnu39003e0mmkdlkmdor8`, rechecking r1
`cmvwjipsd003a0mmke26d63tv`. Its two concrete findings were resolved at the
**design** level; the dormant claim now tests the first boundary below, not
an executable gate pipeline:

* Authorization linearizes at the PostgreSQL QUEUED-to-DISPATCHING claim.
  Lock the current operation, bot profile and tag assignment; compare the
  current DB-owned grant and catalog revision in that same transaction.
  Revocation/catalog edits committed before the claim deny admission.
  After the claim the invocation is authorized in flight; revocation is
  prospective, not a promise to cancel an already-running sandbox.
* LOST/timed_out/missing receipts never alone admit another attempt. Before
  retry, trusted code must inspect deterministic runner/sidecar identities,
  positively establish exit/removal, check late receipts, and record a durable
  retirement fence under the repo+SHA admission lock. Unknown Docker/identity
  state refuses. Active-attempt/job identity fences trusted publication so
  an old runner cannot overwrite newer proof.

No production path uses the claim yet, and retry retirement/publication
fences remain unimplemented. The public Docker spec validator still rejects
`batch_gate`; a synthetic internally-created ledger fixture is not an enabled
gate executor. Refused stored receipt callbacks must also be durably parked
when actual admission/callback wiring lands; the current inert receipt seam's
poison-row retries are not presented as a completed admission design.

The future adapter must export only committed bytes with matching private
Git HEAD; pin and verify source/toolchain/image/assets; run repository code
only in the socketless, unprivileged sandbox with a per-job private sidecar;
keep host credentials, Docker socket and authoritative host receipt roots
out of that sandbox; and publish only through trusted job/attempt-bound code.
See [finite asset inventory](OFFLINE_RUNNER_ASSETS.md) for its owned/frozen
staging assumption and explicit incomplete-offline/toolchain non-attestation.
The prior 11,193-test sandbox result is feasibility evidence, not SHIP proof.

Production source activation, catalog enable/create, worker build/repin,
policy changes and actual dispatch are outside this slice. Publishing these
inert contracts does not depend on TASK-1838 activation.
