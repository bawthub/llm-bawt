# Durable operations runner (TASK-861)

> **Architecture transition:** [Ops service: current and intended architecture](OPS_SERVICE_ARCHITECTURE.md)
> is the working design for TASK-1169–TASK-1173, including the intended FastAPI
> service boundary. Those tasks are not yet implemented or activated. This runner
> document describes the existing execution contracts; do not treat the proposed
> architecture as a live runbook. Keep both references reconciled during migration.

## Execution and recovery contract

Operations remain DB-configured Docker JSON action specs, not arbitrary shell.
There is no SSH, systemd, host script, app daemon-thread executor, or implicit
package installation. The LAN trust boundary is unchanged. The Docker socket is
privileged: network isolation/capability dropping do not make a socket-owning
worker safe for untrusted code. Only the operator-controlled worker image and
catalog are executable.

1. The service persists the full immutable invocation in `ops_jobs` (`queued`).
2. A conditional SQL claim reserves a per-operation concurrency slot and changes
   `queued -> dispatching` **before** any side-effecting Docker call.
3. It publishes/fsyncs an immutable `request.json` on a dedicated Docker volume,
   then creates `llm-bawt-ops-<full job UUID>` with an immutable image ID/digest.
   The worker has its own process/container lifetime, independent of app/bridges.
4. Docker accepting the worker means `accepted`, **not succeeded**. The worker
   persists `accepted`, owns the start delay, persists `running`, executes one
   Docker action, and atomically fsyncs `receipt.json` with the final result.
5. Status reads and the independent reconciler import receipts into canonical DB
   metadata. No success is inferred from submission, container existence, a
   stopped worker, or an app-owned timer.

A missing/abnormally exited worker without a terminal receipt is `lost`: the
side effect is unknown, and recovery never creates/replays a replacement.
A deterministic container still in `created` can be started if no prior start
attempt was persisted. Per-job volume locks and a durable start-attempt marker
serialize the inspect/start race; an uncertain prior start is not blindly retried.
A submission lock plus abandonment marker prevents a paused submitter dispatching
after recovery has declared its missing worker lost. A worker restarted manually
checks its receipt/started marker and will not repeat an action already begun.

The stdlib worker enforces a wall-clock execution deadline with a process alarm;
start delay is excluded. Docker `stop_grace_seconds` (spec field, default 10) is
separate from `timeout_seconds` (job execution deadline). **A timed-out Docker
request can continue inside the daemon**: `timed_out` does not mean the target
was untouched or rollback occurred. Neither timed-out nor lost jobs auto-retry.
An operator must inspect the actual target before deliberately requesting a new
invocation. Success means Docker completed its API action, not application
health/readiness. Pull downloads an image; it does not recreate its container.

Read-only prerequisite failures leave a queued job with an error. Fixing a
missing image/volume allows the same snapshotted request to proceed. Changing
the configured image cannot rewrite queued jobs: historical image IDs and volume
mounts must remain available for their lifetime. `max_concurrent` counts claimed,
accepted, and running jobs; the smallest limit among overlapping snapshots wins.

## Same-image routing repair (TASK-1179; activated on echo 2026-10-19)

See [OPS_ROUTING_REPAIR.md](OPS_ROUTING_REPAIR.md) for the disabled-by-default
`bawthub.reconcile-prod-routing` action, approval binding, routed verification,
and worker/bootstrap activation gates. Echo has the tested worker and enabled,
approval-gated operation. Approved job `389fd3f7eb2742a8944f82158b51154b`
completed the repair and both hostname-routed health endpoints returned HTTP 200.
A catalog row alone does not upgrade an immutable worker on other installations.

## Quiesce start fence: worker rollout (TASK-1233)

A quiesce job dispatches with `fence.start_not_after`, and the worker refuses a
later start (`worker.fence_expired`). The worker runs from its immutable image,
not the bind-mounted source, so the fence is enforced only by a worker image
built from a tree that has it. Such an image carries the label
`llm-bawt.ops.quiesce-fence="1"` (`docker/Dockerfile.ops-worker`).

The app fails closed without it. `DockerExecutor.check_quiesce_fence` inspects
the configured worker image, and a quiesce operation is refused with
`quiesce_unavailable` when preparing its approval, again when creating the job
(including from an approved snapshot), and in the driver's preflight before any
hold opens. Operations without quiesce are unaffected.

**Rollout step, before `quiesce_enabled` is set on any operation.** This needs
Nick's approval; it is not an ops catalog action:

1. Build `docker/Dockerfile.ops-worker` from the commit that has the fence, as
   in the setup checklist below, and record its immutable image ID or digest.
2. Confirm the label: `docker image inspect <id> --format
   '{{ index .Config.Labels "llm-bawt.ops.quiesce-fence" }}'` prints `1`.
3. Repin `LLM_BAWT_OPS_WORKER_IMAGE` to it and recreate app and ops-reconciler
   so both read it (a restart alone does not change the environment).

Jobs already created keep the worker image in their snapshot.

## Quiesce receipt callback and open deadline (TASK-1233)

A quiesce job whose scope covers its own caller tells that caller to end its
turn (`caller_instruction`), so the caller must get the terminal receipt as a
new turn. An approval-gated job already does, through its approval
continuation. A direct `ops_run` job records `receipt_callback_json`
(`{bot_id, user_id, session_id}`, from the caller's signed turn capability) at
creation, exactly when the scope covers the caller and the turn has a session.
Out-of-scope callers and approved jobs record none.

The app's receipt loop (`service/ops_receipt_outbox.py`, started in the app
lifespan, 2 s idle) runs `QuiesceReceiptSender` (`ops/receipt_callback.py`).
It runs in the app, not the ops reconciler, because the app owns the delivery
dispatcher: each new receipt row is announced at once (QUEUED published, the
dispatcher woken), and a duplicate only wakes it. For every terminal job with a callback and no
`receipt_callback_delivery_id` it enqueues one durable inter-bot delivery from
`system` to the caller (`message_kind=OPS_RECEIPT`, `on_gate="hold"`, a new
turn bound to the caller's session, never a steer), framed by the
prompt-registry key `ops.quiesce_receipt`, then stamps the delivery id. The
delivery's idempotency key is `ops-receipt:<job_id>`, so a crash before the
stamp re-reads the same delivery. Pre-dispatch failures are terminal too, so
the receipt also reports `quiesce_timeout`, `quiesce_lost` and a disabled
operation, and an operator cancel. Under a live hold the delivery waits at the
claim and is delivered after release.

*Rollout note.* Before TASK-1233 (f) the ops reconciler sent receipts itself.
While an old reconciler image still runs beside a new app, whichever process
enqueues first wins; the idempotency key keeps it to one delivery, but a
receipt enqueued by the old reconciler is never announced and shows no QUEUED
event (the dispatcher's recovery sweep still delivers it).

**Open deadline.** A job that has not opened its hold within
`quiesce_timeout_seconds` of `submitted_at` fails `quiesce_timeout` with no
action submitted. This bounds a preflight that keeps failing (its error is kept
in `error_text` meanwhile) and a concurrency slot that never frees. The drain
deadline after the hold opens is unchanged.

## Quiesce visibility and operator cancel (TASK-1233 (f))

**Job reads.** Every job read (`GET /v1/ops/jobs`, `GET /v1/ops/jobs/{id}`,
`ops_job_status`) carries, for a quiesce job, `quiesce.phase` =
`{name, since, drained, total, observed_at}` and `quiesce.members`.

- `phase.name` while QUIESCING is the driver's last recorded phase:
  `opening` (no step yet), `waiting_slot`, `preflight_blocked`, `draining`
  (some held turn not yet drained) or `dispatch_pending` (all drained, the
  dispatch CAS not yet taken). After that it follows the job state:
  `dispatched`, `running`, or the terminal state. `drained`/`total` count the
  driver's last roster read at `observed_at`, so they can lag the live turns by
  one reconciler interval.
- `quiesce.open_deadline_at` is `submitted_at + quiesce_timeout_seconds` while
  the hold is unopened (null once opened); `quiesce.deadline_at` is the drain
  deadline once open.
- `quiesce.members` is one row per turn in the job's current gate generation:
  `bot_id, turn_id, backend, disposition, control_state, control_attempts,
  control_error, control_changed_at, ack_status, ack_note, acked_at,
  enrolled_at, drained, drain_reason`. `control_state` and `ack_status` are the
  stored values, not inferred. `control_changed_at` is
  `gate_members.control_changed_at`, stamped by every UPDATE that changes
  `control_state` (rows that pre-date the column read `created_at`).
  `drained` is the driver's last verdict (null for a turn not yet judged).

Each phase change logs one INFO line from `llm_bawt.ops.quiesce`:
`quiesce job=<id> op=<slug> phase <old>-><new> gate=<generation>
drained=<n>/<m> deadline=<iso>`, with the failure outcome or the cancel's
claimed actor in parentheses. Steady-state steps log nothing.

**`GET /v1/ops/holds`.** Every live gate (`released_at IS NULL AND expires_at
> now()`): `{gate_id, owner_kind, owner_id, mode, scope, reason, allow_actors,
created_at, expires_at, members: {total, by_disposition, by_control_state},
job}`. `job` is `{id, operation, state, phase}` for a job-owned gate and null
for an operator-owned gate.

**`POST /v1/ops/jobs/{job_id}/cancel`** with `{actor, reason?}` cancels a
QUIESCING job before dispatch. `actor` is a self-asserted label and is
recorded as claimed (`quiesce_json.cancelled_by.actor_claimed`, and in
`error_text`). In one transaction, in a driver step's lock order, the job moves
QUIESCING -> CANCELLED (state CAS), every gate generation is released and
pending steers are cancelled; no action was submitted. 200 `{job, cancelled:
true}`; an already-cancelled job is 200 `{cancelled: false, already: true}`;
any other state (dispatched, running, terminal, or not a quiesce job) is 409
`not_cancellable` with the state; unknown is 404. The cancelled job's receipt
goes out on the normal paths (approval continuation, or the receipt loop above).

## Deployment setup checklist (original TASK-861 setup; installed on echo)

The base compose app mounts Docker's socket and `./.logs:/app/.logs`; the dev
overlay mounts `./src:/app/src`. Do not guess host bind sources from a bridge
container. `docker-compose.ops.yml` adds a dedicated **named volume** without
altering existing compose hunks. The original implementation lacked a live
Docker check; the ops worker and receipt path are now deployed on echo. For a
new installation or topology change, verify each step below again.

An operator setting up a new installation must:

1. Inspect the current app image ID, current compose file list, Docker daemon,
   app mounts and networks. Verify the daemon is the one owning the target
   containers. Do not infer host bind paths from `/app/src` in a bridge.
2. Build `docker/Dockerfile.ops-worker` with `OPS_PYTHON_BASE` set to an
   operator-verified local Python image ID/digest. It COPYs `worker.py` and
   `image_deploy.py`; no pip/uv/apt install is required. Record the immutable image ID
   (or published digest) as `LLM_BAWT_OPS_WORKER_IMAGE`. Mutable tags are rejected.
3. Choose a dedicated per-stack volume name and set
   `LLM_BAWT_OPS_RECEIPT_VOLUME`. Set `LLM_BAWT_OPS_RECONCILER_IMAGE` to the
   verified existing CPU app image ID/digest. These are topology configuration,
   not catalog metadata. Preserve historical image/volume availability while
   jobs are active.
4. Append `docker-compose.ops.yml` **last** to the existing compose configuration
   (preserve any dev/prod overlays). Review merged config before applying. The
   override mounts the volume at `/var/lib/llm-bawt-ops` in app and reconciler,
   and starts a separate `python -m llm_bawt.ops.reconciler` service. It uses
   the existing app image, `.env` DB configuration, default stack network and
   read-only dev source mount. It also mounts the app's existing credential key
   file read-only, not the entire app config directory; reconciliation needs the
   same key to decrypt DB-stored GitHub tokens. Validate the merged Compose config
   and separately authorize recreation of an existing reconciler to add the mount.
   Adapt the source mount/network for a baked image or alternate DB network.
5. With explicit deployment permission, recreate **only** app and the new
   ops-reconciler service. A restart alone cannot add mounts/environment. Never
   restart bridges/Redis as part of this setup. Applying the override/building
   images/creating volumes is an operator action, not performed by tests.
6. Verify both submitter containers' inspected `Mounts` contain the configured
   writable named volume at the exact configured root. The executor performs
   this check itself using `HOSTNAME` (or `LLM_BAWT_OPS_APP_CONTAINER` if the
   container has a custom hostname). The worker image must already exist;
   dispatch never implicitly pulls it or creates a misspelled volume.
7. Run staged non-self-affecting smoke operations, then explicitly approved
   self-restart tests. Confirm accepted/running state, app-independent worker,
   receipt survival and DB reconciliation. The SQLite/fake-Docker tests are
   strong regression evidence, **not proof of live Docker survival**.

The worker is launched with no network, read-only rootfs, dropped capabilities,
no-new-privileges, bounded memory/PIDs and restart policy `no`. It receives only
the receipt volume and Docker socket, not DB credentials, app source mounts, or
app environment. The reconciler has DB access and needs the app's Docker SDK;
the standalone worker uses only Python's standard library. No Python dependency
changes are required.

Retain receipts/requests and exited worker containers until the job is terminal
and audit retention allows removal. No automatic Docker/volume garbage collector
is included. `compose down -v`, daemon-wide pruning, volume loss, host power loss,
manual receipt edits and disk-full conditions are not covered by an exactly-once
claim. Receipts contain resolved arguments (possibly secrets); restrict volume
access and backups. Public job JSON uses schema-sensitive argument redaction;
operator revision endpoints intentionally expose operator configuration.

## Internal ops health API (TASK-1157)

The **ops-reconciler process** owns Docker health inspection. The frontend health
aggregator consumes `http://ops-reconciler:8650`; it does not need Docker socket
permissions for monitoring. Port 8650 is exposed only on the Compose network:
no host publish, public proxy route, arbitrary target selector or Docker proxy.
This migration does **not** remove the app executor or existing Docker-page
controls/socket mounts; those are separate work.

The fixed GET-only contract (`ops/health_http.py`):

* `GET /health/live` → HTTP 200, `{"status":"ok"}` while HTTP is listening,
  independently of DB, Docker, collector or reconciler progress.
* `GET /health/ready` → `{status,checkedAt,checks:[{name,status,reason}]}`.
  Status is `ok|degraded|down`; only `ok` gets HTTP 200, otherwise 503. Checks:
  `database`, `docker`, `ops.executor`, `reconciler.loop`. Reasons are fixed
  sanitized messages or null, never exception text/DSNs/Docker response bodies.
* `GET /v1/health/containers` → `{schemaVersion:1,checkedAt,status,reason,containers}`.
  Each container is **only** `{name,state,health,startedAt}`. Exact Docker names
  lose their leading slash. State is `created|running|paused|restarting|removing|exited|dead`;
  health is `healthy|unhealthy|starting|null`; startedAt is timezone-bearing ISO
  or null for never-started containers. No IDs, env, mount details, logs, labels,
  image/config metadata or ports are returned. Healthy inventory gets 200/`ok`;
  unavailable/malformed/over-budget/stale inventory gets 503/`down`, a fixed
  reason, and an empty list. An unsuccessful empty list is **not** evidence a
  configured target is absent. No partial inventory is advertised as complete.

`checkedAt` is the collection **start**, not HTTP request time or completion.
Readiness uses the oldest contributing sample. The first startup snapshot is
`down`/`not yet collected`. Samples older than 30 seconds (monotonic or wall
clock) or more than 5 seconds in the future fail closed on **every GET**, even
if collection hangs after previously publishing success. Container observations
remain independently usable when DB/reconciler readiness is down. Consumers
should also enforce timestamp freshness; a critical ops-health row describes
collector failures, while affected target rows should say observation unavailable
rather than falsely report containers down.

A fixed HTTP worker reads immutable cached samples and never performs DB/Docker
I/O, schema work, job-status reads, preflight, dispatch, or reconciliation.
Separate fixed workers collect DB and Docker data on a 5-second cadence and run
the actual reconciler loop. Successful/failed **real ticks**, not synthetic timer
heartbeats, update loop health. A successful tick taking over 30 seconds is stale
from its start, not newly healthy. Reconcile cadence remains controlled by
`LLM_BAWT_OPS_RECONCILE_INTERVAL` (default 5s); intervals at/above the 30s freshness
window intentionally cannot sustain continuously-ready observations.

Docker RPC timeout is 2s; inventory has a 15s overall acceptance budget and a
1000-container cap. One list plus one inspect per observed container supplies
accurate start/health fields. Oversize/duplicate/malformed inventories, including
containers disappearing before inspection, fail closed and retry on the next
sample. This is not an atomic Docker snapshot; changes after collection are
observed on a later sample. Readiness reuses the executor's
read-only prerequisite checks, **not** `preflight()` (which can pull images for
some actions). The shared DB pool is reused with a read-only `SELECT 1` transaction
and 2s statement/lock timeout. Pool/connect/driver stalls are contained to one
collector: no retry-thread fan-out; stale detection is the backstop, not a claim
that Python can forcibly cancel a blocked driver call.

SIGTERM/SIGINT mark readiness down, stop polling and close HTTP. The HTTP worker
join is bounded to 2s; collector/reconciler joins share another 2s bound. Stuck
workers are daemon threads and cannot prevent process exit. Docker inventory
and executor prerequisites share one collector: a stuck prerequisite eventually
expires inventory too, without blocking HTTP or the DB collector. The Compose
readiness healthcheck replaces the inherited app-port probe.
Adding this listener/healthcheck is a reviewed source/Compose change, **not**
authorization to restart or recreate any service. Verify it with fake adapters
and local loopback HTTP tests first; activation requires its own authorization.

Frozen producer contract: `tests/fixtures/ops_health_v1.json` contains synthetic
`ready` and `inventory` payloads plus a fixed `now` for consumer tests (not live
container evidence). `test_frozen_cross_stack_fixture_is_real_producer_output`
asserts exact equality against collector/state output. Run
`.venv/bin/python -m pytest tests/test_ops_*.py -q` for the ops regressions.

## Digest-bound image deploy/rollback (TASK-997)

Two fixed actions extend the same worker; they are **not** a generic image
runner. `deploy_image` / `rollback_image` specs pin `container_name`,
`compose_project`/`compose_service`, `image_repository` (`ghcr.io/...`),
`github_repository`, `workflow_path`, `canonical_branch` and
`health_timeout_seconds`; selector arguments are rejected
(`executor.validate_image_spec`). Seeds `bawthub.deploy-prod-image` and
`bawthub.rollback-prod-image` ship **disabled** with dedicated
`require_approval` policies (orders 7/8), separate from the GitHub build
approval (`workflow_dispatch` of `.github/workflows/release-frontend.yml`).

Before an approval snapshot exists, `prepare_invocation` binds (module
`ops/release.py`):

* **deploy** args `workflow_run_id`, `digest`, `source_sha`, `version` (regex
  re-checked in code). `GitHubReleaseVerifier` requires the run to be a
  completed, successful `workflow_dispatch` of the pinned workflow on the
  canonical branch in the pinned repo; its single `release-receipt` artifact
  (`bawthub.release-receipt/v1`, `status: complete`) must equal the args, with
  `base_sha` = the run head SHA; tag `v<version>` (annotated tags peeled) must
  resolve to `source_sha`. The signed artifact redirect is followed WITHOUT the
  GitHub token. The token is the DB-stored `github-release` provider credential,
  read per verification. Any failure → `release_unverified`; no snapshot, no job.
* **rollback** arg `deploy_job_id`: a SUCCEEDED deploy of the same target with
  a complete deployment record; its `previous` image becomes the target.
* both: a compare-and-swap on the image the target runs now
  (`expected_current_image_id` / `from_image_id`). `_verify_snapshot` rejects a
  release/rollback binding that disagrees with the args or appears on an
  ordinary operation.

Deploy preflight (app side, `DockerExecutor.ensure_release_image`) pulls
`repo@digest` with the DB-stored `ghcr-pull` provider credential when it is not
already local. Pulling touches no container; a missing credential or failed pull
leaves the job QUEUED with the reason visible. The worker holds no credential.

The worker (`ops/image_deploy.py`, stdlib only) takes a per-target flock
(`<receipt volume>/.target-locks`), re-checks Compose identity and the CAS,
refuses (known, nothing changed) if the approved image is not on the daemon,
requires the `RepoDigest` and release labels to match, then creates a clone of
the live container (Compose labels/env/host config/networks preserved, old
image's labels/env/cmd not carried), stops and renames the old one to
`-prev-<job>`, starts the new one, waits for Docker health and verifies
`/api/health` reports `healthy` and the approved baked release. On any failure
after the stop it restores the previous container; a verified restore is a
known `failed`, an unverified restore is `lost` (never retried). Rollback never
pulls. The receipt `output_tail` is a JSON deployment record
(`llm-bawt.ops.image-deploy/v1`), surfaced as `job.deployment` in the API.

Activation checklist (echo activated 2026-10-03; recheck for a new
installation, not permission for another deployment):

1. Rebuild the worker image (it now COPYs `worker.py` + `image_deploy.py` into
   `/ops/`) and update `LLM_BAWT_OPS_WORKER_IMAGE`. Old queued jobs keep their
   snapshotted image.
2. Connect two provider credentials (encrypted in the CredentialStore; Providers
   UI or `POST /v1/providers/{id}/connect/api-key`, validated against GitHub):
   `github-release` — fine-grained, Actions:read + Contents:read on
   `bawthub/bawthub` only; `ghcr-pull` — classic, `read:packages` only (ghcr.io
   rejects fine-grained tokens; the GitHub login becomes the registry user).
   No new env var is needed for connection changes. The reconciler must have the
   existing app encryption key mounted read-only before it can use these tokens.
3. Reload the app with the ops worker configuration, review the seeded rows and
   policies, then enable — each a separate explicit decision.

On echo both operations are enabled. The first approved deploy (v0.1.58, run
`37079842420`, job `e728475475c0433ba72c83a42bd5381f`) succeeded. No live
rollback has been verified; a rollback needs its own explicit request and
approval, not an automatic test of the enabled action.

## Durable one-command BawtHub release orchestration (TASK-1030)

`bawthub.release-prod` is a typed release executor operation. It is not a shell,
SSH, Make, Docker-build, or generic GitHub executor. One approved call starts one
durable parent ops job and one `ops_release_runs` row. The caller supplies only:

```json
{
  "operation": "bawthub.release-prod",
  "args": {
    "release_task": "TASK-N",
    "bump": "patch"
  },
  "idempotency_key": "stable-logical-release-key"
}
```

`bump` defaults to `patch`. The agent does
not dispatch Actions, find a run, download a receipt, transcribe a digest, create
a deploy job, poll either job, or recover a partial release. The parent job stays
active while the server-owned coordinator advances this lifecycle:

```text
PREFLIGHT -> DISPATCHING_BUILD -> BUILDING
          -> BUILD_FAILED_SAFE | BUILD_PARTIAL | BUILD_COMPLETE
BUILD_COMPLETE -> AWAITING_DEPLOY_APPROVAL -> DEPLOYING
               -> DEPLOYED | DEPLOY_FAILED_RESTORED
any uncertain external side effect -> LOST_REQUIRES_INSPECTION
```

The release run is the orchestration source of truth and has append-only events,
a lease/claim for multi-process reconciliation, the parent ops job id, a unique
`release_request_id`, exact remote source SHA, GitHub workflow run/attempt,
verified receipt, deployability, deployment approval id, and child deploy
job id. Transitions are compare-and-swap updates. State is persisted before each
external side effect. App or reconciler restarts resume from the row; agent polling
is never the scheduler.

### Build authorization, source authority, and dispatch correlation

The ordinary fail-closed `ops_run` approval is the **build/release authorization**.
Before that approval is shown, read-only preflight resolves the canonical remote
BawtHub branch head through a dedicated `github-release-dispatch` credential.
Echo working trees are not consulted: dirty, ahead, behind, divergent, or absent
local clones cannot alter a remote release. The immutable approval snapshot
includes that SHA and the release plan. A release never reads or tags llm-bawt.

GitHub `workflow_dispatch` does not return a run id. The coordinator therefore
creates one unique `release_request_id`, passes it as a workflow input, requires it
in the workflow `run-name` and receipt, and persists dispatch intent before the
HTTP request. A missing/ambiguous dispatch response is reconciled by that exact
correlation before any retry. Zero matches remains pending or fails safe after a
bounded window; multiple matches are `LOST_REQUIRES_INSPECTION`. Never redispatch
merely because the run id was not returned.

Workflow receipts are attempt-aware and bind request id, repository, workflow,
run id, run attempt, source SHA, semantic version/tag and image digest. The
coordinator verifies this identity before a release becomes deployable. A rerun
uses GitHub's rerun-failed-jobs API only for a classified-safe failed job and stays
on the same release row, run id, version and tag. It never dispatches a second
workflow and never increments semver again. Only a `complete` receipt is
deployable. Recovery never runs `make rebuild-prod`, `make snapshot-rebuild`, or
`make -o version-release snapshot-rebuild`.

### Separate server-owned deployment approval

A verified deployable build creates a second, deterministic approval request of
kind `orchestration`. It is bound to the release run and the canonical invocation:

```json
{
  "operation": "bawthub.deploy-prod-image",
  "args": {"release_run_id": "<durable-release-id>"}
}
```

This is the separate **production deployment authorization**. It is always
required by the coordinator and rendered in the existing approval UI; agent
`ops_run` calls still go through the compiled approval-policy bundle. It has no MCP execution claim, bridge grant, synthetic agent turn, or
continuation outbox. The first terminal human decision wins. Denial/cancellation
leaves the verified release undeployed and terminates the parent safely.

Approval lets the coordinator create exactly one child
`bawthub.deploy-prod-image` job with a deterministic idempotency key. The existing
credential-free, network-disabled Docker worker remains the only production
mutator. Its release verifier loads the stored release by `release_run_id` and
re-verifies the frozen receipt binding; agents no longer supply workflow run id,
digest, SHA, or version. A known failed deployment with verified restoration maps
to `DEPLOY_FAILED_RESTORED`; an ambiguous deploy remains
`LOST_REQUIRES_INSPECTION` and is never blindly retried. Parent success requires
the child job to succeed and final production health/identity to equal the
approved SHA, version, digest and workflow run.

The Actions credential is separate from `github-release` (read-only verification)
and `ghcr-pull` (registry pull). It is least-privilege Actions read/write plus
Contents read on the release repositories. The release executor runs in the app
and independent reconciler processes; no GitHub credential enters the Docker
worker. The operation ships disabled. Implementation, catalog activation, each
build approval, and each deployment approval are distinct decisions; implementing
this contract authorizes no live release.

## Approval integration

Trusted interception calls:

```python
snapshot = ops.prepare_invocation(operation_slug, args)
# Persist snapshot with approval request BEFORE presenting approval.
ops.dispatch_job(operation_slug=operation_slug, args=args,
                 idempotency_key=stable_key, approved_snapshot=snapshot, ...)
```

`prepare_invocation` returns detached JSON with `snapshot_version`, full
`operation` (`version`, `script_hash`, exact `command_script`, schema/default JSON,
all settings and attribution), parsed `spec`, `schema`, `defaults`, `input_args`,
`resolved_args`, `execution` (including immutable worker image and receipt
configuration), and `snapshot_hash` (canonical-JSON SHA256). Hash checking detects
corruption; it is **not authorization**. The snapshot must come from persisted
trusted approval context, never a caller-supplied tool field.

`ApprovedCallerContext.approved_snapshot` is the integration field read by
`ops_tools`; the public `ops_run` signature does not accept snapshots. An approved
context missing this snapshot is rejected: pre-migration approvals need fresh
approval, never silent execution of today's catalog. Contextvars
are captured/propagated before offloading blocking DB/Docker work with
`asyncio.to_thread`. A valid approved snapshot executes the old revision despite
later edits, but current disabled/soft-deleted state vetoes new dispatch. Existing
idempotent replay remains a read even if the operation was subsequently disabled.

No explicit key means a new UUID on every direct invocation. The same explicit
key + identical original operation/input returns the original job (including
queued jobs), without re-dispatch. A different payload or approved snapshot with
that key raises `idempotency_conflict`. An approved call with no explicit key uses
its trusted approval request ID. This is global key uniqueness, so integrations
should namespace their own explicit keys.

## API and migrations

- `GET /v1/ops/operations` and `/v1/ops/jobs` accept `limit` (1..200) and
  nonnegative `offset`; responses are `{operations|jobs, total, limit, offset}`.
  `total` is the count for the filters, not the current page length.
- Dispatch responses include both `id` and compatibility alias `job_id`.
- Direct HTTP dispatch accepts `actor`, `caller_user_id`, `caller_bot_id`,
  `caller_turn_id`, `caller_session_key` under the existing trusted LAN operator
  model. Backend is `http-operator`; no new global authorization layer is added.
- CRUD accepts operator `actor`; enable/delete are revisioned. Slugs are immutable
  to preserve invocation/history identity. Nullable fields can be cleared with
  explicit JSON null. Invalid schema/default/limits/spec updates are atomic errors.
- `GET /v1/ops/operations/{slug}/revisions` returns full historical configuration,
  actor and timestamp in `{revisions, total, limit, offset}`.
- Bootstrap safely adds nullable `invocation_snapshot_json`, `request_payload_json`,
  `caller_actor` to existing jobs and creates `ops_operation_revisions`; captures
  only the existing current revision as baseline. Earlier historical revisions
  cannot be reconstructed and are not fabricated. Legacy active jobs without
  snapshots reconcile to lost; never execute them using today's catalog.
- The supported schema subset is explicit; malformed/unsupported keywords are
  rejected rather than ignored. Empty `{}` means no args, all nested objects are
  closed, array items/types and numeric/string constraints are checked. Defaults
  must themselves be valid declared values; required caller fields can remain
  absent in the defaults object.
