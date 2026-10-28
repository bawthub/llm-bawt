# Ops service: current architecture and intended consolidation

**Status:** design baseline; implementation and rollout are queued, not complete.

**Updated:** 2026-10-21 (TASK-1220 approved-admission move, implemented, not deployed). **Accountable owner:** Snark.

**Tasks:** TASK-1169–TASK-1173. Related health delivery: TASK-1157.

**Review:** Caid requested revisions in `cmved9mrf000x0mnxbjwkf9pj`, then approved the revised scope for bounded implementation planning in `cmvedeozu00180mnx4hha2089`. The explicit FastAPI service-host choice below records Nick's subsequent direction; it is not a claim of implementation acceptance.

## 1. Purpose and boundaries

Nick wants a proper operations service, not Docker access distributed throughout the application. A future move to Kubernetes and replicated services must not require rewriting every caller. Reduce operational machinery and duplicated code rather than build another platform.

The target is **one logical ops service hosted by FastAPI**, with typed control, status and health endpoints, durable background execution, and a runtime adapter. One logical boundary does not require exactly one physical replica. Start with the existing ops deployment; preserve/test safe job claims rather than implement speculative distributed infrastructure.

This document is the migration design and decision record. [OPERATIONS_RUNNER.md](OPERATIONS_RUNNER.md) describes the existing runner's detailed safety contracts. Neither document authorizes a restart, recreation, deployment, permission expansion or cleanup. Do not interpret intended behavior below as a current capability.

## 2. Current architecture — verified baseline

### Control and execution

```text
UI/operator or agent MCP
          |
          v
llm-bawt app: catalog + approval integration + OpsService
          |                         |
          | Docker target checks    | immediate job dispatch
          v                         v
       Docker daemon <------ per-job Docker worker container
          ^                         |
          |                         v
ops-reconciler process <---- durable request/receipt volume
          |
          +---------------- canonical ops job database
```

The app constructs `DockerExecutor`, resolves targets during invocation preparation, and dispatches directly submitted jobs immediately: a directly allowed MCP `ops_run` (MCP server process) and `POST /v1/ops/jobs`. A `get_job_status()` read still dispatches/reconciles active jobs unless the caller passes `reconcile_if_active=False`. The separate reconciler also pumps the ledger; conditional claims coordinate dispatchers. **Approved** MCP `ops_run` calls are the exception: since TASK-1220 the app records only the decision and the reconciler admits the job (below).

### Approved MCP `ops_run` admission (TASK-1220) — implemented, not yet deployed

Incident: approving an `ops_run` made the app lazily import the MCP tool tree at approval time. The app imports code from the live bind-mounted tree, so newer on-disk modules met older loaded ones and the approved call died with an `ImportError` after the human had approved it. Approved ops work now has one owner, which is ops:

```text
human approves ──> app resolve route: record decision only (no import, no execution)
                     returns ok + execution_state=pending + awaiting_admission, no `result`
ops-reconciler tick ──> reconcile active jobs  ─┐ independent; first failure reported
                   └──> ApprovedOpsAdmission.admit_due(limit=10)
                           claim (lease 300s) → invocation hash → ops_run schema → snapshot
                           → dispatch_job: existing job for key → report it (any age)
                                           else create it behind execution_fence
                                           (claim held + approved ≤600s ago)
                           → result + continuation enqueue in one transaction
app continuation outbox ──> reads the job (no reconcile), writes the outcome to the tool card,
                            continues the agent once the job is terminal
```

- **App** (`service/routes/approval_policies.py`): an approved `ops_run` is never claimed or run in the app. The card gets only its approval status, and the `approval_resolved` event omits `result`/`is_error`, so no client can finalize it as a null success. App recovery skips APPROVED `ops_run`; denied/cancelled/responded `ops_run` rows are still completed by the app.
- **Admission** (`ops/approved_admission.py`, run by `ops/reconciler.py`): it uses the existing fences: the claim token and lease, the job idempotency key (caller key, else approval id), and the fenced result commit. It adds one more: job creation runs in the same transaction as `execution_fence` (→ `OpsStore.create_job(fence=)`), which checks that the claim token is still held and the approval is inside the 10-minute window, and renews the lease. A claimant that stalls past its lease therefore can't create a job after a newer claimant has already refused or expired the row, and a lone admitter that stalls past the window can't create one late. Concurrent admitters produce one job. A claim superseded after lease expiry is logged and left to its newer owner. Job provenance comes from the persisted approval row. Refusals complete as error results with a machine-readable code: `[invocation_mismatch]`, `[args_invalid]`, `[snapshot_invalid]`, `[admission_expired]` or an ops dispatch code. An approval not admitted within 10 minutes expires without a job, but an existing job for its key is always reported. The window is checked only at creation, after the invocation checks, so an expired row whose stored invocation or operation no longer validates reports that code (for example `[snapshot_invalid]` or an ops dispatch code such as `[operation_disabled]`) rather than `[admission_expired]`. Either way, no job is created.
- **One contract** for both paths: the receipt and the `[code] message` error format are shared with the direct tool. The direct tool refuses an approved caller context with `[admission_owned]`.
- **No lazy imports** on these paths. The startup imports are hoisted, a unit test bans function-level imports in the path functions, and an offline-integration test checks in a cold interpreter that resolve, admission and continuation load no new `llm_bawt`/`agent_bridge` modules.
- **App health:**
  - `ops.admission` (on `/v1/health/checks`) reports DEGRADED when an approved `ops_run` has waited more than 60s for admission, and DOWN beyond 600s.
  - The app registers MCP tools eagerly at startup. If that fails, `mcp.tool_registry` reports DEGRADED and approved generic MCP calls are refused with `[tool_registry_unavailable]` rather than lazily importing.
- **Deploy order:** reconciler first, then app. An app that records decisions only must not run without an admitting reconciler.
- **Preflight deadline (TASK-1169 S1e):** an admitted job whose executor preflight fails stays QUEUED with `error_text`, and each reconcile pass retries it. Once `submitted_at` is older than `LLM_BAWT_OPS_PREFLIGHT_DEADLINE_SECONDS` (default 300) plus the snapshot's start delay, the next failing pass fails it with `preflight_failed: <last reason>` (`OpsStore.fail_stale_queued`, a QUEUED-only CAS on the DB clock that a claim committing first always wins). A per-job advisory try-lock (`OpsStore.preflight_lock`) keeps preflight passes from overlapping, so only the latest outcome is noted, expired or claimed on. A job waiting on a concurrency slot never expires. Its continuation completes on the terminal job; until then the blocked-ops notifier warns the agent.

Each operation uses a short-lived Docker worker with an immutable request, deterministic identity and durable receipt. This allows execution to outlive app restarts. Workers have no network or DB credentials, but carry the privileged Docker socket. Isolation settings do not make a socket-owning worker safe for untrusted code. Exited workers and receipts remain for audit; ordinary jobs therefore leave container objects behind.

The durable ledger, approval snapshots, idempotency, recovery markers, bounded deadlines and receipt semantics are valuable. Their current physical distribution is not the desired boundary.

### What runs in the ops container

The observed entrypoint is `python -m llm_bawt.ops.reconciler`: **one Python process**, containing:

- A reconciliation/queued-dispatch loop.
- A database health collector thread.
- A Docker inventory/executor-prerequisite collector thread.
- A separate standard-library HTTP server thread serving cached observations.

It is **not currently a FastAPI operations control service**. Its HTTP interface is GET-only:

| Current route | Meaning |
| --- | --- |
| `/health/live` | The HTTP server answers; not proof the reconciliation loop progresses. |
| `/health/ready` | Database, Docker, executor prerequisites and actual loop progress are fresh/ready. |
| `/v1/health/containers` | Versioned, sanitized inventory keyed by exact Docker names. |
| `/v1/health/services[/{id}]` | Services enabled catalog ops target, by `<project>.<service>`, with opaque instance ids and capabilities (TASK-1169 S1c; contract in `OPERATIONS_RUNNER.md`). |

The HTTP handlers do not probe Docker/DB or dispatch jobs. This separation keeps health HTTP responsive during blocked background calls. The collection-start timestamp and freshness limits prevent old success from remaining healthy forever.

### Recovery behavior today

The inspected runtime uses `restart: unless-stopped`, with Docker's healthcheck disabled (`Test: ["NONE"]`). Process exit can trigger a Docker restart. A blocked background thread can leave the Python process and health HTTP alive; readiness reports failure, but the current runtime has no automatic stuck-loop restart mechanism.

A readiness healthcheck exists in Compose source but has not been activated in the inspected container. A plain restart cannot apply that configuration. Adding a Docker healthcheck alone would mark unhealthy, **not automatically restart** it.

### Remaining direct consumers

| Consumer | Current dependency | Migration |
| --- | --- | --- |
| llm-bawt app | Docker executor/preflight, image pulls, job dispatch and status reconciliation | TASK-1169 |
| Ops collectors/execution | Separate Docker access implementations | Cohesive runtime adapter, TASK-1169 |
| Hono dev/prod | Docker list/sync/state/start/stop/restart helpers | TASK-1170 |
| Hono repository browsing | `repos.ts` imports `getContainerMounts` for path discovery | Explicit deployment/repo-path configuration, TASK-1170 |
| Voice backend | Bundled `restart_tts_container.sh` (fixed argv, no shell; `BAWTHUB_TTS_RESTART_CMD` retired, TASK-1606) directly restarts the label-verified TTS container through the socket | TASK-1171 |
| GPU handoff in app | Reads STT/TTS from the reconciler's `/v1/health/services/bawthub.{stt,tts}` (`LLM_BAWT_OPS_HEALTH_URL`); refuses on anything but one fresh, typed instance. No Docker client. Mutations use consented ops jobs | Done in source, TASK-1171 |
| Traefik | Docker provider watches labels through a socket mount | Deployment-owned routing, TASK-1172 |

#### TASK-1171: source vs activation

Source changes ship inert. Each row below needs its activation step before the live system matches source; TASK-1173 runs them and records the acceptance.

| Change | In source | Activation (operator, TASK-1173) |
| --- | --- | --- |
| `bawthub.restart-tts` (typed `{restart, bawthub, tts}`, no args, high risk) | Seeded **disabled** | `POST /v1/ops/seed-defaults` inserts it; enable it only after the guard and fence rows below are live in both app and ops-reconciler. It needs no `ops_run` policy: the guard refuses it on every path but the handoff route |
| GPU voice guard (`ops/gpu_guard.py`): every action but `stop`, `pull` and release orchestration that targets `bawthub` `stt`/`tts` needs a `GpuHandoffFence`, checked at job creation and again at dispatch against the live container's labels | In `OpsService.dispatch_job` and `_dispatch_queued`/quiesce preflight | app **and** ops-reconciler restart (the reconciler dispatches queued jobs) |
| TTS restart fence: `POST`/`GET /v1/media/local-video/handoff/tts-restart[/{request_id}]` (service bearer). The fence (`phase=tts_restarting`) and the job commit in one transaction; only the status read releases it, on the job's outcome plus start evidence | Routes, `GpuHandoff.tts_restart`, `gpu_handoff_v6` column `tts_restart_key` | app restart (the column is added on first store construction) |
| BawtHub TTS admin restart through that route, backend socket removed | BawtHub slice 2 | BawtHub deploy, then the backend recreate above |
| `bawthub.reconcile-backend-config` drops `/var/run/docker.sock` and `BAWTHUB_TTS_DOCKER_CONTAINER`/`BAWTHUB_COMPOSE_PROJECT` | Seed row changed | Seeds never update existing rows: `PATCH` the live row, then recreate the backend without the socket |
| GPU handoff observation via ops health | App has no Docker client for it | `LLM_BAWT_OPS_HEALTH_URL` only if the reconciler isn't at the default `http://ops-reconciler:8650`; app restart |
| `restartCount` on `/v1/health/services` instances | Collector + schema | ops-reconciler restart |

Production Hono has a socket mount but its non-root user lacks socket permission; dev Hono has effective access. A `:ro` socket mount, as used by Traefik, does not restrict Docker API methods. Agent bridges have no direct socket mount in the inspected deployment, but host SSH/operator privileges are a separate surface, not removed by this design.

### Original health issue: resolved, not blocked on this refactor

Production v0.1.78 routes health observations through ops. The last verified aggregate was **20 OK / 0 degraded / 0 down**. The former false outages caused by frontend socket permission failures are fixed. These are dated observations, not a perpetual health guarantee. The broader consolidation can remain queued without undoing that fix.

## 3. Intended architecture

```text
UI / agents / application services
        |
        | existing user identity, admission and approval flow
        v
llm-bawt facade / ops client (no runtime SDK or execution)
        |
        | typed internal control/status API
        v
Ops FastAPI service — one logical owner
  +-- Catalog / preparation / verified submission / read-only job queries
  +-- Existing DB job ledger and immutable approval bindings
  +-- Bounded execution and recovery under managed application lifespan
  +-- Cached service observations and health reporting
  +-- Runtime adapter boundary
        |
        +-- Docker implementation today
        +-- Other runtime implementation only when actually needed
```

FastAPI owns HTTP routing and startup/shutdown wiring. It does not itself supply durable execution. A request validates and persists work, then returns a job ID; it does not run a deployment inline. Plain FastAPI `BackgroundTasks` is not the job ledger or recovery mechanism.

Use bounded worker processes/internal execution within the ops service by default, not a Docker container per job. Keep blocking SDK work off the HTTP event loop and retain bounded shutdown and independent observation paths. Do not create a new queue/broker or general workflow framework.

### Ownership

- **App/UI/MCP:** caller identity, user interactions and existing approval presentation; typed requests and receipt consumption. No Docker preflight, image pulls, worker creation or execution on a status read.
- **Ops service:** operation preparation/validation, runtime target binding, approved job admission, execution, observations and reconciliation. Own the existing ledger rather than introducing parallel job state. Client-side approval presentation does not grant callers authority to manufacture approved snapshots.
- **Runtime adapter:** translate service identity/capabilities into Docker-specific selection and actions; consolidate SDK construction/observation/execution here. Existing `Executor` abstraction is the starting point, not a mandate for a second abstraction hierarchy.
- **Deployment configuration:** runtime mapping, connectivity, routing and credentials. Docker/Compose details do not belong to ordinary product callers.
- **Each application service:** its own readiness and functional health. Orchestration observations are not a substitute for application health.

## 4. Interface principles

These are intended responsibilities, **not newly implemented endpoints**. Preserve existing external operation slugs and `/v1/ops/*` compatibility where practical; finalize internal route/schema names during TASK-1169 instead of inventing an additional public API now.

| Surface | Intended semantics |
| --- | --- |
| Operation catalog/preparation | Typed capabilities, validated arguments, approved target scope and revision binding. Preparation may consult ops for target evidence, not the app's Docker client. |
| Job submission | Durable acceptance of a verified invocation; idempotency key and job ID. Submission is not completion. |
| Job status/results | Read persisted state/evidence only; no dispatch or reconciliation side effects. |
| Service observations | Logical service IDs, opaque instance identities, readiness/availability, capabilities and collection freshness. |
| Health | Distinct process liveness, execution progress, dependency readiness and observation freshness. |

Normal consumer contracts refer to services such as `bawthub.tts`, not container names, Docker IDs, socket paths or Compose labels. Runtime mappings are ops/deployment-owned and versioned/bound sufficiently to prevent approval target drift. Historical snapshots/receipts remain immutable; migrate new invocations without rewriting history.

Service observation schemas must represent zero, one or multiple instances. Unavailable/stale inventory is not zero instances. Define the scope of actions explicitly: restarting one instance and rolling a service are not interchangeable. Expose only capabilities the current adapter actually supports. GPU handoff must reject unsupported replica/resource layouts instead of choosing the first instance; distributed GPU scheduling is out of scope.

The existing exact-container-name health contract needs a versioned migration with bounded compatibility. The frontend and GPU handoff share the new observation contract. No env, mounts, credentials, raw config or arbitrary logs in health payloads. Repository path discovery uses explicit deployment/repo settings, not an expanded privileged inventory API.

The explicitly Docker/Unraid host-admin UI is distinct from product service orchestration. Preserve its functionality without letting its runtime-specific details leak into normal service contracts. Existing Unraid GraphQL features are not being rewritten in this effort. Unsupported arbitrary host-admin requirements require a narrow decision, not a generic Docker proxy.

## 5. Durability, authorization and recovery

Preserve these existing invariants while changing execution ownership:

1. Persist trusted approval context and immutable invocation/target binding before execution. A content hash detects modification; it does not prove authorization. Reuse the established trusted approval provenance and internal service controls; do not accept a caller-supplied `approved` flag.
2. Conditional claims and concurrency limits precede side effects. Test competing dispatchers and crash windows; existing claims are not proof every new execution path is replica-safe.
3. Recheck current enablement/interlocks and the approved operation/target binding. Drift must fail visibly, not silently execute a different target.
4. Duplicate delivery cannot create a second side effect. Persist execution intent before dispatch and durable evidence afterward.
5. Accepted/running is not succeeded. Lost/timed-out may mean the operation happened; do not blindly replay or assume cancellation/rollback.
6. Preserve release source/digest verification, compare-and-swap target binding and recorded rollback identity. Move runtime calls, not unrelated release workflow logic.
7. Do not leak credentials to callers or job output. Review the trust change when moving execution from credential-free workers into the privileged ops service; isolate necessary credentials and never execute untrusted scripts.

### Automatic recovery and self-management

The new service must distinguish dependency failure from a stuck executor. Database/Docker unavailability should make readiness fail; restarting indefinitely will not repair an external outage.

Define a bounded progress watchdog for internal execution/collector failures, with real progress evidence, startup grace and retry/backoff limits. A recoverable stalled worker can be terminated/reconciled by its supervisor without replaying uncertain mutations. An unrecoverable service stall can cause a controlled process failure for the deployment supervisor to recover. Docker restart policy or future Kubernetes liveness then supplies process recovery. A completely wedged process still needs an external supervisor/liveness mechanism; FastAPI alone cannot guarantee recovery.

Explicit self-restart requires durable correlated intent and outcome verification; a newer container start time alone is not proof our operation succeeded. Self-update/recreation requires a bounded approved maintenance design. Do not secretly retain another privileged executor to solve this case. If the simpler design cannot preserve a required guarantee, present the concrete tradeoff before implementation.

These recovery semantics need tests and activation evidence; they are not deployed today. Do not wire every readiness failure to a restart.

## 6. Routing is not an ops feature

Prefer supported static/templated Traefik file-provider configuration with stable service upstreams. Keep one deployment-owned source rather than duplicate labels and hand-maintained route files. Validate all actual dev/prod/tenant routes, auth, priorities, websocket/SSE behavior, health filtering and DNS refresh.

Stable DNS alone is not proof of replica load balancing or stale-endpoint recovery; test the real provider/upstream behavior. Do not build custom route discovery in ops, a Docker compatibility proxy, or another privileged sidecar. A future Kubernetes deployment uses its routing/Ingress configuration without changing product ops consumers.

## 7. Bounded implementation and acceptance

| Task | Deliverable | Dependency |
| --- | --- | --- |
| TASK-1169 | FastAPI ops host; portable contracts; existing ledger/admission separation; adapter and bounded execution; recovery tests | Foundation |
| TASK-1170 | Frontend controls/health plus repo-path discovery migrated; direct helpers removed | 1169 |
| TASK-1171 | TTS restart and GPU observations use the shared contract | 1169 |
| TASK-1172 | Socket-free deployment routing using supported provider configuration | Independent of 1169 |
| TASK-1173 | Integrated authorized activation, actual mount removal, healthcheck/recovery configuration and final evidence | 1169–1172 |

Start an architecture ratchet in TASK-1169: list current offenders with owning task IDs, reject new ones immediately, and remove exceptions as migrations land. Cover SDK imports, direct HTTP/CLI/shell helpers, socket/TCP configuration and parent-directory mounts across Compose and tenant artifacts. Separate legitimate adapter code and operator test tooling from product dependencies.

**Implemented (S1a):** `tests/test_docker_access_ratchet.py` scans `src/` and every `docker-compose*.yml`. `OPS_ADAPTER` and `OPS_COMPOSE` name the ops boundary; `OFFENDERS` lists each current offender with its owning task (`media/gpu_handoff_adapter.py` → TASK-1171, compose `app` socket mount → TASK-1173). A new offender fails, and so does an entry that no longer matches, so a migration deletes its own exception. BawtHub carries the twin for Hono, the voice backend and its compose files.

Required evidence includes consumer tests with Docker unreachable; zero/one/many-instance contract fixtures; pure status reads; authorization/target-drift and duplicate dispatch tests; competing claims and crash recovery; blocked-execution health responsiveness; routing replacement; and proof that ordinary jobs no longer create per-job containers.

Final acceptance requires actual runtime mounts/configuration and behavior, not just green unit tests or a healthy dashboard. Preserve old receipts and rollback assets. Any worker cleanup must be bounded and separately authorized, never daemon-wide pruning. No automatic bridge/Redis restart or production deployment is implied by these tasks.

**Non-goals:** Kubernetes adapter now; autoscaling; distributed GPU scheduling; new broker/job/auth framework; generic shell/Docker proxy; homegrown routing discovery; rewriting unrelated Unraid tools; removing operator SSH access by stealth. Extra scope requires concrete evidence and a bounded decision.

## 8. Keep this document current; graduate verified guidance to skills

This is the single working design for the five tasks, not a frozen proposal.

- Each task must update affected sections when implementation changes a decision, contract or ownership boundary. Mark implemented-but-not-activated separately from live-verified behavior.
- Keep a concise decision/verification record; replace stale diagrams and update the source map. Never present a proposal as the installed runbook.
- TASK-1173 owns final reconciliation: document the actual accepted architecture, remove obsolete runner instructions and distinguish historical receipts from current procedures.
- **After the tasks are complete and live acceptance is recorded**, transition the durable architecture and operational guidance into the agent-skills repository. Load the skill-authoring/maintenance instructions then; do not pre-publish intended behavior as current instructions.
- Separate portable product architecture from private deployment coordinates/policies. Update the applicable `bawthub-ops`, `llm-bawt` ops-actions and local infrastructure guidance, using their ownership rules and normal reviewed publication/sync workflow.
- Retain this repo document as a short decision/history and navigation entry pointing to the final skills references, or keep repo-specific contracts here and clearly define ownership. Do not leave two independently maintained contradictory runbooks. Link the exact final skill paths and acceptance receipts in TASK-1173.

## 9. Current source map

Paths below are relative to llm-bawt unless prefixed with `bawthub/`:

- `src/llm_bawt/ops/service.py`: admission, target preparation, dispatch and status reconciliation currently combined.
- `src/llm_bawt/ops/store.py`: durable ledger and claims.
- `src/llm_bawt/ops/executor.py`, `worker.py`, `image_deploy.py`: Docker execution and per-job worker lifecycle.
- `src/llm_bawt/ops/release_executor.py`: release execution/coordinator integration.
- `src/llm_bawt/ops/reconciler.py`, `health_runtime.py`, `health_http.py`, `health_collector.py`, `health_state.py`: current independent loop and monitoring.
- `src/llm_bawt/ops/approved_admission.py`: approved MCP `ops_run` admission and the shared `ops_run` receipt/error contract (TASK-1220).
- `src/llm_bawt/approval_mcp_store.py`: approval execution claims, plus the shared awaiting-execution predicate used by the claim, app recovery and admission.
- `src/llm_bawt/service/routes/approval_policies.py`, `service/approval_continuations.py`: approval decision recording and the continuation outbox, which reads ops results without reconciling them.
- `src/llm_bawt/service/routes/ops.py`: existing app HTTP surface.
- `src/llm_bawt/media/gpu_handoff_adapter.py`: GPU-service observation through the reconciler's `/v1/health/services` (TASK-1171).
- `docker-compose.ops.yml`: ops deployment and source healthcheck configuration.
- `bawthub/frontend/server/lib/dockerApi.ts`, `routes/docker-unraid.ts`, `routes/repos.ts`: direct frontend Docker consumers (route paths relative to `frontend/server/`).
- `bawthub/bawthub/main_websocket.py`, `restart_tts_container.sh`: voice restart path (helper relative to `bawthub/bawthub/`).

### Decision record

- **2026-10-18:** Health monitoring fixed independently of full isolation; broader tasks remain queued.
- **2026-10-18:** Caid review narrows scope: reuse ledger, portable service contracts, shared observations, routing owned by deployment, guardrails first. Follow-up found repository mount inspection; TASK-1170 explicitly covers it.
- **2026-10-18:** Nick asks for a proper FastAPI operations service and a repo design document maintained through implementation, then transitioned to skills upon completion. This document records that direction; no service/code rollout performed.
- **2026-10-21 (TASK-1220):** After the approval-time stale-import incident, Nick approved moving admission of approved MCP `ops_run` from the app into the ops reconciler. Snark's design review added these must-fixes: the admission deadline, recovery exclusion, the pending resolve contract, the shared contract, the hash check, no lazy imports, and per-row isolation. Nick set a 10-minute admission window, and chose to degrade and refuse (not crash) when MCP tool registration fails at startup. Implemented and tested; not yet deployed. This is the first step of the §3 ownership move ("approved job admission" to ops). Directly submitted jobs still dispatch in the app until TASK-1169.
