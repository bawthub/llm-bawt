# Ops service: current architecture and intended consolidation

**Status:** TASK-1169–1173 activated and live-verified on 2026-11-03 (TASK-1173). The ops reconciler is the only Docker-socket holder among first-party services. Sections 3–5 still describe intended work that is **not** live: jobs still run in per-job worker containers (finished ones are now removed, TASK-1991).

**Updated:** 2026-11-04 (TASK-1173 activation and final reconciliation). **Accountable owner:** the coordinator.

**Tasks:** TASK-1169–TASK-1173. Related health delivery: TASK-1157.

**Review:** The reviewer requested revisions in `cmved9mrf000x0mnxbjwkf9pj`, then approved the revised scope for bounded implementation planning in `cmvedeozu00180mnx4hha2089`. The explicit FastAPI service-host choice below records the operator's subsequent direction; it is not a claim of implementation acceptance.

## 1. Purpose and boundaries

The operator wants a proper operations service, not Docker access distributed throughout the application. A future move to Kubernetes and replicated services must not require rewriting every caller. Reduce operational machinery and duplicated code rather than build another platform.

The target is **one logical ops service hosted by FastAPI**, with typed control, status and health endpoints, durable background execution, and a runtime adapter. One logical boundary does not require exactly one physical replica. Start with the existing ops deployment; preserve/test safe job claims rather than implement speculative distributed infrastructure.

This document is the migration design and decision record. [OPERATIONS_RUNNER.md](OPERATIONS_RUNNER.md) describes the existing runner's detailed safety contracts. Neither document authorizes a restart, recreation, deployment, permission expansion or cleanup. Do not interpret intended behavior below as a current capability.

## 2. Current architecture — verified baseline

### Control and execution

```text
UI/operator or agent MCP
          |
          v
llm-bawt app: catalog + approval integration + admission-only OpsService
          |                                   |
          | prepare (:8651, bearer)           | create job QUEUED (+ NOTIFY ops_job_created)
          v                                   v
ops-reconciler process ---------------> canonical ops job database
   | target checks, dispatch,  ^ LISTEN wakes the loop; the interval is the fallback
   | reconciliation            |
   v                           |
Docker daemon <------ per-job Docker worker container ----> durable request/receipt volume
```

**TASK-1169 S3 (live since TASK-1173, 2026-11-03):** the ops-reconciler is the only executor. The app's `OpsService` is admission-only (`build_admission_ops_service`). It builds no `DockerExecutor` and registers no executor: `_resolve_executor` raises `ops_admission_only`. It asks the ops service to prepare each invocation over the internal control listener (`ops/control_http.py`, `:8651`, never published), then creates the job QUEUED (or QUIESCING) and returns. `get_job_status()` in the app is a pure read. The job insert notifies `ops_job_created` in its own transaction, which wakes the reconciler loop at once; a lost notify costs one interval. Callers already handle a queued result: the BawtHub Run dialog, `ops_run` and the GPU handoff adapter, which polls. A release job's `release_id` stays null until the reconciler creates the release.

- **Control channel.** `POST /v1/ops/prepare` returns the ops service's snapshot or a coded refusal, which keeps its own status. Auth is a bearer derived from the master key under `ops-control-token/v1`, not the BawtHub service token: the reconciler mounts the key but not BawtHub's token override. The client verifies the snapshot hash. It retries once, and only where nothing ran: a connection that never opened, or `ops_busy`. Everything else (timeout, 5xx, an old reconciler's 404, a non-protocol body) is `ops_unavailable`, which maps to HTTP 503 and creates nothing. The server runs at most `LLM_BAWT_OPS_CONTROL_CONCURRENCY` (default 2) requests at a time under a 20s budget; one more gets 503 `ops_busy`, and an over-budget request gets 504 `ops_timeout`. `GET /v1/ops/executors` feeds the app's `ops.executor.*` health rows; when it is unreachable they collapse into one DOWN `ops.control` row.
- **GPU guard target.** Prepare records `gpu_target` in the snapshot, so the app's creation check needs no Docker read. The app checks it against whatever the spec alone proves. A snapshot without the field whose target needs a Docker read is refused in the app with `gpu_guard_unverifiable`. Dispatch still re-checks live in the reconciler.
- **Quiesce fence.** The app no longer re-checks it at creation. The reconciler's driver does, before opening any hold. A worker that provably lacks the fence (`QuiesceFenceUnsupported`) fails the job at once with `quiesce_unavailable`, with no hold opened. A fence that can't be read right now waits out the job's open deadline, as before.
- **Deploy order:** ops-reconciler first, with `8651` exposed (`docker-compose.ops.yml`, so `up -d`, not a restart), then the app. An S3 app facing an older reconciler fails closed with `ops_unavailable`.

Approved MCP `ops_run` admission (TASK-1220, below) already ran in the reconciler.

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

Each operation uses a short-lived Docker worker with an immutable request, deterministic identity and durable receipt. Since TASK-1991 the reconciler removes stopped workers whose job is terminal (once a minute, bounded batches); dispatch refuses any job that already has a receipt, so a removed worker is never replaced. This allows execution to outlive app restarts. Workers have no network or DB credentials, but carry the privileged Docker socket. Isolation settings do not make a socket-owning worker safe for untrusted code. Receipts remain for audit.

The durable ledger, approval snapshots, idempotency, recovery markers, bounded deadlines and receipt semantics are valuable. Their current physical distribution is not the desired boundary.

### What runs in the ops container

The entrypoint remains `python -m llm_bawt.ops.reconciler`. **TASK-1169 FastAPI host slice (implemented in source, not activated)** runs Uvicorn/FastAPI on the existing private health port `8650`, with FastAPI lifespan owning:

- The existing reconciliation/queued-dispatch daemon thread.
- A fixed database health collector daemon thread.
- A fixed Docker inventory/executor-prerequisite collector daemon thread.
- The separate bounded, bearer-authenticated FastAPI control host on `8651` (its own socket, Uvicorn loop thread and slots; see below).

HTTP runs on the main async loop, outside the blocking producers. Uvicorn handles SIGTERM/SIGINT; lifespan stops readiness, signals producers and closes the control listener, with the existing two-second shared producer join budget. Partial startup is cleaned up; a runtime cannot start twice or spawn replacements for blocked producers. HTTP request draining has a two-second budget. A protocol-level, absolute one-second initial-header deadline closes silent or trickling clients before ASGI parsing, preventing incomplete connections from indefinitely exhausting the concurrency limit; it is cancelled on parsed headers or disconnect. Every health response closes its connection. Docs/OpenAPI, proxy-header trust and access-path logging are disabled. Health responses remain cache-only, exact-path, GET-only, `no-store`; query strings and encoded aliases are refused. No listener is published and no control protocol, bearer, dispatch, admission or ledger semantics change. **TASK-1169 FastAPI control slice (implemented in source, not activated):** the `8651` control listener (`ops/control_http.py`) is now a FastAPI app with exactly two typed routes, `POST /v1/ops/prepare` and `GET /v1/ops/executors`, on its own private socket and Uvicorn loop thread. It never shares the health host's loop or slots, and signals stay with the main-thread health host. The wire contract is unchanged: bearer check before any body read, 411/413/400, 401, 404/405 protocol bodies, 503 `ops_busy` with `Retry-After`, 504 `ops_timeout`, 500 `ops_internal`, protocol field, `no-store` and `Connection: close`. The prepare concurrency semaphore and the per-request worker-thread wall budget are kept as they were; the abandoned call still holds its slot until it returns. Slow clients are bounded by an absolute 5s initial-header deadline (the health host's protocol, at the old 5s read bound) and a 5s body deadline (408, no slot taken), with at most 32 connections. Query strings, encoded aliases, trailing slashes, docs and OpenAPI are refused, and there is no access-path logging or proxy trust. Port, protocol, client, bearer derivation, compose and catalog are unchanged; activation belongs to TASK-1173.

The health routes remain:

| Current route | Meaning |
| --- | --- |
| `/health/live` | The HTTP server answers; not proof the reconciliation loop progresses. |
| `/health/ready` | Four gating checks (database, Docker, shared executor prerequisites, loop progress) remain fresh/ready; the `ops.executor.modes` object is **non-gating** per-mode diagnostics. A 200 does not mean every enabled operation can run: process mode may lack its immutable interpreter image pin, or an isolated operation may lack a capability label. Admission/preflight still refuses those jobs explicitly. The process-mode sample comes from the actual reconciler's child table (live count, DB slot bound, oldest running age); isolated mode reports pinned worker image availability and first-party capability label booleans. Unknown/stale samples show null metrics, never cached live counts. |
| `/v1/health/containers` | Versioned, sanitized inventory keyed by exact Docker names. |
| `/v1/health/services[/{id}]` | Every observed Compose service plus every service enabled catalog ops target, by `<project>.<service>`, with opaque instance ids and capabilities (`[]` when no op targets it, `null` when the catalog is stale) (TASK-1169 S1c/S2a; contract in `OPERATIONS_RUNNER.md`). |

The HTTP handlers do not probe Docker/DB or dispatch jobs. This separation keeps health HTTP responsive during blocked background calls. The collection-start timestamp and freshness limits prevent old success from remaining healthy forever.

### Recovery behavior today

Live (verified 2026-11-03, TASK-1173): `llm-bawt-ops-reconciler-1` runs `restart: unless-stopped` with the Compose readiness healthcheck active (`GET 127.0.0.1:8650/health/ready`), status healthy; the `8651` control listener is up and unpublished. Process exit triggers a Docker restart. A blocked background thread fails readiness and marks the container unhealthy. **TASK-1991:** a watchdog thread exits the process (code 70) when any producer (database, docker, executor, reconciler loop) has published nothing for `LLM_BAWT_OPS_WEDGE_SECONDS` (default 180; 0 disables), and the restart policy brings it back. A producer that keeps reporting `down` (for example the database is unreachable) is an outage, not a wedge, and does not trigger it. Isolated workers and receipts survive the restart; process-mode children end with the process and are reconciled as after any reconciler restart.

### Direct Docker consumers (live state after TASK-1173)

| Consumer | Current dependency | Migration |
| --- | --- | --- |
| llm-bawt app | Admission-only: prepares through the ops control channel; no executor, dispatch or reconciliation. No socket mount and no worker env | Live: llm-bawt `feba0310`, app recreated 2026-11-03; socket connect inside the container fails |
| Ops collectors/execution | One Docker SDK construction path: `DockerExecutor` (`executor.py`), reused by `health_collector.docker_client` | Done in source, TASK-1169 |
| Hono dev/prod | None in code. `/tools/docker` reads `/api/system/ops/services` (ops health) and runs actions through the catalog Run dialog; `/api/docker/containers*` return 410; `dockerApi.ts` is deleted. No socket mount on `frontend` or `frontend-prod` | Live: TASK-1170 code; mounts removed in BawtHub `b19ba64f`, containers recreated 2026-11-03 |
| Hono repository browsing | `repos.ts` takes the host path from `REPOS_HOST_ROOT`, never Docker | Done in source, BawtHub `39d2ee30` (TASK-1170) |
| Voice backend | Fenced llm-bawt restart client (`tts/restart.py`) using the existing service token; direct shell/Docker restart helper removed | TASK-1171 source published; backend observed token-configured and socket-free on 2026-11-01. Catalog enablement and actual Restart acceptance remain pending below |
| GPU handoff in app | Reads STT/TTS from the reconciler's `/v1/health/services/bawthub.{stt,tts}` (`LLM_BAWT_OPS_HEALTH_URL`); refuses on anything but one fresh, typed instance. No Docker client. Mutations use consented ops jobs | Done in source, TASK-1171 |
| Traefik | File provider reading `bawthub/deploy/traefik/dynamic/routes.yml` (watched, `:ro`); no socket, no `traefik.*` labels | Live: TASK-1172 cutover (BawtHub `e31c6421`) with Traefik recreated 2026-11-03; `bawthub.reconcile-prod-routing` disabled |

#### TASK-1171: source, observed readiness and remaining acceptance

The original source migrations are published; readiness is not the same as a completed restart. The following is **dated read-only evidence from 2026-11-01 04:01–04:04Z**, with the disabled catalog row rechecked at 04:17Z. Detailed receipts are in TASK-1171 step `cmvxb7yvd018s0mmrk4mc9rkj`; these observations are not perpetual health guarantees or deployment instructions for another installation.

| Change | Published source / observed readiness | Remaining work |
| --- | --- | --- |
| `bawthub.restart-tts`: exact `{action:restart, compose_project:bawthub, compose_service:tts}`, no args, high risk, isolated Docker execution | Existing row is **disabled, version 1**; no seed/bootstrap needed | Only an explicitly authorized enable of this exact unchanged row; preserve existing policy, target, argument, isolation and quiesce settings. Enablement alone does not dispatch anything |
| GPU voice guard (`ops/gpu_guard.py`) and fenced TTS restart | Creation and dispatch checks require the scoped `GpuHandoffFence`; app exposes `POST`/`GET /v1/media/local-video/handoff/tts-restart[/{request_id}]`. Fence and job commit atomically | Keep existing policy and creation/dispatch checks; generic ops Run cannot supply this fence. A service bearer is not human Restart consent |
| BawtHub TTS admin restart client and socket removal | Published `tts/restart.py` uses the fenced route; shell helper removed. Observed healthy backend has the token mount, no Docker socket or retired restart environment; capabilities report `restart_supported=true` and `restart_configured=true` | Actual separately consented Restart acceptance. No backend image/recreate or socket-removal action is presently evidenced necessary for this flow; configured capabilities do not prove the catalog is enabled or a restart succeeded |
| `bawthub.reconcile-backend-config` | Row is disabled, but the observed backend is already socket-free | Not a prerequisite to this Restart flow; do not patch/enable/recreate it to repeat an achieved state |
| GPU handoff observations and start evidence | App reads the shared service contract without a Docker client; ops ready checks pass and observations show exactly one fresh healthy STT and TTS instance, with integer `restartCount=0` | No observation-driven restart/rebuild is needed solely to repeat this evidence. Reject stale/unavailable or unsupported replica/resource arrangements; broader GPU handoff acceptance still needs its own consent |
| Services capability link to TTS Admin | Reviewed consumer delta `40aad4dc` routes only the exact no-argument `bawthub.tts` / `bawthub.restart-tts` restart capability to `/tools/tts`, not generic Run | Integration/publication and applicable frontend release remain tracked separately in TASK-1171; review is not deployment evidence |

At that observation point, GPU ownership was voice/idle with no open claims or pending work, voice admissions were idle/unparked, and the four Moshi start/stop operations were enabled. The existing worker and telemetry were ready. These are readiness observations, **not** live Restart, GPU handoff, inference or calibration acceptance. Do not blanket-block the already observed TTS path on every remaining TASK-1169 or TASK-1173 rollout item; their separately owned activation and final isolation evidence remain outside this bounded correction.

**Enable-only authority is still withheld.** ONE pinned proposal, `TASK-1171-catalog-enable-v1` (step `cmvxbm31h01ar0mmr78g5d3dy`), records the exact row, reversible enable/reversal and verification. The ops catalog has no enable/update operation; the existing catalog API is a manual alternative, not an `ops_run` approval card. Its enable endpoint has no expected-version/CAS check, so the proposal requires an explicitly coordinated short single-writer window plus fresh exact pins and revision verification, or no execution. Technical review and the `approve_ops` grant do not provide manual execution authority. Do not use ad hoc SQL, grants, seed/bootstrap, another operation or an invented approval card to enable it. The task artifact owns the specific pending request; this document does not grant it.

**Restart acceptance remains a separate genuine-human TTS Admin action.** Do not simulate operator consent through the service bearer or direct backend calls. Record the correlated request/job, terminal receipt, exactly one fresh running/healthy TTS instance with `startedAt > dispatched_at` and integer `restartCount == 0`, two consecutive `build_info` successes in the voice client, evidence-based fence release and unchanged STT. The status route can release a proven never-dispatched refusal without start evidence; a dispatched success or supported uncertain terminal outcome needs the fresh start evidence. Timeout/PID absence alone cannot release uncertainty, and an uncertain transport/5xx response (including 503) is not a definite refusal or permission to create a fresh restart. Disabling the catalog row prevents new admissions but cannot cancel an in-flight job or clear its fence. Ready-to-click is not restarted, UI acceptance or task completion.

After TASK-1173 only the ops reconciler and its per-job workers hold the Docker socket. A `:ro` socket mount does not restrict Docker API methods, so that boundary is the reconciler, not the mount flag. Agent bridges have no direct socket mount in the inspected deployment, but host SSH/operator privileges are a separate surface, not removed by this design.

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

**Verified topology and prepared state (TASK-1172, 2026-11-03).** Only the echo BawtHub stack uses Traefik: three routed services (`frontend` for dev.bawthub.com, `frontend-prod` for app.bawthub.com, `backend` as the host-agnostic `/api` catch-all with the `strip-api` middleware), no auth middleware, Traefik on `bawthub_default` only. Tenant installs publish ports directly and have no Traefik. Ops image deploys clone containers with their network aliases, so the Compose service names stay resolvable after a release.

- `bawthub/deploy/traefik/dynamic/routes.yml` is the one routing source: the same routers, priorities, middleware and upstreams by Compose service name. Until the cutover, a parity test keeps it identical to the live labels.
- `bawthub/deploy/traefik/check_routes.py` runs it in a disposable real Traefik with stub upstreams and no socket. It checks loading, the host/path/priority matrix, `/api` stripping, unbuffered SSE, a websocket upgrade, upstream recreation (Docker DNS follows the new IP), two replicas behind one name (requests succeed; spread is not promised), a stopped upstream (502, other hosts unaffected), and a broken watched edit (logged; last good routes keep serving).
- Behaviour changes at cutover: routing no longer follows container health, so a down or starting upstream answers 502 instead of Traefik's 404; and Traefik no longer needs a restart after a recreate. Scaling a service needs its replicas listed in the file, or an orchestrator Service/Ingress.
- The cutover commit must ship with Traefik's recreation: merged alone, a recreated `frontend`/`backend` would lose its labels while Traefik still reads them. The `bawthub.reconcile-prod-routing` ops action (TASK-1179) edits labels, so TASK-1173 disables it at the same time; route changes become reviewed edits to the routes file.

## 7. Bounded implementation and acceptance

| Task | Deliverable | Dependency |
| --- | --- | --- |
| TASK-1169 | FastAPI ops host; portable contracts; existing ledger/admission separation; adapter and bounded execution; recovery tests | Foundation |
| TASK-1170 | Frontend controls/health plus repo-path discovery migrated; direct helpers removed (done; socket mounts left to 1173) | 1169 |
| TASK-1171 | TTS restart and GPU observations use the shared contract | 1169 |
| TASK-1172 | Socket-free deployment routing using supported provider configuration (live via TASK-1173) | Independent of 1169 |
| TASK-1173 | Integrated activation, mount removal, healthcheck and final evidence (done 2026-11-03; receipts in the task) | 1169–1172 |

Start an architecture ratchet in TASK-1169: list current offenders with owning task IDs, reject new ones immediately, and remove exceptions as migrations land. Cover SDK imports, direct HTTP/CLI/shell helpers, socket/TCP configuration and parent-directory mounts across Compose and tenant artifacts. Separate legitimate adapter code and operator test tooling from product dependencies.

**Implemented (S1a):** `tests/test_docker_access_ratchet.py` scans `src/` and every `docker-compose*.yml`. `OPS_ADAPTER` and `OPS_COMPOSE` name the ops boundary; `OFFENDERS` listed each offender with its owning task; since TASK-1173 it is empty in both repos. A new offender fails, and so does an entry that no longer matches, so a migration deletes its own exception. BawtHub carries the twin for Hono, the voice backend and its compose files.

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

- `src/llm_bawt/ops/service.py`: admission, target preparation, dispatch and status reconciliation; `admission_only` limits the app's instance to admission and reads (TASK-1169 S3).
- `src/llm_bawt/ops/control_protocol.py`, `control_http.py`, `control_client.py`: the app/ops control channel and `build_admission_ops_service` (TASK-1169 S3); `control_http.py` is the FastAPI control host (implemented, not activated).
- `src/llm_bawt/ops/store.py`: durable ledger and claims.
- `src/llm_bawt/ops/executor.py`, `worker.py`, `image_deploy.py`: Docker execution and per-job worker lifecycle.
- `src/llm_bawt/ops/release_executor.py`: release execution/coordinator integration.
- `src/llm_bawt/ops/config_reconcile.py`, `app_config_reconcile.py`, `tagged_config_reconcile.py`: worker-side same-image Compose reconcile per target kind (frontend, Compose-built with source guard, public tagged image); see `docs/OPS_CONFIG_RECONCILE.md`.
- `src/llm_bawt/ops/plan_reconcile.py`, `plan_approval.py`, `probe_contract.py`: the cutover service plan (TASK-1476). The worker runs the ordered recreate, end probes and revert-all; the app binds the plan at approval; the probe contract is the one-line JSON schema shared with `service/cutover_probe.py` and `claude_code_bridge/cutover_probe.py`.
- `src/llm_bawt/ops/reconciler.py`, `health_runtime.py`, `health_http.py`, `health_collector.py`, `health_state.py`: FastAPI health host/lifespan and independent fixed reconciliation/collection producers (implemented, not activated); cached observation contracts unchanged.
- `src/llm_bawt/ops/approved_admission.py`: approved MCP `ops_run` admission and the shared `ops_run` receipt/error contract (TASK-1220).
- `src/llm_bawt/approval_mcp_store.py`: approval execution claims, plus the shared awaiting-execution predicate used by the claim, app recovery and admission.
- `src/llm_bawt/service/routes/approval_policies.py`, `service/approval_continuations.py`: approval decision recording and the continuation outbox, which reads ops results without reconciling them.
- `src/llm_bawt/service/routes/ops.py`: existing app HTTP surface.
- `src/llm_bawt/media/gpu_handoff_adapter.py`: GPU-service observation through the reconciler's `/v1/health/services` (TASK-1171).
- `docker-compose.ops.yml`: ops deployment and source healthcheck configuration.
- `bawthub/frontend/server/routes/ops-services.ts`, `lib/opsHealth.ts`, `routes/docker-containers.ts` (410 tombstones), `routes/repos.ts` (`REPOS_HOST_ROOT`): the frontend's ops-backed service view; no direct Docker consumer remains (paths relative to `frontend/server/`).
- `bawthub/deploy/traefik/dynamic/routes.yml`, `deploy/traefik/check_routes.py`: deployment-owned Traefik routes and their real-proxy check (TASK-1172).
- `bawthub/bawthub/main_websocket.py`, `bawthub/bawthub/tts/restart.py`: voice restart controller and fenced llm-bawt client; the direct `restart_tts_container.sh` helper is removed (TASK-1171).

### Decision record

- **2026-11-03 (TASK-1173 activation):** App recreated without socket/worker env (`feba0310`); Traefik switched to the file provider and `frontend`/`frontend-prod` sockets removed (`e31c6421`, `b19ba64f`); `bawthub.reconcile-prod-routing` disabled; image IDs unchanged. Verified live: `check_routes.py` PASS, dev/prod pages and APIs, HMR, SSE, and with Docker unreachable from app/frontends: service list, a catalog restart, speech synthesis, GPU handoff status and the release card render. Both ratchets have no remaining consumer offenders. Still open, not live: per-job worker containers and automatic restart of an unhealthy ops container.

- **2026-11-02 (TASK-1169 FastAPI health host):** Reconciled the published prepared-hold correction and existing process/adapter work without replay. Migrated only the dedicated health host and runtime ownership to FastAPI/Uvicorn lifespan; kept the separate bounded control transport, immutable wire contracts, fixed producers and execution/recovery semantics. No dependency, Compose, catalog, runtime, mount or activation change; TASK-1173 retains rollout.
- **2026-11-03 (TASK-1169 FastAPI control host):** Moved the `8651` control listener from the standard-library server to FastAPI/Uvicorn on its own socket and loop thread, keeping the wire contract, bearer, slots and wall budget unchanged and adding header/body deadlines. With this, the ops service's control, status and health endpoints are all on FastAPI in source. No dependency, Compose, catalog, runtime, mount or activation change; TASK-1173 retains rollout.
- **2026-11-03 (TASK-1170 reconciliation):** BawtHub frontend code has no direct Docker consumer: service view and actions go through ops, legacy container routes are 410, repo paths come from `REPOS_HOST_ROOT`. The BawtHub ratchet's only frontend offenders are the two unused socket mounts, now owned by TASK-1173.
- **2026-11-03 (TASK-1172 source prep):** Traefik routing moves to one file-provider routes file using Compose service names; no ops route discovery, proxy or sidecar. The routes file, parity test and real-proxy check are in source. A separate cutover commit (provider switch, labels and socket mount removed, ratchet entry dropped) is held for TASK-1173 to integrate alongside Traefik's recreation, together with disabling `bawthub.reconcile-prod-routing`. No live routing or service change.

- **2026-10-18:** Health monitoring fixed independently of full isolation; broader tasks remain queued.
- **2026-10-18:** review narrows scope: reuse ledger, portable service contracts, shared observations, routing owned by deployment, guardrails first. Follow-up found repository mount inspection; TASK-1170 explicitly covers it.
- **2026-10-18:** The operator asks for a proper FastAPI operations service and a repo design document maintained through implementation, then transitioned to skills upon completion. This document records that direction; no service/code rollout performed.
- **2026-10-21 (TASK-1220):** After the approval-time stale-import incident, the operator approved moving admission of approved MCP `ops_run` from the app into the ops reconciler. The coordinator's design review added these must-fixes: the admission deadline, recovery exclusion, the pending resolve contract, the shared contract, the hash check, no lazy imports, and per-row isolation. The operator set a 10-minute admission window, and chose to degrade and refuse (not crash) when MCP tool registration fails at startup. Implemented and tested; not yet deployed. This is the first step of the §3 ownership move ("approved job admission" to ops). Directly submitted jobs still dispatch in the app until TASK-1169.
- **2026-10-28 (TASK-1169 S3):** The ops-reconciler becomes the sole executor and the app admission-only (design r1 and v1.1, reviewed by Codex). The app prepares over an internal bearer-authenticated listener on `:8651`, with no local fallback. Job creation notifies the reconciler. The quiesce fence check moves to the reconciler's driver. Deviation from the design: the control bearer is derived from the master key, not the BawtHub service token, because the reconciler doesn't mount that token's override. Implemented, not deployed.
- **2026-10-28 (TASK-1476):** The llm-bawt cutover is one approval-gated plan job (`llm-bawt.cutover-plan`) over existing per-target reconcile rows, not a new executor. Each target keeps its own reconciler; the plan adds ordering, a credential-class check with an explicit `accept_loss`, end probes and revert-all. It ships disabled and needs a rebuilt ops worker (`service-plan-reconcile` label). Not activated.
- **2026-11-01 (TASK-1171):** Read-only reconciliation found the published fenced client, socket-free/token-configured voice backend, fenced routes and fresh shared voice observations already available; only the exact restart catalog row remains disabled v1. Removed stale seed/recreate prerequisites for this flow. The one enable-only proposal is pending actual manual authority and technical review; a later genuine-human fenced Restart and its correlated acceptance remain separate. No catalog mutation, restart, handoff, inference or calibration was performed for this readiness/documentation work.
