# Exact-commit batch gate: inert contracts (TASK-1865)

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

## Before any trusted dispatch wiring

The recorded security design r2 approval is
`cmvwjnu39003e0mmkdlkmdor8`, rechecking r1
`cmvwjipsd003a0mmke26d63tv`. Its two concrete findings were resolved at the
**design** level, not by this inert contract:

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

Neither rule is wired or claimed enforced for gates yet. The public Docker
spec validator still rejects `batch_gate`; a synthetic internally-created
ledger fixture is not an enabled gate executor.

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
