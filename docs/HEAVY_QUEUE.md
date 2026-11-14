# Non-SHIP heavy-check queue

`scripts/heavy-queue` is a host script, not a bot or a service. It runs a generic
argv command in one capped, disposable container and enqueues a durable callback
to the requester through the existing inter-bot delivery endpoint. It does not
create a SHIP receipt, change gate evidence, or broaden TASK-1865's fixed-workload
ops action into an arbitrary command executor.

## Submit

Expose this checkout's `scripts/heavy-queue` on the test host's PATH, or call its
absolute path. Use the host-visible task checkout path, not a bridge-only alias.
Supply a locally prepared toolchain image with the dependencies for the check:
the queue neither installs dependencies nor borrows another task's mutable
environment. All paths, images, origins and requester IDs are deployment input.
The prepared image must include `/bin/sh` and GNU `/usr/bin/timeout`.

```sh
heavy-queue submit --cwd /work/task/frontend --image prepared-checks:revision \
  --notify-url http://llm-service:8642 --notify-bot requester --task-id TASK-123 \
  -- pnpm test
heavy-queue submit --cwd /work/task/frontend --image prepared-checks:revision \
  --notify-url http://llm-service:8642 --notify-bot requester --task-id TASK-123 \
  -- pnpm exec tsc --noEmit
```

The JSON response names the job directory and worker PID. The detached script
continues after SSH disconnect. `heavy-queue status <job-directory>` reads the
durable receipt. `run.log` contains check output; `worker.log` contains launcher
diagnostics. The callback contains exit status and paths, **not arbitrary test
output**. Enqueue retries with a stable idempotency key until accepted; HTTP calls
have a 15-second timeout, with backoff capped at 60 seconds. If the host/worker
is killed, inspect the receipt/container and use `heavy-queue notify <directory>`
to retry notification. Never replay a job automatically after an uncertain exit.

## Containment and admission

- One heavy host job, FIFO within class; **gate > queue**, without preemption.
- One shared flock: `$TMPDIR/llm-bawt-test-suite.lock`, default `/tmp`.
  Every participant must use the same host TMPDIR. Existing llm-bawt runners
  retain the lock and their `LLM_BAWT_TEST_LOCK_WAIT_SECONDS` semantics.
- Locked tickets in the sibling `.tickets` directory establish gate-pending
  priority. Queue admission reserves its FIFO ticket before returning to the
  submitter; inherited lock ownership survives the submit process exiting.
  Dead/unlocked tickets do not hold capacity.
- Queue waits for `MemAvailable >= 20 GiB` before acquiring the heavy slot.
  Low host memory waits, rather than launching optimistically.
- Each job uses `--init`, **16 GiB RAM, 16 GiB memory+swap (no swap), 8 CPUs,
  pids-limit 4096**, no restart policy, a separate cgroup, and no network.
- A **30-minute wall deadline** includes create/start/command execution.
  Deadline/interruption kills the actual named container, not just its attach
  process. A second, container-owned timeout survives the host worker dying.
  Bounded Docker cleanup follows before the host slot is released. Gate wait
  defaults to 2700 seconds, covering a maximum-length queue job plus cleanup.
- Only explicitly named task directories are mounted, at identical host/container
  paths. No implicit home, Docker socket, provider credentials, DB environment
  or application settings enter the job.
- A normally exiting shell records cgroup-v2 `memory.peak`; the receipt reports
  measured bytes, not an invented RSS estimate. OOM/forcibly killed jobs may
  have no peak sample and say so.

**Private-PG limitation:** the exhaustive llm-bawt suite's private PostgreSQL
fixtures need host Docker. Do not mount that socket into a queue container.
The integrator's host `make gate` is the only exhaustive private-PG path (the
typed batch-gate operation is retired). The frontend PG gate likewise retains its
own throwaway-DB provisioning. The queue never repoints a profile or runs a
migration against a shared database.

## What the queue can and cannot run

The queue runs prepared, self-contained checks: focused or broad test subsets,
frontend tests, project-wide typechecks and lint, with dependencies already in
the image or task-local, and nothing that needs the network or an external
service. It is not a full-unit or full-gate environment. The stock house image
in particular lacks the frozen project lock and memory extra, the offline
tokenizer cache, a local `redis-server` and a worktree's Git common directory,
so it cannot support the full llm-bawt unit partition. A full run there is
dominated by failures from those missing prerequisites, which masks whatever
else fails, so it is neither a pass nor a usable product verdict. Do not patch
images per task to chase parity.

When an owner genuinely needs a full pre-check, the integrator runs one solo
host `make gate` on the reviewed tip. A passing receipt for that exact SHA
(`make gate-receipt REV=<sha>`) is reused for publication; nobody reruns the full
suite for the same SHA. A changed integration tip needs its own gate, and a
queue or base receipt never stands in for it.

## Gate integration and failure recovery

SHIP gates retain their commands, snapshots, throwaway PostgreSQL, receipt format
and publication checks. Only host-slot admission changes:

- llm-bawt `scripts/test_suite.py` takes a gate-class ticket around its existing
  legacy flock.
- BawtHub `scripts/ci_gate.py` takes the same host slot before running its unchanged
  gate; receipt lookup does not acquire a slot. Its small standalone adapter
  implements the same ticket protocol without a cross-repository import.

Do not activate either host entrypoint or start queue work while an old unreviewed
gate is running. Review the two-repository contract first and coordinate the
initial cutover with the integrator.

Container removal requires a **successful empty Docker inventory**. A daemon
failure is not absence. Before Docker create, `.tickets/active.json` durably names
the workload; only positive exact-container removal clears it. SIGKILL/OOM of the
Python worker releases its flock but leaves this marker, so both gate adapters
refuse admission rather than overlapping a surviving daemon workload.
Uncertain cleanup also writes `.tickets/hold`, fails the
receipt and stops subsequent queue **and gate** admission. The callback identifies
the job; admission refusals identify the marker, container and recovery step. An operator must
reconcile that exact container, confirm its absence, then remove that hold file.
For an orphan, remove its active marker too after confirming absence. Do not
clear a marker merely to get tests moving.

## Bridge enforcement

Before dependency setup, Make full suites and mypy refuse with queue guidance.
The Python suite runner repeats admission; root pytest config catches direct
broad `pytest` runs, including an expanded whole-suite list. Explicit focused
files/subnodes remain local.

Frontend strict/PG/package typecheck entrypoints guard before workers or DB setup.
The bridge image's Node preload discovers the repository's
`scripts/heavy-check-guard.cjs` policy from cwd/CLI ancestors, covering direct
`tsx`, Vitest, Node test discovery and `tsc` calls too. This is coordination, not
a security boundary: it does not attempt to stop a user deliberately removing
startup policy. Hosts and non-bridge containers are unchanged. An explicit queue
flag cannot override a bridge marker.

Node full-suite lists are compared with the repository's test inventory; no
arbitrary "maximum focused files" rule is introduced. A real focused filename is
required; option values, globs, directory discovery and appended filenames after
`pnpm test` do not disguise a full run. Project-wide typechecks are heavy even
when a previous instruction called `tsc --noEmit` "focused."
