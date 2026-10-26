# Test suites (TASK-1189)

Use the Make targets below on a Unix runner with Python 3.12+, `uv`, and Docker
(for PostgreSQL). `make` and CI invoke the same `scripts/test_suite.py` runner.
It rejects concurrent suites on the same host and exits nonzero on failed
selection, missing PostgreSQL suites, PostgreSQL/local skips, or a slow unit.
Do not use `pytest -m ""`: that bypasses the external-target boundary.

| Command | Selection | Requirements |
| --- | --- | --- |
| `make test` | Hermetic unit tests; **setup + call + teardown <1 s** per case, <1.5 s for `real_process` cases (`LLM_BAWT_UNIT_TIME_BUDGET_SECONDS`, `LLM_BAWT_REAL_PROCESS_TIME_BUDGET_SECONDS`, see below) | Python only |
| `make test-pg-acceptance` | All manifest-listed private SQL tests, zero skips | Docker with pinned pgvector image |
| `make test-offline-integration` | Real local HTTP/thread lifecycle and real SDK wire integration, zero skips | Loopback/in-memory transport |
| `make test-sdk-probe` | Optional installed SDK subprocess against fake localhost upstream, zero skips | Installed SDK CLI; never operational bot/provider |
| `make test-integration` | Private SQL, then local HTTP/thread tests | Docker + loopback |
| `make test-all` | **Every offline test exactly once**: unit, SQL, local | Docker + loopback |
| `make test-cov` | Hermetic coverage (not a latency gate) | Python only |
| `make test-profile` | Detailed hermetic setup/call/teardown timings | Python only |
| `make test-target NODES=tests/test_file.py::test_name` | Explicit offline target, no skips | Relevant private fixtures |
| `make test-snapshot PATHS="…" AUTHOR=<bot>` / `PATCHES="…"` | `test-all` (or `test-target` with `NODES`) on exactly BASE + patches in a throwaway worktree | Test host only; see below |
| `make test-stress [STRESS_BASE=origin/master] [REPEAT=10] [FILES="…"]` | Optional diagnostic: test files added or modified since `STRESS_BASE` forked, `REPEAT` times under CI's conditions | Linux, 2+ CPUs; see below |
| `make gate REV=<sha> AUTHOR=<bot>` / `make gate-receipt REV=<sha>` | The llm-bawt SHIP gate (`test-all` on exactly `REV`) / its newest receipt | Test host only; see below |
| `make test-sweep` | One round of every offline test under CI's conditions | Docker + loopback |
| `make test-service` / `make test-paid` | **Refuse** until a disposable service/DB and credential scope are provisioned and verified | Never run against live :8642 |

## Unit time budget (TASK-1312)

Every unit case must finish setup + call + teardown within the budget, 1 s by
default, or the run fails. That default is the standard every commit meets on
echo through `make test-snapshot` before it is pushed. GitHub's hosted runner
has fewer, slower cores, cold caches and a virtual disk, so cases that take
0.8 s here by fixed waits exceed 1 s there with no regression;
`.github/workflows/tests.yml` sets `LLM_BAWT_UNIT_TIME_BUDGET_SECONDS=3` so CI
still fails on hangs and new sleeps without failing on runner noise. The value
must be a number in (0, 60]; anything else fails closed. Keep echo at the
default.

Cases marked `@pytest.mark.real_process(reason=...)` spawn a real binary
(bash, node, git), whose cold start on the hosted runner is outside the test's
control: a node launcher that takes 0.1 s on echo once took 3.3 s there. They
are judged on their own budget, `LLM_BAWT_REAL_PROCESS_TIME_BUDGET_SECONDS`:
1.5 s by default (about twice the slowest such case on echo) and 7 s in CI
(about twice the slowest seen there). It is enforced like the unit budget, and
the failure names which budget tripped. Every gated run prints its slowest
real_process case, so the CI value can be revisited from its own logs. Fixtures that need a file-backed SQLite engine use
`tests/sqlite_support.py::fast_sqlite_engine`, which turns off per-commit fsync
(`PRAGMA synchronous = OFF`); that is the dominant cost of such fixtures on a
virtual disk and no test relies on durability across a power loss.

## CI's conditions on echo: the optional stress diagnostic (TASK-1457)

echo is fast enough that a timing-dependent test passes there and fails on
GitHub's 4-vCPU runner. `make test-stress` reruns the tests a push adds or
changes the way CI runs them: this process, its pytest workers and the private
PostgreSQL container pinned to 2 CPUs (the container through
`docker run --cpuset-cpus`, from `LLM_BAWT_TEST_PG_CPUSET`), with CI's worker
count and unit budget read from the `make test-all` step's `env` in
`.github/workflows/tests.yml`, so the two can't drift. Each changed file runs in
the partition and with the flags `make test-all` gives it; a changed
integration test that isn't in a gate suite fails the step. Any failure in any
of the `REPEAT` rounds (default 10) fails it, naming the round and partition.
It is a diagnostic for chasing a suspected flake, not a push requirement:
repeated rounds only find a timing dependence by chance, and they cost minutes.

**SHIP gate.** An llm-bawt SHIP needs a PASSED `make gate REV=<tip> AUTHOR=<bot>`
receipt for the exact commit being pushed. It runs `make test-all` on exactly
`REV` in a throwaway worktree. `make gate-receipt REV=<tip>` prints the newest
such receipt and exits 0 only if it passed (1 failed, 2 none). A `PATCHES`
trial never counts, because it tested other bytes.

## Timing patterns the guard rejects (TASK-1457)

`tests/test_timing_guard.py` (scanner: `tests/timing_guard.py`) fails on the
two patterns behind most CI-only failures:

- **Hand-rolled `pg_stat_activity` polling.** PostgreSQL snapshots that view
  once per transaction, so a loop polling inside one keeps its first view. Wait
  for a lock with `tests/pg_lock_wait.py::wait_blocked_by(engine, holder_pid,
  worker)`.
- **A fixed upper bound on wall-clock time** (`assert elapsed < 0.5`,
  `assert time.monotonic() < deadline`). Prove the behaviour instead: an event
  or barrier, the call's own timeout result, a count of the work done, or, for
  asyncio code, `tests/virtual_clock.py::run`, whose clock jumps to the next
  timer so elapsed time is exact and takes no real time (`follow=[module]` makes
  a module's own `time` reads follow it).

A justified exception carries `# timing-ok: <reason>` on the statement or the
line above. The reason is what a reviewer judges: typically a hang guard on a
real thread, process or PostgreSQL lock whose bound is far from both the normal
cost and the failure it rules out.

## Exact-bytes gate: `make test-snapshot` (TASK-1137)

The shared checkout carries other bots' uncommitted work, so `make test-all`
there tests a mix. `make test-snapshot` runs the same suite on a detached git
worktree outside `~/dev` (`~/.cache/llm-bawt-test-snapshot/wt/`) holding only
BASE plus the given patches, then removes it on every exit path, including
failure and interrupt. Run it on echo
(`ssh nick@172.18.0.1 'cd ~/dev/llm-bawt && make test-snapshot …'`); it refuses
inside a container.

| Variable | Meaning |
| --- | --- |
| `PATHS="a.py b.py"` + `AUTHOR=<bot>` | Build the patch from this checkout's working tree for exactly these files (new, modified or deleted) vs HEAD. BASE must be HEAD. |
| `PATCHES="/abs/one.patch /abs/two.patch"` | Apply these patch files in order with `git apply --index` (no fuzz; any reject fails the run). |
| `BASE=<commit>` | Default `HEAD`. |
| `NODES="tests/…"` / `SUITE=offline\|target\|gate\|sweep` | `NODES` alone selects `test-target`; the default is `test-all`. `gate` is `test-all` recorded as a SHIP receipt; `sweep` is `test-sweep`. |
| `LOCK_WAIT=<s>` | How long to wait for another suite to finish (default 1800). The runner lock is never bypassed. |

**Ownership.** `PATHS` refuses a file unless its uncommitted bytes provably
belong to `AUTHOR`: they must chain back to HEAD through only `AUTHOR`'s
recorded edits (`turn_changed_files`, via `pgq`), every link an exact sha256
match. Line-level attribution (`commit-recon`) is not accepted as proof: it can
credit you with identical text somewhere else. Each file is read once; the
ownership proof and the patch use those same bytes, so a write that lands
during the run is simply not in the patch. Symlinks are refused (use
`PATCHES`). Edit tracked files only
with Edit/Write: a turn's record of a file runs from the bytes before its first
Edit/Write in that turn to the bytes after its last, so a Bash edit outside that
span can't be proven. If you edited a file through Bash, restore it to HEAD and
redo the change with Edit/Write in a new turn. When another bot has uncommitted
changes in the same file, `PATHS` refuses; sequence the work on the task.
`AUTHOR` is self-declared and recorded in the receipt for the reviewer. Known
limit: another bot's *unrecorded* edit made inside your turn, between your
edits, is absorbed into your turn's snapshot.

**Receipt.** Every run writes `runs/<run_id>/receipt.json` (status, exit, base,
each patch's sha256 and pinned copy, `diff_sha256`, changed files, runner and
pytest summary lines, log path) and prints the same summary. `diff_sha256` is
sha256 of `git diff BASE` in the snapshot: the pin reviewers use. Once committed
on BASE it equals sha256 of `git diff <commit>^ <commit>`. It is not the patch
file's sha256 (patch text carries its own headers); the receipt records both.
An interrupted or finished suite has its whole process group stopped, so no
descendant outlives the run holding the runner lock. The run id and
receipt path print first, so a caller that disconnects can read the receipt
later; the run continues without stdout. Exit codes: 0 passed, 1 failed /
patch rejected / error, 2 refused before testing, 130 interrupted. A patch that
changes the runner itself (`Makefile`, `scripts/test_suite.py`, `tests/conftest.py`,
…) is flagged in the receipt: that gate ran the patched runner. Run directories
are pruned after 14 days, so commit (or re-gate) a reviewed patch before then.

**Flow.**
1. Edit in the shared checkout.
2. `make test-snapshot PATHS="…" AUTHOR=<bot>` → receipt plus
   `runs/<run_id>/01-paths.patch`.
3. The reviewer reruns with `PATCHES="<that patch>"` on the same BASE, adding a
   mutant patch to prove a test fails without the fix.
4. Commit exactly the pinned patch: `git apply --cached <patch>`, then
   `git commit` with **no pathspec**, so another bot's unstaged hunks in the same
   files stay out of the commit.

Only `.git/worktrees/<run_id>` metadata (and new blob objects) are written in
the canonical repository; the working tree, index, HEAD, branches and stash
are untouched. A snapshot left by a killed run is removed by the next run.

The required offline gate compares pytest's safe collection with the union of
three disjoint suites and reports selected counts before executing any test.
`tests/suite_manifest.py` owns the SQL selectors used by both pytest's
`--require-pg-acceptance` guard and the runner. Unknown new integration tests
make the gate fail until assigned to a suite. `service`, `llm_call`, and
`external` are excluded; optional `sdk_probe` cases run with `make test-sdk-probe`
using fake local providers and never run implicitly. The unit gate now requires
zero skips; its former ten skips were opt-in SDK probes misclassified as units.
Raw pytest still has its historical hermetic default but is not a full gate.

The PostgreSQL fixture runs one pinned, networkless Docker container per pytest
session (one per xdist worker if workers are used), bound only to a
pytest-owned Unix socket with **no published port**. Each module receives its
own randomly named `_test` database; function-level schema/DB tests retain
independent transactions/databases. Only the `postgres` maintenance database
is used for CREATE/DROP DATABASE and connection identity checks. The DBAPI
connection guard and test-specific configuration remain mandatory; module DBs
are force-dropped after engine disposal, then the container stops. No test may
use an operational service or an inherited PostgreSQL address.

Current external-only tests are the CLI module (its subprocess targets are
hardcoded for an unverified service and can mutate real bot data) and the
read-only live-bot timeline parity class. Their `service`/`external` markers
make them explicit omissions from offline coverage. A paid-LLM target must
also be provisioned with separately scoped credentials before that suite can
be enabled. An explicit command that fails safely is preferable to one that
appears to pass by skipping every case or that contacts production.

For profiling, compare `make test-profile` and the unit gate on a quiet runner.
The initial default-suite sample (3,830 passed, 10 skipped, 404 deselected,
156.97 s) had 60 cases above 1 s when fixture time was included. A later
isolated host sample (3,832 passed, 10 skipped, 404 deselected, 46.93 s) had
only three: two real HTTP shutdowns (~2 s each, correctly classified as
local integration) and one delivery poll (~1 s, now driven by an event).
A real OpenAI-SDK-over-mock-HTTP wire test sometimes exceeded 1 s in later
runs; it is also local integration, not a pure unit.
Transient load should be measured again rather than treating a slow run as
proof that each test's own work takes a second.
