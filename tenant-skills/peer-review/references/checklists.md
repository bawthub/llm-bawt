# Peer review checklists

Use only the sections that apply. Each item is a question to answer with
evidence (file:line, query, test), not a box to tick.

## High-risk areas (Tier 2 only when the change can break these paths)

**Concurrency, leases, idempotency**
- Every check-then-act: can the state change between the check and the act,
  including across an `await`, a lease expiry or a process restart? Deadlines
  and ownership have to be enforced in the same transaction as the side effect
  (TASK-1220 r2: an expiry check made before a fenced insert let a stalled
  admitter create a job after the cutoff).
- Two claimants, one stalled claimant, a crash after the side effect but
  before completion: are all three tested?
- Idempotency keys: does a replay return the original result instead of doing
  the work again?

**Live-tree code loading**
- llm-bawt source is bind-mounted into running containers (`llm-bawt` skill,
  "Deploying code changes"). A function-level import on a request, approval or
  continuation path loads whatever is on disk *now*, not what the process
  started with (TASK-1220 incident). New lazy imports on those paths are a
  Blocker unless justified.

**Approvals and trust boundaries**
- Can an agent reach the new path directly and skip approval? Any surface an
  agent can call, the MCP server included, counts.
- Is caller identity taken from trusted request context, never from tool
  arguments?
- Is the approved invocation integrity-checked before execution? Is a denial
  ever treated as permission for a fallback?

**Ops and service actions**
- Mutations go through the ops catalog (`ops_run`), never manual Bash/SSH.
- Queued, approved and completed are distinct states. Does the code and its UI
  say which one actually happened?

**Prod data writes, backfills, migrations**
- Is there a dry-run count first and a reviewed write set?
- Updates idempotent (`WHERE` still unset), an audit table written in the same
  transaction, reversible by batch id?
- Protected bots' partitions excluded. DB access only through `pgq`; no inline secrets.
- Migrations: reversible? Lock duration on big tables? Batched backfill? A
  verification query afterwards that should return zero rows?
- Irreversible or destructive writes are the user's call, whatever the review
  verdict.

**Bridge and turn lifecycle**
- Restarting a `*-bridge` kills live agent sessions. Is a restart required to
  ship this, and has it been requested from the user?
- Timers, watchdogs and stop paths: can anything outlive its cap, or kill a
  healthy sibling?

## Platform rules (any tier)

- **Prompts:** all LLM instruction text comes from the prompt registry. No
  string-literal prompts in Python or in BawtHub `.ts`/`.tsx`.
- **No hardcoded model, provider, bot or endpoint** as a feature preference;
  use the DB-backed settings and catalog.
- **No bot-slug literals in code; behaviour comes from bot tags/data.** A
  `slug == "<bot>"` check or a set of slugs is a Blocker; read the tag resolver
  (llm-bawt `bot_tags`, BawtHub `src/lib/botTags.ts`) instead. The guard tests
  (llm-bawt `tests/test_no_bot_slug_literals.py`, BawtHub
  `src/lib/noBotSlugLiterals.test.ts`) pass without growing their allowlists
  (TASK-1329).
- **Duplication:** is the new rule a second copy of logic that exists
  elsewhere? Shared classification and contract logic belongs in one neutral
  module that every consumer imports (TASK-1220 M4).
- **Tests:** run through the Make targets. New PG/integration test files are
  registered in `tests/suite_manifest.py`. The regression test fails without
  the fix.
- **BawtHub:** both `tsc` checks pass. Every new backend state has a
  deliberate UI rendering, not a fallback error string. Check visible changes
  with a Playwright screenshot on dev HMR. A UI review includes a split-screen
  screenshot (chat pinned, 1440 viewport).
- **Health:** does a new failure mode show up in health checks, or fail
  silently?
- **Docs and skills:** architecture docs and skills that describe the old
  behavior are updated in the same change.
- **Protected:** no reads, writes or tests against protected bots. No secrets in code,
  logs, commands or the review body.

## Timing (any diff touching tests/)

A test that passes because the machine was fast is a Blocker, even when the
gate is green (TASK-1457). Reject:
1. `sleep`/`asyncio.sleep` used to let background work get there. Instead,
   wait on an Event, a future or the result itself.
2. A timeout under ~5s used for anything but a hang guard. Instead, use a
   generous hang guard, and make the asserted outcome independent of it.
3. An assertion on elapsed time. Instead, use the virtual clock
   (`tests/virtual_clock.py`) and assert on `loop.time()`, or assert on order.
4. A real subprocess or network call in a unit test under a production
   deadline. Instead, fake the boundary, or move the test to integration with
   a hang-guard-only timeout.
5. A thread wait that can time out silently: an `Event.wait(t)` whose bool
   is ignored, or a `join(t)` with no `is_alive()` check after it. Instead,
   `assert event.wait(5)`, and `thread.join(5); assert not thread.is_alive()`.
   (`Future.result(t)` raises on timeout, so it is already checked.)
6. Hand-rolled `pg_stat_activity`/lock polling. Instead, use
   `tests.pg_lock_wait.wait_blocked_by`.

## Reward-hacking checks (agent-written code)

Look in the diff for:
- Tests deleted, skipped, marked xfail, or with loosened assertions.
- Expected values rewritten to match new output. This is legitimate only when
  it follows an explicit ruling, which the request must cite (e.g. TASK-1210
  ranking change).
- Mocks or fakes widened until they hide the real path; special cases for the
  exact inputs the test uses.
- Timeouts, budgets or gate thresholds raised; checks moved out of the gate.
- Errors swallowed so a test passes (`except Exception: pass`, defaults on
  failure).
- A receipt saying "all tests pass" with no command or count. Ask for the
  command and its result.

## Design review questions

1. What is broken or missing today? Show it traced to file:line, not from
   memory.
2. Goals and non-goals: what plausible thing is deliberately left out?
3. What alternatives were considered, and what does each cost? (One real
   alternative at minimum.)
4. Does it fit the recorded architecture and ownership (architecture doc,
   task family)? Is it a first slice of planned work or a fork from it?
5. Premortem: assume it failed in production a week from now. What is the
   most likely cause? Cover dependency down, crash mid-operation, mixed
   versions during rollout, and a stale or replayed request.
6. How will we see it failing (health, logs, UI)?
7. Rollout and rollback: what needs a restart, migration or backfill, and in
   what order? Can it be undone?
8. Which decisions belong to the user (scope, product, irreversible)? List them
   explicitly with a recommended default. Don't decide them in review.
