# Task-qualified worktrees

`bawt-wt` resolves a task by repository, owner and task branch. For new tasks it
uses `<dev-root>/.worktrees/.wt-<repo>-<bot>-TASK-N` with branch
`<bot>/TASK-N-<slug>`. Canonical source repositories stay directly under the dev
root. The namespace must be a real directory, not a symlink. Use the same
absolute dev root on host and bridge; symlinked dev roots resolve to their real
path before Git records the worktree pointer.

## Rollout boundary

The helper source, installed helper and application hook rollout are separate.
The canonical TASK-1881/1882/1883 guards validate the worktree basename and
owner/task branch, not its parent depth; common-dir hook dispatch and TASK-1884
Repos discovery already support registered nested trees. New placement does
not bypass or require weakening those guards. Keep the published/live hook
checks below; disposable fixture success is not publication evidence.

Even after helper rollout, creating a new task-qualified path requires a fresh
successful fetch and the exact capability line in the published executable
canonical `.githooks/pre-commit`:

```sh
# bawt-wt-task-path-v1: legacy and task-qualified owner paths are guarded.
```

The canonical checkout's executable hook must also contain it, since installed
common-dir hook stubs dispatch there. A marker only on a local/WIP branch, an
old remote-tracking ref, or hook-installer success is not enough. A repository
without support refuses creation; do not fake the marker, bypass hooks, hand
create an alternate tree or lower the gate. Coordinate hook publication and
activation with its owner.

## Existing trees and task isolation

- Resolve the task branch in Git's registered worktrees first, including legacy
  `.wt-<repo>-<bot>` trees, older root-level task-qualified trees and registered
  parked legacy paths inside the dev root. Detached task-qualified registrations
  retain their task identity; duplicate task paths refuse rather than allocating
  a second tree. The path owner must match the requested owner and branch prefix; a
  task suffix, when present, must match the requested/branch task.
- Resume a clean matching tree in place. Dirty same-task trees remain intact:
  `status` reports them; `start`, `sync` and `done` refuse mutation. Continue
  editing the existing owned WIP without moving or recreating it.
- A new task gets its own path and cannot switch a legacy or sibling task's
  branch. Dirty/unmerged sibling trees are not touched or tested for merge.
  Detached legacy trees stay untouched. A detached task-qualified path may be
  resumed only for its original task, clean and with its detached tip already
  in published canonical history.
- Duplicate task branches, a branch checked out at a mismatched owner/task or
  outside the dev root, unregistered path collisions, symlinked paths, missing
  registered directories and locked mutable targets refuse. No automatic
  repair, pruning, reset, force, ref deletion or migration is provided.
- `status` does not fetch; it reports task/owner/path, branch, HEAD, dirty/lock
  state and cached canonical origin. `sync` preserves the existing rule against
  rebasing a separately published task branch. `done` requires a clean tree,
  no unfinished Git operation and merged/patch-equivalent published history;
  it then removes the tree and keeps the branch. If the tree holds ignored
  files other than known caches/dependencies (for example `.env` or notes), it
  only detaches and lists those paths. `done --keep` only detaches.

## Automatic removal (TASK-1996)

Task trees do not outlive their task. A timer on the host runs
`bawt-wt --dev-root <shared-root> reap --apply` in the bridge container about
every 30 minutes (`reap` alone is a dry run). It removes a
`.worktrees/.wt-<repo>-<bot>-TASK-N` tree only when the task is
COMPLETED or archived with no task activity for 30 minutes, no live turn is
working on that task (a live turn with no task protects every tree its bot owns
or is associated with), no visible process has its
cwd, open file, executable or mapped file inside it, the tree is clean,
unlocked, has no unfinished Git operation, holds only disposable ignored files,
and its HEAD is merged or patch-equivalent (no unique merge commits) on the
freshly fetched `branch.canonical`. Branches are always kept, so a fix round
recreates the tree with `start`. Other registered trees, including legacy-named
trees, are skipped with a reason in the receipt under
`<shared-root>/.agent-work/logs/worktree-reaper/`. `start`, `done` and `reap`
share a per-tree lock, and `reap` rechecks everything under it. A live turn
protects only its own task's trees, so a turn that uses another finished task's
tree (an integrator reading a REVIEW tree) associates that task first or locks
the tree. Work outside a bot turn that must keep a tree (a long-lived server, a
manual hold) uses `git worktree lock`; locked trees are never reaped. Run
`git worktree` commands from the same path alias the tree was registered under
(the host's dev root vs `/home/bridge/dev`); `reap` does this itself. Never remove
review-held/unmerged/dirty work or a branch/ref by hand to make room.

### Orphan cache trimming (TASK-2132, Nick24h)

The same `reap` also discovers task directories with `.git` **files** in
`.worktrees/`, root-level `.wt-*` directories and `.task-worktrees/`, including
nested/parked legacy `.wt-<repo>-<bot>` names. A legacy path needs one unambiguous
`TASK-N` in its path; otherwise it is retained with an explicit reason. Registered
trees continue through the existing rules above, not this orphan path.

An orphan requires positively absent repository `.git/worktrees/<name>` admin
metadata. Git-command failure alone is not evidence: existing metadata, permission
errors, unsafe/unknown pointers and inaccessible alternate aliases are retained.
The helper recognizes `/home/bridge/dev` and the host's dev root as shared aliases
and checks the corresponding local admin path too. It never repairs metadata.

**Nick24h is a separate, stricter rule:** the task must be `COMPLETED` **or**
archived, and the root **and every nested entry** (including source, metadata and
symlinks) must have mtime strictly older than 24 hours. `REVIEW` or `CANCELLED`
alone is insufficient. A recent nested source edit protects the entire tree.
No arbitrary live-turn TTL is applied; the existing task-bound/taskless live-turn
query and protection semantics are unchanged.

Only known regenerable **ignored directories** may be trimmed: `node_modules`,
`.venv`, `.venv-host`, `.next`, `coverage`, `__pycache__`, `.pytest_cache`,
`.ruff_cache`, `.mypy_cache`, `.vite` and `.turbo`. The orphan's ignore rules are
checked using the canonical Git admin/index, without creating orphan metadata;
canonical tracked paths veto deletion. Symlinked ignore files cannot supply
deletion evidence. `build`/`dist` additionally require a sibling strict-JSON
`tsconfig.json` explicitly declaring that directory as `compilerOptions.outDir`.
Only TypeScript-emitted JS, declarations, maps and build-info files are accepted
there; mixed-in source/assets or symlinks veto the whole output directory.
Names alone, JSONC/inherited/other build configurations, and ambiguous output are
not proof: those directories stay. Unignored caches stay too.

All source outside these proven disposable outputs, the orphan `.git` file and
the task directory stay, even if source is dirty/uncommitted or HEAD is unknown.
Receipts include owner, filesystem UID, logical bytes retained, cache paths/bytes
and a specific skip/keep reason; source-only orphans remain visible as
`kept-orphan`. The helper never follows directory symlinks, crosses detected
mounts, deletes a symlinked cache root, or deletes a branch. Cache removal uses
no-follow directory descriptors and symlink-safe `rmtree`. Task-qualified locks
are shared with `start`/`done`, also for legacy paths. Before the cache-only
mutation batch it repeats task, live-turn, visible-process, container-mount,
metadata, inventory and whole-tree age checks under that lock.

Process protection has the existing **visible-process limit**: unreadable
`/proc` entries and other PID namespaces are not covered; cwd/fds alone are
insufficient, so executable and mapped-file paths are checked too. There is no
root privilege machinery. If Docker is on PATH, running-container mounts sourced
at/inside the task tree (or task-specific parked parents) protect it; Docker lookup
failure skips orphan cleanup.
Without Docker the helper has only visible-process/local-mount protection.
Broad shared dev-root mounts are not evidence that every task is in use. These
checks and the lock protect cooperating users, not arbitrary concurrent writers;
use an explicit hold/live task rather than relying on an invisible process.

`--apply` retains at most 30 days of this helper's safely timestamp-named regular
JSON receipts in `.agent-work/logs/worktree-reaper/`, validating receipt identity
and contents before expiry. It never expires unrelated artifacts, symlinks or
hardlinks. Dry runs write a new receipt but delete neither caches nor old logs.

The soft warning at four registered trees per owner/repository remains.

Measured examples were about 2.4 GiB for a provisioned BawtHub tree (.venv
~1.4 GiB, frontend/node_modules ~1.1 GiB) and 485 MiB for llm-bawt (.venv
~456 MiB). These are observations, not storage guarantees. Share only safe
immutable/content-addressed package caches. Keep mutable virtual environments,
node_modules, generated outputs and editable source bindings task-local; never
symlink another task's writable environment. This helper provisions no cache,
environment, dependencies or application files.

## Offline checks

Run the helper and consumer fixtures in the task source checkout. They create
only disposable local repositories; their pushes/ff/moves are fixture actions,
not publication or migration of real worktrees:

```sh
python3 scripts/test_bawt_wt.py
python3 scripts/test_bawt_wt_hooks.py
python3 scripts/test_commit_slice.py
python3 scripts/test_commit_recon.py
python3 scripts/test_skills_sync.py
python3 scripts/test_skill_frontmatter.py
```

The consumer fixtures cover private-index/hook isolation and attribution in
both legacy and task-qualified trees. Use current-checkout-relative or absolute
paths for commit-recon. Its existing normalization strips leading dots, so a
`.wt-.../file` basename-prefixed relative path is not a supported alias; arbitrary
sibling/canonical aliases are not added by this slice.
