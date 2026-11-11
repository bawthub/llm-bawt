# Changed-file commit reconciliation (TASK-885)

## Authority

Turn snapshots remain immutable historical evidence. `commit_requested_at` means
only that a user requested an action; it never proves a successful commit.

The conversation aggregate and trigger-keyed history reads now annotate each file
with `commit_state`: `committed`, `pending`, `unknown`, or `not_applicable`.
`commit_evidence` contains tracked repository ID, relative path, checked HEAD and,
when proved, the commit hash. It makes **no push claim**.

The app has no repository mount. It posts bounded snapshot descriptors (hashes,
paths, capture time; no file contents) to BawtHub's internal-only
`POST /internal/changed-files/commit-evidence`. BawtHub resolves paths exclusively
through its tracked-repository catalog and read-only `/repos` mount. Explicit
capture-root mappings bridge container/host/tilde paths to that catalog; basename
and arbitrary suffix matching are not authority.

The verifier reads reachable first-parent commits after capture, checks exact
SHA-256 blob content, and requires an actual path transition. Deletion also needs
matching parent bytes; rename needs matching old bytes, new bytes and old-path
removal. Clean working-tree status alone proves nothing. Matching dirty bytes
not in HEAD are pending; missing, discarded, incomplete or unavailable evidence
is unknown. Errors never turn into success.

Conversation results retain `history_files` for every snapshot. Latest snapshots
control actionable scope (canonical repo ID/path where available), while older
snapshots retain their own content and commit evidence. `Commit all` excludes only
verified covered files (plus explicitly dismissed scratch files). Requested but
unverified files remain visible. The client no longer consumes the current-turn
Bash-output hash parser as commit authority. Completion/reconnect/focus invalidates
the conversation query; all visible turn cards share its evidence.

## Deployment boundary

- App setting: `BAWTHUB_COMMIT_EVIDENCE_URL`, default `http://frontend-prod:3002`.
- BawtHub setting: `CHANGED_FILES_CAPTURE_ROOTS`, colon-separated mount roots;
  defaults to `/home/bridge/dev:~/dev` in
  `frontend/server/routes/changed-file-evidence.ts`. Additional host roots belong
  in that deployment setting, not a personal-home source default.
- Dev evidence route is available through Hono watch at `http://frontend:3002`.
- The app must reload the Python changes and reach a Hono instance containing the
  new internal route before live API reconciliation works. A missing route yields
  unknown, not committed. This task's implementation did not activate services or
  change environment configuration. Do not make production depend on optional dev
  HMR as an automatic fallback.

## Conservative bounds / limits

- At most 200 newest snapshot descriptors per read; older overflow stays unknown.
- Four Git workers, batch start budget 8 seconds, per-snapshot traversal deadline,
  2.5-second subprocess timeout, 100 first-parent path commits, 1 MiB file limit.
- Binary/truncated/content-less snapshots stay unknown.
- Exact content coverage is deliberately conservative: later combined edits that
  never existed as the exact captured blob cannot be claimed committed. No fuzzy
  hunk-overlap success. Earlier same-path snapshots remain browsable even when a
  newer committed snapshot removes that path from actionable scope.
- Same-second capture/commit ordering is ambiguous with Git's second-resolution
  timestamps; such evidence stays unverified rather than allowing a pre-capture
  commit to count. Clock skew can similarly cause conservative unknown results.
- Unmapped capture roots and relative paths without canonical identity fail closed.
- No durable Git evidence cache: refresh recomputes from current reachable history,
  so resets/history rewrites do not leave stale success records.

## Verification receipt

- Real incident: 11 stored snapshots of one bot from session
  `5ff520ee-d6fb-41d2-ba68-fff13605a29a`, independently checked against mounted Git.
  Five frontend files match `4ded2dae3caaa20f63cdb2867062f7c6e570546f`; six backend
  files match `e507a58a4d4514514cbf7591c2d5320f8a0693a6`. Python reconciliation with
  the dev evidence service produced zero pending, 11 history files; no DB writes.
- Backend tests cover request-vs-success, two-repo/two-turn scope, immutable content,
  later snapshots, unknown service responses and scratch files.
- Real temporary Git repository tests cover multi-repo commits, partial commit,
  subsequent edits, discarded content, deletion, rename, empty files, literal
  paths and unavailable identity/content.
- Browser-only component replay: `frontend/scripts/verify-changed-file-commits.mjs`.
  At 1280px and 375px: both groups committed, all 11 historical diff controls,
  no Commit all for covered scope, unknown/requested scope actionable, no overflow.
  This is a component replay, not an activated app end-to-end test.
