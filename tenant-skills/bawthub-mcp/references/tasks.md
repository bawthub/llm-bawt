# Tasks, steps, projects

Use `agent-system` for implementation lifecycle and the trusted association
architecture. MCP names below are stable; harnesses may prefix `mcp__bawthub__`.

## Normal workflow

1. Look up cheaply. `tasks_board(q=<keyword>)` answers "is X started / who
   has it" in one small call. `tasks_list` filters literal `q`, status,
   project (`project_id="none"` means unassigned). `tasks_search_semantic(query)`
   finds conceptual matches when keywords miss.
2. `tasks_get_context(task_id)` loads the task/project briefing. `tasks_get`
   returns the full structured record. Both accept UUID or `TASK-N`.
3. Begin actual work with `tasks_associate_current(task_id)` or
   `tasks_update(..., status="IN_PROGRESS", bot_id=<self>, associate_current_turn=true)`.
   Mentioned/quoted task text alone does not authorize linking.
4. Maintain steps and record review (Tier 0: self-check), pushed attributable
   commit, and test/live-check receipts in `response`; set `outcome` to the
   plain-language change. Verify objective UI acceptance yourself on dev
   (Playwright, phone and desktop width), record it and attach the screenshots
   (`add_image_urls=[<asset_id>, ...]` from each screenshot result); Snark or Rook closes
   that as `COMPLETED`. Only when Nick must judge it himself (subjective
   look/feel, his device/account, a product call), submit `REVIEW`
   with `check_links` ("Where to look", required for bots): the pages to open,
   each `{label, path, steps}` with an app-relative path ("/tools/approvals",
   no host) and 1-4 plain steps saying what to check. Open every link yourself
   first and confirm the named controls are on the page. Otherwise Snark or Rook
   checks receipts and closes complete backend/infra/tooling/skill work as
   `COMPLETED`, including their own reviewed work; no extra closure reviewer.
   Incomplete work stays open. Agents cannot leave `BUG` (human-only transition).
   Passing bot_id defaults the REVIEW owner unless agent_bot_id is explicit.

For new work use `tasks_create(title, description, bot_id=<self>,
associate_current_turn=true)` when starting now; omit association for queued
follow-ups. The server generates shortId and correlation identifiers. Callers
without trusted request-local context fail closed for association. A mutation
may succeed while its returned `currentTurnAssociation` fails: inspect both;
do not create the task again to repair a link.

## Fields and states

- Task status: QUEUED, PLANNING, REFINED, IN_PROGRESS, REVIEW, BLOCKED,
  WAITING, COMPLETED, FAILED, BUG, CANCELLED, DELETED. An agent may flag BUG but cannot
  clear it (human-only transition). BLOCKED is an open status, not a terminal
  outcome; the blocked-reason field clears when the task leaves BLOCKED.
- WAITING needs `waiting_on` (RESTART, RELEASE, DECISION, EXTERNAL) and
  `waiting_reason` (400 without; either is rejected outside WAITING). BawtHub
  stamps `waitingSince` on entry and clears all three on exit. The kanban
  shows WAITING cards in Doing with a "Waiting · kind · age" chip; the status
  view gives them their own lane. `tasks_board(waiting=...)` keeps one kind.
- Follow `agent-system` for the one-executable-task-per-bot rule and task
  lifecycle. BLOCKED includes internal and external dependencies when no useful
  next step can proceed. Set `blocked_reason` to the concrete dependency, its
  responsible owner and release condition; identify Nick-only decisions rather
  than presenting bot-owned blockers as his action items. An empty reason or a
  reason without BLOCKED is rejected (400). Surplus deliberately paused or
  unstarted work belongs in QUEUED, not IN_PROGRESS or an invented blocker.
- REVIEW is for what only Nick can judge (see step 4), and a bot, cron or system caller entering
  it must send `check_links` (400 without; people moving a card are exempt).
  Links show at the top of the task popup and as a pill on REVIEW cards;
  `check_links` replaces the list. Paths are app-relative and open on whatever
  origin the reader is on, so for dev-only changes add a step saying to open it
  on the dev site. For backend, infra,
  tooling and skill work, record the review, pushed commit and verification;
  Snark or Rook closes `COMPLETED` after checking the receipts. Do not park
  non-visual work, or UI a bot can verify, in REVIEW for Nick.
- Priority: URGENT, HIGH, MEDIUM, LOW, NONE.
- `tasks_update` patches provided non-null fields; do not assume null clears a
  field. `model_id` is the actual execution model; `agent_bot_id` assigns owner.
- Three text fields, three audiences — keep them separate:
  - `description`: the request plus technical spec (engineer-facing, may be long).
  - `response`: the technical receipt — what was done, verified, risky.
    Writing it replaces the whole field, and activity keeps no copy. To add a
    close-out or addendum to someone else's receipt, read the current
    `response` first and write it back with your text appended.
  - `outcome` is the user's "What changed": 1–3 plain sentences, at most 400 characters; no task IDs, receipts, tests or file names. Technical notes go in `response`; REVIEW instructions go in `check_links.steps`, never `outcome`.
    Draft the intended outcome when planning; overwrite it with what actually
    shipped when moving to REVIEW. Example: "Tasks now show a plain-English
    'What changed' summary above the technical notes, and search finds it."
- Lists and update receipts are compact. Use `tasks_get` for full description,
  outcome, response, checklist and dependencies. Semantic search reports fallback `mode`.

## Images

Attach every relevant image from the conversation (screenshots, mockups, error
captures). Images reach every bot that works the task; don't make them
re-capture with Playwright.

- `tasks_create(image_urls=[...])`, `tasks_update(add_image_urls=[...],
  remove_image_urls=[...])`. Name an image by chat upload id (`ma_…`) or any
  upload URL (`/api/chat/uploads/<id>`, `/v1/uploads/<id>`, a task-UI upload);
  chat uploads are stored as `/api/chat/uploads/<id>`. Unknown or non-image
  ids are refused and nothing is written. Adding or removing twice is a no-op.
- `associate_current_turn=true` also attaches the images on the message that
  triggered this turn. `turnImagesAttached` lists what it added; drop an
  irrelevant one with `remove_image_urls`. Pass `attach_turn_images=false` when
  the turn's images don't belong on the task. Re-linking the same turn adds
  nothing; `turnImagesError` reports a lookup failure without blocking the write.
- An update sends only what to add and remove; BawtHub applies it to the list
  under the task's row lock, so an image someone else attaches meanwhile is
  kept. A BawtHub release without this answers "BawtHub doesn't support image
  edits on update yet; the images were not changed" (the other fields still
  apply). Attach the images later, after a BawtHub release; don't fall back
  to replacing the whole list.
- Upload URLs must be on BawtHub's domain or an internal service name (`app`);
  an upload-looking path on any other host is refused.

## Board focus and archive

Any bot or user may change these; BawtHub records who did.

- Focus is Nick's ranked list. `tasks_list(focused=true)` returns it in rank
  order (`focusRank` 1 first); `focused=false` returns the unfocused backlog.
- `tasks_update(focus="on"|"off", focus_position, focus_anchor)`. `on` alone
  appends an unfocused task at the bottom and leaves a focused task in place.
  `focus_position` is `top|bottom|before|after` and implies `on`; `before`/
  `after` need `focus_anchor` (TASK-N), which must itself be focused.
  Finished tasks can't be focused (409), and finishing clears focus. A 409
  returns `focused`, the current order: re-anchor from it, don't blind-retry.
- Archive takes a finished task off the board: `tasks_update(archived=true)`
  only when the task is or becomes COMPLETED, FAILED or CANCELLED (DELETED is
  not archivable). It combines with status in one call
  (`status="COMPLETED", archived=true`). A non-finished status returns 400;
  409 means the status changed meanwhile, so re-read first. `archived=false`
  restores; reopening or deleting a task unarchives it.
- `tasks_list` includes archived tasks unless `archived=false`;
  `archived=true` lists only archived ones. Filters combine with each other.
  Rows are lean by default (shortId, title, status, priority, owner,
  updatedAt, project name, plus BLOCKED/REVIEW/focus/archive fields);
  `detail=true` returns the fuller rows. `limit` defaults to 10 (max 25);
  `more: true` means `total` exceeds the rows returned, so narrow the filter.
- `tasks_search_semantic` returns 10 rows by default (max 25), each with
  `descriptionChars` instead of the description; open one with
  `tasks_get_context`.
- `tasks_board` reads the board as `/tasks/board` draws it, in one call. Its
  args are the board URL's: `view` (`kanban`|`status`), `updated`
  (`1h|6h|24h|48h|7d|30d|all`, default `7d`), `bot`, `q`, `project`,
  `priority`, plus `limit` rows per lane (default 20, 1-50) and `rows`.
  Each lane gives `shown` and `older` (left out by `updated`) counts,
  `active`, and compact rows (no descriptions); a drawer lane names its
  column in `in`. Every column is returned, including ones the page hides by
  default. Without `q`, `rows` defaults to `active`: only the open columns
  (Blocked, To do, Doing, Review) carry cards, while Backlog, Done and
  Archived keep their counts and say `rowsOmitted`. With `q` it defaults to
  `all`; pass `rows="all"` for every card. Any other `updated` value is
  rejected, and a `view`/`updated` the board did not apply returns an error
  instead of a different board. A note saying the BawtHub build ignores
  `rows` means every card came back (older BawtHub).
- An error saying the server "does not support focus/archive" means it wasn't
  applied (older BawtHub); `updated` names the fields that were. Never report
  such a change as done.

## Checklists

- `steps_add(task_id, steps)` appends; each needs `title`. Defaults: PLAN/PENDING.
- Types: PLAN, READ_FILE, EDIT_FILE, CREATE_FILE, DELETE_FILE, RUN_COMMAND, SEARCH,
  ASK_USER, REVIEW. States: PENDING, RUNNING, COMPLETED, FAILED, SKIPPED.
- `steps_update(task_id, step_id, status, output)` patches one step.
  Nonempty `updates` selects batch mode, overriding scalar args. Each entry needs
  `step_id` (also accepts `stepId`) plus status/output/title/type/file_path
  (`filePath` alias). IDs outside the task are ignored; inspect returned steps.
- `steps_set` atomically replaces the ENTIRE checklist; [] clears it. Entries
  support title/type/status/output/file_path or filePath. Do not use to update
  progress unless you intentionally want to delete/recreate step identities.
- `steps_delete`: nonempty step_ids wins over step_id; outside-task IDs are
  ignored. Prefer SKIPPED to retain the audit trail.

## Reviews

Peer reviews are an append-only history on the task, separate from `response`
and from task status `REVIEW`. The workflow (who reviews, request contract,
severities, verdict meaning, re-review) is the `peer-review` skill.

- `tasks_reviews_add(task_id, stage, verdict, body, bot_id)` appends one
  review. `stage` is `DESIGN` or `IMPLEMENTATION` (a plan review is `DESIGN`).
  `verdict` is `APPROVE`, `APPROVE_WITH_CHANGES`, `REQUEST_CHANGES` or
  `COMMENT`. Pass `target_revision` (local commit SHA; for r2 include its
  follow-up fix SHA; for a catalog-only review use the canonical catalog ref
  below) and a stable `idempotency_key`, conventionally
  `<bot>:task<N>:<stage>-review:r<N>`. An exact retry replays; reusing a key
  with different content returns 409, so each round needs its own key. It never
  changes status, description, response or assignee.
- IMPLEMENTATION `FIXED` needs `revision`; DESIGN `FIXED` also accepts no
  revision with a required 1–2000 character `note` explaining the design fix.
  When provided, `FIXED.revision` accepts a bare 7–64 hex Git commit SHA or
  `ops-catalog:<slug>@v<positive-version>#sha256:<64-hex-command-script-hash>`.
  Use **exactly the same tagged form** for catalog-only `target_revision` and
  `FIXED.revision`; old free-text targets remain valid. The case-sensitive slug
  grammar is `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`; only hex is lowercased.
  This validates format, not existence in Git or the live ops revision store.
  Unknown `FIXED` revision kinds return 400. Blockers still require a signed
  reviewer's one r2; direct resolution is for non-Blockers only.
- `tasks_reviews_list(task_id)` returns reviews newest first; follow
  `nextCursor` for older pages.
- Recording a review does not notify the author. Call back with
  `bots_send_message`.

## Projects and other actions

- `projects_list` is compact; `projects_get` includes context and tasks.
  `projects_get_context` reads just the name/instructions briefing.
- `projects_create`/`projects_update`: context_prompt is shared instruction text;
  agent_bot_id assigns the project bot. Use bot_id for attribution.
- `tasks_add_dependency(task_id, depends_on_id)` makes the first wait for the
  second; cycles/self-links fail. `tasks_remove_dependency` removes that edge.
- `tasks_promote` creates a project from a task and moves the task into it.
- `activity_get` reads task/project changes, bounded at 100 entries.
- `tasks_delete` permanently removes task + steps; prefer CANCELLED.
  `projects_delete` removes the project but leaves tasks unassigned.
- `tasks_regenerate` uses server-side inference and replaces title/checklist.
  Prefer writing the plan yourself via tasks_update + steps_set.
