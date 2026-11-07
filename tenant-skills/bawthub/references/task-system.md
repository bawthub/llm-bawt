# Agent task system: models, dispatch, cron, API routes

> Part of the bawthub skill. Moved verbatim from `bawthub/SKILL.md` (TASK-1640); the rules stay in the parent.

Task pipeline: humans direct → bots plan, execute and verify → Nick reviews visually confirmable UI/UX only. Every bot work interaction goes through this.

**Data models** (Prisma): `AgentTask` (`shortId` TASK-N, status QUEUED → PLANNING → REFINED → IN_PROGRESS → REVIEW → COMPLETED/FAILED, relations steps[]/project/activities[]/uploads[]) · `AgentStep` (`orderIndex`, PENDING → RUNNING → COMPLETED/FAILED, typed PLAN/READ_FILE/EDIT_FILE/RUN_COMMAND/…) · `AgentProject` (groups tasks; `contextPrompt` injected into agent docs; optional PM bot) · `AgentActivity` (audit log via `recordActivity()`).

**REVIEW is Nick's visually confirmable UI/UX queue**: set `check_links` (app-relative pages with plain steps, required for bots; open each first). For non-visual work, Snark or Rook checks peer-review (Tier 0: self-check), attributable pushed commit and test/live-check receipts, then closes complete tasks as COMPLETED. Never close incomplete work or clear BUG. Steps MAY be COMPLETED by agents.

**Dispatch:** `POST /api/tasks/tasks/[id]/dispatch` {botId, userId} → IN_PROGRESS immediately → background `after()` builds prompt (task + project context + steps) → llm-bawt `/v1/chat/completions` → agent PATCHes progress → REVIEW with summary.

**Cron** (`/api/tasks/cron/project-manager`): Phase 1 planning (PM bot specs unplanned tasks, 30-min per-project cooldown), Phase 2 auto-dispatch (queued+planned → agents by priority, max 3/run). Don't bypass the cooldown/concurrency logic.

### API routes (all under `/api/tasks/` — old `/api/agents/*` paths are gone)

`tasks` (GET/POST) · `tasks/[id]` (GET/PATCH/DELETE) · `tasks/[id]/steps[/[stepId]]` · `tasks/[id]/dispatch` · `tasks/[id]/promote` · `tasks/[id]/regenerate` · `tasks/[id]/dependencies` · `tasks/reorder` · `projects[/[id]]` · `projects/[id]/context` · `projects/[id]/chat` · `activity` · `cron/project-manager` · `docs` (`?task=ID`) · `grid`

Live API reference: on the host, `docker exec bawthub-frontend-1 curl -s http://localhost:5173/api/tasks/docs`. The `DOCS_TEMPLATE` in the docs route is a fallback — llm-bawt's `agents.docs` prompt may shadow edits. `shortId` is server-generated; don't let callers set it.
