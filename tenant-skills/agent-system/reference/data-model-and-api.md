# Agent system: data model, API and file locations

> Part of the agent-system skill. Moved verbatim from the parent SKILL.md (TASK-1640).

### Data Models (Prisma)

**AgentTask** — the core work unit
- `id`, `shortId` (e.g. "TASK-42"), `title`, `description`, `imageUrls`
- `status`: QUEUED → PLANNING → REFINED → IN_PROGRESS → COMPLETED;
  `REVIEW` for visual UI/UX, `BLOCKED` for concrete internal or external
  dependencies (lifecycle and one-task-per-bot rule: parent agent-system skill;
  also FAILED, BUG, CANCELLED, DELETED)
- `priority`: URGENT | HIGH | MEDIUM | LOW | NONE
- `response` — agent's summary when done
- `modelId` — which model worked on it
- `planned` (boolean) — whether a project manager has spec'd it out
- `projectId`, `agentBotId`, `position` (for ordering)
- Relations: steps[], project, activities[], uploads[]

**AgentStep** — sub-units of work within a task
- `taskId`, `orderIndex`, `title`, `output`, `filePath`
- `status`: PENDING → RUNNING → COMPLETED | FAILED | SKIPPED
- `type`: PLAN, READ_FILE, EDIT_FILE, CREATE_FILE, DELETE_FILE, RUN_COMMAND, SEARCH, ASK_USER, REVIEW

**AgentProject** — groups tasks with shared context
- `name` (unique), `description`, `contextPrompt` (injected into agent docs)
- `color`, `icon`, `agentBotId` (optional project manager bot)
- Relations: tasks[], activities[], uploads[]

**AgentActivity** — audit log
- `type` (e.g. task.created, task.status_changed, step.status_changed, task.dispatched)
- `actorType`: user | bot | system | cron
- `actorId`, `taskId`, `projectId`, `meta` (JSON)

**Upload** — files attached to tasks or projects
- `sha256` (unique), `filename`, `mimeType`, `sizeBytes`, `width`, `height`
- Can belong to a task or project

### API Endpoints

All under `/api/tasks/` (renamed from `/api/agents/` — old paths no longer exist):

| Route | Methods | Purpose |
|---|---|---|
| `tasks` | GET, POST | List/Search/Create (see query params below) |
| `tasks/[id]` | GET, PATCH, DELETE | Single task CRUD |
| `tasks/[id]/steps` | POST | Create steps (single or batch) |
| `tasks/[id]/steps/[stepId]` | PATCH, DELETE | Update/delete a step |
| `tasks/[id]/dispatch` | POST | Send task to a bot for execution |
| `tasks/[id]/promote` | POST | Convert task into a project |
| `tasks/[id]/regenerate` | POST | AI-regenerate title + steps from description |
| `tasks/[id]/dependencies` | GET, POST, DELETE | Task dependency management |
| `tasks/reorder` | POST | Reorder tasks (update positions) |
| `projects` | GET, POST | List / Create projects |
| `projects/[id]` | GET, PATCH, DELETE | Single project CRUD |
| `projects/[id]/context` | GET | Plain-text project context for agents |
| `projects/[id]/chat` | POST | Stream chat with project manager bot |
| `activity` | GET | Activity log (filterable by type, actorId, taskId, projectId) |
| `cron/project-manager` | POST | Automated planning + dispatch |
| `docs` | GET | Agent-readable docs (optionally with `?task=ID`) |
| `grid` | GET | Grid view data |

### GET /api/tasks/tasks Query Params

| Param | Description |
|---|---|
| `q` | **Full-text search** across title, description, and shortId (case-insensitive substring) |
| `status` | Filter by status (QUEUED, IN_PROGRESS, REVIEW, COMPLETED, FAILED, etc.) |
| `projectId` | Filter by project ID, or `"none"` for unassigned tasks |
| `agentBotId` | Filter by assigned bot |
| `sort` | `"position"` (default if projectId set) or `"updated"` (default otherwise) |
| `limit` | Pagination limit (default 50) |
| `offset` | Pagination offset (default 0) |

### File Locations

- API routes: `frontend/src/app/api/tasks/`
- Prisma schema: `frontend/prisma/schema.prisma` (models start around line 118)
- Activity helper: `frontend/src/lib/agentActivity.ts`
- Frontend pages: `frontend/src/app/(app)/(dashboard)/tasks/`
- Task dispatch prompt: `frontend/src/app/tasks/taskDispatchPrompt.ts`
- Docs endpoint: `frontend/src/app/api/tasks/docs/route.ts`

## When Enhancing This System

- **New endpoints**: Add route files under `frontend/src/app/api/tasks/`, follow the existing pattern (prisma queries, JSON responses, activity recording).
- **Schema changes**: Edit `frontend/prisma/schema.prisma`, then run `npx prisma migrate dev --name description`.
- **New activity types**: Use `recordActivity()` from `lib/agentActivity.ts`. Follow the `noun.verb` naming convention (e.g. `task.created`, `step.status_changed`).
- **Docs updates**: If you add new endpoints or change the workflow, update the `DOCS_TEMPLATE` in the docs route so agents stay informed.
- **Frontend**: Agent UI lives in `frontend/src/app/(app)/(dashboard)/tasks/`. Uses TanStack Query for data fetching.
