# Task <-> conversation architecture (TASK-700-706) — bawthub side + worker callbacks

> Part of the bawthub skill. bawthub-side detail. The canonical cross-stack reference is `agent-skills/agent-system/task-conversation-architecture.md` (ER diagram, sequence diagrams, 16 invariants) — read it before touching the association surface (schema, routes, aggregator, dispatch resolver, chat task-binding).

**Canonical reference: `agent-skills/agent-system/task-conversation-architecture.md`.**
That file has the full ER diagram, five Mermaid sequence diagrams (ordinary-chat
association, dispatch/non-activating-session, task-context aggregation,
commit-recon), the 16 verified invariants, the deferred ledger, and pgq/curl
verification recipes. Read it before touching any of the association surface
(schema, routes, aggregator, dispatch resolver, chat task-binding). This section
is the fast summary.

**Three association tables (bawthub DB, `frontend/prisma/schema.prisma:177-311`):**

- `AgentConversationIdentity` — durable `sessionId → (botId, userId)` ownership
  anchor. Survives task detach (FK `RESTRICT`), so a `sessionId` can never be
  re-bound to different provenance after first claim.
- `AgentTaskSession` — `(taskId, sessionId)` grain, with a partial unique index
  enforcing at most one primary per task (`WHERE isPrimary=true`).
- `AgentTaskTurn` — `(taskSessionId, triggerMessageId)` grain, `turnId` nullable
  (delivery-shape turn ids stay null).

Enum `TaskAssociationSource`: `DISPATCH | CHAT | AGENT | MANUAL | RECON`.

**Two write surfaces — different trust models:**

1. **Internal listener** (`INTERNAL_PORT` — served only on the internal port,
   never published beyond the docker network; every `/internal/*` path on the
   main port answers 404; no per-endpoint auth since TASK-793). Accepts all 5
   sources. Used by llm-bawt's trusted current-turn capability, by
   `taskDispatchService` for `DISPATCH`, and by `commit-recon --repair-attach`
   for `RECON`. Endpoints in `frontend/server/routes/task-conversations.ts`.
2. **Public app** (main port, user-scoped, source whitelist `{MANUAL, CHAT}`
   only). Endpoints in `task-link.ts` (attach/link/unlink), `task-primary.ts`
   (set primary), `task-context.ts` (purpose-built browser context). Ownership
   verified out-of-band via llm-bawt `GET /v1/sessions/:id` — `bot_id` from the
   response is authoritative; client `botId` in the body is a hint only; null
   `user_id` fails closed.

**Ordinary-chat trusted association (TASK-701)** — for a claude-code bridge
turn only, llm-bawt mints a short-lived Fernet capability containing server-known
`session_id`, `turn_id`, `trigger_message_id`, `bot_id`, `user_id`. The bridge
forwards it as MCP HTTP header `X-LLM-Bawt-Task-Turn-Context`; the MCP server
verifies it and writes `source: AGENT` through the internal listener. Raw
identifiers are never MCP tool arguments. Native Codex cannot safely vary MCP
headers per turn today and fails closed.

Agent actions:

- `tasks_associate_current(task_id)` links an existing task to this exact turn.
- `tasks_update(..., associate_current_turn=true)` updates/claims and links; with
  no other fields it behaves as an association-only claim.
- `tasks_create(..., associate_current_turn=true)` creates, then immediately
  links the new task.

**Detection is never authority** — `TASK-N` text pattern-matching may suggest
an action, but quoted/negative/incidental references are not evidence.

**Dispatch (TASK-702)** — `taskDispatchService` reuses the newest compatible
`(bot, user)` task session; failing that, calls llm-bawt `POST /v1/sessions
{activate: false}` for a **born-archived** thread that never rotates the user's
active conversation. Persists `source: DISPATCH`. The dispatch client throws
`INVALID_RESPONSE` if llm-bawt returns `rotated: true` for a non-activating
request — a contract-violation guard.

**Aggregation (TASK-703/704)** — `GET /api/tasks/tasks/:ref/conversation-context`
returns owned sessions and their exact turns joined to llm-bawt changed-file
summaries (chunk 50, concurrency 10, hard cap 500 turns → 413). Per-turn
`associationStatus` (`unique | ambiguous | stale`) and `changedFilesStatus`
(`available | no_changes | missing_provenance | unavailable`); `partial: true` +
`failures[]` on any degrade — never fabricates attribution. Purpose-built:
never returns trigger-message IDs, association row IDs, source metadata, or
another user's sessions.

**Manual attach/link/unlink (TASK-705)** — the public write surface for user
correction. Foreign-owned vs nonexistent both uniform 404 (no existence leak).

**Recon repair (TASK-706)** — `commit-recon --repair-attach TRIGGER:TASK-N`
(repeatable, gated by 4 conditions: both keys unassociated, full identity tuple
present, attach entry hits this trigger, single task per trigger). Never
auto-repairs from regex. `--no-repair` = strictly read-only. Delivery-shape
turn IDs (`turn-delivery-<hex>`) are filtered from resolve batches and PUT
bodies alike; attribution rides on `trigger_message_id` alone for those turns.

**Runtime config** — default internal URL `http://frontend-prod:3002` (the
always-on service, NOT the optional HMR `frontend`). No per-endpoint auth
(TASK-793): the internal port stays inside the docker network, and that
boundary is the auth layer.

### Event-driven worker callbacks

Project managers should not poll workers as their primary scheduler. The manager
delegates with default asynchronous `bots_send_message`; every async send is
durable and returns a delivery ID. The worker updates task steps and persists its
READY/BLOCKED receipt in the task response first, then calls default
`bots_send_message(task_id=..., message_kind=...,
idempotency_key="TASK-N:READY|BLOCKED")`. If the manager has an active
steer-capable Claude turn, the callback redirects that same run; otherwise exactly
one safe idle fallback turn starts. Use `delivery="when_idle"` only when a
separate turn is intentional. `force=true` never authorizes concurrent agent
turns. Delivery state is inspectable through
`bots_delivery_get`/`bots_deliveries_list`; polling is fallback diagnostics only.
For large independent assignments, managers may inspect `agent_context_health`
and choose `session_policy="reset_retain_history"` or
`reset_without_history`. llm-bawt owns context measurement, capability gating,
durable thread rotation, and overflow recovery; BawtHub only renders the normal
delivery/task lifecycle. See llm-bawt `docs/INTER_BOT_COMMUNICATION.md` and
`docs/AGENT_CONTEXT_CONTROL.md`.

## Hot facts (moved from SKILL.md, TASK-1640)

- Three association tables (bawthub DB): `AgentConversationIdentity` (durable session→bot/user anchor, FK RESTRICT), `AgentTaskSession` (one primary per task), `AgentTaskTurn` (`turnId` nullable — delivery-shape ids stay null). Sources: `DISPATCH | CHAT | AGENT | MANUAL | RECON`.
- Two write surfaces: internal listener (bearer, internal port only, all 5 sources) vs public app (user-scoped, `{MANUAL, CHAT}` only, ownership verified via llm-bawt).
- Trusted ordinary-chat association rides llm-bawt's Fernet capability header — raw identifiers are never MCP tool args. **Text detection is never authority.**
- Dispatch reuses the newest compatible task session or mints a born-archived (`activate: false`) thread — never rotates the user's live conversation.
- Workers persist their READY/BLOCKED task receipt FIRST, then call durable `bots_send_message(task_id=…, idempotency_key="TASK-N:READY")`; managers never poll as primary scheduler.
