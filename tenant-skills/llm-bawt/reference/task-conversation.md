# Task ↔ conversation surface (TASK-700–706) — llm-bawt side

> Part of the llm-bawt skill. llm-bawt-side summary. The canonical cross-stack reference is `agent-skills/agent-system/task-conversation-architecture.md` (ER diagram, 5 sequence diagrams, 16 invariants, pgq/curl recipes) — read that before touching the trusted capability, non-activating sessions, MCP task tools, or `/v1/turn-changed-files*`.

#### Trusted ordinary-chat capability (TASK-701)

For claude-code bridge turns only, `chat_completion_stream` resolves the
canonical BawtHub session (explicit `request.session_id` or the active
`(bot, user)` thread), ensures a trigger-message UUID exists, and mints a
short-lived Fernet capability containing `session_id`, `turn_id`,
`trigger_message_id`, `bot_id`, `user_id`, `issued_at` (TTL 6 h). Header is
`X-LLM-Bawt-Task-Turn-Context`. Mint site: `service/chat_streaming.py:347-373`
— three preconditions all required: `is_agent_backend`, `backend_name ==
"claude-code"`, `session_id` present from `thread_binding`. Mint failure
logs a warning; the chat turn is never blocked over association plumbing.

Propagation trail: app → Redis command fields → `claude-code-bridge` (pops
capability) → Claude Agent SDK MCP env → MCP HTTP header →
`TaskTurnCapabilityMiddleware` binds ASGI header to `ContextVar` for one
request → `associate_current_task()` verifies the Fernet token → PUT to
BawtHub internal listener (no bearer since TASK-793; network-bounded).

Tools: `tasks_associate_current(task_id)`, plus `associate_current_turn=true`
on `tasks_update` / `tasks_create`. **Text detection alone never writes links**
— `TASK-N` in a quote, negative reference, or incidental mention is not
authority. BawtHub owns the association tables (`AgentTaskSession`,
`AgentTaskTurn`, `AgentConversationIdentity`); llm-bawt does not copy them.
Native Codex cannot safely vary MCP headers per turn today and fails closed
with a clear error.

Runtime config: default internal URL `http://frontend-prod:3002` (the
always-on service, NOT the optional HMR `frontend`). No per-endpoint auth
(TASK-793): the internal port stays inside the docker network, and that
boundary is the auth layer.

#### Non-activating sessions for dispatch (TASK-702)

`POST /v1/sessions` accepts `activate: bool` (default `true`). `activate=false`
mints a session that is **born archived**: it does NOT rotate the user's
active thread and does NOT participate in the one-active-per-`(bot, user)`
partial unique invariant. Response carries `status: "archived"`, `rotated:
false`; `session_metadata.created_inactive = true` marks the row so operators
can distinguish background/task threads from UI-archived ones. Background
task dispatch uses this so a queued task never disturbs the user's live
conversation. Subsequent `POST /v1/chat/completions` with an explicit
`session_id` continues the archived thread untouched.

#### Turn-changed-files surface (TASK-661/703)

- `GET /v1/turn-changed-files?message_ids=...` → trigger-message-keyed
  summaries (chat UI parity with `/v1/tool-calls`). Bawthub's task-context
  aggregator batches trigger IDs into this endpoint.
- `GET /v1/turn-changed-files/{turn_id}/content?repo_key=X&path=Y` → immutable
  turn-time bytes. NOT a live worktree diff. Content-addressed via
  `before_blob_key`/`after_blob_key` in `turn_changed_files`.
- `GET /v1/turn-changed-files/uncommitted?bot_id=X&user_id=Y&session_id=...&anchor_turn_id=...`
  and `POST /v1/turn-changed-files/commit-requested` accept a `session_id`
  request parameter — that param is resolved through the canonical session
  scope (`changed_files_store._resolve_session_id` verifies against `sessions`
  OR joins from an anchor turn via `turn_changed_files → messages →
  session_id`). **The `turn_logs` TABLE has no `session_id` column** — the
  request param is a convenience, not evidence of a column. Do not add one.

## Summary formerly in the parent SKILL.md (TASK-1640)

### Task ↔ conversation surface (TASK-700–706)

Canonical cross-stack reference: **`agent-skills/agent-system/task-conversation-architecture.md`** (ER diagram, sequence diagrams, 16 invariants). llm-bawt-side summary: **`reference/task-conversation.md`** — read one of them before touching the trusted capability, non-activating sessions, MCP task tools, or `/v1/turn-changed-files*`. Hot facts:

- Claude-code-bridge turns get a short-lived Fernet capability (`X-LLM-Bawt-Task-Turn-Context`, TTL 6h) minted in `chat_streaming.py` — this is what authorizes `tasks_associate_current` / `associate_current_turn`. **Text detection alone never writes links.** BawtHub owns the association tables.
- `POST /v1/sessions` with `activate: false` mints a born-archived session (task dispatch never disturbs the live thread).
- `turn_logs` has NO `session_id` column — the request param on `/v1/turn-changed-files/*` is resolved through session scope; do not add one.
