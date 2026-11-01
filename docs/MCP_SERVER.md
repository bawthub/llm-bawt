# BawtHub MCP Server

The FastMCP instance and core memory/message tools are implemented in
[src/llm_bawt/mcp_server/server.py](../src/llm_bawt/mcp_server/server.py).
Focused registration modules keep tool families separate: task tools in
[`task_tools.py`](../src/llm_bawt/mcp_server/task_tools.py), durable/immediate
inter-bot tools in
[`inter_bot_tools.py`](../src/llm_bawt/mcp_server/inter_bot_tools.py), session
tools in [`session_tools.py`](../src/llm_bawt/mcp_server/session_tools.py),
profile tools in [`profile_tools.py`](../src/llm_bawt/mcp_server/profile_tools.py),
and self-handoff tools in
[`self_tools.py`](../src/llm_bawt/mcp_server/self_tools.py).

The server name is `bawthub`.

## Transport

- HTTP transport uses FastMCP streamable HTTP.
- `stdio` is also supported for local development.
- DNS rebinding protection is enabled; allowed hosts come from
  `LLM_BAWT_MCP_ALLOWED_HOSTS`.

Start it directly:

```bash
uv run python -m llm_bawt.mcp_server --transport http --host 0.0.0.0 --port 8001
```

## Tool groups

| Group | Examples |
|---|---|
| `memory_*` | `memory_store`, `memory_search`, `memory_search_all`, `memory_consolidate` |
| `messages_*` | `messages_get`, `messages_add`, `messages_search_all`, `messages_ignore_recent` |
| `context_*` | `context_get_recent` |
| `facts_*` | `facts_extract` |
| `system_*` | `system_stats`, `system_run_maintenance` |
| `bots_*` | `bots_list_available`, `bots_send_message`, `bots_delivery_get`, `bots_deliveries_list`, `bots_delivery_cancel` |
| `agent_context_*` | `agent_context_health`, `agent_context_compact`, `agent_context_reset` |
| `tasks_*` | `tasks_list`, `tasks_get`, `tasks_create`, `tasks_update`, `tasks_get_context`, `tasks_reviews_list`, `tasks_reviews_add` |
| `steps_*` | `steps_add`, `steps_update`, `steps_delete` |
| `projects_*` | `projects_list`, `projects_get`, `projects_create`, `projects_update`, `projects_delete`, `projects_get_context` |
| `activity_*` | `activity_get` |
| `self_*` | `self_recap`, `self_tail`, `self_fwd`, `self_system_prompt` |
| `profile` | unified user/bot profile tool |

## Notes on inter-bot tools

Default asynchronous `bots_send_message` is durable: it returns a stable delivery
ID, steers an active Claude Code turn in place when possible, and otherwise starts
one safe idle fallback turn. `delivery="when_idle"` deliberately skips steering.
`force=true` never authorizes concurrent agent turns. `session_policy` accepts
`continue`, `reset_retain_history`, or `reset_without_history`; reset policies wait
for idle and rotate non-destructively before delivery. Use `bots_delivery_get`,
`bots_deliveries_list`, and `bots_delivery_cancel` for lifecycle inspection.

`agent_context_health` exposes active-thread resident usage/headroom and backend
capabilities. `agent_context_compact` queues one idempotent Claude `/compact`
maintenance turn through the durable when-idle scheduler. `agent_context_reset`
performs an idle-only safe reset without deleting durable transcript rows. Full
accounting, capability, and overflow recovery contract:
[AGENT_CONTEXT_CONTROL.md](AGENT_CONTEXT_CONTROL.md).

## Notes on task tools

`tasks_*`, `steps_*`, `projects_*`, and `activity_*` are thin MCP wrappers over
the BawtHub task API. They call `LLM_BAWT_TASK_API_URL` plus `/api/tasks/...`
internally.

`tasks_get(task_id)` and `tasks_get_context(task_id)` return a compact structured
briefing, **not the full historical task**. Current owner/status/blocker, response
and description excerpts, running/next/recent steps, and three recent reviews are
included. `tasks_get_context` also includes project instruction excerpts. Large
fields carry `complete=false`, head/tail excerpts, `total_chars` and a detail path;
collection summaries carry whole counts and the exact path. Small requested
budgets can explicitly omit history while retaining current task state.

Both tools accept the same bounded detail parameters:

- `path`: JSON pointer under `/task`, `/reviews`, or `/project`. Examples:
  `/task/description`, `/task/steps/0/output`, `/reviews/reviews/0/body`,
  `/project/contextPrompt`. RFC 6901 escapes `~1` and `~0` address `/` and `~`.
- `offset`: character offset, normally supplied by the returned `next` call.
- `max_chars`: budget for the **whole serialized reply**, clamped to 2,000–8,000.
  Escaping and non-ASCII data count toward that budget, not just source text.
- `version`: selected-value SHA-256 from the first detail page. It is required
  for nonzero offsets; changed values return `detail_changed`, never a mixed
  snapshot. Restart that path at offset zero after a change.
- `cursor`: existing opaque backend review cursor, only for `/reviews` paths.

Details return `text`, `encoding` (`text` or `json`), `total_chars`, `offset`,
`next_offset`, and an executable `{tool, arguments}` `next` descriptor. Follow
`next` until `next_offset=null`, concatenate text, and decode JSON only when
`encoding=json`. `/task` retrieves **every raw task field**, including unknown
fields; `/task/steps` pages the complete ordered list without dropping a huge
individual output. `/reviews` pages the complete raw review page (including
findings and long bodies); after finishing its text chunks, follow `next_page`
to the next backend review page. New backend pages have their own version.
No unbounded `full=true` escape or harness-output exemption is provided.

Compatibility: old `tasks_get` callers expecting the raw object should read
`result.task` for current summary fields, or reconstruct `path=/task` explicitly.
`tasks_get_context` now returns a dictionary rather than an unbounded markdown
string. No internal Python callers depended on either old result shape; the
BawtHub HTTP API, recall readers and approval continuation envelope are unchanged.
Source publication does not activate these signatures: the MCP-serving app needs
a separately authorized activation and fresh tool discovery. Old resumed SDKs
can retain their old schemas; do not reset sessions/history as a workaround.

`tasks_reviews_list(task_id, limit=20, cursor=None)` reads a bounded-row, newest-first
page of task-owned reviews from `GET /api/tasks/tasks/{id}/reviews`; pass the
returned opaque cursor to fetch the next page. `tasks_reviews_add(task_id,
stage, verdict, body, bot_id, target_revision=None, idempotency_key=None)`
appends a review via `POST /api/tasks/tasks/{id}/reviews`. Stage is `DESIGN` or
`IMPLEMENTATION`; verdict is `APPROVE`, `APPROVE_WITH_CHANGES`,
`REQUEST_CHANGES`, or `COMMENT`. The POST body contains `stage`, `verdict`,
`body`, and optional `targetRevision` and `idempotencyKey`; **never an author**.
Pass your own `bot_id` as the actor in `X-Agent-Bot-Id`; BawtHub derives the
stored author. Use a stable `idempotency_key` when retrying an uncertain write.
Reviews do not change task status or overwrite `response` (the executor's
summary); a design review may occur before the task enters REVIEW.

## Important environment variables

| Variable | Default | Purpose |
|---|---|---|
| `LLM_BAWT_MCP_ALLOWED_HOSTS` | `127.0.0.1:*,localhost:*` | Host allowlist for HTTP clients |
| `LLM_BAWT_MCP_SERVER_VERBOSE` | `1` | Verbose logging |
| `LLM_BAWT_MCP_SERVER_DEBUG` | `0` | Debug logging |
| `LLM_BAWT_TASK_API_URL` | `http://frontend-prod:3000` | Task API base URL for task-related tools (BawtHub prod frontend's service name on the shared `llm-bawt_default` network; no Traefik hostname alias) |
| `LLM_BAWT_APP_BASE_URL` | `http://localhost:8642` | Used by some self tools |

## Task association middleware (TASK-701)

`src/llm_bawt/mcp_server/task_association.py` provides the trusted
current-turn context path for the BawtHub task tools:

- **`TaskTurnCapabilityMiddleware`** — an ASGI middleware that reads the
  MCP HTTP header `X-LLM-Bawt-Task-Turn-Context` from the request scope
  and binds it to a `ContextVar` for the lifetime of one request. No
  process-local state assumptions across Redis/SDK/MCP hops.
- **`open_task_turn_context()`** — Fernet-verifies the capability token
  (6 h TTL), returns a `TaskTurnContext` dataclass with the app-known
  `session_id`, `turn_id`, `trigger_message_id`, `bot_id`, `user_id`,
  `issued_at`. Raises `TaskTurnContextError` on absent/expired/invalid
  tokens — MCP tools fail closed with a clear error message.
- **`associate_current_task(task_ref)`** — reads the capability from the
  ContextVar, PUTs `source: AGENT` to BawtHub's internal listener at
  `${BAWTHUB_TASK_ASSOCIATION_INTERNAL_URL}/internal/tasks/{task_ref}/associations`.
  The listener is container-only. Every BawtHub call also carries the service
  bearer (TASK-1472): derived from the master Fernet key by default, or the
  `BAWTHUB_SERVICE_TOKEN` / `BAWTHUB_SERVICE_TOKEN_FILE` override. Raw
  identifiers are never accepted as MCP tool arguments.

Task tools that use this path (`src/llm_bawt/mcp_server/task_tools.py`):

| Tool | Trusted-context behavior |
|---|---|
| `tasks_associate_current(task_id)` | Requires capability; fails closed. |
| `tasks_update(..., associate_current_turn=true)` | Applies update; attempts association after. Association-only branch when no update fields set. |
| `tasks_create(..., associate_current_turn=true)` | Creates task; attempts association on returned `shortId`. |

Text detection (`TASK-N` in prompts) is never authority — quoted, negative,
and incidental references must not persist a link.

See
[`agent-skills/agent-system/task-conversation-architecture.md`](../../agent-skills/agent-system/task-conversation-architecture.md)
for the cross-system diagrams, all 16 verified invariants, and the
internal-listener contract on the BawtHub side.

## Related docs

- [docs/INTER_BOT_COMMUNICATION.md](../docs/INTER_BOT_COMMUNICATION.md)
- [docs/BACKGROUND_SCHEDULER.md](../docs/BACKGROUND_SCHEDULER.md)
- [docs/CLAUDE_CODE_BRIDGE.md](../docs/CLAUDE_CODE_BRIDGE.md)
