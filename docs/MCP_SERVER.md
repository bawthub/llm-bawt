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

`tasks_reviews_list(task_id, limit=20, cursor=None)` reads a bounded, newest-first
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
  No bearer (TASK-793) — the listener is container-only and the docker network
  boundary is the auth layer. Raw identifiers are never accepted as MCP tool
  arguments.

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
