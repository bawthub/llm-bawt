# llm-bawt architecture map

Moved from the parent `SKILL.md` (TASK-1640). Read when you need the layout, routes, request pipeline or config names.

## Tech Stack

Python 3.12+/FastAPI (port 8642) · PostgreSQL 15+ with pgvector · Redis (agent bridge transport) · MiniLM embeddings via `local-model-bridge` HTTP (app is CPU-only) · LLM clients: OpenAI, Grok/xAI, OpenAI Responses API · Local inference: GGUF (llama-cpp), vLLM in `local-model-bridge` · Agent SDKs: Claude Agent SDK, Codex SDK · Package manager: **uv**

## Architecture

```
BawtHub Frontend (port 3000)
  ↓ /v1/chat/completions
llm-bawt (port 8642)
  ├─ Bot Manager       → personality, model selection
  ├─ Memory System     → pgvector semantic search
  ├─ Tool System       → memory, web search, home assistant
  ├─ History Manager   → conversation persistence
  ├─ Profile Manager   → user attributes
  └─ Agent Backends    → Redis → Claude Code Bridge (native Anthropic or proxy)
```

## Directory Structure

```
src/llm_bawt/
├── service/            # api.py (FastAPI app), background_service.py (core orchestrator,
│                       #   mixins), chat_streaming.py, chat_stream_worker.py,
│                       #   turn_lifecycle.py, turn_logs.py, tasks.py (bg queue),
│                       #   routes/, jobs/, usage/ (subscription usage endpoint)
├── clients/            # base.py (LLM client ABC), openai_client.py, grok_client.py,
│                       #   responses_client.py, agent_backend_client.py (Redis routing)
├── memory/             # postgresql.py (pgvector), embeddings.py (HTTP → bridge),
│                       #   consolidation.py, extraction/
├── mcp_server/         # server.py (port 8001, runs INSIDE app container), task_tools.py,
│                       #   media_tools.py, self_tools.py, storage.py
├── search/             # Brave, Tavily, DuckDuckGo, Reddit
├── media/              # Media library (arr integration, object store)
├── models/  adapters/  agent_backends/  tools/  integrations/  core/  shared/  utils/  cli/
├── bots.py             # Bot personality system (DB-driven)
├── approval_policies.py  profiles.py  prompt_registry.py  runtime_settings.py

src/agent_bridge/       # Shared bridge package (approval engine, events, publisher,
                        #   subscriber, session queue)
src/claude_code_bridge/ # Claude Agent SDK bridge
    └── proxy/          # Anthropic-compat proxy → reference/anthropic-proxy.md
src/codex_bridge/       # Legacy native Codex backend (no mansion service)
src/local_model_bridge/ # GPU inference + embeddings (ports 8683/8684)
```

## API Endpoints

| Route | Method | Purpose |
|-------|--------|---------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat (streaming/non-streaming) |
| `/v1/chat/abort` | POST | Abort in-flight generation by turn ID |
| `/v1/chat/session/reset` | POST | Reset agent backend session |
| `/v1/bots` · `/v1/bots/{slug}/profile` | GET/POST/PUT/PATCH | Bot list + personality management |
| `/v1/models` | GET | List available models |
| `/v1/memory/*` · `/v1/history/*` | GET/POST | Memory + history |
| `/v1/tasks` · `/v1/tasks/{id}` | POST/GET | Background tasks (`?wait=true&timeout=30`) |
| `/v1/personas` · `/v1/personas/{id}` | CRUD | Persona registry → `reference/personas.md` |
| `/v1/admin/*` | POST | Reload bots, models, settings, tool-approval policies |
| `/v1/usage` | GET | Subscription/plan usage → `reference/claude-credentials-usage.md` |
| `/v1/uploads` · `/v1/uploads/{id}[/thumb|/preview]` | POST/GET/DELETE | Content-addressed media store on Garage S3 — **any file type** (TASK-847): images → WebP + variants, everything else verbatim. Response carries `public_url` (BawtHub `/api/chat/uploads/<id>`). Agents: `bawt-upload <path>` prints the user-clickable link — paste it as a markdown link; `http://app:8642/...` URLs are curl-only |
| `/v1/inter-bot-deliveries*` | POST/GET | Durable deliveries → `reference/inter-bot-deliveries.md` |
| `/v1/turn-changed-files*` | GET/POST | Turn-time file snapshots → `reference/task-conversation.md` |
| `/health` | GET | Service diagnostics |

## Chat Request Pipeline

Every `/v1/chat/completions` request: 1 Pre-process → 2 Context build (system prompt from bot config + profile) → 3 Memory retrieval (pgvector, if `augment_memory`) → 4 History filter (token budget) → 5 Message assembly → 6 Execute (LLM client + tool loop, or agent bridge) → 7 Post-process (memory extraction, history persist, turn log).

Request extensions beyond OpenAI spec: `bot_id`, `augment_memory`, `extract_memory`, `include_summaries` (dead legacy — see below), `tts_mode`, `user` (memory/history scoping), `session_id`.

## Agent backend routing

```
App → AgentBackendClient → Redis command queue → bridge → external agent
                                              ↘ Redis event stream → app
```
Current mansion: Redis → claude-code-bridge → Claude Agent SDK (stdio),
using native Anthropic or the in-process proxy for GPT/other providers.
GPT model choice does not imply a Codex harness. The separate native
`codex-bridge` service is retired; its Python backend and optional Codex houses
remain compatibility code. Both harnesses use the shared `AgentEvent` format.

### Bridge incident routing

"Claude bridge bots not responding" → check the llm-bawt Claude bridge path first (bridge container health/logs, Redis flow, app routing) — NOT legacy `ask-llm`/webhook docs.

## Agent Task Execution

BawtHub owns task state; llm-bawt is the execution engine. Dispatch = BawtHub calls `/v1/chat/completions` with the task prompt → bot resolved → agent bridge (or direct LLM) → SSE stream back → agent PATCHes step status to BawtHub's `/api/tasks/*` during work → turn log records everything.

## Memory System

pgvector, namespaced by `(bot_id, user_id)` — never cross-contaminate. Local MiniLM embeddings (no API calls). Temporal decay (facts ~180d, events ~30d half-life), contradiction supersede, consolidation dedup >0.85 similarity, access tracking. Exposed as MCP tools on port 8001 (server runs inside the app container).

## Tool System

Native function calling (OpenAI/Grok) or ReAct text mode (GGUF/vLLM). Tools: `memory`, `history`, `profile`, `search`, `web_fetch`, `home`, `model`, `time`.

## Key Config

`LLM_BAWT_SERVICE_PORT=8642` · `LLM_BAWT_POSTGRES_*` · `LLM_BAWT_REDIS_URL` · `LLM_BAWT_DEFAULT_BOT` · `LLM_BAWT_DEFAULT_MODEL_ALIAS` · `LLM_BAWT_MEMORY_DECAY_ENABLED` · `LLM_BAWT_SCHEDULER_ENABLED` · `OPENAI_API_KEY` · `LLM_BAWT_XAI_API_KEY`
