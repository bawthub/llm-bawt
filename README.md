# llm-bawt

`llm-bawt` is the backend behind [BawtHub](https://bawthub.com/). It keeps each bot, thread, memory, profile, and task on the server so the same conversation can continue across the web app, voice surfaces, and the `llm` CLI.

The public site describes BawtHub as "one conversation, every device." This repo is the part that makes that true: one API, one history store, one memory system, one bot registry, and optional agent bridges that let bots do real work.

## What This Repo Powers

- Persistent bot threads shared across BawtHub web, voice, and CLI
- Bot profiles stored in Postgres, including prompts, voices, tool access, and default models
- OpenAI-compatible chat streaming at `POST /v1/chat/completions`
- Per-bot history, summaries, semantic memory, and user/bot profiles
- Agent backends for OpenClaw and Claude Code, with Codex bridge support still present
- Task submission/status APIs used by the BawtHub agent dashboard
- Media upload/storage routes for chat attachments and generated assets
- Local embedding and optional local model support through `local-model-bridge`

## BawtHub Architecture

```text
BawtHub surfaces
  web app          voice pipeline         llm CLI
      \                 |                  /
       \                |                 /
        -> llm-bawt service (:8642) <----
              |- chat streaming + tool loop
              |- bot profiles + model catalog
              |- history + summarization
              |- semantic memory + profile storage
              |- tasks + turn logs + media
              `- OpenAI-compatible API

        -> llm-bawt MCP server (:8001)
              |- memory tools
              |- profile/history tools
              `- task/self-recap tools

        -> optional bridges
              |- claude-code-bridge
              |- openclaw-bridge
              `- codex-bridge

        -> backing services
              |- PostgreSQL + pgvector
              |- Redis
              |- local-model-bridge
              `- Crawl4AI / Playwright MCP / provider APIs
```

## Main Runtime Pieces

### Core service

The FastAPI app on port `8642` is the system of record for BawtHub conversations. It owns chat streaming, bot resolution, model selection, memory retrieval, history persistence, profile access, task APIs, media routes, and admin endpoints.

### MCP server

The MCP server on port `8001` exposes memory/history/profile/task tools to agent runtimes. BawtHub's agent surfaces and bridge workers rely on this layer for structured tool access.

### Bridges

- `claude-code-bridge` runs Claude Agent SDK bots and now also supports provider-routed models through its Anthropic-compatible proxy path.
- `openclaw-bridge` connects Redis-backed agent dispatch to OpenClaw workers.
- `codex-bridge` is still available for Codex SDK based bots.

### Storage

- PostgreSQL + `pgvector` store bot profiles, message history, summaries, memories, turn logs, tasks, and entity profiles.
- Redis carries bridge command/event traffic.
- `local-model-bridge` serves embeddings and optional local inference.

## Key API Surfaces

These are the endpoints BawtHub depends on most:

- `POST /v1/chat/completions` — streaming chat/tool execution
- `POST /v1/chat/session/reset` — reset an agent-backed bot session
- `GET /v1/models` — OpenAI-compatible model list
- `/v1/models/catalog/*` — normalized model, access-path, endpoint CRUD and
  harness cascade filters. The alias-keyed `/v1/models/definitions/*` routes
  remain as a guarded compatibility surface during cutover.
- `GET /v1/bots` and `/v1/bots/{slug}/profile` — bot registry and config
- `GET /v1/history`, `/v1/history/around`, `/v1/history/search` — thread history
- `GET/POST /v1/memory/*` — semantic memory operations
- `GET/PATCH /v1/profiles/*` — user and bot profiles
- `POST /v1/tasks` and `GET /v1/tasks/{task_id}` — agent task pipeline entrypoints
- `GET /health` — service health

## Quick Start

### Docker stack

This is the closest match to how BawtHub uses the repo.

```bash
git clone https://github.com/bawthub/llm-bawt.git
cd llm-bawt

# Create .env manually in the repo root.
make up
```

Important compose services:

- `app` — FastAPI service + MCP server
- `redis` — bridge transport
- `claude-code-bridge`
- `openclaw-bridge`
- `codex-bridge`
- `local-model-bridge`
- `crawl4ai`
- `playwright-mcp`

Useful commands:

```bash
make up
make docker-dev
make logs
make restart-app
make docker-status
```

### Local development

```bash
git clone https://github.com/bawthub/llm-bawt.git
cd llm-bawt
./install.sh --dev
./server.sh start --dev --stdout
```

Useful commands:

```bash
make dev
make test
make typecheck
make lint
```

## Required Configuration

There is no checked-in `.env.example` right now. Create `.env` in the repo root for Docker, or `~/.config/llm-bawt/.env` for local CLI/service use.

Minimum practical config for the BawtHub stack:

```bash
LLM_BAWT_DEFAULT_USER=nick
LLM_BAWT_DEFAULT_BOT=nova

LLM_BAWT_POSTGRES_HOST=your-postgres-host
LLM_BAWT_POSTGRES_PORT=5432
LLM_BAWT_POSTGRES_USER=llm_bawt
LLM_BAWT_POSTGRES_PASSWORD=your-password
LLM_BAWT_POSTGRES_DATABASE=llm_bawt

REDIS_URL=redis://redis:6379/0
LOCAL_MODEL_EMBED_URL=http://local-model-bridge:8684

OPENAI_API_KEY=...
# or / and
LLM_BAWT_XAI_API_KEY=...
```

Notes:

- PostgreSQL must have the `vector` extension enabled.
- Bot profiles and model definitions are DB-backed now; YAML is legacy.
- Redis is required if you are using agent bridges.
- `LOCAL_MODEL_EMBED_URL` is the current embedding path; embeddings are no longer loaded in-process by the app.

## Repo Map

```text
src/llm_bawt/
  service/         FastAPI app, chat streaming, routes, tasks, usage
  mcp_server/      MCP server and tool implementations
  memory/          pgvector memory, extraction, consolidation, summaries
  media/           attachment/media storage and generation helpers
  profiles.py      entity profile management
  runtime_settings.py
                   bot profile persistence and cleanup helpers
  bots.py          runtime bot resolution from DB-backed profiles
  model_manager.py model catalog and alias handling
  search/          web search providers
  integrations/    Home Assistant, nextcloud, web fetch, provider links

src/agent_bridge/          shared bridge transport/runtime pieces
src/openclaw_bridge/       OpenClaw bridge
src/claude_code_bridge/    Claude Code bridge + proxy path
src/codex_bridge/          Codex bridge
src/local_model_bridge/    embeddings + local inference service
```

## BawtHub-Facing Behavior

The important design choice is that the thread does not live in the client. BawtHub surfaces are just views into server-side state held here:

- A phone voice turn and a desktop text turn can hit the same bot thread.
- A CLI `llm -b loopy "status?"` turn can resume the same conversation BawtHub shows in the browser.
- Tool calls, summaries, memories, and task progress are stored centrally instead of inside one app session.
- Bots can hand work to other bots through bridge and MCP tooling while the UI stays attached to the same thread.

## Additional Docs

- [docs/README.md](docs/README.md)
- [docs/MCP_SERVER.md](docs/MCP_SERVER.md)
- [docs/CLAUDE_CODE_BRIDGE.md](docs/CLAUDE_CODE_BRIDGE.md)
- [docs/CODEX_BRIDGE.md](docs/CODEX_BRIDGE.md)
- [docs/INTER_BOT_COMMUNICATION.md](docs/INTER_BOT_COMMUNICATION.md)
- [docs/BACKGROUND_SCHEDULER.md](docs/BACKGROUND_SCHEDULER.md)
- [docs/approval-policies.md](docs/approval-policies.md)
- [docs/usage-endpoint.md](docs/usage-endpoint.md)

## License

Elastic License 2.0 — see [LICENSE](LICENSE).
