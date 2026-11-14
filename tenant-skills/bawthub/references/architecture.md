# BawtHub architecture, environments and layout

> Part of the bawthub skill. Moved verbatim from `bawthub/SKILL.md` (TASK-1640); the rules stay in the parent.

## Frontend surfaces

Two frontend surfaces, both fronted by Traefik on the host: a **dev/HMR** surface (`bawthub-frontend-1`, live `./frontend` mount, instant `.ts/.tsx/.css` pickup) and a **baked prod** surface (`bawthub-frontend-prod-1`, changes only on a prod image rebuild). Each installation configures its own hostnames and sign-in; read them from its Traefik routes and environment rather than assuming another installation's.


**Host-only Make commands** (on the host from `~/dev/bawthub`; from a bridge, run them over SSH to the host): `make mode-status` (always run first if unsure) · `make dev-mode` / `make prod-mode` (toggle HMR only — **neither changes that canonical = baked prod**) · `make up-dev` · `make logs / logs-backend / logs-prod`.

## Architecture

```
Traefik (host:80)
├─ <prod host>                                     → frontend-prod  (baked)
│   └─ frontend-owned /api/* prefixes (priority=200) → frontend-prod
├─ <dev host>                                      → frontend       (vite dev + HMR)
│   └─ same /api/* prefixes (priority=200)         → frontend
└─ /api/* catch-all (priority=100)                 → backend (Python FastAPI)
```


```
External: llm-bawt :8642 (BAWTHUB_LLM_URL=http://host.docker.internal:8642) · STT :8090 · TTS :8089 · PostgreSQL :5432
```

**Background jobs:** the delivery dispatcher, approval callbacks, scheduled resets, maintenance, recovery and the ops reconciler all run in llm-bawt, not here. Before answering what one is doing, read **`../llm-bawt/reference/background-jobs.md`**: cadence, leader-only or not, what it writes, and what's built but not wired in.

## Directory Structure

```
bawthub/
├── frontend/
│   ├── vite.config.ts               # aliases (@/ → src/) + dev proxy /api → Hono
│   ├── Dockerfile.vite              # prod image: baked SPA + Hono on :3000
│   ├── hot-reloading.Dockerfile     # dev image: vite (5173) + tsx-watch Hono (3001)
│   ├── prisma/schema.prisma
│   ├── server/                      # Hono API (was Next.js src/app/api/*)
│   │   ├── index.ts                 # entry; serves dist/client + /api/*
│   │   ├── routes/                  # chat.ts (llm-bawt proxy + approvals), tasks.ts,
│   │   │                            #   docker-unraid.ts, remaining.ts, health.ts
│   │   ├── middleware/              # auth (entity_id from proxy headers), proxy helper
│   │   └── lib/                     # prisma client, llmUrl, agentActivity, botInfo…
│   └── src/                         # Vite SPA (client ONLY — no Prisma/server code)
│       ├── main.tsx  routes.tsx     # React Router 7; lazy-imports @/app/* pages
│       └── app/                     # (app)/(fullscreen)/: chat/ voice/ avatar/
│                                    # (app)/(dashboard)/: tasks/ tools/ relay/ studio/ docker/
│                                    # dir names are kept-from-Next.js convention, not routing
├── bawthub/                         # Python voice backend: main_websocket.py (/v1/realtime),
│                                    #   handler.py (STT→LLM→TTS), stt/ tts/ llm/
├── voices.yaml                      # TTS voice definitions (seed only — DB is truth)
└── docker-compose.yml
```

## Voice Pipeline

Mic → browser WebSocket → backend `/v1/realtime` (subprotocol "realtime", OpenAI Realtime API event format) → STT (Moshi) → llm-bawt `/v1/chat/completions` → TTS (Moshi) → speaker. Client sends `input_audio_buffer.append` (base64 opus); server sends `response.audio.delta` / `response.text.delta`.

## Chat Proxy Routes

Frontend proxies `/api/chat/*` → llm-bawt `/v1/*`: `POST /api/chat` → `/v1/chat/completions` · `bots` → `/v1/bots` · `models` → `/v1/models` · `history/*`, `memory/*` → same-named `/v1/*` · `/api/chat/proxy/[...path]` → catch-all `/v1/*`.

## Environment

Check where you are first (`whoami`/`pwd`). The host has node/npm/gh/docker/git locally. Agent bridges = `bridge` (`/home/bridge/dev`): Node 24.21.0, npm/npx 11.19.0, Corepack 0.36.0, pnpm 10.32.1, make and gh are installed (TASK-1061/1062). Run focused checks (named test files, lint on changed files) in the worktree; project-wide `tsc`, full suites and Python suites follow [houses and test placement](../../llm-bawt/reference/houses.md).


## When a rebuild is needed beyond a code edit (moved from SKILL.md, TASK-1640)

| You changed… | Dev | Prod (only when shipping) |
|---|---|---|
| `.ts`/`.tsx`/`.css` (client) | Vite HMR auto-reloads | prod rebuild |
| `server/**` (Hono API) | `tsx watch` auto-restarts (~1s) | prod rebuild |
| `vite.config.ts`, `package.json`, `tsconfig.json`, `.env` | dev server auto-restarts (kills SSE/ws — one reconnect is normal) | prod rebuild |
| add a dependency (`pnpm add`) | Authorized `make restart-ui` workflow: recreates frontend, installs shared deps and generates Prisma; **not equivalent to the plain restart op** | prod rebuild |
| `Dockerfile.vite` / `hot-reloading.Dockerfile` | `make rebuild-ui` | prod rebuild |
| `prisma/schema.prisma` | `make db-push-ui` then `make db-generate-ui`, then request `bawthub.restart-frontend` through `ops_run`; generation does not restart the frontend (push re-applies the HNSW DDL Prisma can't model). If it warns about data loss, **stop and fix the drift** — never work around it with hand-written SQL. Raw-SQL tables must declare their DB defaults in the schema (`@default(now()) @updatedAt`); removing a model/field = drop it in the DB too | — |

**Adding a hostname or a frontend-owned `/api/*` prefix:** → **`references/routing-hostnames.md`** (AdGuard + `deploy/traefik/dynamic/routes.yml` + Vite allowedHosts; both `frontend-api` AND `frontprod-api` rules must stay aligned or the Python backend's priority-100 catch-all swallows the route → 404).
**Background jobs:** the delivery dispatcher, approval callbacks, scheduled resets, maintenance, recovery and the ops reconciler all run in llm-bawt, not here. Before answering what one is doing, read **`../llm-bawt/reference/background-jobs.md`**: cadence, leader-only or not, what it writes, and what's built but not wired in.

## Frontend conventions (moved from SKILL.md, TASK-1640)

- **Theme:** `slate` palette (never `zinc`); text `white/90` primary, `white/60` muted, `white/25` placeholder. Fonts: Satoshi (body/UI), SF Mono stack — no new imports.
- **Prisma 7:** `@prisma/adapter-pg`; connection in `prisma.config.ts`. Prisma is **server-only** (Hono `server/lib/prisma.ts`) — never import into the Vite client bundle.
- **API routes are Hono, not Next.js:** `c.req.param()`/`c.req.query()`, return `c.json(...)`; routes register under `server/routes/*`.
- **Client pages are plain React** (React Router 7) — no SSR, no server components, no `force-dynamic`. Dev bundler is Vite (`pnpm dev` = vite + tsx watch).
- **`(app)/` import paths:** files under `src/app/(app)/` importing from `src/app/` need `../` — the group dir adds a nesting level.

## MCP memory

Use **`bot_id="claude"`** for conversation-history MCP tools (`get_messages`, `get_recent_context`, `search_memories`, …) — the default `bot_id` returns empty/test data.
