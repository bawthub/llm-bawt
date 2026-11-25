---
name: bawthub
description: Work in BawtHub, the chat, voice and task dashboard. Routes to frontend, Hono, Prisma, voice, authentication and deployment references.
---

# BawtHub

User-facing chat, voice and task dashboard. Vite/React + Hono + Prisma; Python voice backend.

## Updating bundled skills on an installed tenant

Bundled skills ship inside the app and agent-bridge images. On the tenant host, use `bash deploy/install.sh update` from the existing Git-connected installation, or copy the whole newer official release folder to the host and run `bash bawthub-release/install.sh` for an offline install. See `deploy/README.md` in the installer for both paths. Custom-pinned images may stay on their current bundle.

Check the manifest in each running image, not the upstream skills checkout. A newer bundled skill doesn't replace a local custom skill or override. Don't send an installed owner to upstream authoring, `skills-sync`, or a Settings upgrade control that isn't available on their install.

## Development

Use the BawtHub checkout configured for your environment. Canonical branch `main`.

Use [git-commits](../git-commits/SKILL.md) for isolated worktrees and publication, [agent-system](../agent-system/SKILL.md) for tasks, and [bawthub-ops](../bawthub-ops/SKILL.md) for service changes. These are their authoritative homes; do not add parallel workflow rules here.

Use the task worktree for edits and focused tests; full suites, project-wide typechecks and branch UI previews follow [houses and test placement](../llm-bawt/reference/houses.md). For visual changes, the owner attaches screenshots of the changed element at phone width (and desktop where the layout differs), in a representative state that stresses it (long label, badge, open menu). Follow the [preview-to-dev workflow](../agent-system/SKILL.md) for handoff. The shared checkout feeds dev HMR with integrated `main`. Production is a separate ops release with live verification. Prisma uses the BawtHub PostgreSQL database; use pgq for live queries. Inspect actual client errors instead of asking users to clear browser state.

Keep changes bounded and verify the actual behavior. Routine design details belong to the owner. Refactoring is justified by the change, not a file-size threshold.

Data refreshes from events, never timers: no polling `refetchInterval` or data `setInterval`. Refresh from the feature's server event on the existing app stream (add one there if missing, never a second app-wide SSE) plus `app:resync` (`onAppResync`/`useLiveRefresh` in `frontend/src/app/chat/stream/`). The stream controller alone detects wake, network return and reconnect; features add no raw `visibilitychange`/`focus`/`pageshow`/`online` listeners. `noPolling.test.ts` enforces this; its allowlist is only for UI clocks, animation, transport and recorded exceptions, each with a reason.

Repos prompt bodies come from the backend's canonical product-default catalog and shared tenant seeder, not frontend seed copies. Prompt saves remain tenant overrides; follow [product defaults](../llm-bawt/reference/product-defaults.md) for approved export, reset and provenance.

To mock up a BawtHub UI change in chat, use a ```html-card title="…" styles="app"``` fence and copy the component's real markup from source; only classes the app already uses are styled, so use inline `style` for new looks.

## Technical references

| Task | Read |
|---|---|
| A requested tool needs an account, credential, or other setup | [BawtHub MCP: When a tool needs setup](../bawthub-mcp/SKILL.md#when-a-tool-needs-setup) |
| Dev vs prod surfaces, host Make commands, Traefik/edge map, directory layout, voice pipeline, chat proxy routes, environment/tooling, MCP memory `bot_id` | [references/architecture.md](references/architecture.md) |
| Shipping to prod | [Service operations](../bawthub-ops/SKILL.md); use the installation’s private release runbook for local activation |
| New hostname or frontend-owned `/api/*` prefix | [Architecture and routing](references/architecture.md) |
| Chat module | [references/chatui-architecture.md](references/chatui-architecture.md) |
| Notifications/toasts, shared dispatcher and Events history | [references/design-system.md — Shared notifications](references/design-system.md#shared-notifications--one-system) |
| UI design treatments, menu layers (section bar, page header row), workspace surfaces, `TaskTag` | [references/design-system.md](references/design-system.md) |
| Relay pills / bot-bar delivery feedback | [references/inter-bot-relay.md](references/inter-bot-relay.md) |
| Task data models, dispatch, PM cron, `/api/tasks/*` routes | [references/task-system.md](references/task-system.md) |
| Task ↔ conversation association | [references/task-conversation.md](references/task-conversation.md) |
| Auth headers, user switcher, Playwright's default identity, Personas UI (backend is llm-bawt) | [references/auth-and-personas.md](references/auth-and-personas.md) |
| Prisma schema change: tenants apply only committed migrations, so a change needs its migration, a regenerated `tenant-expected-catalog.json` (`--write-catalog`) and a `tenant-migration-audit.json` entry; run `check-tenant-migrations.mjs` against an empty DB before handoff | repo `deploy/README.md` (tenant migrations), `frontend/scripts/check-tenant-migrations.mjs` |
| Mobile keyboard/scroll/layout | [references/mobile-troubleshooting.md](references/mobile-troubleshooting.md) |
| Any other symptom | [references/pitfalls.md](references/pitfalls.md) |
