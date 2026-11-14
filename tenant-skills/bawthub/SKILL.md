---
name: bawthub
description: Work in BawtHub, the chat, voice and task dashboard. Routes to frontend, Hono, Prisma, voice, authentication and deployment references.
---

# BawtHub

User-facing chat, voice and task dashboard. Repo `~/dev/bawthub` on the host, `/home/bridge/dev/bawthub` in bridges. Vite/React + Hono + Prisma; Python voice backend. Canonical branch `main`.

Use [git-commits](../git-commits/SKILL.md) for isolated worktrees and publication, [agent-system](../agent-system/SKILL.md) for tasks, and [bawthub-ops](../bawthub-ops/SKILL.md) for service changes. These are their authoritative homes; do not add parallel workflow rules here.

Use the task worktree for edits and focused tests; full suites, project-wide typechecks and branch UI previews follow [houses and test placement](../llm-bawt/reference/houses.md). UI work runs in a house so its preview link goes on the card for the user's live feedback. A change to how something looks is not sent for publication until its owner has screenshotted the changed element on that preview at phone width (and at desktop width where the layout differs), in the state that stresses it (longest label, badge showing, menu open), and attached the screenshots to the card. The integrator returns a visual change whose card lacks them. The shared checkout feeds dev HMR with integrated `main`. Production is a separate ops release with live verification. Prisma uses the BawtHub PostgreSQL database; use pgq for live queries. Inspect actual client errors instead of asking users to clear browser state.

Keep changes bounded and verify the actual behavior. Routine design details belong to the owner. Refactoring is justified by the change, not a file-size threshold.

Repos prompt bodies come from the backend's canonical product-default catalog and shared tenant seeder, not frontend seed copies. Prompt saves remain tenant overrides; follow [product defaults](../llm-bawt/reference/product-defaults.md) for approved export, reset and provenance.

To mock up a BawtHub UI change in chat, use a ```html-card title="…" styles="app"``` fence and copy the component's real markup from source; only classes the app already uses are styled, so use inline `style` for new looks.

## Technical references

| Task | Read |
|---|---|
| Dev vs prod surfaces, host Make commands, Traefik/edge map, directory layout, voice pipeline, chat proxy routes, environment/tooling, MCP memory `bot_id` | [references/architecture.md](references/architecture.md) |
| Shipping to prod | references/prod-release.md (not included in this install) |
| New hostname or frontend-owned `/api/*` prefix | references/routing-hostnames.md (not included in this install) |
| Chat module | [references/chatui-architecture.md](references/chatui-architecture.md) |
| Notifications/toasts, shared dispatcher and Events history | [references/design-system.md — Shared notifications](references/design-system.md#shared-notifications--one-system) |
| UI design treatments, section bar (page controls), workspace surfaces, `TaskTag` | [references/design-system.md](references/design-system.md) |
| Relay pills / bot-bar delivery feedback | [references/inter-bot-relay.md](references/inter-bot-relay.md) |
| Task data models, dispatch, PM cron, `/api/tasks/*` routes | [references/task-system.md](references/task-system.md) |
| Task ↔ conversation association | [references/task-conversation.md](references/task-conversation.md) |
| Auth headers, user switcher, Playwright's default identity, Personas UI (backend is llm-bawt) | [references/auth-and-personas.md](references/auth-and-personas.md) |
| Mobile keyboard/scroll/layout | [references/mobile-troubleshooting.md](references/mobile-troubleshooting.md) |
| Any other symptom | [references/pitfalls.md](references/pitfalls.md) |
| Vite migration history | references/vite-migration-architecture.md (not included in this install) |
