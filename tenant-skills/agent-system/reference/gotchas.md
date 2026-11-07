# Agent system: implementation gotchas

> Part of the agent-system skill. Moved verbatim from the parent SKILL.md (TASK-1640).

Rules 1, 3, 4, 5 and 7 of the parent's former Rules list (rules 2, 6, 8 and 9 stay in the parent).

1. The `DOCS_TEMPLATE` in the docs route is the **fallback** — the system first tries to fetch a `agents.docs` prompt template from the LLM backend. If you only edit the hardcoded template, your changes may be shadowed by the remote one.
3. The docs endpoint accepts both `shortId` (e.g. `TASK-42`) and `id` (cuid) in the `?task=` param — it checks `startsWith("TASK-")` to decide which to query.
4. API routes are **Hono** (`frontend/server/routes/*`), not Next.js. Read path/query params with `c.req.param("id")` / `c.req.query("foo")` and return with `c.json(...)` — no `params`/`searchParams` Promises.
5. Prisma 7 uses `@prisma/adapter-pg` — there's no `datasourceUrl` on the client constructor. Connection config lives in `prisma.config.ts`.
7. **Frontend theme**: Popups/dropdowns use `bg-slate-900/95 backdrop-blur shadow-2xl` with `border-white/15`. Text colors follow `white/60` (muted), `white/90` (primary), `white/25` (placeholder/empty). Never use `bg-zinc-*` — the app uses `slate`. Match existing components in `agents/projects/page.tsx` and `agents/_components/` for reference.
