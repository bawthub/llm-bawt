# Authentication, user model and Personas UI

> Part of the bawthub skill. Moved verbatim from `bawthub/SKILL.md` (TASK-1640); the rules stay in the parent.

## Authentication & user model

User identity depends on the installation's configured sign-in path: owner enrollment/password session, trusted reverse-proxy identity, or legacy identity selection. Inspect `GET /api/chat/auth` (`sign_in`, `user`, `email`, `entity_id`) and the auth configuration; LAN placement or a single route's 401 does not establish that password/session enforcement is enabled. `sign_in="enroll"` with null identity is not an authenticated owner session.

Trusted proxy headers (`x-forwarded-email`/`x-forwarded-user`) are not interchangeable with a local password session. Email maps to llm-bawt `entity_id` through `/v1/profiles/lookup` (cached 5 min). The legacy user-switcher (`AppMainNav.tsx`) stores `selectedUser` in Zustand and localStorage (`bawthub-selected-user`); a real lookup `entity_id` beats stale localStorage. `display_name` comes from the llm-bawt profile (`/tools/profile`).

**Backend caller authentication is independent.** Hono/voice service keys identify those programs to llm-bawt; they do not authenticate the browser user. The backend's `api_auth_mode` and MCP's `mcp_auth_mode` are separate from BawtHub sign-in. Finishing an observation window does not change any of these settings automatically. See [llm-bawt security boundaries](../../llm-bawt/reference/security-model.md).

A Playwright MCP context can be pre-seeded with a `bawthub-subject=<user>` cookie and `bawthub-selected-user` localStorage for legacy identity selection; use the installation's owner identity, never a guessed one. Those values do not create a password session or bypass enrollment/session enforcement.

### Personas (TASK-477)

Backend (registry, `prompt_override_id`, `/v1/personas`) is llm-bawt — see that skill. Frontend: chat 3-dot menu (`StatusActionsPopover.tsx`) → Personas tile → `PersonaSwitcher.tsx` modal (switch/edit/create/delete; agent bots get one-tap session reset on switch; edits are SHARED across bots using the persona). `QuickSystemPromptEditor.tsx` `auto` mode edits whichever prompt is live. Persona CRUD in `tools/api.ts`; `prompt_override_id` threaded through `lib/useBots.ts`.
