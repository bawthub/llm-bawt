# Authentication, user model and Personas UI

> Part of the bawthub skill. Moved verbatim from `bawthub/SKILL.md` (TASK-1640); the rules stay in the parent.

## Authentication & user model

Reverse proxy injects `x-forwarded-email`/`x-forwarded-user` on OAuth-gated hosts; absent on `<dev-host>` (LAN, no auth). Email → llm-bawt `entity_id` via `/v1/profiles/lookup` (cached 5 min). User-switcher (`AppMainNav.tsx`) lists `entity_type=user` profiles, writes `selectedUser` to Zustand + localStorage (`bawthub-selected-user`); a real lookup `entity_id` always beats stale localStorage. `display_name` comes from the llm-bawt profile (`/tools/profile`).

A Playwright MCP context can be pre-seeded with a `bawthub-subject=<user>` cookie and the `bawthub-selected-user` localStorage record so the dev host opens without the "Who are you?" picker; use the installation's own owner identity, never a guessed one.


### Personas (TASK-477)

Backend (registry, `prompt_override_id`, `/v1/personas`) is llm-bawt — see that skill. Frontend: chat 3-dot menu (`StatusActionsPopover.tsx`) → Personas tile → `PersonaSwitcher.tsx` modal (switch/edit/create/delete; agent bots get one-tap session reset on switch; edits are SHARED across bots using the persona). `QuickSystemPromptEditor.tsx` `auto` mode edits whichever prompt is live. Persona CRUD in `tools/api.ts`; `prompt_override_id` threaded through `lib/useBots.ts`.
