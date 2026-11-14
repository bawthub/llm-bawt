# Personas — switchable system-prompt overrides (TASK-477)

> Part of the llm-bawt skill. Read before creating/switching personas or debugging persona/system-prompt override behavior.

A bot can swap its base system prompt for a **persona** on the fly (e.g. make
`helper` a therapist) without ever mutating its stored prompt.

- **Storage:** personas are rows in the **prompt registry** (`prompt_templates`,
  `src/llm_bawt/prompt_registry.py`) with `category='persona'`,
  `scope_type='global'` — so they are **global / shareable across bots** and get
  append-only versioning (`prompt_template_versions`) on every edit. `key` is
  `persona.<slug>`.
- **Which one is active:** `bot_profiles.prompt_override_id` (nullable int,
  `src/llm_bawt/runtime_settings.py`) points at the active persona.
  **`NULL` = use the bot's base `system_prompt` (default persona).** Switching
  **never** overwrites the base column — clearing back to `NULL` restores it
  verbatim. Mirrored onto the `Bot` dataclass in `bots.py`.
- **Injection:** `core/base.py::_resolve_base_prompt()` (~L223) REPLACES the
  "base_prompt" section with the persona body when an override is active. A
  stale / missing / non-persona / empty id falls back to the base prompt and
  logs a warning — **a bad override never crashes a turn**.
- **REST API:**
  - `GET /v1/personas` · `POST /v1/personas` (`{title, body}`)
  - `GET|PUT|PATCH|DELETE /v1/personas/{id}` (PUT/PATCH `{title?, body?}` bumps a
    new version; edits are **shared** — every bot using that persona sees them)
  - **Switch:** `PATCH /v1/bots/{slug}/profile` with `{"prompt_override_id": <id>}`
    to activate, or `{"prompt_override_id": null}` to clear back to base.
    Bots switch another bot with the MCP tool `bots_profile_patch(bot_id,
    {"prompt_override_id": <id>}, reason)` (audited compare-and-set; bawthub-mcp
    references/sessions-delivery.md), not a raw PATCH.
  - `/v1/bots` and the bot-profile response now expose `prompt_override_id`.
- **Agent-bot caveat (claude-code/codex):** the system prompt is the
  cacheable prefix, so switching persona mid-session changes that prefix. **Pair
  a switch with a session reset** — `POST /v1/chat/session/reset {"bot_id":"<slug>"}`
  — and `POST /v1/admin/reload-bots` so the running BotManager picks up the new
  `prompt_override_id`.
- **Frontend:** the chat 3-dot menu → **Personas** tile (PersonaSwitcher modal);
  see the bawthub skill.

```bash
# Create a persona, then point a bot at it with the MCP tool
# bots_profile_patch(bot_id="helper", changes={"prompt_override_id": 21}, reason="...")
curl -s -X POST http://app:8642/v1/personas -H 'Content-Type: application/json' \
  --data @/tmp/persona.json          # {"title":"Therapist","body":"..."}
curl -s -X POST http://app:8642/v1/admin/reload-bots
curl -s -X POST http://app:8642/v1/chat/session/reset -d '{"bot_id":"helper"}'
```

## Summary formerly in the parent SKILL.md (TASK-1640)

- **Personas (TASK-477):** a bot can swap its base prompt via `bot_profiles.prompt_override_id` → prompt-registry row (`persona.<slug>`, global/shared, versioned). `NULL` = base prompt; switching never overwrites the base column; bad override falls back safely. Agent bots: pair a switch with session reset + `reload-bots` (prompt is the cache prefix). Full API + gotchas: **`reference/personas.md`**.
