# System prompt generation — end to end (chat bots & agent bots)

How the `system` message is assembled for every llm-bawt bot: which sections,
in what order, gated by which flag, and where each byte comes from. Also covers
the per-turn user-message prefixes (agent bots) and the multi-layer augmentation
the agent bridges bolt on.

> **Read this before** touching prompt assembly, adding a bot flag, debugging
> "why isn't my prompt change showing", or answering "why aren't summaries
> going in". For the caching/resume mechanics specifically, pair it with
> [`agent-system-prompt.md`](agent-system-prompt.md).

---

## 0. TL;DR

- **One builder, two paths.** Both chat and agent bots build the same system
  prompt via `PromptBuilder` in `core/base.py`. Then agent bots **early-return
  with only the current user message** (no history, no summaries); chat bots
  append windowed history + summaries.
- **The live entry point is `BaseLLMBawt.query()` → `_build_context_messages()`**
  in `src/llm_bawt/core/base.py`. `core/pipeline.py` (`RequestPipeline`) is a
  parallel implementation that the current `query()` **does not invoke** — the
  base.py docstring calling it "the RequestPipeline" is stale. Don't edit
  pipeline.py expecting production impact; verify first.
- **Agent bots get downstream augmentation layers** the app prompt builder never
  sees: a `<runtime-context>` model block, an optional access-path + harness
  provider system message, a `## MCP Tool Context` block, and the Claude Agent
  SDK's own outer system prompt.
- **Summary/history delivery is governed by `history_scope`** (4-value enum) for
  **both** bot types, via the shared `build_context_payload` handler (TASK-493/518).
  The old `include_summaries` and `seed_summary_on_new_session` flags are dead
  legacy. On agents, `history_scope` is delivered through the **session seed**
  (`build_context_seed`), not the base.py message list. See §7 and the
  authority: [`history-context-flow.md`](history-context-flow.md).

---

## 1. `PromptBuilder` — the assembly mechanics

`src/llm_bawt/core/prompt_builder.py`

- Sections are named, each with a `position` (lower = earlier) and `enabled`
  flag. `build()` sorts enabled sections by position and joins with `\n\n`.
- Empty/whitespace-only content is silently dropped (`add_section` early-returns).
- Standard positions (`SectionPosition`):

  | pos | constant | section name(s) |
  |----:|----------|-----------------|
  | -1  | `DATETIME` | `temporal_context` — **removed from live prompt (TASK-288)**, still used by pipeline.py |
  | 0   | `USER_CONTEXT` | `user_context` |
  | 1   | `BOT_TRAITS` | `bot_traits` |
  | 2   | `BASE_PROMPT` | `base_prompt` |
  | 3   | `MEMORY_CONTEXT` | `cold_start_memory` |
  | 4   | `TOOLS` | `tools` |
  | 5   | `CLIENT_CONTEXT` | `client_context` |
  | 6   | `GLOBAL_INSTRUCTIONS` | `global_instructions`, `agent_global_prompt` |
  | 50  | `CUSTOM` | `tts_output` and any bot-specific additions |

  > Position 6 is shared by two different sections (`global_instructions` and
  > `agent_global_prompt`). Both can coexist; join order between equal positions
  > follows insertion order.

The builder is created once in `_init_system_prompt()` (stable sections) and
`.copy()`-ed per turn in `_build_context_messages()` so per-turn sections don't
accumulate.

---

## 2. Stable sections — built once (`_init_system_prompt`, base.py:259)

Added at bot init, cached on `self._prompt_builder`:

| Section | pos | Gate | Content | Source |
|---------|----:|------|---------|--------|
| `user_context` | 0 | `_db_available and profile_manager` | `## About the User\n{summary}` | `profile_manager.get_user_profile_summary(user_id)` |
| `bot_traits` | 1 | `_db_available and profile_manager` | `## Your Developed Traits\n{summary}` | `profile_manager.get_bot_profile_summary(bot_id)` |
| `base_prompt` | 2 | always (if body non-empty) | the bot's personality prompt **or active persona** | `_resolve_base_prompt()` |
| `global_instructions` | 6 | `bot.requires_memory` | `GLOBAL_SYSTEM_PROMPT` (conversation-recall + cross-bot memory/search guidance) | `prompt_builder.GLOBAL_SYSTEM_PROMPT` |

**Persona resolution (`_resolve_base_prompt`, TASK-477):** if
`bot.prompt_override_id` points to a valid, non-empty row whose
`category == PERSONA_CATEGORY`, its `body` replaces `system_prompt` for the
`base_prompt` section. Invalid/missing/empty/non-persona → warn and fall back to
`bot.system_prompt`. The stored `system_prompt` is never mutated; `NULL`
override = default persona.

---

## 3. Per-turn sections — added on each query (`_build_context_messages`, base.py:498)

Starts from `self._prompt_builder.copy()`, then conditionally adds:

| Section | pos | Gate | Notes |
|---------|----:|------|-------|
| `tools` | 4 | **chat bots only**, plus `tool_format != "none"` AND (`bot.uses_tools` → full tool set, else `self.memory` → read-only `MEMORY_TOOL`) | Native agent harnesses own their tool schemas/protocol and receive no app-generated tool prompt. Chat bots use the model-specific ReAct/native prompt from `get_tools_prompt(tool_format=...)`. |
| `client_context` | 5 | `self._client_system_context` set | `## Client Context\n{...}` — e.g. HA device list passed by the caller. |
| `cold_start_memory` | 3 | `self.memory and not _ha_mode` AND `history ≤ 3 msgs` | Injects top memories only when history is thin; steady-state bots use the memory tool on-demand instead. |
| `tts_output` | 50 | `_tts_mode` **and NOT agent backend** | `chat.tts_output_instructions` from prompt registry. Agent bots get a user-message prefix instead (see §6) — injecting both double-doses the constraint. |
| `agent_global_prompt` | 6 | **agent backend** AND `agent_global_prompt_enabled` setting true | `agents.global_prompt`, bot-scoped resolve → global fallback. Empty body injects nothing even when flag on. Rides the cacheable prefix. |

`temporal_context` (date/time) is **deliberately NOT added here** (TASK-288) — it
changed every minute at the top of the prompt and busted the cache. The removal
comment lives at base.py:511-516; do not re-add it to the system prompt.

After building, `system_content = builder.build()` becomes the `system` message.

---

## 4. Where the paths fork — agent vs chat (base.py:664)

```python
if self.model_definition.get("type") in ("agent_backend", "claude-code"):
    # AGENT: current user message only (+ per-turn prefixes), then RETURN
    ...
    return messages
# CHAT: fall through to history + summary assembly
```

### Chat bots (base.py:725-786)
- `history = self.history_manager.get_context_messages(max_tokens=budget)`.
  Budget = `max_context_tokens` setting, or auto `ctx_window - max_output`.
- Iterate history: include `user`, `assistant`, and `summary` roles, plus
  `system` messages that start with `[Tool Results]`/`[Tools used:`.
- **Two independent gates (TASK-518):** raw messages kept iff `include_history`,
  `summary` rows kept iff `include_summaries` — both derived from `history_scope`
  via `scope_flags()`, ANDed with the `session_memory_continuity` mirror. The
  per-request `include_summaries` flag can only force summaries *off*, never on.
- `_ha_mode` special-case: drop summaries entirely, keep only last 6
  user/assistant/tool-result messages (stateless device commands).
- Inline response-style directive (see §6) is stamped onto the outbound **copy**
  of the latest user message (never the stored one).

### Agent bots (base.py:664-723)
- Only `Message(role="user", content=<current prompt + prefixes>)` is appended,
  then the function returns. **No history, no summaries, no memory context in the
  message list** — the agent backend owns its own transcript.

---

## 5. Agent-bot augmentation layers (added AFTER the app hands off)

For agent bots the app's system prompt is passed down and downstream layers wrap
it before/around it. Order as the model finally sees it:

1. **`<runtime-context>` model block** — `agent_backends/claude_code.py:~60`.
   Prepended to the system prompt: `<runtime-context>\nmodel: {model}\n</runtime-context>`
   plus the "report exactly this model" instruction. Grounds the model's
   self-identity (the SDK doesn't surface its own model id). Claude-code backend
   hard-fails if `agent_backend_config.model` is missing — no silent fallback.
2. **App-built bot/persona prompt** — the stable `PromptBuilder` output described
   above.
3. **Provider+harness system message** — optional
   `access_paths.system_prompt_overrides[harness]`, resolved by
   `ModelCatalog._dispatch_config()` as `provider_system_prompt` and appended by
   `agent_backends/claude_code.py`. It is trimmed, interpolation-free, cacheable,
   and refreshed from the catalog at dispatch time, so an edit applies on the
   next turn without a bot reload or `/new` (the changed prefix causes one cache
   miss). A `claude-proxy` entry on shared `openai-oauth` never reaches Codex.
4. **`## MCP Tool Context` block** — `claude_code_bridge/bridge.py:~1054`.
   Appended: tells the agent its `bot_id` and how to pass it to bawthub MCP
   tools (memory/message/profile entity ids). Only when `system_prompt` +
   `_mcp_servers` + `bot_slug` all present.
5. **Claude Agent SDK outer prompt** — the harness's own system prompt (tool
   defs, skills list, deferred-tool machinery, etc.). Not llm-bawt code.

The effective-config inspector implementation lives in
`core/effective_config.py`; it reports the provider section's access-path source,
active harness, character length, and downstream order.

Delivery:
- `clients/agent_backend_client.py:~180` extracts every `system` message from
  the message list and joins into `config["system_prompt"]`.
- `agent_backends/agent_bridge.py:~191` passes `system_prompt=config.get(...)`
  into the Redis `send_command`.
- **claude-code** (`bridge.py:~1282`): `ClaudeAgentOptions(system_prompt=...)` on
  **every** `query()` (TASK-288 — SDK does NOT lock it on resume; byte-stable →
  full-prefix cache read).
- **codex** (`codex_bridge/bridge.py:~859`): no system slot — prepended to the
  user message as `f"{system_prompt}\n\n---\n\n{message}"`.

---

## 6. Per-turn user-message prefixes (agent bots only, base.py:682-721)

These ride on the **outbound user message**, never the system prompt (keeps the
cacheable prefix byte-stable) and are never persisted. Slash commands
(`/new`, …) bypass all of them. Stack order when multiple fire:

```
[agent_user_prefix]\n\n[agent_voice_prefix]\n\n[response_style]\n\n---\n\n{prompt}\n\n[context_suffix]
```

| Prefix | Gate | Source |
|--------|------|--------|
| `chat.agent_user_prefix` | `_inject_user_prefix` (request flag / composer toggle) | prompt registry |
| `chat.agent_voice_prefix` | `_tts_mode` | prompt registry |
| `response_style` | inline keyword in prompt: `\btldr\b` / `\beli5\b` / `\bdeep[ -]?dive\b` | hardcoded (base.py:616-633) |
| `context_suffix` | TASK-391 attachment manifest present | caller-supplied, appended last |

For **chat** bots only `response_style` applies, stamped onto the latest user
message copy (base.py:776).

---

## 7. Summaries & history delivery — the unified model (TASK-493/518)

**Summaries are generated for every bot** (background summarization writes
`role="summary"` rows into the LIST-partitioned `messages` table, keyed by
`bot_id`). Whether they — and raw recent messages — reach the model is now governed by
**one control for both bot types**: the `history_scope` enum, decoded into two
**independent** flags (`include_history`, `include_summaries`) and applied by the
shared `build_context_payload` handler.

| `history_scope` | raw messages | summaries |
|---|---|---|
| `inline+summaries` (default) | ✅ | ✅ |
| `inline` | ✅ | ❌ |
| `summaries` | ❌ | ✅ (dense summary-only) |
| `none` | ❌ | ❌ |

`session_memory_continuity` is a **mirror** (`= scope != "none"`), not a separate
switch. The old `include_summaries` (chat request shim, can only force-off) and
`seed_summary_on_new_session` (agent) flags are **dead** — do not use them.

### Chat bots
`build_context_payload` fills the `summary`/`regular` buckets per the two flags;
the caller reads `payload.inline_history`. (`core/base.py` chat branch.)

### Agent bots
Agents early-return in base.py before message assembly (the SDK transcript is
their memory). Summaries/history reach them via the **session seed**: on
`/new`/cold-start, `maybe_build_session_seed` → `build_context_seed` runs the
**same** `build_context_payload` with `delivery="seed"`, decoding the **same**
`history_scope`. So an agent set to `summaries` gets a dense summary-only seed;
`none` gets a cold session. No separate agent flag.

**The old summary-delivery finding is obsolete:** it described summaries failing to reach an
agent because `include_summaries` was inert and `seed_summary_on_new_session` was
absent. That whole split is gone — a single `history_scope` now drives delivery
for every bot, verified end-to-end (seed endpoint + turn logs, TASK-518).

---

## 8. Complete bot-flag → prompt impact catalog

Fields in `bot_profiles` (see `bots.py`, `runtime_settings.py`). "Prompt impact"
= does it change the assembled `system` message or the outbound messages.

| Flag | Type / default | Prompt impact |
|------|----------------|---------------|
| `system_prompt` | text / "You are a helpful assistant." | **`base_prompt` section** (pos 2), unless persona override active. |
| `prompt_override_id` | int / NULL | **Replaces** `base_prompt` body with a persona template (TASK-477). NULL = base. |
| `requires_memory` | bool / true | Gates `global_instructions` (GLOBAL_SYSTEM_PROMPT), cold-start memory, and profile sections. On chat bots it also allows the read-only memory tool; native agent harnesses provide their own MCP memory tools. |
| `uses_tools` | bool / false | Chat bots: gates the **full** app-generated tools section. If false but `requires_memory`, a chat bot still gets the read-only memory tool. Native agents receive no app-generated tools section because Claude Code/Codex own their schemas and protocol. |
| `uses_search` | bool / false | Adds web-search tools **inside** the tools section (requires `uses_tools`). |
| `uses_home_assistant` | bool / false | Adds HA tools inside the tools section (requires `uses_tools`). |
| `history_scope` (runtime setting) | str / `inline+summaries` | **The live control (both bot types):** 4-value enum → two independent flags gating raw messages + summaries (§7). |
| `session_memory_continuity` (runtime setting) | bool / true | **Mirror** of `history_scope` (`= scope != "none"`), not a switch. Legacy readers use it. |
| `include_summaries` | bool / true | **DEAD legacy.** Chat-only per-request shim that can only force summaries *off*. Inert on agents. |
| `tts_mode` (default) / request `tts_mode` | bool / false | Chat: adds `tts_output` section. Agent: adds `chat.agent_voice_prefix` user-message prefix instead. |
| `voice_optimized` | bool / false | **Not a prompt flag** — post-processes the *response* (scrub markdown/emotes for TTS). |
| `agent_global_prompt_enabled` (runtime setting) | bool / false | Agent bots: adds `agent_global_prompt` section (`agents.global_prompt`). |
| `agent_backend` | str / NULL | Selects the bridge (claude-code/codex). Determines the augmentation layers in §5. No app-side section of its own. |
| `agent_backend_config.model` | str | claude-code: injected as `<runtime-context>` block. Hard-required (no fallback). |
| `agent_backend_config.seed_summary_on_new_session` | bool / false | **DEAD legacy.** Superseded by `history_scope` on the seed path (§7). |
| `bot_type` | "chat"/"agent" | Picks the fork at base.py:664 — history+summaries vs early-return. |
| `default_model` / `default_voice` / `color` / `avatar` / `avatar_render` / `name` / `description` / `slug` / `nextcloud_config` / `include_in_global_search` | — | **No prompt impact** — routing, model selection, UI metadata, or search-scope only. |

Request-level flags (`service/schemas.py`) that shape the prompt:
`augment_memory` (skip memory sections when false), `include_summaries`,
`tts_mode`, `inject_user_prefix`, `client_system_context`, `ha_mode`.
`extract_memory` is post-processing only.

Generation params (`temperature`, `top_p`, `max_output_tokens`) and context
budgets (`max_context_tokens`, summarization tunables, memory tunables) affect
**how much** history/memory is included or the sampling — not the prompt's
section content directly.

---

## 9. Caching / byte-stability rules (TASK-288)

- The system prompt is the **cacheable prefix**. Anything that varies per turn
  and sits in it busts the cache for the whole transcript behind it.
- Kept OFF the system prompt on purpose: temporal grounding (removed),
  response-style keywords (→ user message), TTS constraint for agents
  (→ user-message prefix).
- The system prompt is **re-sent every turn** (resume included) and is
  byte-stable, so it reads from cache (~90% hit) instead of busting it.
- Agent system prompts are pinned per session. Edits take effect at the next supported session boundary; see [`agent-system-prompt.md`](agent-system-prompt.md).

---

## 10. Debugging checklist

- **"My system_prompt edit isn't showing"** → check `prompt_override_id`; an
  active persona replaces `base_prompt`. Also confirm you edited the DB row
  (there is no file fallback; `bots.yaml` was removed in `da881c2`).
- **"Summaries aren't in context"** → check `history_scope`: it must include
  `summaries` (`inline+summaries` or `summaries`). Confirm summary rows exist in
  `messages` for the bot (`WHERE bot_id = ...`) and that
  `summarization_max_in_context` (default 5) isn't 0.
  For agents, remember delivery is via the **seed** — only rebuilt on `/new`/cold
  start, so change scope *then* start a new session. (Ignore `include_summaries`/
  `seed_summary_on_new_session` — dead flags.)
- **"Prompt cache keeps missing"** → something per-turn leaked into the system
  prompt. Diff two turns' prompts; check for temporal/response-style/TTS leakage.
- **Inspecting the live prompt** → `PromptBuilder.get_verbose_summary()` /
  `build_with_debug()` list sections, positions, and sizes. The agent bridges log
  the system-prompt length per turn.
- **Editing pipeline.py expecting a behavior change** → don't. The live path is
  `base.py`; `RequestPipeline` is dormant. Verify with a grep for an actual
  `RequestPipeline(...).run()` call before relying on it.

---

*System-prompt generation verified 2026-07-10 against llm-bawt @ `2a6802d`
(TASK-477). §7 + summary-delivery claims updated 2026-07-11 for TASK-518
(history/summaries decoupled into independent `history_scope` axes; legacy
`include_summaries`/`seed_summary_on_new_session` retired) — verified end-to-end
via the seed endpoint and turn logs. See `history-context-flow.md` for authority.*

## Summary formerly in the parent SKILL.md (TASK-1640)

### System prompt generation (chat + agent, every flag)

**→ `reference/system-prompt-generation.md`** for full assembly order + every bot flag's prompt impact. Key gotcha: `include_summaries` and `seed_summary_on_new_session` are **dead legacy flags** — `history_scope` is the live control, `session_memory_continuity` its derived mirror (TASK-493/518).
