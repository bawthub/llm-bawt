# llm-bawt data model (PostgreSQL, DB `askllm`)

Read this before reasoning about conversation storage, session/thread scoping,
history retrieval, tool-call rendering, or any query against the chat tables.
Query the live DB only via `pgq llm-bawt` (never inline the password).

## The one fact that explains the whole schema

**Conversation storage has three partitioned parents; most relationships remain logical, but some are FK-enforced.**
- **The conversation timeline, memory, and forgotten-message stores are 3 LIST-partitioned
  tables keyed by `bot_id`** (TASK-571, cutover 2026-07-20): `messages`, `memories`,
  `forgotten_messages` — each a partitioned parent with one partition per bot
  (`messages_p_<bot>`, etc.). **This replaced the old per-bot sharding** (`<bot>_messages`
  × 11); those shard tables are gone. A new bot = one auto-provisioned partition on first
  write (`ensure_partitions`), not DDL. Schema changes are now ×1 on the parent.
- **Composite PK `(bot_id, id)`** on all three (the partition key must be in the PK). `id`
  is a UUID and still effectively unique within a partition, so bot-scoped `WHERE id=:id`
  queries are unaffected. Anything referencing a message by id from another table must
  carry `bot_id` too.
- **Two access modes:** the `PostgreSQLMemoryBackend` points its `_*_table_name` attrs at
  the **partition** (`messages_p_<bot>`) for bot-scoped work (partition-direct — cross-bot
  leakage is physically impossible); cross-bot features (`search_all_messages`, spotlight,
  scheduler sweeps) query the **parent** with `bot_id` in the select/`WHERE` (pruning to a
  partition). The old UNION-ALL-over-`information_schema` machinery is deleted.
- **FKs are selective.** `messages.session_id → sessions.id` and
  `tool_call_records.turn_id → turn_logs.id` are enforced; the middle
  `turn_logs.trigger_message_id → messages.id` edge remains logical, as do many
  other application joins. FK-driven ERDs therefore show only part of the
  conversation domain — use a DB row browser such as NocoDB, not a complete
  model map.

## The conversation core — four tables in a chain

```
sessions ──< messages ──< turn_logs ──< tool_call_records
  (thread)   (timeline,    (per request)   (per tool call)
              partitioned
              by bot_id)
```

Only the middle edge is **logical** (no FK):
- `messages.session_id` → `sessions.id` (**FK enforced**)
- `turn_logs.trigger_message_id` → `messages.id` (**logical only**: the **user** message that triggered the turn; carry `bot_id` to hit the composite key)
- `tool_call_records.turn_id` → `turn_logs.id` (**FK enforced**, ON DELETE SET NULL)

| Table | Grain | Role | Key columns |
|---|---|---|---|
| `sessions` | 1 per thread `(bot,user)` | thread registry | `id`, `bot_id`, `user_id` (nullable — leak source), `status` (`active`/`completed`), `started_at`, `ended_at` |
| `messages` (partitioned by `bot_id`) | 1 per rendered bubble | **canonical timeline** | PK `(bot_id, id)`, `role` (`user`/`assistant`/`summary`/`system`), `content`, `session_id`, `timestamp`, `reasoning`, `attachments`, `summarized` |
| `turn_logs` | 1 per `/v1/chat/completions` | request audit | `id`, `bot_id`, `user_id`, `trigger_message_id`, `response_text` (snapshot, NOT canonical), `tool_calls_json` (audit copy), `token_usage_json`, `latency_ms`, `parent_turn_id` |
| `tool_call_records` | N per turn | **what the chat renders as tool cards** | `id`, `turn_id`, `bot_id`, `user_id`, `tool_name`, `arguments_json`, `result_text`, `iteration`, `tool_use_id`, `approval_status` (~97k rows) |

### Non-obvious facts that will bite you

- **The chat view = timeline + enrichment, two fetches.** `/api/chat/history` →
  `messages` (the bubbles); `/api/chat/tool-calls` → `tool_call_records`,
  merged onto messages by turn/message id. Tool cards come from
  **`tool_call_records`**, NOT `turn_logs.tool_calls_json` (that's the audit copy).
- **You cannot rebuild the timeline from `turn_logs`.** Turns attach only to
  **user** messages; `assistant`/`summary`/`system` rows have **no** turn. Summaries
  and system messages are produced by background jobs / seeding and exist **only**
  in `messages`. And not every user message has a turn_log (older imports).
- **`turn_logs.response_text` ≠ displayed text.** It's the raw execution snapshot;
  the canonical, edited, displayed assistant text is the `messages` row.
- **`messages` is the source of truth** for display: mutable (edit/delete via
  `forgotten_messages`, recall), paginated by `timestamp`, searchable
  (`content_trgm` gin index), and it carries `session_id` + `idx_messages_session` (templated per partition).

### Session / thread scoping (the correct approach)

Filter **`messages` by `session_id`** — it's the canonical timeline and the
only table in the chain with a `session_id` column + index. Tool cards ride along
by the message→turn→tool chain. `turn_logs` and `tool_call_records` have **no
`session_id`** — they are enrichment reached via the trigger message, never the
thing you scope on. Making turn_logs primary would drop summaries/system/edited
text — a strictly dumber view.

`sessions` invariant (TASK-284): one `active` row per `(bot_id, user_id)`. Legacy
NULL-`user_id` rows leaked (~503) and are reaped by the migration step.
`get_or_create_active_session` / `rotate_session` / `activate_session` live in
`memory/postgresql.py`. Indexes: `idx_sessions_bot_user_active`.

## The other clusters (singleton tables)

| Cluster | Tables |
|---|---|
| **Memory** | `memories` (pgvector, partitioned by `bot_id`), `forgotten_messages` (soft-deleted bubbles, partitioned), `entity_profiles`, `entity_profile_attributes`, `user_profiles` |
| **Agent execution** | `agent_runs`, `agent_events`, `agent_session_state` |
| **Approvals** | `tool_approval_policies` (rules), `tool_approval_requests` (audit), `chat_pending_questions` |
| **Config / models** | `bot_profiles`, `runtime_settings`, `prompt_templates` (+`prompt_template_versions`), `models`, `model_endpoints`, `access_paths`, `model_definitions`, `model_definitions_compat` |
| **Jobs** | `scheduled_jobs` → `job_runs` |
| **Media** | `media_generations`, `media_assets` |

## Enforced foreign keys (verify live; not every join is enforced)

As checked 2026-10-27, the live parent/non-partition tables have 12 FK
constraints; `messages` partitions inherit their parent's FK and also appear
in a raw `pg_constraint` listing. The conversation edges are called out above.

| Table | Column | → |
|---|---|---|
| `job_runs` | `job_id` | `scheduled_jobs.id` |
| `model_endpoints` | `model_id` | `models.id` |
| `model_endpoints` | `access_path_id` | `access_paths.id` |
| `bot_profiles` | `endpoint_id` | `model_endpoints.id` |
| `bot_profiles` | `last_session_id` | `sessions.id` |
| `messages` | `session_id` | `sessions.id` |
| `tool_call_records` | `turn_id` | `turn_logs.id` (ON DELETE SET NULL) |
| `prompt_schedules` | `job_id` | `scheduled_jobs.id` |
| `prompt_occurrences` | `job_id`, `run_id` | `scheduled_jobs.id`, `job_runs.id` |
| `gate_members` | `gate_id` | `system_gates.id` |
| `bot_profile_tags` | `tag_slug` | `bot_tag_catalog.slug` |

Recheck rather than assuming the count after schema changes:

```sql
SELECT c.conrelid::regclass AS tbl, c.conname, c.confrelid::regclass AS target
FROM pg_constraint c JOIN pg_class t ON t.oid = c.conrelid
WHERE c.contype = 'f' AND c.connamespace = 'public'::regnamespace
  AND NOT t.relispartition ORDER BY tbl, c.conname;
```

## Cross-database task ↔ conversation joins (TASK-700–706)

BawtHub owns three additional association tables in its own database
(`echo`) that logically join to the llm-bawt conversation chain. **No
foreign key crosses the boundary** — everything is a logical join by
canonical identifier.

| BawtHub table (`echo` DB) | Logical join to llm-bawt (`askllm` DB) | Meaning |
|---|---|---|
| `AgentConversationIdentity.sessionId` | `sessions.id` | Durable session ownership anchor; survives task detach (intra-bawthub FK RESTRICT). |
| `AgentTaskSession.sessionId` | `sessions.id` (via identity FK) | `(taskId, sessionId)` grain — navigation/context. Partial unique `WHERE isPrimary=true`. |
| `AgentTaskTurn.triggerMessageId` | `messages.id` (partition PK `(bot_id, id)`) | `(taskSessionId, triggerMessageId)` grain — attribution. |
| `AgentTaskTurn.turnId` (nullable) | `turn_logs.id` (`turn-<32 hex>`) | Optional; NULL for delivery-shape turns (`turn-delivery-<hex>` from `bots_send_message`). |

Attribution reads (from BawtHub's task-context aggregator) resolve
`AgentTaskTurn.triggerMessageId → turn_changed_files.trigger_message_id`
and use `/v1/turn-changed-files/{turn_id}/content` for immutable
turn-time file bytes. Session-scope for chat continuation still uses
`messages.session_id` — not `turn_logs`, which has no `session_id` column
by design (see the invariant list in the canonical ref).

**Canonical cross-system reference:**
[`agent-skills/agent-system/task-conversation-architecture.md`](../../agent-system/task-conversation-architecture.md)
— cross-DB ER diagram, five sequence diagrams, all 16 invariants, and pgq
verification recipes. Read that file before touching either side of the
join.

## Model catalog — where the context window actually lives

The **normalized catalog is `models` → `model_endpoints` → `access_paths`** (the
FK triple above). `model_definitions` (+`_compat`) is the **legacy alias table**
— its `extra` jsonb holds only descriptive bits like `pricing`, **NOT the
window.** Don't go looking for `context_window` in `model_definitions`; it isn't
there.

- **The window is `model_endpoints.context_window_override ?? models.default_context_window`** (a positive-int CHECK is enforced on `models.default_context_window`; the override is nullable). `models.default_context_window` is the model fact; the endpoint override is a per-access-path exception. Same pattern for `*_max_tokens`/`tool_support`.
- **Endpoint writes:** `PUT /v1/models/catalog/endpoints/{model}/{access}` is a full replace; any omitted field (`upstream_model_id`, pricing, overrides) is nulled. For a one-field change use `PATCH` on the same path: only supplied fields change, `serving_config` merges key by key (`null` removes a key), `endpoint_id` is a compare-and-set guard, and invalid values return 422.
- **Provider/harness compatibility instructions live on `access_paths.system_prompt_overrides`** as JSONB `{harness: text}`. `_dispatch_config(endpoint, harness)` selects only the active harness entry and exposes it as `provider_system_prompt`. This lets one provider setup cover every attached model without leaking Claude-proxy instructions into a native Codex bot sharing the same access path. Values are trimmed; blank entries are removed; API writes reject unknown harness keys and non-string values.
- **App-side resolution authority (TASK-602/609):** `Config.get_model_context_window(alias)` → `resolve_context_budget()` is the single Tier-2 window authority. Under the hood `resolve_model(ref, harness)` → `ModelCatalog.resolve()` → `_dispatch_config()` surfaces `context_window` onto the resolved def.
- **A bot resolves via its pinned `bot_profiles.endpoint_id`, not the bare alias.** A model with >1 endpoint (e.g. `grok-4.5` = `xai-chat` + `xai-responses`) is **`AmbiguousModelError` on a bare alias — even with the harness** (both endpoints are proxy-compatible). Resolve by the **int `endpoint_id`** (single-row, unambiguous) when you need a specific bot's window. `resolve_model()` only swallows `ModelNotFoundError`; Ambiguous/Incompatible propagate.
- **`bot_profiles.harness`** ∈ `chat | claude-code | codex | claude-proxy` (CHECK-enforced). **`claude-proxy`** is the value for proxy-routed bots that ride the claude-code bridge to a non-Anthropic provider (grok/gpt/glm) — distinct from `claude-code` (native Anthropic). The DB trigger requires every bot's `endpoint_id` to be set and harness-compatible.
- **Bridge side (TASK-609):** the app resolves the window and sends it as the `context_window` command field; the claude-code bridge consumes that scalar and overrides the Claude CLI's 200k default for **proxy** models only (direct-Anthropic defers to the SDK). There is **no** bridge-side window table anymore (the old `_proxy_context_window` was retired).

```bash
pgq llm-bawt -c "SELECT m.key, m.default_context_window, e.id AS endpoint_id, e.context_window_override, e.upstream_model_id FROM models m JOIN model_endpoints e ON e.model_id=m.id ORDER BY m.key;"
pgq llm-bawt -c "SELECT slug, harness, default_model, endpoint_id FROM bot_profiles WHERE bot_type='agent' ORDER BY harness, slug;"
```

## Regenerate / verify

```bash
pgq llm-bawt -c "\d sessions"                 # or \d messages / \d messages_p_loopy / \d turn_logs / \d tool_call_records
pgq llm-bawt -tAc "SELECT count(*) FROM information_schema.table_constraints WHERE constraint_type='FOREIGN KEY' AND table_schema='public';"  # → 4
pgq llm-bawt -tAc "SELECT role, count(*) FROM messages WHERE bot_id='<bot>' GROUP BY role;"   # user/assistant/summary/system only
```

## Summary formerly in the parent SKILL.md (TASK-1640)

## Data model

**→ `reference/data-model.md`** before reasoning about conversation storage, session scoping, tool-call rendering, or the model catalog. Load-bearing facts:

- Conversation and memory storage uses LIST-partitioned parent tables `messages`, `memories`, and `forgotten_messages` (keyed by `bot_id`); per-bot `<bot>_messages` shards are gone. `messages.session_id → sessions.id` is FK-enforced; not every relationship is merely conventional.
- Conversation core: `sessions ──< messages ──< turn_logs ──< tool_call_records`. `messages.session_id` and `tool_call_records.turn_id` are FK-enforced; `turn_logs.trigger_message_id → messages.id` is logical (no FK). `messages` is the canonical timeline; `tool_call_records` is what chat renders as tool cards (NOT `turn_logs.tool_calls_json`); you cannot rebuild the timeline from `turn_logs`.
- Session scoping = filter `messages` by `bot_id` and `session_id` (only timeline table in the chain with `session_id`); bot-scoped backend queries can target `messages_p_<bot>` directly.
- Model catalog: live = `models → model_endpoints → access_paths`; `model_definitions` is legacy alias. Context window = `model_endpoints.context_window_override ?? models.default_context_window`, resolved via the bot's int `endpoint_id`.
