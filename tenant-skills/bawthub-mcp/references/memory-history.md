# Memory, history, profiles

## Selection and scope

- **Active work:** `tasks_board` shows compact lanes and counts; open only the
  relevant task with `tasks_get_context`. See [tasks.md](tasks.md#board-focus-and-archive).
- **Past decisions or conversation:** `context_recall(request={"query": "…"})`
  uses a separate utility-model context and returns bounded cited quotes, not a
  transcript. Its default `sources` are messages and memories; explicitly add
  `tasks` for shared task evidence and optionally narrow with `task_ids`.
  Inspect the coverage for every requested source. `no_match` is meaningful
  only with complete successful coverage; `insufficient_evidence` and unavailable
  sources are not absence. A historical owner quote does not prove current WIP.
  Follow-ups repeat `sources`/`task_ids`, retain intent and refresh evidence.
- **Locate a literal, identifier or time window across permitted bots, browse
  recent history, or follow up insufficient recall:** `messages_find` returns
  compact session cards with best/recent matches and bot/age/speaker facets.
  Narrow with `bot`, `speaker`, `since` or `until`; a `handle` alone pages.
  It is not permission to read a protected namespace.
- **Expand only what is relevant:** `messages_context` expands one find card
  using its `handle` and `card_ref`, or a permitted bot plus message ID.
  `messages_get_by_id` finds a UUID/unique prefix (minimum 8 characters) with
  optional before/after neighbors; a missing message returns null. Use bounded
  `messages_get` (one thread) or `messages_search_all` for exact text/IDs, full
  context or diagnostics — these return bulky history, not search cards.
- **Extracted facts, not conversation evidence:** `memory_search(query,
  bot_id=<self>)` is local; `memory_search_all` searches allowed bots and may
  restrict by bot_id. `memory_list_sources` lists allowed namespaces/counts;
  `memory_search_source(source, query)` narrows to one allowed source. Do not
  infer access from a hit or treat a memory alone as the full conversation.
- `context_get_recent` combines recent messages and memories; query optionally
  guides memory retrieval. It does not accept user_id/max_age_seconds arguments.

Time formats differ intentionally: `messages_find` since/until accept bounded
relative `Nh`/`Nd` (e.g. `24h`, `7d`), ISO date/datetime or Unix seconds;
invalid values return `unavailable` with `invalid_request` and accepted formats.
`messages_search_all` since/until accept ISO dates, ISO datetime or Unix seconds;
naive timestamps are UTC. `memory_search_all` since/until take Unix seconds;
`messages_get.since_seconds` is a look-back window. Message search sort_by is
relevance or recent. Similarity/importance thresholds are 0–1; bound results.

`messages_get(session_id=...)` scopes a durable thread. summaries_only selects
summary records. exclude_summarized prevents repeating raw text already covered
by summaries. Without session_id the existing summary-aware history path applies.
`messages_get_for_summary` expands one stored summary; `messages_mark_recalled`
marks expansion metadata (a write).

## Memory writes

Use your own bot_id. `memory_store` accepts content, tags, importance and source
message IDs. `memory_update` patches non-null fields; do not assume null clears
values. `memory_update_meaning` changes intent/stakes/emotional_charge,
recurrence_keywords or updated_tags. `memory_supersede` marks an old record
superseded by a replacement ID (or the DELETED marker).

`memory_delete` deletes one record. `memory_delete_by_source_messages` removes
facts derived from specific messages. `memory_clear` deletes the entire bot's
distilled memory: never use as routine cleanup. `memory_list_recent` returns
backend-native rows; `memory_list_high_importance` ranks by importance.

`facts_extract` takes role/content messages and explicit bot_id/user_id;
store=true persists extracted facts. use_llm=false selects heuristics.

## History writes

`messages_add` inserts history, not an execution. Supply session_id or user_id
for the intended thread; roles are user/assistant/system. message_id and Unix
 timestamp are optional. Attachments use asset reference objects such as
`{"asset_id":"ma_...","kind":"image"}`; they are not inline file bytes.

Preview before bulk changes:
- messages_preview_recent / messages_preview_since_minutes show affected windows.
- messages_ignore_recent / messages_ignore_since_minutes move history to ignored
  storage; messages_ignore_by_id targets one message.
- messages_preview_ignored reads ignored history;
  messages_restore_ignored restores ALL ignored messages for the bot.
- messages_remove_last_partial deletes the latest message if its role matches.
  Despite its name, the MCP storage path does NOT check whether it is partial.
  Do not use it as an automatic safe cleanup; verify the exact message first.
- messages_clear deletes ALL messages. It is not archive, reset or ignore.

## Working memory (TASK-1281)

`working_memory_list` / `working_memory_create` / `working_memory_update` keep
your own short todo list for the current user. Spec:
llm-bawt `docs/WORKING_MEMORY.md`.

- **Off by default.** Each bot needs the bot-scope runtime setting
  `working_memory_enabled=true`, and only claude-code bots qualify. Otherwise
  you get `not_enabled`; protected bots get `protected_bot`.
- **No bot or user arguments.** Identity comes from the signed turn
  capability. Outside a trusted turn every call, reads included, returns
  `unauthorized`. You can't read or write another bot's or user's items.
- **create:** `idempotency_key` must be a canonical UUID. Use a fresh one per
  new item and reuse it only to retry the same create. A same-key retry returns
  the existing item (`replayed: true`); the same key with different fields is
  `idempotency_conflict`; once that item has been deleted by the owner, the
  key returns `removed`. 20 open items (planned/active/blocked) at most
  (`cap_exceeded`). `kind=initiative_link` needs `initiative_task_id` and
  takes no next_action/blockers.
- **update:** pass `expected_revision` from your last read. A stale revision
  returns `revision_conflict` with the current item and writes nothing; re-read
  and retry. Omitted fields stay as they are; to clear next_action,
  checkpoint or priority, list them in `clear`. `done`/`cancelled` archives.
  There's no delete tool: Nick deletes via REST `/v1/working-memory`.
- **Errors are JSON** `{"error": code, ...}`. `unavailable` means storage
  failed. It is NOT an empty list: don't conclude you have no items.

## Profiles and maintenance

`profile` handles structured attributes, not semantic facts. entity_type is
user or bot; entity_id is the corresponding identifier. Actions:
summary/list/get/set/delete. category/key select attributes; set requires value.
Only set/delete mutate. Use your own bot for bot profile changes.

`system_stats` reads counts. memory_regenerate_embeddings writes embeddings.
memory_consolidate defaults dry_run=true; system_run_maintenance defaults
 dry_run=false and may consolidate, detect recurrence, prune decay or clean
orphans according to flags. Preview and scope maintenance deliberately; tool
availability does not authorize deleting someone else's state.
