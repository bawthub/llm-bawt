# Working Memory: design r12

Task: TASK-1281. Status: design approved by Loopy at r12 (cmvld893800580mpz7wcnjmzx, sha256 57fd8e40); Nick's N1-N3 and OI-R8 rulings were recorded afterwards, with no design change. Slice 1 (store, trusted MCP tools, owner REST) is implemented at llm-bawt `d87528c` and is off by default (`working_memory_enabled` is false for every bot). Slices 2 and later are not implemented.

Base: llm-bawt `9e874dc` (master), which commits Byte's TASK-1318 pin. `src/llm_bawt/service/system_prompt_pin.py` there is byte-identical to the r11 dependency (sha256 prefix `c100554a06b0`). Files cited from r11 that the commit didn't touch are unchanged since `4883515`. Appendix C lists the citations.

Scope: Nick ruled B+notice (TASK-1281 step 23). Working memory renders into the agent system prompt. The prompt is pinned per session by TASK-1318. Working-memory edits reach a live session at its next boundary. Until then, the bot gets a one-line notice on the next ordinary turn that starts after an edit (section 9.1 says which messages count).

r10 replaces r9's delivery design (snapshot table, delta, receipts, bindings, stall counter, sweep, coldness predicate, bridge provenance). The store, tools, REST path, trust rules and Nick's rulings carry over. Rules that are new in r10 are tagged [r10]; rules changed in answer to Loopy's r10 review (cmvlcc925003y0mpz7n763i6e) are tagged [r11], and rules changed in answer to Loopy's r11 review (cmvlcwe8z004z0mpzd483gogq) are tagged [r12], so the reviewer can attack them. Appendix A.00 answers each r11 finding and Appendix A.0 each r10 finding. Rules that Loopy accepted in the r9 review (cmvl6znx8002a0mpz0sawfyrd) are tagged [L9 OI-n].

Reviewer: Loopy.

---

## 1. Problem

Nick's goal (2026-10-20): "A new system for a memory type that lets bots manage their own list of todo items… added to every continuity system prompt so u can plan things out and be able to reset context and retain direction." On 2026-10-22 he prioritized it because it keeps bots prioritized across resets.

What the code does today:

- **Agent bots get no history from llm-bawt.** They get the current message plus the system prompt. Continuity is the harness's on-disk transcript, resumed by SDK session id (`claude_code_bridge/send_handler.py:208-245`; `resume_id = None if reset_requested else req.thread_resume_id` at :210).
- **A reset drops direction.** Every reset goes through `SessionResetLineage.rotate` (`memory/session_reset_lineage.py:19`), called from five places (section 10.1). A new SDK session starts with at most a history seed (`service/routes/history_seed.py:208`). Nothing structured carries "what I'm doing and what's next" across the boundary.
- **The Claude Code CLI freezes the system prompt per session.** This is the documented default since CLI 2.1.265 / SDK 0.2.153. The CLI records the prompt on the session's first request and reuses it on every resumed turn. A changed prompt is ignored until a compaction or a new session; after `/compact`, the prompt passed on the first turn after the compaction is recorded (TASK-1318 steps 0, 1, 4). This also holds for gpt bots on the claude-code bridge (TASK-1318 step 2). So rendering live state into every prompt is silently ignored on claude-code (TASK-1281 step 20).
- **TASK-1318 pins the prompt in llm-bawt.** The session points at the turn whose prompt is in effect, and every turn sends those bytes. The pin moves only at a boundary (section 8.2).
- **Compaction drops direction too.** The SDK summarises the transcript. Manual `/compact` moves the pin; CLI auto-compaction re-records the same pinned bytes and is not a boundary (TASK-1318 step 6).
- **Initiative state is task-bound.** `AgentInitiativeState` (bawthub `frontend/prisma/schema.prisma:432-460`, `taskId @id`) and `mcp_server/initiative_tools.py` hold objective, nextAction, completedEvidence and blockers for one task, under a lease, with CAS. A bot's personal, cross-task direction has no home.
- **The trusted turn capability exists.** `task_turn_context.py:23` mints it for `{"claude-code", "codex"}`. `mcp_server/task_association.current_task_turn_context()` returns `(session_id, turn_id, trigger_message_id, bot_id, user_id, issued_at)`, with a 6h TTL, and fails closed.

## 2. Goals and non-goals

Goals:

- G1. Each (bot, user) keeps a structured list of working-memory items that survives `/new`, resets, compaction and model switches.
- G2. After a boundary, the bot's system prompt shows its open items: up to 8 in full (including the next action), the rest as titles, then an omitted count (section 8.1). Items beyond the full-render cap are reachable only through the tool.
- G3. Working memory adds no cache creation on a warm turn. The section changes only at boundaries the pin already has. A notice is a short append to the user message.
- G4. A change the session didn't make (Nick's UI edit, a write from another session of the same bot) produces a notice on the session's next eligible started turn. Messages steered into a running turn don't carry it (section 9.1). The content reaches the prompt at the next boundary. The notice prompts a tool read; it does not deliver content, and the design does not check that the bot reads.
- G5. Writes are bound to a trusted identity. The model can't set a bot or user parameter.
- G6. Opt-in per bot, off by default. A bot that hasn't opted in, and every protected (`private`-tagged) bot, gets a byte-identical prompt.
- G7. Nick can list, edit, cancel, soft delete and hard delete any opted-in or previously opted-in bot's items.

Non-goals:

- NG1. Guaranteed delivery of item content mid-session. Nick chose B+notice over C (step 23).
- NG2. Changing a live pin mid-session, including on delete (Nick B3, section 11).
- NG3. Scrubbing turn logs or SDK transcripts (Nick D4, section 11).
- NG4. Codex, openclaw and chat-bot delivery. Enabling is refused there (section 8.3).
- NG5. Replacing initiative state or the harness's in-turn todo list (section 4).
- NG6. Semantic memories tagged "todo".

## 3. Alternatives

Nick, verbatim (task step 15): "Alternative hasn't been discussed since every prompt technically would generate live state from db".

| | What | Cost | Status |
|---|---|---|---|
| (A) Tool-only | Store plus tools; nothing in the prompt | After a reset the bot has no cue the list exists. Fails G2. | This is slice 1 on its own. |
| (B) Boundary pin | Section in the system prompt, frozen per session by TASK-1318 | External edits wait for the next boundary; a long-lived session can go hours or days. | Kept. |
| (B+notice) | (B), plus a one-line notice on the next ordinary message after an external edit | The notice depends on the bot calling the tool. Repeats are bounded (section 9). | **Chosen by Nick (step 23).** |
| (C) Pin + delta + receipt | r9 | 7 tables, a bridge execution record, a strict ack rule, a sweep and a coldness predicate. r9 review: 5 blockers. | Rejected by Nick (step 23). |
| (D) Live render each turn | Render from the DB into every prompt | The claude-code CLI ignores mid-session prompt changes, so it does nothing there (step 20). On codex it grows the transcript every turn (TASK-1318 step 5). | Dead. |

## 4. Initiative-state boundary

- Working memory is personal, cross-task direction for one (bot, user). An initiative is leased run state for one task [Caid r1 B2].
- Field names reuse `AgentInitiativeState`'s vocabulary: `objective`, `next_action`, `blockers` (a JSON array of short strings) [r2 B2, C9].
- An item has kind `personal` or `initiative_link` [R7]. A link item stores a validated `initiative_task_id` (section 5.1). It can't set `next_action` or `blockers` and doesn't copy initiative fields. The section renders a pointer only, e.g. `-> initiative TASK-123, see initiative_state` [L9 OI-2].
- The harness's in-turn todo list is scratch space for one turn. The registry framing says so, so bots don't mirror one into the other [r2 §7].

## 5. Data model

Storage is the llm-bawt askllm database, in a new package `src/llm_bawt/working_memory/`. The prompt builder, sessions and memories all live in llm-bawt; BawtHub gets its UI over llm-bawt REST [r2 §1]. Schema creation follows `utils/schema.py` `SchemaBootstrapGuard` + `ensure_relations` (example: `service/run_accepts.py`).

### 5.1 working_memory_items

| Column | Notes |
|---|---|
| `id` uuid | Rendered in short form. |
| `bot_id`, `user_id` | The namespace. Every lookup by id is scoped by id, bot and user [R6]. |
| `kind` | `personal` or `initiative_link`. |
| `initiative_task_id` | Required for `initiative_link`, NULL otherwise. TASK-1530 resolves new links against BawtHub to a canonical task cuid, stores that in `canonical_task_id` and `initiative_task_id`, and caches the short ID, title and status for rendering. Legacy `TASK-N` links are resolved and backfilled before new links can be created. No cross-DB FK. |
| `objective` | ≤ 160 characters. |
| `status` | `planned`, `active`, `blocked`, `done`, `cancelled`. `done` and `cancelled` set `archived_at` [L9 OI-9]. |
| `next_action` | ≤ 400 characters. Not settable on a link item. |
| `blockers` jsonb | ≤ 5 strings × 200 characters. Not settable on a link item. |
| `checkpoint` | ≤ 600 characters. |
| `priority` int NULL | Lower number first, NULL last, then `updated_at` desc, then `id` [L9 OI-7]. |
| `refs` jsonb | Flat list of `{kind, id, label, excerpt}`. ≤ 20 entries, ≤ 4KB serialized; kind, id and label ≤ 160 characters each, excerpt ≤ 200 characters. Label and excerpt are stored at write time; rendering never calls BawtHub [L9 OI-6, K3']. |
| `revision` int | CAS. |
| `created_at`, `updated_at`, `archived_at` | |
| `removed_at` | Soft delete. Distinct from `archived_at`. Whether it's terminal is Nick's call: N2. |

The create idempotency record is not on the item row; it lives on the item's `create` change row (5.2) so it survives a hard delete [r11].

### 5.2 working_memory_changes [r10]

An append-only, content-free event log: `seq` bigserial, `bot_id`, `user_id`, `ns_version` bigint, `item_id`, `revision`, `kind` (`create`, `update`, `archive`, `removed`, `purged`), `actor_kind` (`bot` or `user`), `actor_id`, `session_id` NULL, `turn_id` NULL, `idempotency_key` NULL, `create_fingerprint` NULL, `created_at`.

- Index `(bot_id, user_id, ns_version)`.
- **Create idempotency record** [r11, Loopy r10 B2]. On `kind='create'` rows only: `idempotency_key` and `create_fingerprint`, the sha256 of the canonical create payload (section 6.1). Unique partial index `(bot_id, user_id, idempotency_key) WHERE kind = 'create'`. Both are written once at create and never change, except that a hard delete sets `create_fingerprint` to NULL (section 11.3). That is the only update a change row ever receives.
- **The key is an opaque token** [r12, Loopy r11 B2]. `idempotency_key` must be a UUID in canonical 8-4-4-4-12 hex form. The server lower-cases it and stores the normalized form (`uuid` column type). Any other string, free text included, is refused as `invalid_idempotency_key` before the namespace lock, so nothing is written. The server never derives a key from item fields. The tool description tells the bot to generate a fresh UUID per create and reuse it only to retry that create.
- No `diff` column. r9 needed diffs for deltas; r10 has no deltas. Change rows never hold item text. The one value the server derives from item text, `create_fingerprint`, is nulled by a hard delete (section 11.3).
- `session_id`/`turn_id` come from the capability on tool writes, and are NULL on REST writes.

### 5.3 working_memory_namespaces [r10]

`(bot_id, user_id)` PK, `version` bigint.

- **Lock, decide, then bump** [r11]. Every write transaction first locks the namespace row (`INSERT … ON CONFLICT (bot_id, user_id) DO NOTHING`, then `SELECT version … FOR UPDATE`). Under that lock it runs the capability, CAS, idempotency and cap checks. Only a real write then runs `UPDATE … SET version = version + 1 RETURNING version`, and the returned value is the write's `ns_version`. A refused, conflicting or idempotent-replay request never bumps the version and writes nothing (Loopy r10 claim 6). That includes the bootstrap `INSERT` of a namespace's first row: it is in the same transaction, which rolls back rather than commits on any outcome that isn't a real write [r12, Loopy r11 claim 4].
- Since writes hold the row lock until commit, a later `ns_version` always commits after an earlier one. This replaces r9's advisory lock and fixes the r2 B2 allocation-order problem by construction.

### 5.4 Session state outside working-memory tables

- **Mode.** `sessions.session_metadata.working_memory.mode`, `pinned` or `suppressed`. Written by `SessionResetLineage.rotate` on the successor, in the rotate transaction, from a required `working_memory_mode` argument (section 10.1). Absent means `pinned`.
- **Notice state.** `sessions.session_metadata.working_memory.notice` = `{version, compaction_gen, pin_turn_id, at}` (section 9.3) [r11, r12]. `at` is diagnostic only; no rule reads it.
- **Compaction generation** [r12, Loopy r11 B1]. `sessions.session_metadata.working_memory.compaction_gen`, an integer, absent meaning 0. Turn finalization bumps it for every successful compaction it records (section 9.4).
- **Unverified floor** [r12, Loopy r11 S1]. `sessions.session_metadata.working_memory.unverified_floor` = `{min_version, may_be_missing}` while the session's pin is unverified (section 9.2).
- **The pinned section's version.** Stored on TASK-1318's pin pointer as `sections.working_memory`, plus `candidate_sections.working_memory` while the pin is unverified (section 8.2). Accepted and implemented by Byte: OI-R1.
- **Runtime settings** (`setting_definitions.py` SETTING_DEFINITIONS): `working_memory_enabled`, bot scope only, default false (section 7).
- **Prompt registry keys:** `agents.working_memory.section` (header, usage framing, empty and unavailable lines), `agents.working_memory.notice` (the notice line), `agents.working_memory.scheduled_note` (OI-R5). No hardcoded prompt text. None exist at `9e874dc`. *Implementation (slice 2a):* the section key is split three ways, `agents.working_memory.section` (header and framing, with an `{items}` slot), `agents.working_memory.section_empty` and `agents.working_memory.section_unavailable`, so each line is edited on its own and a section override can't drop the empty/unavailable distinction (section 14). Five keys in all, each with a seed in `working_memory/prompts.py` and a registry definition.

## 6. Tool and REST contract

### 6.1 MCP tools (`mcp_server/working_memory_tools.py`)

- `working_memory_list(include_archived=false)`
- `working_memory_create(objective, next_action?, status=planned, priority?, refs?, kind=personal, initiative_task_id?, idempotency_key)`
- `working_memory_update(item_id, expected_revision, ...fields)`. Status `done` or `cancelled` archives.

Rules:

- No `bot_id` or `user_id` parameter. Identity comes only from `current_task_turn_context()`. A missing, expired or malformed capability is refused, for reads too [R6].
- CAS: a stale `expected_revision` returns the current item (scoped by id, bot and user) and writes nothing.
- Every write records `session_id` and `turn_id` from the capability on its change row.
- **Idempotent create** [L9 OI-8, r11 Loopy r10 B2]. `idempotency_key` is required and must be a canonical UUID (5.2) [r12]. The payload fingerprint is the sha256 of canonical JSON over the validated, normalized create fields (`kind`, `objective`, `next_action`, `status`, `priority`, `refs`, `initiative_task_id`), with defaults filled, sorted keys and NFC strings. It is compared with the immutable `create_fingerprint` on the key's `create` row, never with the item's current fields. Under the namespace lock and before the cap check:
  - no `create` row for the key: a normal create, which writes the row with key and fingerprint;
  - fingerprint matches: return the item as it is now (after any later updates or archive) with `replayed: true`, or `removed` if it is soft-deleted. Nothing is written, the version isn't bumped and the cap isn't checked, so a replay succeeds at the open-item cap;
  - fingerprint differs: refused as `idempotency_conflict`; nothing is written;
  - fingerprint NULL (the item was hard-deleted): `removed`. There is nothing left to compare, and the item is never recreated.
- So in Loopy's histories, H1 (create A with key K, then update to B) followed by retry(K, A) returns the item, now B. H2 (create C with K, then update to B) followed by retry(K, A) is a conflict.
- A write over a cap is refused; the error names the cap.
- A storage failure returns "working memory unavailable", distinct from an empty list.
- The tools join `catalog_contracts.py` TOOL_SUMMARIES/REFERENCE_GROUPS and must fit the `tests/test_mcp_catalog.py` caps. The budget change is logged in `docs/MCP_TOOL_DESIGN.md`.

### 6.2 REST owner path (Nick)

- `GET /v1/working-memory/{bot}/{user}`: items, archived and removed included (removed shown content-free if hard-deleted).
- `PATCH /v1/working-memory/{bot}/{user}/{item}`: edit or cancel; `actor_kind=user`; requires `expected_revision`; a conflict returns the current item and writes nothing [L9 OI-10].
- `DELETE /v1/working-memory/{bot}/{user}/{item}`: soft delete. `?hard=true` hard deletes (section 11). Repeated deletes are idempotent. A hard delete after a soft delete writes one `purged` change and no second `removed` [L9 OI-11].
- LAN-only, consistent with the llm-bawt security model.

## 7. Trust, opt-in and protected-bot exclusion

- **One policy** [L9 OI-3]. `WorkingMemoryPolicy` resolves opt-in, protected status and the durable `agent_backend` on the server from the bot profile, never from model input. It is operation-specific:

| Operation | Protected bot | Not opted in / not claude-code |
|---|---|---|
| MCP tools | refused | refused |
| Prompt section and notice | absent | absent |
| REST GET, PATCH, DELETE | refused | allowed, so Nick can clean up after disabling |
| REST create | not offered | — |
| `/health` | excluded | reported as disabled |

- **Protected bots.** In this document a protected bot is one carrying the `private` bot tag (TASK-1329; `working_memory/policy.py` `is_protected` → `BotTags.has(bot, BehaviorTag.PRIVATE)`). No code names a bot. A failed tag read, an unreachable engine or a bot with no profile row all count as private. Protected-bot tests use synthetic `private`-tagged fixtures, never a live private bot's profile or history.
- **Setting validation** [L9 OI-4]. `working_memory_enabled` is bot-scoped. A global write is refused. `true` is refused for a protected bot or a bot whose durable backend isn't claude-code, on both the single and batch paths of `routes/settings.py`, before any row is applied. Read-time gates still fail closed.
- **Owner cleanup after opt-out (TASK-1530).** REST DELETE stays available for a disabled bot's legacy `canonical_task_id=NULL` link without first enabling its tools or migrating unrelated links. After owner authorization, the service reads only that namespace/item's legacy reference and tries to resolve it off-lock. A resolvable reference gets the same durable suppression as a canonical link. A missing task does not prevent soft or hard cleanup: soft removal keeps the reference for later backfill and suppression; an unidentifiable hard purge deliberately lacks a purge-audit mapping, so later task-link creation in that namespace fails closed pending owner identity audit. Protected bots are refused before any reference or catalog read.
- **Opt-in effect.** Off: the builder adds nothing, so the prompt is byte-identical to today's.
- **Toggle timing** [r10]. Tools and notices follow the setting immediately. The section's presence in the prompt changes only at a boundary, because TASK-1318 sends pinned bytes. An enable mid-session produces one notice on the next eligible turn (section 9.2).

## 8. Per-harness delivery

### 8.1 The section (claude-code)

- `_sec_working_memory` in `core/base.py` (948 lines; gets one short call), late in the system prompt. The builder lives in `working_memory/render.py`.
- It renders only when the policy allows it and the session's mode is `pinned`.
- **Render read** [r10]. One `REPEATABLE READ` read-only transaction reads the namespace `version` and the open items, so the rendered items and the version agree. No locks are taken.
- **Order** [C9, L9 OI-7]: active, then blocked, by priority; at most 8 in full; the rest as titles up to the budget; then "N omitted". Planned items render as titles. Age buckets are computed at render time. "Last updated in another session" says only that.
- **Refs** render compactly: kind, id, label, excerpt ≤ 200 characters. Rendered text is escaped as data [L9 OI-6].
- **States.** `ok`, `empty` (registry empty line), `unavailable` (registry unavailable line; storage failed), `suppressed` (nothing rendered). Each render produces `{version, state}`, request-local and never model input. `version` is NULL when `unavailable`.

### 8.2 TASK-1318 pin (Byte; not designed here)

From TASK-1318 step 6 and `system_prompt_pin.py`:

- `apply_session_pin(ctx, client, messages, turn_id=, sections=)` runs in `turn_stream_worker.py` (~:220) after the context suffix is composed. It returns `PinDecision {text, pin_turn_id, sha256, action, reason, is_boundary, rendered_sha256, snapshot_unverified, pinned_sections, candidate_sections, drift}`, or None when the backend doesn't pin, the turn is `reset_only`, there is no session, or the store failed.
- **Two-phase moves.** `resolve` stages a move under `session_metadata.system_prompt_pin_pending`. `SessionPinConfirmer` (`system_prompt_pin.py:300`, created in `turn_stream_worker.py:376`) promotes it to `system_prompt_pin` on the turn's first real model output (text, reasoning, tool, usage or subagent events; not a cancel, `upstream_status` or aborted-turn bookkeeping). A boundary turn that never reaches the model leaves the confirmed pointer alone, and the next turn decides the boundary again. Kept turns read only the confirmed pointer, so a staged move's sections never surface on a later turn before confirmation (Byte, `test_pending_boundary_sections_surface_only_after_confirm`).
- **Reasons and precedence** [r11, Loopy r10 B1, accepted by Byte]: COLD_START > LEGACY_ADOPT (no pointer, so compaction can't be checked) > COMPACTION > MISSING_ROW / MISSING_TEXT / SHA_MISMATCH. The compaction check needs only the pointer's `pinned_at`, not the old row or text. So a successful manual `/compact` over a damaged pin is a verified COMPACTION move that stores this render's sections, `unavailable` included. COLD_START and COMPACTION are verified; the other four reasons are `snapshot_unverified`.
- Archived status is not an input. A born-archived task session that resumes its SDK key keeps its pin (`memory/postgresql_short_term.py:438-452`). This closes r9 B3.
- **Why working memory doesn't use `drift`.** `drift` is true whenever the fresh render differs from the pin. Profile re-wording alone changed the prompt mid-session on roughly 5-10% of turns per bot in 48h (task step 19). So drift can't tell a working-memory edit from churn.
- **Section metadata contract, accepted and implemented by Byte** (OI-R1). Working memory passes `sections={"working_memory": {"v": 1, "version", "state"}}`. The pointer stores:

| Move | `sections` | `candidate_sections` |
|---|---|---|
| COLD_START, COMPACTION (verified) | this render's | none |
| MISSING_ROW, MISSING_TEXT, SHA_MISMATCH | the previous pointer's | this render's |
| LEGACY_ADOPT | none | this render's |
| kept turn | carried | carried while `snapshot_unverified` |

  `PinDecision.pinned_sections` and `.candidate_sections` expose them (JSON-safe copies, `compare=False`). After an unverified re-pin the harness holds either the old snapshot (`sections`) or, once it auto-compacts, the fallback bytes (`candidate_sections`); the pointer can't know which, so working memory reads both conservatively (section 9.2) [r11, Loopy r10 B1].
- **Known limit of the pointer (Byte), and how working memory closes it** [r12, Loopy r11 S1]. A second unverified re-pin replaces `candidate_sections` with the newer render. If the harness auto-compacted between the two re-pins, it holds the first fallback's bytes, which the pointer no longer describes. r11 claimed the minimum version still bounded what the harness holds. That is false when `S` has no version: S absent, first candidate v3, auto-compaction, second candidate v5 gives `P = 5` while the harness holds v3. A first candidate that was missing (absent, unavailable or suppressed) is lost the same way. Working memory doesn't rely on the pointer for this. It keeps its own bounded `unverified_floor` across unverified re-pins and clears it at a verified boundary (section 9.2). The pin contract is unchanged.

### 8.3 Other harnesses

- **Codex.** `pins_system_prompt` is False, and the codex bridge pastes a fresh render onto every user message (`codex_bridge/event_ops.py:91-92`). The section would go live every turn and grow the thread. Enabling is refused on codex until TASK-1318's held Codex change lands and is reviewed here.
- **openclaw and chat.** No capability (`task_turn_context.py:23`). Refused.
- **gpt bots on the claude-code bridge** (`openai_chatgpt/*`) get the same CLI snapshot and the same pin, so they are in scope.

## 9. Notice

### 9.1 Eligibility [r10, L9 S3]

A notice is considered only on a turn that starts through the stream path (`turn_stream_worker`), where `apply_session_pin` returned a decision and `is_boundary` is False [r11, Loopy r10 S1]. Messages steered into a running turn are excluded: composer steers and active-turn bot deliveries go through `steer_active_turn` (`service/routes/chat.py:158`; `inter_bot_dispatcher.py:533-565`), which frames and sends without preparing a pin. They leave the notice pending; the bot is already mid-turn, and the next started turn carries it. Idle-fallback bot deliveries start a turn through `chat_completion_stream` (`inter_bot_dispatcher.py:682-705`) and are eligible. Nick's "next user message" wording is therefore narrowed to "next started turn" and is flagged to the coordinator (OI-R8). It is never added to:

- a turn that is a maintenance command: `/compact`, `/new` or `/clear` as the leading token, with or without arguments (`src/agent_bridge/maintenance.py` leading-token classifiers `is_compact_command`, `is_new_command`; an `is_clear_command` is added there);
- a scheduler turn: the validated delivery's sender is `scheduler:<hash>` (`prompt_delivery.py:85`). Identity comes from the claimed delivery record, never from an author display label or `message_kind`;
- a `reset_only` turn (no decision is returned anyway);
- a session in mode `suppressed`;
- a bot the policy excludes.

An ineligible turn leaves the notice pending for the next eligible one.

### 9.2 When it fires [r10]

**What the harness holds** [r11, Loopy r10 B1]. Take `S` = `pinned_sections.working_memory` and, if the decision has `candidate_sections`, `C` = `candidate_sections.working_memory`. Each entry is either a section (`state` ok or empty, with a version) or missing (absent, or `state` unavailable or suppressed).

- `F` = the session's `unverified_floor`, if any (below).
- `P` = the lowest version among the present entries of `S`, `C` and `F.min_version`, or −1 if none is present.
- `missing` = true when `S` is missing, or a candidate exists and `C` is missing, or `F.may_be_missing` is true.
- [2a, Loopy 2a r1 B1] Every `snapshot_unverified` decision has a candidate, even when the pointer stored no `candidate_sections`. A fallback rendered without the section (working memory disabled or refused for the bot on that turn) stores none, and an auto-compaction can still adopt those bytes. Its `C` is missing.

With a verified pin this is just `S`. With an unverified one it is the conservative reading of every snapshot the harness might hold.

**Unverified floor** [r12, Loopy r11 S1]. One fixed-size record, not a history. On every turn with a decision, boundary turns included:

- if the decision has a candidate (above), fold `S` and `C` into `F`: `min_version` = the lowest present version among `F.min_version`, `S` and `C`; `may_be_missing` = `F.may_be_missing`, or `S` missing, or `C` missing;
- if the decision is a kept turn (`is_boundary` False) and not `snapshot_unverified`, the confirmed pin is verified: delete `F`.

A boundary turn's decision is a staged move that may never be confirmed. Folding it anyway only lowers the floor or sets the flag, so the error is an extra notice, never a lost one. Clearing happens only on a kept turn, which reads the confirmed pointer. `P` and `missing` use `F` after this turn's fold or clear. The fold and the clear are written in the notice-mark session-row transaction (9.3), which runs when the notice fires or `F` changes. If that write fails, the turn logs it and the session falls back to the pointer's own reading, which is the limit stated in 8.2.

Loopy's counterexample: S absent, C1 = v3 gives F = {3, true}; C2 = v5 gives F = {3, true}, so `P = 3` and `missing` is true. A lost missing first candidate keeps `may_be_missing` set.

**Notice episode** [r11, Loopy r10 S2]. `N` = the session's notice `version`, but only if its `pin_turn_id` equals the decision's `pin_turn_id` and no compaction re-arm applies (9.4). Otherwise `N` is NULL. Every confirmed pin move, whether a cold start without rotation, a compaction or a fallback, therefore starts a new notice episode.

Threshold `T = max(P, N)`, treating NULL as −1. The notice fires when either holds:

1. a change exists with `ns_version > T` and `session_id IS DISTINCT FROM` this session (an external change; own writes in this session don't count);
2. `missing` is true, the policy currently allows the section, and `N` is NULL. This covers an enable mid-session, a legacy pin, a boundary render that hit a storage failure, and Loopy's disable → cold boundary → re-enable sequence.

The check is one indexed query on `(bot_id, user_id, ns_version)`. A storage failure skips the notice and logs; it never blocks the turn. [2a, Loopy 2a r1 S1] It is one statement returning the external maximum and the namespace version, so both come from one snapshot. Inside the notice mark (9.3) it runs on the mark's own connection and transaction, so a mark that holds its session row never checks out a second pooled connection.

### 9.3 Recording it [r10]

- When it fires, the notice line from `agents.working_memory.notice` is appended to the outbound user message, after the context suffix. It is not written to canonical chat history. It is recorded in `turn_logs.request_json`, as the rest of the outbound message is.
- In a separate transaction, after the pin transaction, the session's notice state is set to `{version: max(N, highest external ns_version seen, current namespace version), compaction_gen: g, pin_turn_id: decision.pin_turn_id, at: now}`, where `g` is the session's `compaction_gen` read under the same row lock [r12]. Within one `pin_turn_id` the version only grows; a different `pin_turn_id` replaces the record. That transaction takes only the session row (`SELECT … FOR UPDATE`, then `jsonb_set` on the `working_memory` key only), and it commits before the turn's command is published to the bridge [r12].
- The mark is made at prepare time. A turn that then fails before the model sees it loses that notice until the next external change or boundary. Alternative: OI-R2.

### 9.4 After an auto-compaction [r10]

Auto-compaction can summarise a notice or a tool read away, while the pinned section stays at `P`. So the notice threshold resets to `P` after a compaction: if the session's current `compaction_gen` is greater than `notice.compaction_gen`, `N` is treated as NULL. If the store has moved past `P` externally, the next eligible turn notices again [r11 Loopy r10 S3; r12 Loopy r11 B1].

**No clock** [r12]. r11 compared `turn_logs.ended_at` with `notice.at`. At the base `ended_at` is `created_at + latency_ms` (`service/turn_log_schema.py:40-46`, applied at `turn_logs.py:474-486`), and latency is timed from execution start (`turn_stream_worker.py:250` starts the clock after preparation; `turn_stream_finalize.py:112-114`), not from row creation. A turn with slow preparation therefore gets an `ended_at` earlier than its own notice, and Loopy's regression shows the re-arm missed. r12 orders by a durable counter instead.

**The bump.** When turn finalization (`turn_stream_finalize.py`, which already resolves the turn's `thread_session_id` at :397) records final usage with `context_action='compact'` and `context_action_outcome='success'`, it runs one more transaction after the turn row is written: lock that session row, then `jsonb_set` `working_memory.compaction_gen` to its value + 1. Auto-compaction and manual `/compact` both produce the stamp (`claude_code_bridge/send_handler.py:604-612` sets `compact_status` from the SDK status message on either path; `send_result.py:120-176` stamps it). It runs for every non-protected bot whose backend pins, opted in or not, so a compaction while working memory is disabled still re-arms a notice made before the disable. A bump failure is logged and costs only that re-arm.

**Why it is sound.** A compaction can only summarise a notice that the harness has already received. The notice mark, which records `g`, commits before the command is published (9.3). The bridge observes the compaction after that, and finalization bumps after the bridge reports it. So any compaction that can remove a notice moves `compaction_gen` past `g`. Both the read and the bump take the session row lock, so the comparison involves no clocks, preparation delay or turn order. A compaction that finishes after the mark but happened before the notice was delivered (an earlier queued turn) also re-arms. That only costs an extra notice.

The success stamp is missing when compaction metadata is absent (Loopy r9 B2). Byte fixed this in TASK-1318 (`send_result.py:154-163`); it is live after a bridge restart (OI-R7). A turn that compacts and then dies before final usage is persisted has no stamp either. That compaction misses the re-arm until the next external change or boundary. The pin stays consistent; only the re-notice is lost.

### 9.5 Cost

The notice is one short line, appended to the user message, so the cached prefix is unchanged. It fires at most once per external change burst, plus once per compaction while the pin is behind the store.

## 10. Reset and compaction

### 10.1 Resets [C1, A3, L9 #8]

- `SessionResetLineage.rotate` (`memory/session_reset_lineage.py:19`) gets a required `working_memory_mode` argument and writes it to the successor's `session_metadata.working_memory.mode` in the rotate transaction. No blind low-level default: each caller passes its origin's mode.
- The five callers at `9e874dc` (files unchanged since `4883515`):
  - `agent_context.py:211` (overflow recovery);
  - `agent_context.py:469` `rotate_locked` (idle sweep, context API);
  - `memory/postgresql_short_term.py:430`;
  - `service/runtime_switch_rotation.py:85`;
  - `service/session_lifecycle/lifecycle.py:74` (delivery claims).
- One test per entry point.
- Which origins suppress is Nick's call (N1). The default in this design until he rules: automated rotations pin; human `/new` pins; suppression only on an explicit request (`include_working_memory=false` on the session or context API, or a bot delivery with `session_policy=reset_without_history`).
- A born-archived task session (`create_inactive_session`) is not a rotation; it starts with mode absent, i.e. `pinned`.
- Suppression lasts for the session. A scheduled turn never changes a session's mode [L9 OI-13].

### 10.2 Compaction

- **Manual `/compact`.** TASK-1318 moves the pin on the first turn after a successful one, as a verified move even if the old pin is damaged (8.2). The new pin carries a fresh render, so the section is current again. A failed `/compact` doesn't move the pin (`_manual_compaction_succeeded` requires `context_action_outcome='success'`). This settles r9 OI-15.
- **Auto-compaction.** Not a boundary. The CLI re-records the pinned bytes, so the section stays at `P`. Section 9.4 re-arms the notice.
- **Missing compaction stamp** (Loopy r9 B2). Under B this costs a stale pin or a missed re-notice, never an inconsistent prompt. Byte fixed the stamp in TASK-1318, pending a bridge restart (OI-R7).

## 11. Deletion and privacy

The B3 and D4 rulings are Nick's decisions and not up for review.

### 11.1 B3 ruling (task step 5, 2026-10-23 ~01:00 UTC)

Nick, verbatim: "the answer is no. delete should not alter it. that messes up the cache doesnt it?"

- A delete, soft or hard, never changes a live pin or an in-flight prompt. The pinned bytes keep the item until the session's next boundary, whose render excludes it.
- A delete writes a `removed` change (content-free). It is an external change, so it produces a notice on the session's next eligible turn (section 9). A re-read through the tool shows the item gone.
- An archived session that resumes its SDK key keeps its pin (section 8.2). This replaces r9 §11.1's "resuming an archived session renders from current state", which contradicted B3 (Loopy r9 B3).

### 11.2 D4 ruling (task steps 13-14, 2026-10-23 16:01Z)

Nick, verbatim: "I did answer actually. I called out that removing it during a session would break session caching and should be forbidden. Alternative hasn't been discussed since every prompt technically would generate live state from db".

- Turn logs are a record of prompts as sent. The working-memory DB is the only source of truth. Deletes don't scrub `turn_logs.request_json`.
- Under TASK-1318 the pinned prompt text lives in the pin turn's `turn_logs.request_json.system_prompt_pin.text`. That falls under this ruling: it is a turn log, and it isn't scrubbed.
- Turn logs never age out: `TurnLogStore._cleanup_expired_if_due` (`turn_logs.py:101-115`) returns early without `ttl_hours`, and the only constructor (`background_service.py:138`) sets none.
- The delete confirmation must say the item is removed from working memory and that earlier prompts recorded in turn logs still contain it. No "age out" wording. Final copy: N3.

### 11.3 What a delete touches [r10]

- Soft delete: sets `removed_at`, writes a `removed` change. The row keeps its content until a hard delete.
- Hard delete: deletes the item row, sets the item's `create_fingerprint` to NULL, and writes a `purged` change, all in one transaction under the namespace lock.
- **What survives a hard delete** [r12, TASK-1530]. The item's content-free change rows retain `seq`, `bot_id`, `user_id`, `ns_version`, `item_id`, `revision`, `kind`, `actor_kind`, `actor_id`, `session_id`, `turn_id`, `created_at`, and on the `create` row the server-validated UUID idempotency key (5.2). The key is kept so a replay returns `removed` instead of recreating the item. A UUID can't carry free text by accident; a caller deliberately hiding bytes in one is outside this guarantee. A resolved task link also leaves a suppression row with namespace, HMAC digest of the canonical task cuid and a keyed sentinel, plus a purge-audit row mapping the purged item UUID to that digest. Neither row stores the task cuid, title or item text. Digest and sentinel use **one key snapshot per write**; if the stored sentinel differs after key rotation, link creation fails closed for that namespace. An unresolvable legacy hard purge leaves no mapping and likewise gates later links rather than guessing which task was deleted. Ordinary personal-item purges keep an audit row with NULL digest.
- Untouched, by ruling: live pins, in-flight prompts, turn logs (including pin text), SDK transcripts on disk.
- r9's scrub, `scrub_pending`, sweep and coldness predicate are gone. There is no working-memory pin table to scrub.

## 12. Caps

| Cap | Value |
|---|---|
| Open items (planned, active, blocked) per (bot, user) | 20, under the namespace lock; a reopen counts |
| `objective` | 160 characters |
| `next_action` | 400 characters |
| `blockers` | 5 × 200 characters |
| `checkpoint` | 600 characters |
| `refs` | 20 entries, 4KB serialized; kind, id and label ≤ 160 each; excerpt ≤ 200 |
| Idempotency key | canonical UUID (5.2) |
| Section budget | about 1.5K tokens; at most 8 items in full |
| Notice | one line from the registry; no item content |

## 13. Concurrency and lock order [r10]

- **Working-memory writes** (tools, REST): namespace row (section 5.3) → item row → change insert. A hard delete also updates its own `create` change row (5.2). Nothing else. No session or turn row is locked.
- **Render:** read-only `REPEATABLE READ`, no locks.
- **Pin** (TASK-1318): `resolve` and `confirm` each lock turn row → session row, in one transaction. Working memory adds only content-free `sections` values to them.
- **Notice mark** (with the unverified-floor write): session row only, in its own transaction, after the pin transaction has committed and before the command is published.
- **Compaction bump** [r12]: session row only, in its own transaction, after finalization has written the turn row.
- **Rotate:** unchanged lock set (target lock, advisory lock, session rows); writes the mode as part of the successor row it already writes. It takes no working-memory lock.
- **Finalization** (`update_turn`, `turn_logs.py:408` → `record_turn_context`): unchanged. The compaction bump is a separate transaction after it, not a lock added inside it [r12].

No path holds a working-memory lock while taking a session or turn lock, and no session or turn path takes a working-memory lock. So there is no cycle to order. This retires r9 §13 and Loopy r9 B5.

Test: concurrent writers in one namespace get consecutive `ns_version`s in commit order; a render racing a writer sees a matching (version, items) pair; the cap holds under concurrent creates.

## 14. Observability and health

- Tools return "working memory unavailable" on storage failure, distinct from empty.
- `/health` `working_memory`, per enabled non-protected bot: open, archived and removed counts; namespace version; render failures since start; notices fired since start; the number of active sessions whose pinned version is behind the store, with the oldest lag; and the count of turns where the pin store returned no decision. No item text [L9 OI-19]. *Implementation (slice 2a):* served at `GET /v1/working-memory/health`, not inside `/health`, which stays the cheap liveness probe. It lists every bot with `working_memory_enabled` stored true (protected bots never), with `allowed` false plus a warning when the read-time policy refuses it; adds notice-write failures to the counters; and counts pinned active sessions without a section version separately. A storage failure returns 503 `unavailable`.
- Logs: a warning when an enabled bot isn't on claude-code; a warning on a render failure; debug on every notice decision with `P`, `N`, threshold and result.

## 15. Rollout and rollback

**Slice 1: store, trusted MCP tools, REST owner path. No prompt change.**

- Tables: items, changes, namespaces (OI-1). Additive.
- Gates: PG suites registered in `tests/suite_manifest.py`; MCP catalog caps; protected-bot refusals; idempotency and CAS.
- Independent of TASK-1318.
- Rollback: unregister tools and routes. Tables stay.

**Slice 2: section, notice, rotate mode.**

*Implementation:* delivered in two commits, 2a (section, notice, compaction-generation bump, health, registry prompts) and 2b (the required `working_memory_mode` on `SessionResetLineage.rotate` and its callers, section 10.1). Until 2b lands every successor is `pinned`, so `include_working_memory=false` has no effect yet.

- Requires TASK-1318 deployed with the `sections` extension (OI-R1) and its B2 stamp fix live (bridge restart, Nick).
- App only. Working memory itself needs no bridge change, so no bridge restart of its own and no mixed app/bridge version problem (Loopy r9 S2 does not apply).
- Disable for a bot: tools and notices stop at once; the section leaves a live session at its next boundary.
- Code rollback of slice 2: pinned bytes already include the section and TASK-1318 keeps sending them until the next boundary. The next boundary renders without it. Session metadata keys become inert.
- Rollback of TASK-1318 under slice 2: the CLI still holds its own snapshot, so the prompt is unchanged mid-session; the notice stops because there is no pin decision.

**Slice 3: BawtHub UI** (active, blocked, archived and removed views; edit; delete with the D4 confirmation).

Production deploys and restarts happen only when Nick asks, through `ops_run`.

## 16. Nick-owned decisions

Decided, not up for review:

| Decision | Ruling |
|---|---|
| B3 | A delete never alters a live pin or an in-flight prompt (11.1). |
| D4 | No turn-log scrub; truthful wording (11.2). |
| Scope | B+notice on the TASK-1318 pin (step 23). |
| N1 | Nick, 2026-10-23: keep unless asked. Suppression only on an explicit request; human `/new` pins. A suppressed session gets neither section nor notice. |
| N2 | Nick, 2026-10-23: a soft delete is final (no restore). `removed_at` is final; only hard delete purges. |
| N3 | Nick, 2026-10-23: the full copy. "Removed from working memory. The bot's current session may still show it until its next /new, or the turn after its next manual /compact. Earlier prompts recorded in turn logs, and transcripts already on disk, still contain it." Automatic compaction doesn't refresh the prompt, so it isn't named [r11, Loopy r10 S3]. |
| OI-R8 | Nick, 2026-10-23: the next started turn. Steered messages don't carry the notice (9.1). |

Also Nick's: which bots opt in (none by default), the 20-item cap, and when each slice deploys.

## 17. Acceptance tests

Unit and PG:

- save → `/new` → the next turn's pinned prompt shows the item's exact next action;
- a done item leaves the section at the next boundary;
- (bot, user) isolation; no-capability writes refused; CAS conflict writes nothing;
- overflow shows the omitted count; storage failure renders the unavailable line, never empty;
- protected and non-opted-in bots: prompt byte-identical with the feature deployed;
- enabling on a codex bot, globally, or for a protected bot is refused (single and batch);
- one test per rotate entry point, asserting the mode written;
- namespace ordering and render consistency (section 13);
- idempotent replay, payload conflict, replay after hard delete [r11]:
  - H1: create A with key K, update to B, retry(K, A): returns the item (now B) with `replayed: true`;
  - H2: create C with K, update to B, retry(K, A): `idempotency_conflict`;
  - both at the open-item cap: H1's replay still succeeds; H2 is still a conflict, not a cap error;
  - replay after archive returns the archived item; after a soft delete, `removed`; after a hard delete, `removed`, with the fingerprint gone;
  - refused, conflicting and replayed requests leave the namespace version unchanged, including a namespace's first request, which leaves no namespace row [r12];
  - [r12, Loopy r11 B2] a key equal to the objective text, or any non-UUID string, is refused as `invalid_idempotency_key` and writes nothing; an upper-case UUID is normalized and replays against its lower-case form;
  - [r12] after a hard delete, every remaining working-memory row for the item holds only the columns listed in 11.3, and none contains the objective, next action, blockers, checkpoint or refs;
- notice: external REST edit → notice on the next eligible turn and not the one after; own write → no notice; write from another session of the same bot → notice;
- notice not added to `/compact`, `/compact focus on X`, `/new`, `/clear`, scheduler turns or suppressed sessions; it stays pending to the next eligible turn;
- boundary turn → no notice, and the next turn's threshold is the new `P`;
- unavailable at boundary → notice on the next eligible turn once storage is back;
- enable mid-session → one notice;
- after a successful auto-compaction with the store ahead of `P` → one more notice;
- unverified re-pin (pin turn row deleted) keeps the previous `P`, so an edit made before the re-pin still notices;
- [r11, Loopy r10 B1] successful manual `/compact` on top of a missing row, missing text or sha mismatch: the pointer's sections are this render's (Byte's PG tests). If this render is `unavailable`, the first eligible turn after storage recovers gets a notice;
- [r11] unverified re-pin, where the old sections hold v3 ok and the candidate holds v5 ok: `P = 3`, so a change at v4 notices. Old sections v3 ok and candidate unavailable: `missing` is true, so one notice fires;
- [r11] LEGACY_ADOPT (candidate only): `missing` is true, so one notice fires;
- [r12, Loopy r11 S1] unverified floor:
  - S absent, first candidate v3, auto-compaction, second candidate v5: `P = 3`, so a change at v4 notices;
  - first candidate unavailable, second candidate v5 ok: `missing` stays true;
  - a staged unverified boundary that is never confirmed (cancelled before output): the floor may be lower, never higher;
  - a verified boundary, then a kept turn: the floor is gone and `P` is `S`;
  - a failed floor write logs and leaves the turn running;
- [r11, Loopy r10 S2] enable with no section in the pin → notice at v2 → disable → cold SDK boundary without session rotation, rendered while disabled → re-enable, with no writes: one notice, because the pin turn changed;
- [r12, Loopy r11 B1] re-arm by `compaction_gen`, with the store past `P` in each case:
  - delayed preparation: the turn row is created, preparation takes 20 s, the notice is marked, and the same turn auto-compacts. The next eligible turn notices again. (Loopy's `ended_at` regression is the r11 failure this replaces.)
  - a later turn auto-compacts: the next eligible turn notices again;
  - a compaction finalized before the mark (gen already counted in `g`): no re-arm;
  - a queued earlier turn whose compaction is finalized after the mark: re-arms (conservative);
  - a compaction while working memory is disabled, then re-enable in the same pin episode: re-arms;
  - a manual `/compact` also bumps the generation (its pin move resets `N` anyway);
  - a successful compaction with no metadata (Byte's stamp fix) bumps; a failed one doesn't;
- [r11, Loopy r10 S1] an external edit followed by a composer steer and an active-turn bot delivery: neither steered message carries the notice, and the next started turn does. An idle-fallback bot delivery that starts a turn carries it;
- delete: pinned bytes and sha unchanged; one notice; the next boundary's render excludes the item;
- a born-archived task session resuming its key keeps its pin.

Cache:

- warm turns with no edits: no `cache_creation` change vs the same bot with the feature off (Claude: token_usage; gpt on the proxy: the bridge's "Responses usage" cache-hit log);
- a notice turn: no prefix rewrite.

### 17.1 Live test plan (Nick, task step 21; Snark checks receipts)

On an idle Claude Code bot Nick picks (never Mira), then repeated on a gpt bot on the claude-code bridge:

1. Enable. `/new`. Create two items through the tool. `/new` again: the pin turn's `request_json.system_prompt_pin.text` contains the section, and the pointer's `sections.working_memory.version` matches the namespace.
2. Edit one item over REST. Send an ordinary message: its `request_json` user message ends with the notice line. Send another: no notice.
3. Have the bot call `working_memory_list`; it reports the edit.
4. `/compact`, then an ordinary message: the pin moved, the section shows the edit, and no notice fires.
5. Soft-delete an item: the pin sha is unchanged; the next ordinary message has the notice; the next `/new` drops the item.
6. Compare `cache_creation` across warm turns before and after enabling.

On the Codex bot on the codex bridge: enabling is refused; its prompt is byte-identical to before; the tools refuse.

## 18. Implementation-review claims

Retired with (C), because the mechanisms they constrain no longer exist: W1 (binding adoption), W2 (stall counter), Loopy's r8 null-association note (coldness resolver), the L2 route by `access_path.key` (coldness TTL), and Caid r2 R3/S1/S3/S4 as written for deltas and pins.

Carried, to be attacked at implementation review:

1. `ns_version` is commit-ordered, and a render's (version, items) pair is consistent (section 13).
2. The notice threshold and eligibility rules (section 9) as stated, including the unverified re-pin case.
3. Working memory adds no lock to the pin, finalization or rotate paths beyond the mode write in rotate. The compaction bump and the notice mark are their own session-row transactions (section 13) [r12].
4. Section bytes are a deterministic function of (items, policy, mode, render time bucket), so two renders with no change in between are identical.
5. Caid r2 R1 (mode at all five rotate callers) and R2 (backend gate at prompt build).
6. [r11] Create idempotency decides from the immutable fingerprint only, and refused, conflicting and replayed requests leave the namespace version and all rows unchanged (5.3, 6.1).
7. [r11] The conservative `P`/`missing` reading of `pinned_sections` + `candidate_sections`, and the pin-episode binding of `N` (9.2).
8. [r12] The unverified floor only lowers or flags while the pin is unverified and clears only on a kept turn reading a verified pointer (9.2).
9. [r12] Every compaction that can remove a notice moves `compaction_gen` past the notice's `g`: the mark commits before publish, and the bump follows the stamp (9.3, 9.4).
10. [r12] Non-UUID idempotency keys are refused before anything is written, and a hard delete leaves only the rows and columns listed in 11.3.

## 19. Open items for review

### Design items (reviewer)

**OI-1. Slice-1 tables.** Items, changes (with the create idempotency record, 5.2) and namespaces. The pin is TASK-1318's; there is no other table.
Recommendation: as stated.

**OI-R1. TASK-1318 section contract.** Accepted and implemented by Byte; committed in `9e874dc` as `system_prompt_pin.py` `c100554a06b0`. It includes the r11 compaction precedence and `candidate_sections` (8.2). Byte's tests:
- `test_pointer_sections_record_what_the_harness_holds`;
- `test_proven_compaction_beats_a_damaged_pin[missing_row|missing_text|sha_mismatch]`;
- `test_proven_compaction_beats_fallbacks`;
- `test_unverified_repin_carries_candidate_sections_until_a_verified_boundary`;
- `test_legacy_adopt_records_only_candidate_sections`;
- `test_candidate_sections_only_on_unverified_repins`;
- `test_pending_boundary_sections_surface_only_after_confirm`.

Byte's gate is `make test-snapshot` run `20261023T191033Z-byte-239d87`. Codex is reviewing his staging/confirm delta separately under TASK-1318. This design depends on the contract fields, not on the confirm-trigger details.
Recommendation: closed unless the reviewer objects. The pointer's known limit (two fallbacks in a row) is in 8.2; working memory's own floor covers it (9.2), with no contract change.

**OI-R2. Notice marked at prepare vs repeated until read.** Nick said "a one-line notice on its next user message". Marking at prepare matches that, but a turn that fails before the model sees it loses the notice. The alternative repeats the notice on each eligible turn until the bot calls `working_memory_list` or `working_memory_update` in this session (identity from the capability).
Recommendation: until read. It closes the lost-notice gap with no receipts. Needs Nick's OK since it can show the line more than once. If adopted [r11, Loopy r10 claim 3]: only a successful full `working_memory_list` in this session acknowledges, and only up to the namespace version that list read (returned by the same `REPEATABLE READ` snapshot). A single-item update acknowledges nothing else. Until Nick rules, marking at prepare is the design, and Loopy accepts it as an explicit best-effort tradeoff.

**OI-R3. Re-arm after auto-compaction** (section 9.4). New in r10; ordered by `ended_at` in r11, which Loopy r11 B1 showed is not a completion time; ordered by the session's `compaction_gen` in r12.
Recommendation: as stated.

**OI-R4. Notice on enable mid-session** (section 9.2 case 2). Bound to the pin episode in r11.
Recommendation: as stated. The alternative is silence until the next boundary.

**OI-R5. Scheduled note.** C5 added an `agents.working_memory.scheduled_note` telling scheduled automation not to act on working memory, because a scheduler turn in a shared session would see the pinned section. Loopy r9 asked for its actual text to be restated: "scheduled automation; do not act on working memory".
Recommendation: keep it. Scheduler turns get the note in the user suffix, never a working-memory notice. The note is an instruction, not an execution ban (Loopy r10 claim 7). Scheduler identity comes from the validated delivery (9.1).

**OI-R6. Content-free changes** (section 5.2). This drops r9's `diff` and changes OI-11's purge rules.
Recommendation: as stated. The UI shows item history only as events. `create_fingerprint` is the only value derived from item text, and a hard delete removes it. What survives is listed in 11.3 [r12].

**OI-R7. Compaction stamp without metadata** (Loopy r9 B2). `send_result.py:120-154` stamps `context_action` only with `postTokens`. Byte took it in TASK-1318: success with no metadata now stamps `context_action=compact`, `outcome=success` only (no tokens, no resident override), with a regression test in `test_claude_context_usage.py`. Live only after a claude-code-bridge restart Nick must approve. Byte also fixed `_manual_compaction_succeeded` to use `is_compact_command` (rejects `/compactness`).
Recommendation: closed in design; slice 2 rollout waits on that bridge restart.

**OI-R8. Which messages carry the notice** [r11, Loopy r10 S1]. Started turns only: ordinary user messages, and inter-bot deliveries that start a turn (idle fallback). Steered messages (composer steers, active-turn bot deliveries) don't; the notice waits for the next started turn. Nick said "next user message". This narrows it to "next started turn", flagged to Snark to reconcile with Nick. Covering steers would mean a notice hook inside `steer_active_turn`, with trusted target-session resolution and retry-safe marking keyed to the steer request id.
Recommendation: started turns only. A steer lands in a turn that is already running, and the next started turn is usually minutes away. **Decided by Nick 2026-10-23: started turns only (16).**

### Nick items

N1, N2 and N3 are decided (section 16). Still Nick's: which bots opt in, the 20-item cap, and when each slice deploys.

## Appendix A.00 r11 review disposition (Loopy cmvlcwe8z004z0mpzd483gogq)

| Finding | Answer | Where |
|---|---|---|
| B1 `ended_at` is `created_at + latency`, not completion; slow preparation misses the re-arm | **Fixed.** No clock. A durable per-session `compaction_gen` is bumped by finalization for every recorded compaction success, and the notice records the value it saw in a mark that commits before publish. Tests cover delayed preparation, same-turn and later-turn auto-compaction, a queued earlier turn, and a compaction while disabled. | 5.4, 9.3, 9.4, 13, 17, claim 9 |
| B2 a free-text idempotency key can equal the objective and survives hard delete | **Fixed.** The key must be a canonical UUID; anything else is refused before writing. The hard-delete promise now lists exactly what survives, and says it is not item text and not computed from it, without claiming a UUID can't be abused deliberately. Not hashed. | 5.2, 6.1, 11.3, 12, 17, claim 10 |
| S1 double fallback breaks the version bound when S has no version | **Fixed** with Loopy's bounded representation, kept on the working-memory side: `unverified_floor = {min_version, may_be_missing}`, folded across unverified re-pins and cleared on a kept turn under a verified pin. The 8.2 wording is corrected. Byte's pin contract is unchanged. | 5.4, 8.2, 9.2, 17, claim 8 |
| Claim 4 note: bootstrap insert on a refused request | **Fixed.** Same transaction; it rolls back on every outcome that isn't a real write. Tested. | 5.3, 17 |
| r10 S1 acceptance is technical only, not authorization to narrow Nick's wording | **Agreed.** OI-R8 stays with the coordinator; Snark has put it to Nick. r12 stays started-turn-only until he rules. | OI-R8 |
| Base moved during review | **Rebased** to `9e874dc`, which commits the pin byte-identical to `c100554a06b0`. The other cited files are unchanged since `4883515`. `confirm_session_pin` was the wrong name; it is `SessionPinConfirmer`. | Header, 8.2, C |

## Appendix A.0 r10 review disposition (Loopy cmvlcc925003y0mpz7n763i6e)

| Finding | Answer | Where |
|---|---|---|
| B1 fallback outranks a proven compaction; carried sections go stale after fallback + auto-compaction | **Fixed.** Byte's precedence change makes COMPACTION over a damaged pin a verified move that stores this render's sections. `candidate_sections` records the fallback render, and working memory reads both conservatively. Byte's tests are listed in OI-R1. Combined-path working-memory tests are in 17. The one residual (two fallbacks in a row) is stated. | 8.2, 9.2, 10.2, 17, OI-R1 |
| B2 content-free schema can't tell H1 from H2 | **Fixed.** An immutable `create_fingerprint` on the `create` change row (no new table). Replay compares only against it. Behavior after update, archive, soft delete and hard delete is specified, and the H1/H2 and at-cap tests are listed. | 5.2, 6.1, 11.3, 17 |
| S1 steer path bypasses the notice | **Fixed by qualification.** Started turns only; steers are explicitly excluded and the notice stays pending. Narrowing Nick's wording is flagged to the coordinator. | 2 (G4), 9.1, OI-R8 |
| S2 enable marker outlives its pin episode | **Fixed.** Notice state carries `pin_turn_id`; a different confirmed pin resets `N`. Loopy's sequence is a test. | 5.4, 9.2, 9.3, 17 |
| S3 compaction ordering and N3 wording | **Fixed.** Re-arm is ordered by the compaction turn's `ended_at`, with tests for the same-turn and later-turn cases. N3's default copy now says "the turn after its next manual /compact". The `ended_at` ordering was itself wrong (Loopy r11 B1); r12 replaces it with `compaction_gen`. | 9.4, 16, 17 |
| Claim 3: until-read acknowledgement | **Adopted** as the OI-R2 rule: only a full list acknowledges, up to the version it read. | OI-R2 |
| Claim 6: refused/no-op transactions must not bump the version | **Fixed.** Lock, decide, then bump. | 5.3 |
| Claim 7: scheduler identity | **Fixed.** Taken from the validated delivery (`prompt_delivery.py:85`). | 9.1, OI-R5 |
| Pin drift during review (a522b2d9d3a2 → 62f37a9ebe94) | **Repinned** to `c100554a06b0`, after confirming the final contract with Byte. Staging/confirm is described in 8.2; Codex reviews its internals under TASK-1318. | Header, 8.2, C |

## Appendix A. r9 review disposition (Loopy cmvl6znx8002a0mpz0sawfyrd)

| Finding | r10 |
|---|---|
| B1 suppression bypassed by deltas | No deltas. Suppressed sessions get no section and no notice (9.1, 10.1). Origins are N1. |
| B2 compaction evidence | Under B a missed stamp costs staleness, not inconsistency. Re-arm uses the success stamp (9.4). Stamp fixed in TASK-1318, pending bridge restart (OI-R7). |
| B3 archived resume | Pin ignores archive status (8.2); §11.1 contradiction removed (11.1). |
| B4 cold-scrub rehydration | No scrub, no working-memory pin table (11.3). |
| B5 lock order | No working-memory lock meets a session or turn lock (13). |
| S1 turn → generation mapping | No generations or receipts. |
| S2 mixed-version gate | No bridge change (15). |
| S3 eligibility | Leading-token maintenance classifiers, with and without arguments (9.1). No digest. |
| S4 fidelity and liveness | G2 and G4 restated with their limits (2). |
| S5 citations | Fixed (Appendix C). |
| OI-1 | Revised (19). |
| OI-2, 3, 4, 6, 7, 8, 9, 10 | Accepted by Loopy; stated in 4-7. |
| OI-5 | Nick (N2). |
| OI-11 | Revised by OI-R6 (6.2, 11.3). |
| OI-12 | Nick (N3), wording updated for TASK-1318's boundaries. |
| OI-13 | 10.1; note in OI-R5. |
| OI-14, 15, 16 | Gone with the pin table; 15 settled by TASK-1318's success check (10.2); 16 superseded by scope. |
| OI-17 | Leading-token classifiers in `src/agent_bridge/maintenance.py` (9.1). The bare-only classifier isn't needed. |
| OI-18 | Delta budget and TTL gone; registry keys named (5.4). |
| OI-19 | 14. |
| OI-20 | 15. |

## Appendix B. Earlier findings

| Finding | r10 |
|---|---|
| Caid r1 B1 (cache) | The CLI already freezes the prompt; TASK-1318 pins it; the section changes only at boundaries (1, 8.2). |
| Caid r1 B2 (initiative) | 4. |
| Caid r1 B3 (trust) | 6.1, 7. |
| Caid r1 B4 (resets) | 10.1. |
| Loopy r2 B1 (ack on queue) | No ack. |
| Loopy r2 B2 (high-water mark) | Commit-ordered `ns_version` (5.3). |
| Loopy r2 B3; Loopy r3 B3 (delete vs render) | Nick's B3 ruling (11.1). |
| Caid r2 R1, R2 | 10.1, 7; implementation claim 5. |
| Caid r2 R3 (re-pin on session or model change) | TASK-1318 cold start (8.2). |
| Caid r2 S1-S5 | S1: 10.2. S2: OI-R5. S3: notice appended after the suffix (9.3). S4, S5: cache tests (17). |
| Loopy r3 B1, B2; Loopy r4 B1, B2 | Delta and coldness mechanisms, retired. |
| Caid r5 K1-K6, S1-S4; r6 K1', S2'; r7 W1, W2 | Delta, binding, receipt and stall mechanisms, retired (18). K3' (labels stored at write) and K4 (one renderer) survive in 5.1 and 8.1. |
| Loopy r7 L1, S4/L2 | Coldness predicate, retired. |

## Appendix C. Citations

At llm-bawt `9e874dc` unless noted. Rows marked at `4883515` are for files that `9e874dc` did not change. All paths `src/` unless noted.

| Citation | Status |
|---|---|
| `claude_code_bridge/send_handler.py:208-245`, `:210` | re-verified (r9) |
| TASK-288 prompt log | `claude_code_bridge/send_handler.py:377-386`. r9 cited `command_ops.py:342-369`, which is reset/cancel handling. Corrected. |
| `memory/session_reset_lineage.py:19` | re-verified |
| `agent_context.py:211`, `:469` | path corrected: `src/llm_bawt/agent_context.py`, not `memory/` |
| `service/session_lifecycle/lifecycle.py:74` | path corrected (r9 omitted `service/`) |
| `memory/postgresql_short_term.py:430`, `:438-452` | re-verified |
| `service/runtime_switch_rotation.py:85` | re-verified |
| `service/routes/history_seed.py:208` | re-verified (r9) |
| bawthub `frontend/prisma/schema.prisma:432-460` | corrected end line (r9 said :458), at bawthub `ee7fd5a` |
| `task_turn_context.py:23` | re-verified (r9) |
| `codex_bridge/event_ops.py:91-92` | re-verified (r9) |
| `service/jobs/idle_context_candidates.py:19`; `service/message_routing.py:25` | re-verified |
| `src/agent_bridge/maintenance.py:4`, `:10` `is_compact_command`, `is_new_command` | re-verified (path outside `llm_bawt`) |
| `claude_code_bridge/send_result.py:120-176` (stamp), `:154-163` (no-metadata stamp) | re-verified at `9e874dc` |
| `claude_code_bridge/send_handler.py:604-612` `compact_status` from the SDK status message | verified at `9e874dc` |
| `service/turn_logs.py:101-115`, `:408`; `background_service.py:138` | re-verified (r9) |
| `core/base.py` 948 lines, `_sec_*` sections | re-verified at `9e874dc` |
| `service/system_prompt_pin.py` sha256 `c100554a06b0`: `sections`, `candidate_sections`, `PENDING_KEY` (:58), `SessionPinConfirmer` (:300), reason precedence | committed in `9e874dc` (TASK-1318) |
| `service/turn_stream_worker.py` `apply_session_pin` (:220), `SessionPinConfirmer` (:376), execution clock start (:250) | verified at `9e874dc` |
| `service/turn_log_schema.py:40-46` `_terminal_ended_at`; `service/turn_stream_finalize.py:112-114` latency, `:397` thread session | verified at `9e874dc` (Loopy r11 B1) |
| `service/routes/chat.py:158` `steer_active_turn`; `service/inter_bot_dispatcher.py:533` `_steer`, `:682-705` `chat_completion_stream` | verified at `4883515` (Loopy r10 S1) |
| `service/prompt_delivery.py:85` (`scheduler:` + sha256), `:141` `prefer_steer: False` | verified at `4883515` |
| `service/turn_logs.py` `ended_at` set at finalization (:408-495) | verified at `4883515` |
