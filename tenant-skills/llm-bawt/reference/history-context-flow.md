# History / Memory / Context Flow — the unified architecture (agent + chat)

> **Status banner (read first).** The unification is **shipped** (TASK-493 +
> follow-ups, all in REVIEW as of 2026-07-11). `build_context_payload` is a real,
> landed function in `utils/history.py`. The **two-axes model changed under
> TASK-518**: history and summaries are now **fully independent** — neither gates
> the other — and `session_memory_continuity` is a derived *mirror*, not a master
> switch. If you read older notes saying "summaries require continuity," that
> coupling is dead. Verify against the bytes; the **"Current status"** table (§9)
> tracks what exists today.

This is the deep dive behind the one-liner in the main skill: **how prior
conversation (history + summaries + system context) reaches the model, for both
chat bots and agent bots, and why the two look different without being two
tangled code paths.**

---

## 1. The one user intent, two independent axes

A user is expressing one thing: *"carry my prior conversation forward into my
next prompt."* Historically this fractured into two badly-scoped flags
(`include_summaries` for chat, `seed_summary_on_new_session` for agents) that
lived in different files, applied to different bot_types, and never referenced
each other. A UI toggle wired to one silently did nothing on the other. That
fracture is the bug the project cured.

The intent decomposes into **two genuinely independent switches** — this is the
TASK-518 correction to the earlier "master gate + sub-choice" model:

- **History** — carry raw recent messages (the `regular_messages` bucket).
- **Summaries** — carry rolling session summaries (the `summary_messages` bucket).

**Neither gates the other.** All four combinations are valid and functional:

| History | Summaries | meaning | scope value |
|---|---|---|---|
| on | on | recent messages + rolling summaries | `inline+summaries` |
| on | off | recent messages only | `inline` |
| off | on | **dense summary-only** (no raw messages) | `summaries` |
| off | off | carry nothing (cold turn / system prompt only) | `none` |

The summary-only mode (`off/on`) is the one the old coupled design made
impossible, and it's arguably the *best* agent seed: dense context without a
raw-message dump.

### The two settings that back this

- **`history_scope`** (str enum, default `inline+summaries`) — the **source of
  truth**. Encodes both bits, substring-tested by `scope_flags()`
  (`utils/history.py:77`): `include_history = "inline" in scope`,
  `include_summaries = "summaries" in scope`. Four canonical values: the full
  cross-product above.
- **`session_memory_continuity`** (bool, default True) — a **MIRROR**, not a
  control. `= scope != "none"` (i.e. "is anything carried?"). Kept as its own key
  so legacy readers (the bridge seed gate, config migrations, the agent
  `maybe_build_session_seed` trigger) need zero changes. The UI writes it derived;
  never treat it as an independent master switch.

`include_summaries` survives only as a per-request chat-only compat shim that can
force summaries **off** for one turn (never on). On agent bots it is inert.

---

## 2. Policy vs Mechanics — the load-bearing boundary

**All intelligence about WHAT to send lives in llm-bawt (the API). One server.**
The bridge is deliberately dumb hands.

- **llm-bawt = the brain.** It alone decides *whether* and *what* to inject on any
  given turn. Every gate — scope, is-this-`/new`, is-there-a-session, future
  "last N messages" — is an llm-bawt decision.
- **The bridge = dumb executor.** Handed `inject_messages`? It appends them. Handed
  nothing? It resumes and appends only the new user prompt. **The bridge holds
  zero policy.** Presence of messages *is* the decision — made upstream.

This boundary is why scattered logic across two servers is a bug, not a style
nit. TASK-508 ripped `_get_seed_setting` out of the bridge for exactly this
reason.

**Corollary (the mechanics the bridge *may* own):** the bridge still chooses *how*
to physically land what it was handed based on session state — cold/`/new` →
write a fresh synthetic transcript; warm → append to the existing transcript.
That's delivery mechanics, not policy. The line: **bridge decides the physical
HOW; llm-bawt decides the WHETHER and the WHAT.**

---

## 3. WHAT vs WHEN — why one handler serves both bot types

Split the problem into two questions that are usually conflated:

- **WHAT context to assemble** → *one shared handler.* Same logic for both bot
  types: pull history + summaries + system context, gate by the two independent
  flags, budget by tokens.
- **WHEN to assemble it** → *the caller's trigger.* This is where chat and agent
  genuinely differ, and the difference is irreducible:
  - **Chat is stateless.** No persistent session — it reassembles the full context
    window from the DB **every single turn**.
  - **Agent is stateful.** The Claude Code SDK on-disk transcript **is** its
    memory. You seed it **once** at birth (cold start / `/new`) and `resume=`
    replays it forever after. Re-seeding every turn would balloon the transcript.

The "everything on `/new`, nothing after" rule for agents is a **WHEN**, not a
**WHAT**. It stays in the caller's trigger (`maybe_build_session_seed`), and must
**never** migrate into the shared handler.

```
maybe_build_session_seed(...)      ← agent-only TRIGGER (WHEN): cold-start gate
        │ if it says "seed now"
        ▼
build_context_payload(*, include_history, include_summaries, delivery, max_tokens)  ← shared WHAT handler
        ▲                                    │
        │ chat calls every turn (delivery=inline)   │ agent ships result once (delivery=seed)
        ▼                                    ▼
   request window                     inject_messages → dumb bridge appends
```

---

## 4. The three delivery modes

| mode | who | when called | system rows | how delivered |
|---|---|---|---|---|
| **inline** | chat bots | every turn | kept inline in the message list | injected into the request window |
| **seed** | agent bots | cold-start / `/new` only | **stripped** (SDK carries its own byte-stable system prompt — see `agent-system-prompt.md` / TASK-288) | `inject_messages` → bridge writes a fresh synthetic transcript, then `resume=` |
| **warm-append** | agent bots (FUTURE) | any warm turn llm-bawt chooses | n/a (append) | `inject_messages` → bridge appends to the **existing** transcript before `resume=` |

**warm-append does not exist yet** (TASK-509, PARKED). It is the future "include
last X messages in my next prompt" use case. The architecture must not foreclose
it: the bridge must consume `inject_messages` *unconditionally when present*, and
llm-bawt must be free to populate it on a non-`/new` turn.

---

## 5. The assembler and the handler (the WHAT)

**The assembler:** `HistoryManager.get_context_messages()`
(`utils/history.py:281`). Returns `system_messages + summary_messages +
regular_messages` with token budgeting. Both paths call it. Its *sizing* behavior
(how many messages, how many summaries) is §6.

**The handler:** `HistoryManager.build_context_payload()`
(`utils/history.py:399`). Signature:

```python
build_context_payload(*, include_history=True, include_summaries=True,
                      delivery="inline", max_tokens=0) -> ContextPayload
```

It calls `get_context_messages()`, then buckets the rows into a `ContextPayload`
(`system_messages`, `summary_messages`, `tool_result_messages`,
`regular_messages`), applying **two independent gates**:

- `summary_messages` filled **iff `include_summaries`**
- `regular_messages` filled **iff `include_history`**

System/tool-result rows are always separated so either delivery composes its own
shape. The caller reads the view matching its delivery:
- chat → `payload.inline_history` (`tool_result + summary + regular`)
- agent seed → `payload.seed_messages` (`summary + regular`, no system rows)

### How the two callers derive the flags

**Chat** (`core/base.py`, ~L964-981): resolves `session_memory_continuity` (the
mirror) and `history_scope`, decodes with `scope_flags()`, then ANDs continuity
in so a legacy row that turned continuity off still carries nothing:

```python
include_history, include_summaries = scope_flags(scope)
include_history    = include_history    and continuity
include_summaries  = include_summaries  and continuity and self._include_summaries
```

**Agent seed** (`service/routes/history.py::build_context_seed`, ~L1707):
`maybe_build_session_seed` already decided we should seed (mirror on), so it just
decodes scope into the two flags and passes them through. `delivery="seed"` drops
system rows.

---

## 6. Sizing knobs — *how much* history and *how many* summaries

The four-value scope decides **which buckets**; these settings decide **how big**
each bucket is. All resolve **runtime_setting (bot → global) → `Config` (.env) →
declared default**. `Config` fields live in `utils/config.py`; per-bot overrides
are read via `HistoryManager`'s `settings_getter`.

### Recent-message sizing (the `regular_messages` bucket)

| knob | runtime key | `Config` field | default | effect |
|---|---|---|---|---|
| Token budget | `max_context_tokens` | `MAX_CONTEXT_TOKENS` | `0` = auto | Total prompt token budget. `0` → auto = `model_ctx_window − max_output_tokens` (`core/base.py`). Trims oldest-first to fit. |
| Message-count cap | `max_context_messages` | `MAX_CONTEXT_MESSAGES` | `0` = no cap | Hard cap on # raw messages before token trimming. Keeps the newest N (`utils/history.py:~322`). |
| Protected recent turns | `memory_protected_recent_turns` | `MEMORY_PROTECTED_RECENT_TURNS` | `3` (= last 6 msgs) | Last N user+assistant turns are shielded from eviction when the budget is tight. |

**Trim order in `get_context_messages(max_tokens=…)`:** (1) apply
`max_context_messages` count cap → keep newest N; (2) if `max_tokens > 0`: system
rows always kept, protect last `N×2` recent messages, fill summaries oldest-first
(capped by `summarization_max_in_context`), then fill remaining budget with
droppable messages newest-first. `max_tokens = 0` → no budget, return everything.

### Summary sizing (the `summary_messages` bucket)

| knob | runtime key | `Config` field | default | effect |
|---|---|---|---|---|
| Max summaries in context | `summarization_max_in_context` | `SUMMARIZATION_MAX_IN_CONTEXT` | `5` | Cap on # past-session summaries injected. |
| Compact summaries | `summarization_compact_context` | `SUMMARIZATION_COMPACT_CONTEXT` | `True` | Strip the `[HISTORICAL SUMMARY]` header, section labels, and empty Tone/Open Loops lines; flatten Key Details onto one line. **Savings are modest (~5–10% on a typical structured summary, not "half")** — the Summary + Key Details *prose* dominates the byte count and survives near-verbatim. **Context-only** — storage keeps the full summary (`_compact_summary_content`, `utils/history.py:~229`). |

### Summary *generation* triggers (when a summary is even created)

These govern the summarizer (`memory/summarization.py`), not per-turn assembly —
but they determine how many summary rows *exist* to be pulled.

| knob | runtime key | `Config` field | default | effect |
|---|---|---|---|---|
| Session gap | `summarization_session_gap_seconds` | `SUMMARIZATION_SESSION_GAP_SECONDS` | `3600` (1h) | A message gap larger than this splits history into a new "session" — the unit that gets summarized. |
| Min messages / session | `summarization_min_messages_per_session` | `SUMMARIZATION_MIN_MESSAGES_PER_SESSION` | `2` | A session with fewer messages than this is not summarizable. |
| Trigger budget | *(none yet)* | `SUMMARIZATION_TRIGGER_TOKENS` | `12000` | How much recent raw convo is kept verbatim before older sessions fold into summaries. **This is the summarization trigger budget, NOT the model context window.** `0` → falls back to the model window (effectively disables summarization on large-context models). |

**Mental model:** `SUMMARIZATION_TRIGGER_TOKENS` sets how far back raw messages
survive before folding; sessions are cut at `session_gap_seconds` boundaries and
must clear `min_messages_per_session` to fold; at assembly time up to
`summarization_max_in_context` of those folded summaries are injected (compacted
if `compact_context`), and raw recent messages fill the rest of
`max_context_tokens` with the last `memory_protected_recent_turns` shielded.

---

## 7. The seed trigger (agent WHEN gate)

`service/routes/history.py::maybe_build_session_seed()` (~L1801) is the single
shared decision point for both dispatch paths (streaming `chat_streaming.py` +
non-streaming `background_service.py`) — unified in TASK-501 so they can't drift.
It returns `inject_messages` (a list of `{role, content, timestamp}`) or `None`.
It seeds iff **all**:
- backend is `claude-code`, AND
- `session_memory_continuity` is on (the mirror — i.e. `scope != "none"`), AND
- `_is_new` (prompt starts with `/new`) **OR** `not _has_session` (no real SDK
  session id yet — a `session_key` containing `:` is a legacy routing key, not a
  session).

On a warm turn it returns `None` → nothing injected → the bridge resumes and the
only new thing that turn is the user's message. **llm-bawt does not resend history
on warm agent turns** — by design; the transcript already holds it.

---

## 8. The bridge contract (dumb hands)

`claude_code_bridge/bridge.py`:
- Parses `inject_messages` from the dispatch.
- `/new`: resets the session, calls `_seed_new_session(injected=inject_messages)`.
- Cold start (`resume_id is None`): calls `_seed_new_session(injected=…)`.
- `_seed_new_session` (~L827): if `injected` present → seed from it directly; else
  fall back to the legacy `_fetch_context_seed` HTTP callback (TASK-501 Phase 1
  keeps the fallback until app-side inject is proven fleet-wide; Phase 2 removes
  it).
- Writes a synthetic transcript (user entry carrying the flattened context +
  assistant ack, `model:"<synthetic>"`), mints a session uuid, persists it, then
  `resume=` on the first real turn. Transcript-safety rules (parentUuid chain
  intact, never end on a dangling `tool_use`) are in `agent-system-prompt.md` /
  TASK-445.

**Remaining dumb-hands gap:** `inject_messages` is only consumed on
`/new`/cold-start; warm-turn injections are silently dropped, and
`_seed_new_session` can only *create* a session, not *append*. **TASK-509**
(parked) generalizes this behind a bridge.py split.

---

## 9. Current status (source of truth for what's built)

| piece | task | status |
|---|---|---|
| Effective-config inspector (`GET /v1/config/effective`) | TASK-487 | ✅ shipped |
| Unified `ConfigResolver` | TASK-488 | ✅ shipped |
| Declared prompt manifest | TASK-489 | ✅ shipped |
| Hardcoded prompt fragments → registry | TASK-490 | ✅ shipped |
| Typed setting registry + `applies_to` | TASK-491 | ✅ shipped |
| `session_memory_continuity` unifies the two legacy flags | TASK-492 | ✅ shipped |
| `inject_messages` dispatch primitive; `maybe_build_session_seed` shared trigger | TASK-501 | ✅ shipped (legacy callback fallback still present) |
| `build_context_payload` shared handler + `history_scope` + composer UI | TASK-493 | ✅ landed (REVIEW) |
| Typed-registry config schema endpoint (`GET /v1/config/schema`) | TASK-507 | ✅ landed (REVIEW) |
| Bridge de-policy (remove `_get_seed_setting`) | TASK-508 | ✅ landed (REVIEW) |
| **Decouple history & summaries into independent axes; 4-value `history_scope`; continuity→mirror; two-toggle UI (admin + composer)** | **TASK-518** | ✅ landed (uncommitted) |
| Warm-turn inject / append-to-session ("inject last X") | TASK-509 | 🅿️ parked (needs bridge.py split first) |
| Proxy-side egress prompt capture | TASK-502 | ⏳ filed |

**Reality check on the legacy flags:** as of 2026-07 all bots are
`bot_type=agent`, so the chat-bot `include_summaries` history path is effectively
dead deployment-wide. `history_scope` (with `session_memory_continuity` as its
mirror) is the live control. `seed_summary_on_new_session` is superseded (do not
describe it as "the agent switch" anymore).

---

## 10. Key file map

| what | where |
|---|---|
| The assembler (sizing/trim) | `utils/history.py::get_context_messages` (L281) |
| Scope decoder (2 bits) | `utils/history.py::scope_flags` (L77) |
| The shared handler | `utils/history.py::build_context_payload` (L399) |
| Chat flag derivation + call | `core/base.py` (~L964-981) |
| Agent seed builder | `service/routes/history.py::build_context_seed` (L1707) |
| Agent seed trigger (WHEN gate) | `service/routes/history.py::maybe_build_session_seed` (L1801) |
| Chat delivery flag (`_include_summaries`) | `service/chat_streaming.py`, `service/background_service.py` (~L316) |
| Setting definitions + defaults | `setting_definitions.py`, `utils/config.py`, `runtime_settings.py`, `config_resolver.py` |
| Sizing config fields | `utils/config.py` (`MAX_CONTEXT_*`, `SUMMARIZATION_*`, `MEMORY_PROTECTED_RECENT_TURNS`) |
| Summarizer (session split / triggers) | `memory/summarization.py` |
| Bridge inject/seed | `claude_code_bridge/bridge.py` (~L827 `_seed_new_session`) |

---

## 11. Task-bound and dispatched flows (TASK-701 / TASK-702)

Beyond the three delivery modes in §4, two variants share the assembler but
change WHICH session gets carried:

- **Task-bound ordinary chat.** During a claude-code bridge turn where a
  task is being worked on (agent called `tasks_associate_current` or
  `tasks_update(..., associate_current_turn=true)`), llm-bawt mints a
  Fernet-encrypted capability containing the resolved `session_id`,
  `turn_id`, `trigger_message_id`, `bot_id`, `user_id`. The capability
  rides an MCP HTTP header to the MCP server, which writes the durable
  BawtHub association. **The history/summary flow itself is unchanged** —
  same session, same buckets, same assembler. Association is a side
  effect for future reopening and attribution, not a context injection.
- **Dispatched task, non-activating session.** `taskDispatchService`
  reuses the newest compatible `(bot, user)` task session; failing that,
  `POST /v1/sessions {activate: false}` mints a session that is **born
  archived** — the one-active-per-`(bot, user)` invariant is untouched,
  the user's live conversation is not rotated. Subsequent
  `/v1/chat/completions` sends carry the explicit `session_id`, so the
  dispatched thread accumulates its own history that stays out of the
  user's active chat until they choose to open it. Reopening later uses
  the association's primary session — same explicit-session path,
  history is carried per §4.

**Canonical cross-system reference:**
[`agent-skills/agent-system/task-conversation-architecture.md`](../../agent-system/task-conversation-architecture.md)
— sequence diagrams for both flows, non-activating session semantics
(`activate=false`, `rotated=false`, `created_inactive=true`), and the
contract guard that throws `INVALID_RESPONSE` if llm-bawt accidentally
rotates during a non-activating dispatch.

## 12. Gotchas

1. **History and summaries are independent (TASK-518).** Do not reintroduce the
   "summaries require continuity/history" coupling. All four scope combos are
   valid; `summaries`-only is a real, tested mode.
2. **`session_memory_continuity` is a mirror, not a switch.** It equals
   `scope != "none"`. The UI writes it derived. Reading it as a master gate is the
   old model.
3. **Never put history/summaries into the system prompt.** It's the byte-stable
   cacheable prefix (~90% cache reads, TASK-288). The seed goes into the
   transcript, not the system field. See `agent-system-prompt.md`.
4. **The bridge is dumb on purpose.** Any "should I?" check belongs in llm-bawt.
   Presence of `inject_messages` is the whole contract.
5. **WHEN ≠ WHAT.** The "seed only on `/new`" rule is a trigger, not part of the
   assembler. Keep it in `maybe_build_session_seed`.
6. **Sizing ≠ scope.** `history_scope` picks the buckets; `max_context_*` /
   `summarization_*` / `memory_protected_recent_turns` decide how big each bucket
   is (§6). Two different questions — don't conflate "carry summaries" with "how
   many summaries."
7. **`SUMMARIZATION_TRIGGER_TOKENS` is not the model window.** It's the verbatim-
   retention budget before folding to summaries; `0` disables summarization on
   large-context models.

## Summary formerly in the parent SKILL.md (TASK-1640)

### History / memory / context flow

**→ `reference/history-context-flow.md`** before touching how history, summaries, or session-seeding reach the model. Key facts: History and Summaries are TWO INDEPENDENT axes (`history_scope` 4-value enum is the source of truth); scope picks buckets, sizing knobs (`max_context_tokens`, `summarization_*`) size them; llm-bawt holds ALL injection policy, bridges are dumb hands; chat reassembles every turn, agents seed once at `/new`/cold-start (the SDK transcript IS their memory).
