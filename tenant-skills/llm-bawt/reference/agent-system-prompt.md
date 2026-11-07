# Agent backend system prompt, history & prompt caching

How the system prompt and conversation context reach **agent backends**
(claude-code / codex bridges) vs **chatbots** (direct LLM clients), and why
prompt caching behaves the way it does. Hard-won; verified against the
claude-code source in `~/dev/claude-code-source` + a live SDK POC (2026-06-21).

> ✅ **TASK-1318 (2026-10-23): the system prompt is pinned per session.** The
> Claude Code CLI keeps the prompt it recorded at session start, so llm-bawt pins
> one composed prompt per session and sends those exact bytes on every
> pin-enabled, session-backed turn whose pin resolves (a reset, a turn without a
> session, or a pin-store error sends the fresh render instead; §7b). A prompt
> edit lands at the session's next boundary (`/new`, reset, model switch, first
> run, or the first turn after a manual `/compact`), **not** on the next turn.
> See §2 and §7. TASK-288's byte-stability work (§5) still holds.
> For current activation, inspect the running service and session pin; the mechanics below describe the supported pin behavior.

## 1. What llm-bawt actually sends — agents get NO history

`core/base.py::_build_context_messages` branches on model type
(`self.model_definition.get("type") in ("agent_backend","claude-code")`, at the
agent early-return ~L590):

- **Agent backends:** early return (~L589). llm-bawt sends **only the current
  user message** (+ optional per-turn prefixes) and the `system_prompt` as a
  **separate field**. It does **NOT** assemble or send conversation history.
- **Chatbots:** the code past the early return runs
  `history_manager.get_context_messages(...)` and packs prior
  user/assistant/summary/tool messages into the array.

**Conversation continuity for agents** comes from the **claude CLI's own
on-disk transcript**, replayed via `resume=<session_id>`. The session id is
persisted in `bot_profiles.agent_backend_config` and reused every turn.
So at the Anthropic wire, an agent turn is still
`system_prompt → CLI-replayed transcript → current message` — but the **CLI**
builds the transcript, not llm-bawt.

The `chat.send` Redis command carries a single `message` string, never a
messages array (`agent_backends/agent_bridge.py` send_command ~L191).

## 2. The CLI DOES freeze the system prompt per session (TASK-1318)

TASK-288 concluded the opposite from source reading; a wire probe proved it wrong
on CLI 2.1.283 / SDK 0.2.160 (Snark, 2026-10-23). The CLI writes a
`prompt_snapshot` attachment into the session transcript and, on resume, sends
**that** snapshot, ignoring a changed `--system-prompt`:

| Turn | Bridge passes | CLI sends |
|---|---|---|
| cold start | V1 | V1 |
| resume | V2 | **V1** |
| `/compact` | V3 | V1 (the compaction call) |
| first turn after the compaction | V4 | **V4** (snapshot re-recorded) |
| resume | V5 | V4 |

An **auto**-compaction inside a turn re-records that turn's bytes, so it is not a
separate boundary. `--bare` / `CLAUDE_CODE_SIMPLE=1` changes how the CLI builds
the prompt; the bridge must not turn it on.

So llm-bawt pins (§7) instead of rendering fresh every turn: a fresh render would
log a prompt the model never saw. The regression is
`tests/test_system_prompt_pin_sdk.py` under `make test-sdk-probe`. It drives the
bridge's real option builder against a fake upstream and fails if the freeze
stops holding or bare mode turns on.

## 3. The bug that was fixed: the bridge used to null it on resume

`claude_code_bridge/bridge.py` (~L787) previously did:
```python
system_prompt = system_prompt if not resume_id else None   # REMOVED in TASK-288
```
Because `resume_id` is set on essentially every turn (session id reused), the
bridge passed the persona only on the first turn after `/new` / model-switch /
first run. TASK-288 read this as the agent running on the stock default prompt
in steady state; that was never wire-tested, and given the §2 snapshot it
likely kept the session's first prompt instead (the TASK-1318 probe did not
test an absent `--system-prompt`, so treat either reading as unverified).

**TASK-288 removed the gate** in both bridges, which then passed
`system_prompt=system_prompt` every turn (Claude as the `ClaudeAgentOptions`
param; Codex, which has no separate system param, by prepending it to the
user-message text via `_build_prompt_input`).

**Current behavior (TASK-1318, §7b):**
- `claude_code_bridge/bridge.py` (~L787) still passes `system_prompt` every
  turn, but llm-bawt now hands it the session's pinned bytes rather than a fresh
  render (full-prefix cache read on resume).
- `codex_bridge` pastes the pinned prompt only on a fresh thread or at a pin
  boundary (`system_prompt_delivery`), not every turn.

## 4. Prompt caching: the system prompt is the cacheable prefix

Caching is **prefix-match**: a request gets a cache read only where the prefix
(from the start, up to a cache breakpoint) is **byte-for-byte identical** to a
previously cached prefix. Any change near the top invalidates everything after
it. This is identical at the wire for the SDK and the raw Messages API — the SDK
is just a convenience layer; `resume=` only saves *re-uploading the transcript*,
not the system prompt. Min cacheable prefix ≈ 1024 tokens (Sonnet) / 2048 (Haiku).

For agents the system prompt sits in front of a **large, growing, already-cached
transcript**, so:
- **Stable** system prompt re-sent every turn → full-prefix **cache read** (~10%
  of fresh input). Effectively free.
- **Volatile** system prompt → busts the cache for the whole transcript behind
  it, every turn, and worsens as the session grows.

**POC (2026-06-21, `ClaudeAgentOptions` + `resume=`, ~26.9k-token stable prompt):**

| Turn | What | cache_creation | cache_read |
|---|---|---|---|
| 1 | new session | 26,884 | 0 |
| 2 | resume, same prompt | 14 | 26,884 |
| 3 | resume, same prompt | 14 | 26,898 |
| 4 | resume, prompt + 1 clock line | 12,134 | 14,813 |

→ Stable prompt caches fully on resume; one changing line (appended at the *end*)
still busted ~12k tokens. A change near the *top* busts nearly everything.

## 5. The llm-bawt system prompt is STABLE (the two leaks are fixed)

It is **not** a "volatile 6.4k prompt" — it's a stable ~6.4k prompt. It used to
have two components that bled per-turn variance into the cacheable prefix; both
were removed in TASK-288:

1. **Temporal line** — `utils/temporal.py::build_temporal_context()` was injected
   at `core/base.py:469-476` (`SectionPosition.DATETIME`, near the top), changed
   every minute, and was the dominant cause of system-prompt char drift.
   **Removed (TASK-288)** — dropped entirely from the live builder for agents
   **and** chatbots; the `build_temporal_context` import is gone from `base.py`.
   Time-grounding is to be redesigned later off the cacheable prefix (follow-up
   task). NOTE: a second, **dead** temporal injection still exists in
   `core/pipeline.py:285-292` — `RequestPipeline` is unwired (only a docstring
   calls `.execute()`); nothing in the live path runs it. Left as-is; flagged as
   a cleanup follow-up.
2. **Response-style block** — `core/base.py` (~L538) scanned the **user message**
   for `tldr` / `eli5` / `deep dive` and mutated the **system prompt** (backwards
   coupling). **Moved (TASK-288)** — it's now computed as a local and stamped on
   the **outbound** user message: appended to `prefix_parts` on the agent path,
   and stamped onto a *copy* of the last user message on the chatbot path (never
   the stored object — see §6 prefix pattern). Keyword match tightened to word
   boundaries (`\btldr\b` / `\beli5\b` / `\bdeep[ -]?dive\b`).

Stable per session (fine to leave): persona/base prompt, and the
`<runtime-context> model:` block (`agent_backends/claude_code.py`
`compose_system_prompt`). Both are inside the pinned bytes. The bridge's MCP
tool-context block is appended **bridge-side** after the pin and cached per bridge
process, so it is **not** in the logged pin text.

**TASK-633 (2026-07-21): native agent prompts no longer receive app-side tool
protocol instructions.** Claude Code and Codex already own their tool
schemas and calling protocol. The retired XML generator injected fake
`<tool_call>` syntax, a partial/incorrect tool inventory, a serial "Call ONE tool
at a time" policy, and a current-minute timestamp into agent system prompts. The
`tools` manifest section is now chat-only, and the obsolete XML format/generator
code is deleted; old `tool_support/tool_format="xml"` configuration maps to ReAct
for non-agent compatibility.

## 6. Agent-only prompt families (so you stop hunting for them)

**Per-turn user-message prefixes** — content that is per-turn or mode-dependent
rides the user message so it stays **off** the byte-stable cached system prefix
(post-TASK-288 rationale; previously framed as "surviving resume," but the system
prompt now survives resume too). Assembled in `core/base.py` (~L590-640). Slash
commands (`/new`) bypass wrapping.

| Prompt key | Fires when | Lands in | Gate |
|---|---|---|---|
| `chat.agent_user_prefix` | every turn, agents | user-message prefix | `inject_user_prefix` flag |
| `chat.agent_voice_prefix` | every turn, **voice only**, agents | user-message prefix | `tts_mode` |
| `response_style` (inline) | agents+chatbots, on `tldr`/`eli5`/`deep dive` | user-message prefix (agents) / copy of last user msg (chatbots) | keyword in user message |
| `chat.tts_output_instructions` | voice, **all bots** | system-prompt section | `tts_mode` |

Important: a prefix added to the **outbound** message must NOT be written to
stored history/memory (else it replays forever / pollutes memory). Augment the
wire copy, keep the original stored. This is the pattern any "ride the user
message" injection must follow.

**Task-dispatch prompts** (`agents.task_spec`, `agents.task_execution`,
`agents.project_plan`, `agents.review`, `agents.docs`) — bodies in
`agent_backends/prompts.py`, registered in `prompt_registry.py`. These are
**not** in the chat flow; BawtHub's `frontend/.../taskDispatchPrompt.ts` fetches
and renders them on task dispatch. Irrelevant to per-turn persona/caching.

## 7. The fixes: TASK-288 (2026-06-21) and the TASK-1318 session pin (§7b)

TASK-288's premise that the prompt "reaches the agent every turn" was wrong (§2).
Its byte-stability work still matters, because the pinned bytes are the cache
prefix.

Made the agent system prompt **byte-stable**, then **stopped nulling it on resume**:
1. response-style block → outbound user message (wire copy only)
2. dropped temporal from the system prompt
3. removed the `resume_id → None` gate at `bridge.py:787` (+ codex equivalent)

Result: a byte-stable prompt is a full cache read on every turn. Verified with a
real-bot-shaped stable prompt (~28.7k tokens) on resume:

| Turn | What | cache_creation | cache_read |
|---|---|---|---|
| 1 | new session | 28,730 | 0 |
| 2 | resume, stable (post-fix) | 14 | 28,730 |
| 3 | resume, stable (post-fix) | 14 | 28,744 |
| 4 | resume, clock line at TOP (pre-fix temporal) | 13,980 | 14,813 |

→ Stable prompt cache-reads the full prefix on every resume for ~free; a
per-minute temporal line at the top busts ~14k tokens **every turn**. That delta
is the recurring cost the fix eliminates.

The older "session identity seed into the first transcript message" idea was built
to route around the §2 myth and is **superseded** — do not implement it.

### 7b. Session pin (TASK-1318)

`service/system_prompt_pin.py`; the hook `apply_session_pin` is called from
`turn_stream_worker.py` for agent backends whose `pins_system_prompt` is True
(claude-code, codex).

- **Pointer:** `sessions.session_metadata.system_prompt_pin` =
  `{v, turn_id, sha256, chars, pinned_at, reason, snapshot_unverified[, sections]}`.
  **Bytes:** `turn_logs.request_json.system_prompt_pin.text` on the pin turn.
  Every pin-enabled, session-backed agent turn whose pin resolution succeeds
  records `request_json.system_prompt_pin`
  (`pin_turn_id, sha256, action, reason?, rendered_sha256?` when the fresh render
  drifted). `system_prompt_pin` is in `AUDIT_KEYS`, so re-arms and backfills keep it.
- **The pin moves only on these turns** (`reason`):
  - `cold_start`: no resume id, a structured reset, or a non-explicit `/new`;
    also the first turn after a bare reset (`reset_only`: bare `/new` or a
    standalone `inter_bot_reset_only`). That turn reaches no model, so it
    leaves `session_metadata.system_prompt_pin_boundary`
    `{turn_id, resume_id}`. The Claude bridge answers a bare `/new` with a seed
    transcript that has no prompt snapshot, so the next turn's bytes become the
    snapshot. The marker counts only once that turn resumes a different SDK
    session id (the reset really ran). It is dropped if the resume id is
    unchanged, and cleared when a pin move is confirmed (TASK-1318, Snark's
    live `/new` finding);
  - `compaction`: the first turn after a `/compact` turn in this session whose
    usage recorded `context_action=compact`, `outcome=success`. It beats every
    fallback below (it needs only the pointer's `pinned_at`);
  - fallbacks, marked `snapshot_unverified` because the harness may still hold
    older bytes until its next boundary:
    - `missing_row`: the pin turn was deleted (delivery recovery, bot purge);
    - `missing_text` / `sha_mismatch`: the pin text is missing or corrupt;
    - `legacy_adopt`: a session from before the pin existed.
- **Two-phase move:** a boundary turn stages its pointer under
  `session_metadata.system_prompt_pin_pending` and marks its audit
  `pending: true`. `SessionPinConfirmer` promotes it on the turn's first
  model output (`proves_prompt_consumed`: text, reasoning, tool or usage
  events — never `turn_cancelled`, `upstream_status` or done-time bookkeeping).
  A boundary turn that is cancelled or fails before that never moves the
  pointer; the next turn decides
  the boundary again and takes its own render, and a kept turn drops the stale
  staged move. An audit still showing `pending: true` = that move was never
  confirmed and the pointer did not change; it does not prove the model never
  received those bytes (a turn can fail after delivery but before any output).
- **Sending:**
  - Claude sends the pinned bytes verbatim; with `system_prompt_pinned` it does
    not compose again.
  - Codex gets `system_prompt_delivery`: `now` on a boundary, otherwise
    `cold_start` (paste only when the thread isn't resuming).
- **Fails open:** if the pin store errors, the turn sends the fresh render and
  logs a warning.
- **`sections`** (TASK-1281): content-free section metadata stored on the
  pointer. A verified boundary stores this render's sections; an unverified
  re-pin keeps the previous pointer's. Unverified re-pins and legacy adopts
  also store `candidate_sections` (this render's), carried on kept turns and
  exposed as `PinDecision.candidate_sections` until a verified boundary clears
  it. Kept turns read only the confirmed pointer.
- **Known gap:** llm-bawt can't see a Codex auto-compaction, so it can't re-paste
  when one happens. If one drops the pasted prompt, the prompt stays missing
  until the next cold start.

**Deploy note:** committed in `9e874dc` but not yet deployed or live-verified.
The app needs a restart, and so do the bridges for their side
(claude: compact-success stamping; codex: delivery modes). Existing sessions
adopt their current render on the first turn after deploy (`legacy_adopt`).

> Line numbers are approximate (code shifts) — grep to confirm before editing.

## Summary formerly in the parent SKILL.md (TASK-1640)

### Agent system prompt, history & caching

**→ `reference/agent-system-prompt.md`** before debugging persona loss, "why isn't my prompt change taking effect," or cache cost on agent bots. Key facts: agents get NO history from llm-bawt (only current message + system prompt; continuity = the CLI's on-disk transcript via `resume=`); the system prompt is **pinned per session** (TASK-1318; committed in `9e874dc`, live only after the app and agent-bridge restarts — check the reference's status line). The CLI freezes the prompt at session start, so llm-bawt sends the pinned bytes on every pin-enabled, session-backed turn whose pin resolves (resets, session-less turns and pin-store errors send the fresh render), and an edit lands only at the session's next boundary: `/new`, reset, model switch, first run, or the first turn after a manual `/compact`. The pinned bytes are also the cacheable prefix, so keep the prompt byte-stable (stable prefix ≈ 10% cost).

**Shared agent rules ("## Shared agent operating rules", every agent bot) = prompt key `agents.global_prompt`**, one global row in `prompt_templates`, gated by the global runtime setting `agent_global_prompt_enabled` (chat bots never see it). Not in source or skills, so don't grep for it. To edit: `GET http://app:8642/v1/prompts/agents.global_prompt?scope_type=global`, splice the change into the full `body`, then `PUT` the same URL with the full body + `title`/`category`/`format`/`metadata` from the GET, `updated_by` (your bot id) and `change_note`. Each write adds a row to `prompt_template_versions`; revert = PUT the previous version's body. The user's UI is the BawtHub Prompt Registry. An edit reaches each agent at that agent's next session boundary (`/new`, reset, model switch, first run, or the first turn after a manual `/compact`), not on its next turn. Each boundary then pays one cache rebuild, so batch edits; don't drip them.
