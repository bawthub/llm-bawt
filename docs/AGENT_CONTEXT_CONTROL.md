# Agent Context Control

Agent sessions are stateful provider transcripts. llm-bawt now separates three
things that were previously conflated:

1. **Resident context** — tokens occupying the model's current prompt window.
2. **Cumulative usage/cost** — repeated cache reads and output across tool-loop
   iterations; this can exceed the context window and is not a fullness gauge.
3. **Durable conversation history** — PostgreSQL session/message rows, which are
   retained across provider-session resets.

## Measured deployment facts (2026-08-02)

- Snark, Al, Loopy, and Codex resolve to `openai_chatgpt/gpt-5.6-sol`.
- Their normalized catalog endpoint resolves to a **372,000-token** context
  window. The deployment does not currently configure one million tokens.
- The historical failing turn (`turn-3e3129ef3361450e84c9ed98625c8d4b`)
  recorded the upstream proxy error `Your input exceeds the context window of
  this model` while the long SDK turn was re-reading roughly 287k–360k tokens
  per model/tool iteration.
- The exact upstream ChatGPT acceptance reserve below 372k remains unmeasured.
  System prompt, MCP/tool schemas, output reserve, and proxy/SDK framing consume
  capacity not represented by a user's visible transcript. The catalog value is
  therefore the configured ceiling, not proof that a 372k user payload is
  accepted byte-for-byte.
- Claude Agent SDK exposes `ClaudeSDKClient.get_context_usage()`. The bridge
  records `totalTokens`, `maxTokens`, percentage, and auto-compact metadata at
  terminal result time. This native resident value is preferred for health.
- The final per-iteration Assistant/stream usage remains the fallback. Cumulative
  `ResultMessage.usage` is used for total output/cost only.
- The installed native Codex Python SDK shells into `codex exec resume`. It has
  no context-health or compact API. It can start a fresh thread, so it supports
  clean reset but not retained-history reset in the current bridge.

## Context-health API

`GET /v1/agent-context/health?bot_id=<slug>&user=<id>` returns:

- active durable `session_id`;
- backend capability matrix;
- `resident_prompt_tokens`, source, effective/configured/provider ceilings;
- warning/critical thresholds, ratio, remaining headroom, and state;
- catalog/provider disagreement;
- the last reset/compaction lifecycle receipt stored on the active session.

Health is scoped to the **active durable thread** by joining its trigger messages
to turn logs. On terminal turn persistence, the owning `(bot_id, user_id)`
session also records `context_stats` (latest turn ID, completed time, provider
session key and usage) in its metadata in the same transaction. An admitted
turn invalidates the old gauge only when its trigger is already linked to that
session; if the message is not yet persisted, projection waits for completion
rather than guessing which active or archived thread owns it. Bot profiles are
shared across users and do not hold a single global context gauge. A reset
therefore starts at `unknown`; the prior thread's critical usage cannot leak
into the new one. Turn logs remain the authoritative audit/recheck source.

Defaults are runtime settings and can be overridden per bot:

- `agent_context_warning_percent = 75`
- `agent_context_critical_percent = 90`
- `agent_context_critical_policy = reset_retain_history`

Before a durable inter-bot send, a critical `continue` delivery is automatically
escalated according to capability. Claude uses retained reset by default; Codex
falls back to clean reset; a backend with no safe rotation primitive continues.
Explicit caller policy always wins.

MCP: `agent_context_health(bot_id, user_id?)`.

For Claude backends, `agent_context_compact(bot_id, idempotency_key,
sender_bot_id?)` queues exactly one durable `/compact` maintenance turn with
`when_idle` semantics. It never interrupts an active turn. Unsupported backends
are rejected before enqueue; Codex does not receive `/compact` as
ordinary prose. The bare-command summary policy, explicit-instruction precedence,
automatic-compaction command-hook finding (distinct from the Python callback
limitation), and activation status are documented in
[COMPACTION_POLICY.md](COMPACTION_POLICY.md).

## Reset policies

Authoritative enum:

- `continue` — current durable thread and provider transcript.
- `reset_retain_history` — wait for idle, archive the active durable thread,
  create a fresh thread, cold-start the provider, and seed bounded history from
  the archived predecessor according to `history_scope` and context budget.
- `reset_without_history` — same non-destructive thread rotation, but send an
  explicit empty seed so the provider starts clean.

Capability matrix:

| Backend | Inspect | Compact | Retained reset | Clean reset |
|---|---:|---:|---:|---:|
| Claude Code / proxy-routed OpenAI | yes | native `/compact` | yes | yes |
| Native Codex | no | no exposed API | no | yes |

Compatibility fields `reset_session_before_delivery` and `retain_history`
normalize into the enum. Contradictory combinations return 422.

Manual idle reset:

```http
POST /v1/agent-context/reset
{"bot_id":"snark","session_policy":"reset_retain_history","reason":"large task"}
```

MCP: `agent_context_reset(bot_id, session_policy, reason, user_id?)`.

Resets are non-destructive: prior `sessions` and partitioned `messages` rows stay
readable. The new session's metadata records the action, reason, predecessor,
and time. A reset requested while the bot has an open turn returns 409.

## Shared new-session lifecycle (TASK-1188)

`service/session_lifecycle` owns typed reset intent, execution, completion and
recovery. Callers use the existing durable delivery queue, not a reset-specific
queue. A database rotation alone is not a completed reset.

| Caller | Continuity | Admission | Completion evidence |
|---|---|---|---|
| Continuous agent `/new` or `/new <task>` | configured | wait for idle | exact provider acknowledgement, keyed lineage, completed persisted transcript |
| Continuous chat-model `/new` or `/new <task>` | configured | wait for idle | keyed lineage and completed persisted transcript; no fabricated provider acknowledgement |
| Bot-send reset policy plus task | retained or clean | wait for idle | exact provider acknowledgement, unchanged persisted task answer, separate app receipt |
| Context-reset API | requested retained or clean | skip if busy/stale | same provider-backed receipt |
| Session-reset API | clean (compatibility) | skip if busy/stale | same provider-backed receipt |
| Scheduled System `/new` | configured | skip if busy/stale | same provider-backed receipt |

Only the exact `/new` token followed by whitespace or end-of-input is a command;
`/newspaper` and `/new/task` are ordinary text. Text-part messages use the same
parser as admission and receipt verification. The original command remains in
history; only its parsed task is sent to inference. Bot-policy task text stays
literal, including a task that happens to start with `/new`.

The target session and message author are independent. Scheduled commands are
ordinary visible user-role `/new` messages authored by **bot `system`**, displayed
as **System** through its entity profile. System is not a runnable bot or a new
role, and does not change the conversation owner.

For bare commands the app owns the confirmation and canonical assistant ID.
For reset-plus-task the task answer is never rewritten; one separate app-authored
confirmation is persisted. Receipt and `thread_switched` outbox commit together.
Fresh history HTTP reads return the same author, command, answer and receipt IDs;
Redis publication is retryable, and duplicate event delivery is an invalidation,
not permission to move the bot's pointer backward.

`bot_profiles.last_session_id` selects the latest accepted bot session. Updates
occur under admission fencing or in the first rotation transaction, never on a
late completion or keyed replay. Claim-time predecessor capture permits queued
`/new` commands to rotate sequentially. Accepted reset deliveries with uncertain
provider/task execution are terminally ambiguous rather than automatically
replayed; crash recovery may finish a receipt only from persisted evidence.

### Explicit-thread compatibility

An explicitly selected thread remains selected: `/new` is literal text there,
not authority to replace the selection or reset another active conversation.
This preserves the agent explicit-thread guard and corrects a chat-only bug:
the old chat interceptor reset the unrelated active session before honoring the
explicit thread for persistence. No reset receipt is emitted for literal text.
Runtime model switches retain lazy provider binding and have no command receipt;
their pointer moves only when it still names the replaced session.

Native Codex retains clean reset support, including standalone reset with a
verified empty successor provider binding. It rejects retained/configured
history it cannot seed instead of silently discarding it.

## Idle oversized-context job (TASK-1185 / TASK-1188)

**The shared lifecycle is implemented and verified across callers.** The app
registers one disabled `IDLE_CONTEXT_RESET` job with a five-minute interval;
source reload never enables it. Activation requires explicit operator approval
and a successful disposable-provider/browser check. Eligibility defaults to **strictly above
100,000 resident tokens** and **strictly more than 15 minutes since completion**,
not turn creation time. TASK-1197 makes these thresholds configurable; see
[SCHEDULER_PARAMETERS.md](SCHEDULER_PARAMETERS.md). Each sweep persists its effective
parameters and passes an immutable snapshot through submission and claim-time checks.

`IdleContextCandidates` selects at most one session per bot through
`bot_profiles.last_session_id`. It never enumerates per-user active sessions or
guesses a missing pointer. Explicitly resumed archived sessions are valid.
Canonical session-linked turn usage supplies the measurement; newer bot activity
invalidates an older gauge. Protected bots are excluded before their sessions or
turns are read. Unknown measurements and invalid/missing pointers are skipped.

The job is a thin caller of `NewSessionLifecycle.submit`: a literal visible
`/new`, authored by **BOT:system**, targeting the existing session's owner.
System is not a new protocol role or conversation owner. The intent requests
`CONFIGURED` continuity (the same policy as typed `/new`), standalone reset,
`SKIP` admission, exact expected session/usage-turn fences, and job/run provenance.
The lifecycle rechecks eligibility, queued/active/reserved work and both fences
under the target lock at submission and claim. It owns execution, history,
provider acknowledgment, confirmation and thread-change events; the scheduler
performs no session writes or event publication.

The normal scheduler owns **one JobRun for the entire sweep**, including start,
finish, duration and a per-bot result dictionary. A transactional due-slot claim
prevents duplicate workers executing the same occurrence. Only a shared-handler
`COMPLETED` result with a persisted receipt counts as a reset; queueing or a
successful transport alone does not. Failed and ambiguous/unconfirmed outcomes
remain visible and make the run fail rather than falsely report success. A
stable job/session/measurement key reattaches to the same delivery across later
sweeps; the run ID is provenance, not a fresh reset key. No per-candidate JobRun
or job-owned event outbox remains.

Verification includes `test_idle_context_reset.py`,
`test_idle_context_reset_pg.py` (test-owned PostgreSQL only), and preserved
projection regressions in `test_turn_context_snapshot_pg.py`. Full cross-caller
parity and disposable-provider/browser proof are separate completion gates;
passing candidate/queue tests alone does not authorize activation.

## Durable inter-bot ordering and idempotency

`bots_send_message` and `POST /v1/inter-bot-deliveries` accept `session_policy`.
A reset policy deliberately disables steering: the FIFO head waits for the
active turn to end. Under the same per-target PostgreSQL advisory lock and in one
transaction, the dispatcher:

1. archives the current durable thread;
2. creates a deterministic fresh thread for the delivery;
3. stores old/new thread IDs with `reset_status=ROTATED` (not completion);
4. rewrites the server-owned payload with the fresh thread and seed source;
5. reserves the deterministic fallback turn.

The provider/SDK id is not caller-authored. The bridge cold-starts because the
fresh thread has no provider key, then writes the minted key onto that thread.
A duplicate idempotency key returns the original delivery; changing its reset
policy is rejected. Retry/restart reuses the same new thread.

## Confirmed overflow recovery

A durable fallback turn that returns a recognized context-window error gets one
controlled recovery:

- same delivery, user-message ID, task association, and fallback turn ID;
- one new deterministic durable thread;
- one new deterministic bridge request ID suffixed `_recovery_1`;
- retained-history reset where supported, otherwise clean reset;
- prior durable history remains unchanged.

A second context overflow dead-letters immediately. Generic 500/overload/timeout
errors do not trigger context reset.

## Verification

```bash
uv run pytest -q \
  tests/test_claude_context_usage.py \
  tests/test_inter_bot_transport.py \
  tests/test_thread_agent_keys.py

uv run pytest -q -m integration \
  tests/test_agent_context.py \
  tests/test_inter_bot_delivery.py
```

Bridge code changes require a coordinated bridge restart for live native-context
and clean-seed proof. Never restart an agent bridge from the agent running inside
it.
