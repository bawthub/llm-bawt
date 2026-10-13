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
to turn logs. A reset therefore starts at `unknown`; the prior thread's critical
usage cannot leak into the new one.

Defaults are runtime settings and can be overridden per bot:

- `agent_context_warning_percent = 75`
- `agent_context_critical_percent = 90`
- `agent_context_critical_policy = reset_retain_history`

Before a durable inter-bot send, a critical `continue` delivery is automatically
escalated according to capability. Claude uses retained reset by default; Codex
falls back to clean reset; OpenClaw continues because it has no safe rotation
primitive. Explicit caller policy always wins.

MCP: `agent_context_health(bot_id, user_id?)`.

For Claude backends, `agent_context_compact(bot_id, idempotency_key,
sender_bot_id?)` queues exactly one durable `/compact` maintenance turn with
`when_idle` semantics. It never interrupts an active turn. Unsupported backends
are rejected before enqueue; Codex and OpenClaw do not receive `/compact` as
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
| OpenClaw | no | no app contract | no | no |

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

## Durable inter-bot ordering and idempotency

`bots_send_message` and `POST /v1/inter-bot-deliveries` accept `session_policy`.
A reset policy deliberately disables steering: the FIFO head waits for the
active turn to end. Under the same per-target PostgreSQL advisory lock and in one
transaction, the dispatcher:

1. archives the current durable thread;
2. creates a deterministic fresh thread for the delivery;
3. stores old/new thread IDs and reset receipt on the delivery;
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
