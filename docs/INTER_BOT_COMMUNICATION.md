# Inter-Bot Communication

Inter-bot messaging is exposed through MCP. Tool registration and immediate-send
compatibility live in
[`src/llm_bawt/mcp_server/inter_bot_tools.py`](../src/llm_bawt/mcp_server/inter_bot_tools.py).
Durable delivery state lives in
[`src/llm_bawt/inter_bot_delivery.py`](../src/llm_bawt/inter_bot_delivery.py),
and the application-owned drain loop lives in
[`src/llm_bawt/service/inter_bot_dispatcher.py`](../src/llm_bawt/service/inter_bot_dispatcher.py).

## Delivery modes

| Mode | Invocation | Busy behavior | Durable | Intended use |
|---|---|---|---|---|
| Steer or idle (default async) | omit `wait_for_reply`/`delivery` | steers the active Claude Code turn in place; otherwise starts one safe idle turn | Yes | Delegation, corrections, READY/BLOCKED/PROGRESS callbacks |
| When idle | `delivery="when_idle"` or `queue_if_busy=true` | never steers; waits for one fresh idle turn | Yes | Work that must begin in a separate turn |
| Waited | `wait_for_reply=true` | rejects a busy target; never overlaps it | No | Bounded synchronous compatibility calls |
| Forced compatibility | `force=true` | uses the same durable steer-or-idle path | Yes | Legacy callers; never authorizes agent concurrency |

Every asynchronous send is durable and immediately returns a stable delivery ID.
A waited timeout can still mean the target is running; do not blindly retry it.

## Durable steer-or-safe-deliver contract

A durable send returns immediately with a stable `delivery_id`,
`user_message_id`, and deterministic fallback `turn_id`. The canonical row is
`inter_bot_deliveries` in llm-bawt PostgreSQL.

States:

- `QUEUED` — persisted, waiting for steer readiness, target idle, or backoff expiry.
- `STEERING` — FIFO head claimed against one exact active Claude Code turn.
- `DISPATCHING` — FIFO head claimed; deterministic fallback turn reserved.
- `DELIVERED` — steer persisted or fallback target turn completed successfully.
- `FAILED` — terminal validation error or retry attempts exhausted (dead letter).
- `CANCELLED` — user/caller cancelled while still queued.

Rows retain sender, target, message/payload, timestamps, attempt count,
`next_retry_at`, `last_error`, response metadata, optional project/task/kind
correlation, and context lifecycle fields (`session_policy`, reset status/reason,
old/new durable session IDs, reset time, overflow recovery count).
`idempotency_key` is unique within `(sender_bot_id, target_bot_id)`; a duplicate
submission returns the original row and cannot create another turn or reset.
Retrying one key with a different session policy is rejected.

### Ordering and atomic steer-or-fallback choice

For each target, PostgreSQL selects the oldest active delivery by insertion
ordinal. A transaction-scoped per-target advisory lock atomically chooses one
path:

1. active Claude Code turn + default mode -> `QUEUED -> STEERING` against that
   exact turn ID;
2. active unsupported backend or explicit `when_idle` -> remain `QUEUED`;
3. no active turn -> reserve deterministic `turn-delivery-*` and transition to
   `DISPATCHING`.

A steer uses stable `steer_delivery_*` RPC and user-message IDs. Redis publishes
that RPC once and retains its result for seven days, so a timeout/restart replays
the same acceptance rather than interrupting twice. The early active-turn race
(`agent_session_key` not registered yet) requeues without consuming the delivery
attempt budget. A definitive unaccepted result marks that exact target turn as
rejected: the delivery waits for it to end, may steer a genuinely newer active
turn, or starts exactly one idle fallback. Once steer acceptance is recorded,
no error may convert the delivery into a fresh turn; retries only repair message
persistence through the same cached RPC.

Ordinary target turn creation takes the same lock and refuses to pass a reserved
fallback turn. If a normal turn wins first, the callback stays queued. If it
appears after a fallback claim but before target execution, `DeliveryTargetBusy`
releases the reservation and atomically returns the delivery to `QUEUED`.

Fallback execution invokes `BackgroundService.chat_completion` with
`stream=false`, so it receives the normal prompt, memory/session, tools, history,
turn log, and bridge runtime. Steering instead redirects the existing Claude SDK
run and persists the inbound message into that same conversation.

### Steer priority: bot deliveries never interrupt (TASK-934)

A steer is delivered at one of two Claude Code CLI queue priorities
(`ChatSteerRequest.priority` → `chat.steer` `priority` field →
`ClaudeActiveRun.steer`):

- `now` (default for `/v1/chat/steer`, i.e. user redirection): SDK `interrupt()`
  plus replacement query. The CLI kills in-flight tool calls.
- `next` (always used by the inter-bot dispatcher): the message is written to
  the CLI's stdin with a uuid and `priority: "next"`. Running tools finish; the
  CLI folds the message into the next model call. If the turn is already
  producing its final answer, the CLI runs the message as a follow-up turn.

The bridge tracks the CLI's `command_lifecycle` frames (`queued → started →
completed|cancelled|discarded|refused`) for its injected uuids. The SDK parser
drops that frame type, so `ClaudeActiveRun.messages()` reads the raw frames. The
steer RPC succeeds once the CLI reports `queued`. A ResultMessage that arrives
while an injected command is still unstarted is held back, so the follow-up
turn's own result ends the llm-bawt request instead of losing the message.
Previously every bot steer was a `now` interrupt. A reply to a sender blocked
in `wait_for_reply` killed that sender's pending tool calls and recorded them
with the CLI's misleading "user doesn't want to proceed" text; steer-killed tool
results are now relabelled as steering interruptions.

**Proxied (OpenAI/Kimi/…) targets — TASK-937.** Against the bridge's proxy
(a custom `ANTHROPIC_BASE_URL`), the CLI sends a `next` message absorbed at a
tool boundary as a positional `role: "system"` message right after the
`tool_result` ("The user sent a new message while you were working: …"). Its
environment/skills context block arrives the same way. Both translators
(`proxy/translate.py`, `proxy/translate_cc.py`) used to drop every
non-user/assistant role, so OpenAI-routed bots never saw `next` steers or
that context. They now keep such messages in place as user text wrapped in
`<system-reminder>` (`inline_system_text`). Moving them into `instructions`
would break its byte stability and with it the prompt cache. Against real
Anthropic, the same CLI instead folds the message into the `tool_result`
string, which Claude models act on.

### Pre-delivery session policy

The authoritative enum is `continue` (default), `reset_retain_history`, or
`reset_without_history`. Reset deliveries deliberately do not steer: they wait
for the target's active turn to end, then rotate the durable thread and reserve
the fallback turn in the same per-target PostgreSQL transaction. Prior messages
are never deleted. Retained mode seeds bounded configured history from the
archived predecessor; clean mode sends an explicit empty seed. Duplicate
idempotency keys perform at most one rotation.

Backend capability is explicit: Claude Code/proxy-routed OpenAI supports both
reset modes; native Codex currently supports only clean reset; OpenClaw supports
neither. `reset_session_before_delivery`/`retain_history` are compatibility
inputs and contradictory combinations fail. At critical context health, an
otherwise-`continue` durable send may be automatically escalated by per-bot
runtime policy. See [AGENT_CONTEXT_CONTROL.md](AGENT_CONTEXT_CONTROL.md).

A recognized context-window overflow during a fallback turn permits exactly one
controlled recovery of the same logical delivery. It rotates once, reuses the
message/task/turn identities, and changes only the deterministic bridge attempt
ID. A second overflow dead-letters; generic overload/timeout errors do not reset.

Terminal turn updates wake the dispatcher immediately. A 15-second sweep is only
a restart/missed-signal fallback.

### Retry and restart recovery

Transient pre-accept failures requeue with bounded exponential backoff (default
five attempts; maximum twenty). Terminal 4xx validation/auth errors fail once.
Unknown/unsupported target bots still produce a stable, inspectable `FAILED`
delivery rather than an untracked rejection, and the enqueue call returns 404 or
422 (plus the delivery ID/status body) instead of HTTP 202.

Each delivery also owns a deterministic `req_delivery_*` bridge request ID. The
shared Redis command publisher atomically deduplicates request IDs across Claude
Code, Codex, and OpenClaw. A retry either publishes once or reattaches to the
same `agent:run:*` stream, while stable user-message/turn IDs make llm-bawt
history and turn creation idempotent. Durable command/run keys use a seven-day
sliding retention window. Within that bounded recovery window, retries and app
restarts reattach instead of launching another SDK turn. If transport state is
lost/evicted beyond that window, the delivery is dead-lettered for inspection
rather than claiming an unbounded exactly-once guarantee.

A PostgreSQL session-level advisory lock elects one active dispatcher across
processes/rolling overlap; lock ownership drops automatically with its database
connection. Each dispatch claim also has a random claim token and a five-minute
lease renewed every minute. Completion/requeue is compare-and-swap on that token,
so a stale worker cannot complete a newer claim. After a crash, the next elected
dispatcher recovers the claim when the short lease expires; live claims cannot be
stolen by a second process:

- completed target turn -> `DELIVERED`/`FAILED` from its terminal status;
- nonterminal/uncertain agent turn -> reset to a reservation and reattach using
  the same turn/message/bridge request IDs;
- exhausted attempts -> `FAILED` dead letter.

Durable mode is capability-gated to Redis-backed Claude Code, Codex, and OpenClaw
agent harnesses. Local/chat backends are recorded as inspectable `FAILED`
deliveries because their tool side effects cannot be safely reattached after an
ambiguous process failure.

## Lifecycle observability

Every state transition publishes `_type="inter_bot_delivery"` to the sender's
normal unified Redis stream. BawtHub shows deduplicated QUEUED, DELIVERED,
FAILED, and CANCELLED notifications; FAILED is sticky and includes `last_error`.
The PostgreSQL row remains the durable source of truth if Redis/UI delivery is
missed.

REST API:

- `POST /v1/inter-bot-deliveries`
- `GET /v1/inter-bot-deliveries/{delivery_id}`
- `GET /v1/inter-bot-deliveries?sender_bot_id=&target_bot_id=&status=`
- `POST /v1/inter-bot-deliveries/{delivery_id}/cancel`

MCP inspection/control tools:

- `bots_delivery_get(delivery_id)`
- `bots_deliveries_list(...)`
- `bots_delivery_cancel(delivery_id)`
- `agent_context_health(bot_id, user_id?)`
- `agent_context_compact(bot_id, idempotency_key, sender_bot_id?)`
- `agent_context_reset(bot_id, session_policy, reason, user_id?)`

REST context controls:

- `GET /v1/agent-context/health?bot_id=&user=`
- `POST /v1/agent-context/reset`

## Manager/worker orchestration contract

The event-driven project workflow is explicit:

1. Manager (for example Snark) delegates work with default asynchronous
   `bots_send_message`; the durable delivery returns immediately and the manager
   does not block waiting for completion.
2. Worker updates task steps and persists the final READY/BLOCKED receipt in the
   BawtHub task response first, so the result exists even if all notification
   transports are unavailable.
3. Worker sends the callback with default `bots_send_message`, plus `task_id`,
   `message_kind`, and a stable idempotency key such as `TASK-700:READY`.
4. If the manager has an active steer-capable Claude turn, the callback redirects
   that same run. During the bridge-not-ready race it remains queued without
   consuming attempts. If no steer is possible, exactly one safe idle fallback
   turn starts. Use `delivery="when_idle"` only to request a separate turn.
5. The manager reloads task/dependency state, verifies the worker result, and
   releases the next safe task. Polling is fallback diagnostics, not scheduling.

Example worker callback:

```python
bots_send_message(
    target_bot_id="snark",
    sender_bot_id="loopy",
    message="TASK-700 READY. Review the task response and release the next dependency-safe task.",
    task_id="TASK-700",
    project_id="project-id",
    message_kind="READY",
    idempotency_key="TASK-700:READY",
    # For a large independent assignment, choose deliberately:
    # session_policy="reset_retain_history",
)
```

`force=true` is only a compatibility input and never authorizes concurrent agent
turns. It follows the same durable steer-or-safe-fallback contract.

## Verification

```bash
# Unit/regression and pure transport tests
uv run pytest -q tests/test_inter_bot_transport.py tests/test_claude_context_usage.py tests/test_thread_agent_keys.py tests/test_bot_send_timeout.py tests/test_self_fwd.py

# Real PostgreSQL ordering/race/restart/context contract
uv run pytest -q -m integration tests/test_inter_bot_delivery.py tests/test_agent_context.py

# BawtHub lifecycle notifications
cd ../bawthub/frontend
npx tsx --test src/app/chat/unifiedInterBotDeliveryHandler.test.ts
npx tsc --noEmit
```
