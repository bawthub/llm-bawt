# Durable inter-agent callbacks + context control (TASK-710/711)

> Part of the llm-bawt skill. Read before touching `bots_send_message` delivery semantics, session_policy resets, or agent context health/compaction. Full contracts: `docs/INTER_BOT_COMMUNICATION.md`, `docs/AGENT_CONTEXT_CONTROL.md` in the repo.

`bots_send_message` has three contracts:

- default asynchronous: PostgreSQL-backed durable FIFO delivery. It steers an
  active Claude Code turn in place; otherwise it starts exactly one safe idle
  fallback turn. Returns a stable `delivery_id` immediately.
- `delivery="when_idle"` (or `queue_if_busy=true`): same durable path but
  deliberately skips steering and waits for a separate idle turn.
- `wait_for_reply=true` (or `fire_and_forget=false`): durable `when_idle`
  enqueue, then a bounded wait for that exact turn's reply. Unkeyed + busy
  target = rejected with no side effect; keyed = re-attaches to the existing
  delivery. Timeout returns `in_flight=true` — retry with the SAME key.
  `force=true` never authorizes concurrent agent turns and async force
  follows the default durable steer-or-idle path.
- TASK-1015: one resolver (`inter_bot_send_mode.py`) rejects conflicting flags
  before any side effect; the idempotency key spans both modes; waited task
  handoffs require a key. Prefer default async for task handoffs.

Durable state is owned by llm-bawt in `inter_bot_deliveries`. A PostgreSQL
per-target advisory lock atomically chooses `STEERING`, safe `DISPATCHING`, or
continued `QUEUED`. Stable `req_delivery_*`, user-message, and turn IDs make
retries/restarts reattach to the same logical operation.

**Steer generations (TASK-1251):** the bridge caches its first answer per steer
request ID, so a rejected ID can never succeed later. The steer ID is
`steer_delivery_<id>` for generation 0 and `steer_delivery_<id>_g<k>` after,
with `k` in `metadata_json.steer_generation`
(`inter_bot_dispatcher_steer.SteerGeneration`). `k` advances only on the
requeue that follows a not-ready or definitive steer rejection, in the same
UPDATE (`requeue(metadata_merge=...)`). Accepted, unpersisted-accept, timeout
and crash replays keep the same ID, so a steer is never delivered twice. Rows
from before TASK-1251 sit at g0 and self-heal: one cached-rejection cycle, then
g1. An open target turn (`reserved|pending|streaming`, no `ended_at`) whose run
hasn't registered yet (`no_active_run`, `maintenance_active`, `Turn not found`)
keeps its target and FIFO place: it retries on the next generation after
1/2/4/8/15 s backoff, refunds the attempt, and counts in
`metadata_json.steer_not_ready` per target. After 20 tries, or once the target
has ended, it falls back on the next generation and waits for that exact turn
to end before one safe idle fallback. App-side not-ready waits also do not
consume attempts. An accepted steer can never be converted into a fresh turn.
`/v1/chat/steer` sends a caller-supplied `steer_request_id` once; only composer
steers (no ID) loop. Redis recovery state is retained seven days, after which
ambiguous accepted work dead-letters rather than duplicating.

**Steer priority (TASK-934):** inter-bot steers use CLI priority `next`. They
never interrupt: running tools finish, and the message is folded into the next
model call or run as a follow-up turn. User `/v1/chat/steer` defaults to `now`,
which interrupts and kills in-flight tools. The bridge tracks CLI
`command_lifecycle` frames via `ClaudeActiveRun.messages()` and holds a
ResultMessage while an injected command is unstarted. Details:
repo `docs/INTER_BOT_COMMUNICATION.md` § "Steer priority".

APIs: `POST|GET /v1/inter-bot-deliveries`, GET by ID, POST `/{id}/cancel`; MCP:
`bots_delivery_get`, `bots_deliveries_list`, `bots_delivery_cancel`.

**User Stop holds the queue (TASK-1327).** A user Stop (`/v1/chat/abort` with
`source` `chat_stop` or `bot_list_stop`) opens or extends a per-bot hold gate
(`owner_id=user_stop:<bot>`, global setting `user_stop_queue_hold_seconds`,
default 600, 0 = off) so the bot's queued deliveries wait instead of starting
the next turn. The user's own chat still runs past it. `GET
/v1/bots/{bot}/queue-hold` shows it; `POST .../resume` releases it; `POST
.../cancel-all` cancels the held rows (reason `Cancelled by user (queue hold)`)
and releases it. The hold also expires on its own. After that, the dispatcher
can claim the held rows on a later pass, and every other gate still applies.
Expiry does not run them itself. When a bot-to-bot delivery is cancelled,
whoever cancelled it, the sender can get one queued `DELIVERY_CANCELLED`
notice. That happens only if all of these hold:
- the setting `delivery_cancel_notice_mode` is `notify`;
- the row is eligible: not user-authored, not a scheduled prompt, not itself a
  notice, and no private bot (`private` tag, legacy
  `include_in_global_search=false`, or no profile) on either end;
- the sender is an agent bot with a user scope.
Like any delivery, the notice itself can be held, cancelled or fail. Full rules
are in background-jobs.md.

TASK-711 adds `session_policy`: `continue` (default),
`reset_retain_history`, or `reset_without_history`. Reset deliveries wait for an
active turn to end, then rotate the durable thread and reserve delivery in one
transaction; old sessions/messages are never deleted. Duplicate idempotency keys
perform at most one reset. Claude Code/proxy OpenAI supports retained + clean
reset; native Codex supports clean reset only.

Inspect with `agent_context_health`; queue idempotent Claude compaction with
`agent_context_compact`; reset an idle bot with `agent_context_reset`. Claude
native resident usage comes from
`ClaudeSDKClient.get_context_usage()`; final-iteration usage is the fallback,
never cumulative cache-read billing. Configured window for the current
OpenAI-backed fleet is 372k; exact upstream acceptance reserve remains
unmeasured. Full contract: `docs/AGENT_CONTEXT_CONTROL.md`.

Manager/worker invariant: persist the READY/BLOCKED task receipt FIRST, then call
default `bots_send_message` with `task_id`, `message_kind`, and a stable
idempotency key (e.g. `TASK-700:READY`). Inspect context before large independent
work and choose reset policy deliberately. Use `when_idle` only when a separate
turn is required. Never poll as the primary scheduler. See
`docs/INTER_BOT_COMMUNICATION.md`.

## Summary formerly in the parent SKILL.md (TASK-1640)

### Inter-bot deliveries + context control (TASK-710/711)

`bots_send_message`: default = durable PostgreSQL-backed FIFO that steers an active Claude Code turn or starts exactly one idle fallback turn; `delivery="when_idle"` skips steering; `wait_for_reply=true` is bounded sync. `session_policy` ∈ `continue|reset_retain_history|reset_without_history`. Inspect with `agent_context_health` / `agent_context_compact` / `agent_context_reset`. Manager/worker invariant: persist the task receipt FIRST, then send with a stable idempotency key; never poll as primary scheduler. Contracts: **`reference/inter-bot-deliveries.md`** + repo `docs/INTER_BOT_COMMUNICATION.md`, `docs/AGENT_CONTEXT_CONTROL.md`.
