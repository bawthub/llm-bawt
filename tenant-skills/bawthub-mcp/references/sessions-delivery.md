# Sessions, delivery, context and persona

## Session versus provider context

A durable conversation thread and a provider/SDK session are different objects.
`sessions_get`/`sessions_list` inspect durable threads; list since accepts ISO
 time or Unix seconds. `sessions_get_active(bot_id, user_id)` reads the active
thread without creating one. `sessions_get_or_create_active` may create one.
`sessions_close` archives without deleting messages. `sessions_rotate` archives
the user's active thread and creates a successor: never use it to silently
prepare background work.

`agent_context_health` reports provider context estimates and supported actions.
`agent_context_reset` only resets an idle provider session, preserving durable
history. session_policy is reset_retain_history or reset_without_history.
Capabilities vary by backend; inspect them first. `agent_context_compact` queues
one Claude /compact turn for idle execution; use a stable idempotency_key.

## Durable bot messaging

`bots_send_message(target_bot_id, message, sender_bot_id=<self>)` defaults to
asynchronous durable delivery and returns delivery_id. It steers a busy capable
Claude turn or starts one idle fallback turn. `delivery="when_idle"` explicitly
requires an idle turn; queue_if_busy=true is the legacy equivalent.

- Use `idempotency_key` per logical handoff, not a new random value on retry.
- `task_id`, `project_id`, `message_kind`, and metadata attach work context.
  Write the task receipt BEFORE sending a READY/BLOCKED callback.
- `wait_for_reply=true` selects bounded synchronous waiting; timeout_seconds
  applies subject to the server ceiling. Async does not require polling.
- Async sends never return the target's reply: its final response stays in its
  own chat. If you need an answer, use `wait_for_reply=true`, or say in the
  message to reply with `bots_send_message(target_bot_id=<you>)`. A recipient of
  an async ask must call back explicitly; its final response does not reach you.
- `force=true` does not authorize concurrent agent turns.
- session_policy: continue, reset_retain_history, reset_without_history.
  For durable delivery the reset waits for safe idle execution. Prefer this
  explicit field over legacy reset_session_before_delivery/retain_history flags.
- fire_and_forget is a compatibility option; prefer the explicit current fields.
- bots_list_available is discovery, not permission to contact protected bots.

### Sender identity and outside agents (TASK-1362)

The sender comes from the signed turn when one exists. A `sender_bot_id` that
contradicts it is rejected, so family bots can omit it. Without a signed turn:

- An explicit `sender_bot_id` is accepted and must still resolve to a real bot.
- An agent outside the family (a Claude Code or Codex session with no bot of its
  own) may omit `sender_bot_id` on `bots_send_message` only. The one bot tagged
  both `external` and `inactive` then sends for it. Zero such bots is a setup
  error, several must be named, and an unreadable tag policy fails closed.
- A capability header that is present but does not open never gets that
  default; name yourself or fix the header.
- `inactive` makes the bot unreachable: sends to it are refused
  (`target_inactive`), and its deliveries carry no "reply with
  bots_send_message" note. Use `wait_for_reply=true` to get an answer back.
- Set up: create an ordinary bot profile for outside agents and give it both
  tags. Nothing in code creates or names it.

### Who is busy: bots_list_available state

`bots_list_available(tag=None, include_state=true)` lists visible targets.
Private and inactive bots never appear. `tag` filters (e.g. `reviewer`); an
unknown tag is an error, not an empty list. Each bot's `state` holds:

- `turn` {kind, turn_id, started_at}: kind is idle / reserved / executing /
  ambiguous, from the same classifier that admits a send, so it is what a send
  would see right now. Your own row shows your current turn as executing.
- `last_turn` {ended_at, status ok/aborted/error, end_reason}.
- `queued_deliveries` {count, oldest_at}: non-terminal inbound deliveries.
- `gated` + `gate` (quiesce/restart hold), `pending_approvals` count,
  `session` {id, started_at}.

It is an advisory, unlocked snapshot: use it to pick who to ask, not as a
lock. A source that couldn't be read is listed in `state.unavailable` with null
fields; treat that bot as unknown, never as idle. Task counts are not here:
use `tasks_list`. `include_state=false` returns profile fields only.

Inspect one receipt using bots_delivery_get; bots_deliveries_list is bounded
status inspection returning compact rows (no metadata; clipped preview and
last_error), newest first, default limit 10, max 50. Filter instead of paging:
`bot_id` matches sender OR target (one call, no overlap), `active_only=true`
means QUEUED/STEERING/DISPATCHING, `status` accepts a comma list, plus
`task_id` and `since_minutes`. Users' own chat queue rows and any
self-addressed row (sender == target, e.g. self_model continuations) are
excluded unless `include_chat=true`. "Who is still running?" is
`bots_deliveries_list(bot_id=<self>, active_only=true)`. bots_delivery_cancel succeeds only while QUEUED; it cannot
abort an already running turn. Accepted/delivered and completed are different.
Event-driven callbacks are primary; status queries are verification/fallback.
This delivery API is NOT future-time scheduling. It does not accept a run date.

## Durable wake-ups (schedule_wakeup)

For any future callback ("check X at 07:17", "re-poll in 30 min", a recurring
nudge) use `schedule_wakeup`, never harness timers (Claude Code `CronCreate`,
`/loop`, `ScheduleWakeup`): those live in one SDK session and die on restart,
`/new` and compaction. `schedule_wakeup` writes a prompt schedule (visible on
the Schedules page) owned by your signed user.

- `create`: `message` plus exactly one of `at` (ISO-8601 with offset),
  `in_minutes` or `cron` (+ `timezone`, IANA). `target_bot_id` defaults to
  you; another bot must be known and not private. `task_id` tags it and the
  delivered message starts `[Scheduled wake-up from <you>, TASK-N]`. Runs in a
  fresh context without memory augment/extract; a run missed by up to 7
  days fires once on recovery. Pass a stable `idempotency_key`: a retry returns
  the original (`duplicate: true`) instead of double-booking.
- `list`: active wake-ups you created or that target you.
- `cancel(schedule_id)`: same scope; never the user's hand-made schedules.
  A fallback wake-up guards one step: cancel it when that step completes, the
  task leaves your hands, or you move the card to REVIEW/COMPLETED. Stale
  fallbacks land in the user's view and in whatever session you're in now.
- Don't set one for work that already reports back: ops jobs and gates deliver
  a durable receipt to the requester, and a replying bot's message wakes you.
  Use wake-ups only where nothing announces completion (your own long runs,
  GitHub CI, timed measurements, recurring checks).

## Self tools

`self_tail(bot_id=<self>, count=...)` loads recent raw bubbles without summaries
or writes. Absorb the returned context; do not paste the transcript back unless
asked. A short acknowledgment of context/topics is enough.

`self_fwd` forwards YOUR raw tail to another allowed bot. sender_bot_id must be
explicit and cannot be "default"; target_bot_id is the recipient. note adds
cover text, count bounds the excerpt. Async by default; wait_for_reply blocks.
Do not forward private context just because a target is listed.

`self_system_prompt(bot_id=<self>, action="view")` reads the durable persona,
not the assembled harness system message. action="edit" and new_prompt fully
replace it, returning old_prompt for recovery. This changes the DB-backed persona
for subsequent turns, not current context. Keep the cache prefix byte-stable
except for deliberate authorized edits.

`self_model(bot_id=<self>)` shows your current endpoint/model/harness and the
compatible endpoints. Despite the name, it also works on **another bot**
(`bot_id=<slug>`, caller is recorded): use it to move a sibling to a different
model instead of hand-PATCHing profiles or querying the DB. Mira is refused
unless she is the trusted caller. Endpoint access-path labels are unreliable for
billing: native claude-code turns authenticate with the bridge's OAuth
subscription token even on endpoints labelled `anthropic-api`. action="switch" with endpoint_id persists the change
through the same profile PATCH as the chat model picker: the server validates
it and moves claude-code <-> claude-proxy as needed. It applies from your NEXT
turn; this turn keeps its model, so never claim you are already on it. To keep
working on the new model, pass continue_prompt in the same call: it queues one
durable when_idle delivery that starts after this turn ends (never steer — a
steered message lands in the old-model turn). Make the prompt self-contained,
because it is the next turn's only instruction. Its delivery records
response_model, which proves the switch. Follow-ups always run in the target's
active default-user chat. continue_prompt is therefore refused, before
anything switches, when there is no trusted turn context, for a non-default
user, or when you switch yourself from a task/archived thread. In those cases,
switch without it and tell the user. The switch is not rolled back if queueing
itself fails; check result.continue.queued. Codex sessions are model-bound, so
a Codex switch starts a fresh provider session. Only switch when the user asked
for it, and say what you switched to (cost and context window differ by
endpoint). Mira is refused unless she is the caller. Switching another bot
sends the continuation as from you.

## Editing another bot's profile or tags

`bots_profile_patch(bot_id, changes, reason, expected=None)` is the supported
way for a bot to change a sibling's tags or profile fields. Use it instead of
curl PATCHes to `/v1/bots/{slug}/profile` (TASK-1521; live after the app
restart that ships it).

- You are recorded from your signed turn (`updated_by = bot:<you>`), with
  `reason` on a `bot_profile_changes` audit row. There is no sender argument.
- `changes` maps field to new value. `tags` is a full-set replace (send the
  whole list, not a delta). Editable: tags, name, description, color, avatar,
  default_voice, system_prompt, prompt_override_id, requires_memory,
  voice_optimized, tts_mode, uses_tools, uses_search, uses_home_assistant.
  Other runtime fields are refused; `harness` is derived, never set.
- Switching another bot's model (TASK-1818): `changes={"endpoint_id": <catalog
  id>}`, optionally with the `default_model` that endpoint must serve. Only a
  caller holding a role with the capability named by the global setting
  `bots_model_switch_capability` may do it; otherwise `capability_required`.
  Unset means no bot can; a signed-in person still switches in the UI. Roles
  carrying that capability can't be added or removed with this tool
  (`tag_locked`). An endpoint outside the target's catalog is
  `unknown_endpoint`; no null endpoint. Your own model is `self_model`.
- Private and unknown bots both answer `access_denied`. The `private` tag can't
  be added or removed here; that is Nick's UI action. You can't change your own
  tags (`self_tag_edit_refused`); ask another bot or Nick. Your other fields are
  editable.
- Every write is a compare-and-set. Pass `expected` with the values you read
  when you are acting on an earlier read; fields you leave out are pinned to
  what the tool reads just before writing. `conflict` (409) means nothing was
  saved: re-read and decide again, never retry blindly. `no_change` means every
  field already had the value.
- The result has `changed{field:{before,after}}` and `rollback{changes,expected}`.
  To undo, pass `rollback` straight back as `changes` and `expected`. A `null`
  before-value (e.g. no color) restores as `null`.
- A `prompt_override_id` or `system_prompt` change on an agent bot lands at
  that bot's next session boundary; pair a persona switch with a session reset
  (llm-bawt reference/personas.md).

For architecture and backend capability details, read the llm-bawt skill's
reference/inter-bot-deliveries.md and reference/history-context-flow.md.
