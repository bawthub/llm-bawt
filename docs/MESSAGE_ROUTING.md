# Opt-in message routing (TASK-1202)

Routing chooses a bot **before** creating a durable delivery. It is default-off per `(owner_user_id, bot_id)` and never opts a bot in automatically. Protected bots cannot be enabled and are excluded before session or message activity is queried. No service restart, enrollment or fleet migration is implied by schema creation.

## API and scope

- `GET /v1/message-routing/config?user_id=<owner>` (alias: `/candidates`): saved eligibility plus current availability and exclusions.
- `PUT /v1/message-routing/config` with `{ "user_id": "...", "bot_id": "...", "enabled": true }`: explicitly change one eligible bot's membership.
- `POST /v1/message-routing/preview`: `user_id`, optional `candidate_bot_ids` (null = all opted-in; `[]` = no bots), `model_endpoint_id`, `session_policy` (`continue`, `reset_retain_history`, `reset_without_history`) and `scope` (`personal` or `schedule`). Does not reserve or send.
- `POST /v1/message-routing/route`: preview fields with `message`, required `idempotency_key`, optional `task_id` and `project_id`; queues exactly one personal turn. Scheduled occurrences use the prompt outbox, not this POST endpoint. Returns the bound route/delivery IDs, selected bot, endpoint ID and canonical reference, reason, policy and status. Queue admission is not a completed answer.
- `bots_route_message`: the MCP adapter derives the caller bot and owner from a signed current-turn capability. An unsigned LAN API request is user-authored under the existing LAN trust boundary. Signed callers cannot override their owner; receipts and stable keys are scoped to owner **and actor**. Reuse the same key and identical intent to retrieve the original selection; changed intent gets 409. Do not supply bot/user identity as tool arguments.

Ranking uses timestamp metadata only: newest non-recalled, original user/bot-authored user or bot-authored assistant message across the owner's sessions for each candidate. Never-messaged first, then oldest last message; ties use fewer queued jobs then slug. Idle candidates precede occupied candidates. Occupancy includes existing durable deliveries, reservations/executing turns, and prepared automatic occurrences. An unreadable activity state is excluded, not treated as never-messaged. Preview is advisory; route admission selects and rechecks under owner routing-pool and target locks.

Each durable receipt also records ranking version 1, selected activity timestamp, availability, pending count, and compatible candidate IDs for audit. The chosen model endpoint is bound request-locally. `continue` snapshots an owner-owned active session and its persisted provider resume identity at admission; it rejects an unsupported provider/model resume and refuses a changed session or resume binding before acceptance. When no active thread exists, admission creates an owned active row transactionally. Reset policies use the shared new-session lifecycle and intentionally do **not** bind an old thread. All routed deliveries set `prefer_steer=false`: they wait for an idle target, never interrupt or run concurrent turns. Revoke membership before acceptance to cancel without retargeting; already accepted work retains its original receipt.

## Scheduled automatic targeting

The existing `/v1/prompt-schedules?user=<owner>` APIs retain `user`, not `user_id`. Set `target_mode="auto"`, `bot_id=null`, and optionally `candidate_bot_ids`; null means the whole opted-in pool, `[]` matches nobody. CREATE/PATCH/RUN-NOW validate a compatible opted-in candidate and requested canonical model. The worker selects once per occurrence using the same ranking, persists the selected bot on the occurrence snapshot, then executes only in a born-archived automation thread. It never resets or reuses the owner's active personal thread. Reused automation threads are keyed to schedule, selected bot, owner, and selected model; clearing context creates a fresh occurrence thread. Runs report `selected_bot_id` from their own snapshot, not from the schedule's later configuration. Revocation before claim cancels that occurrence's unaccepted delivery.

Lock order: personal route takes owner routing-pool → target advisory → delivery identity; automatic occurrence preparation takes owner routing-pool → schedule job and subsequently its outbox takes target → schedule job → delivery. Claim and consent changes take target before delivery. Never hold a delivery lock while acquiring a target or pool lock.

## Offline verification

Run on echo, where the private PostgreSQL Docker fixture can start; do not run routing tests against the operational database or actual bots:

```bash
make test-target NODES=tests/test_message_routing.py
make test-target NODES='tests/test_prompt_delivery.py tests/test_prompt_schedules_api.py'
```

No app restart, live dispatch, release, commit, or fleet opt-in is part of these checks.
