# TASK-1431 design r3: one send path (fan-out, modes, control lane), quiesce pump folded in

Author: Rook. Base: llm-bawt master 4df476c (re-checked at 21858b1; no touched files changed). Status: design,
r2 after Byte DESIGN r1 REQUEST_CHANGES (review cmvn1wbfe00ee0mlfibt7zsy0). No code yet.

r2 changes: B1 → §3 rewritten (absorbing vs successor rows, claim-time member resolution, link carry-forward,
reference-counted release). S1 → §2 settle reasons from turn evidence. S2 → pin immutable in metadata (§2.2).
S3 → §2.1 item 3. S4 → §4 step 3 user. S5 → §2.2 pinned-row check. S6 → §2.3 grant checks. S7 → §6 rollback.
S8 → §3 duplicate check. N1 → §2.1 item 4. N2 → §4 step 3.
r3 (Byte DESIGN r2 APPROVE_WITH_CHANGES, cmvn20k8j00eq0mlf3rdonwzi): R1 → §3 dead-link re-link + FOR SHARE carry,
§4 selector, test (j). Final reviewed design; the S3 implementation review checks R1 and (j).
Design record. When S1–S3 land, fold the contract into `docs/INTER_BOT_COMMUNICATION.md` and remove this file.

## 0. Facts this design rests on (read at 4df476c)

- `inter_bot_delivery_claim.claim_next` is the one target-locked claim. It already has one bypass precedent:
  under a live `user_stop:<target>` gate, `inter_bot_preclaim.load_user_bypass` lets the user's own chat row run
  ahead of the held FIFO (TASK-1327) without reordering anything else.
- `gate_blocks` at the claim leaves a held head QUEUED with no attempt spent; `check_steer` refuses every steer
  under a live gate unless it carries a `ControlGrant` that `_admit_grant` re-validates under `lock_target`.
- `service/quiesce_control.py` (434 lines) rebuilds claim / lease / CAS settle / backoff / attempts /
  steer-generation on `gate_members`, and imports `DEFINITIVE_NOT_ACCEPTED` from the dispatcher. The delivery
  dispatcher already owns all of that (TASK-1251 generations, TASK-1300 fence, heartbeat lease, recovery).
- Drain is evidence-based (`turn_gates._drain_status` / `execution_exit_drains`). It never reads `control_state`.
  `control_state` is read only by `quiesce_status` (operator view), `cancel_pending_controls` (release) and the pump.
- The quiesce path has never run live: `ops_run restart-app` with quiesce on is refused by the worker fence
  (TASK-1430). Nothing live depends on the pump or its columns.
- `/v1/*` has no service auth (TASK-1420 is QUEUED). Any field on the HTTP model is reachable by `curl` from a bot.
- `inter_bot_delivery.py` is 997 lines and `inter_bot_dispatcher.py` 977: monolith gate. New logic goes in new
  modules; those two files get call-site edits only.

## 1. Shape

One request type, `SendRequest`, and one service function, `DeliverySender.submit(request, principal)`, in a new
module `llm_bawt/inter_bot_send.py`. Every producer calls it:

| Caller | Surface | Principal |
|---|---|---|
| Bots | `bots_send_message` (MCP) → `POST /v1/inter-bot-deliveries[/batch]` | `bot` (signed turn capability) |
| Nick | BawtHub "Message bots" → same HTTP routes | `user` (BawtHub-authored) |
| System (quiesce, future ops) | in-process call, no HTTP | `system` |

`create_delivery`'s current per-target body (validation, privileged-field refusal, author checks, durable
capability, session-policy normalization, enqueue, FAILED-row receipts) is extracted unchanged into
`DeliverySender._admit_one(target, request, principal)`. The existing single-target route becomes a one-target
call to it, byte-compatible.

### 1.1 Targets

`targets` is a selector, exactly one of:

```json
{"bots": ["byte", "nova"]}      // explicit list (a bare list is accepted as shorthand)
{"all_agents": true}
{"backends": ["claude-code"]}   // agent_backend
{"tags": ["reviewers"]}         // bot_tags role tags
```

Optional `exclude: [slug...]`, `include_self: false` (default).

- Expansion reuses `GateScope.covers` for all_agents/backends/bots (same agent-only rule), plus a tag predicate
  from the `bot_tags` resolver. A `bot_type` selector is dropped: only agent bots take durable delivery, so
  `all_agents` is the bot-type selector.
- Server-side exclusions, applied after expansion, for every principal: `private` (protected bots) and `inactive`
  bots never come out of a selector. An explicit list that names one gets a per-target refusal. For bot callers a
  private target reads as `unknown_target`, so its existence isn't revealed. The sender's own bot is excluded
  unless `include_self=true`.
- Expansion is read once, at submit, and recorded (`metadata_json.batch.targets`). A bot added later isn't sent to.

### 1.2 Batch semantics

- One normal delivery per resolved target, each in its own target-locked transaction, in sorted target order.
  Per-target, not all-or-nothing: a deny-gate refusal or an unknown target on one doesn't abort the rest.
- `batch_id` = uuid5 of (sender, idempotency_key) when keyed, else uuid4. It's stored in `metadata_json.batch_id`.
  The per-target idempotency key is the caller's key unchanged: the unique index is (sender, target, key), so one
  key already yields one row per target. Retrying a keyed batch returns `duplicate=true` per target and never
  creates a second row or turn.
- Response: `{batch_id, mode, dispatched, deliveries: [<today's receipt>...], refused: [{target, error}...]}`.

### 1.3 Fan-out limits

- Bot principals: multi-target sends require an `idempotency_key`. `session_policy` must be `continue` (no
  mass resets). `reply_mode=waited` is refused (no fan-out waits). Rate: at most 3 new batches per sender per
  10 minutes, counted in SQL on `metadata_json.batch_id` rows (duplicates don't count). The 4th returns 429
  `fanout_rate_limited` before any side effect.
- User and system principals: no rate limit. User may reset per target as today.
- Hard cap: 64 resolved targets per batch (above today's bot count; a guard, not a quota).

## 2. Modes and the control lane (parameters, one code path)

`mode` ∈ `steer_or_idle` (default) | `when_idle` | `steer_only`. Maps to today's `prefer_steer` plus a new
`steer_only` flag in `payload_json`. No schema change for mode.

- `steer_only`: steer the target's executing Claude Code turn (or pinned run, 2.2). Never reserves or starts a
  turn: no branch of `claim_next` or `_steer_failure_action` may write a reservation or fall back to an idle
  turn for a steer_only row.
- **Settle reasons come from turn evidence, never from the steer error alone (r2 S1).** When a steer_only row
  can't be steered (idle at claim, NOT_READY_BOUND, a definitive pre-accept rejection such as
  `no_active_run` / `active_run_mismatch`):
  - unpinned: the target has no executing turn at claim → settle `not_needed:no_active_run`;
  - pinned, and the pinned turn's row is ended or gone → settle `not_needed:run_ended`;
  - pinned, and the pinned turn is still open → requeue with the dispatcher's normal backoff, spending an
    attempt. At `max_attempts` → `FAILED` with `last_error='unconfirmed:<last reason>'`. That's the pump's
    rule today: retry until the turn ends, unconfirmed after the cap.
  - non-steerable backend → settle `not_needed:unsupported_backend` at claim.
- **Terminal state for "nothing to steer":** `CANCELLED` with `delivery_mode='skipped'` and a machine-readable
  `last_error='not_needed:<reason>'`. No status CHECK change and no BawtHub lifecycle change (Byte r1 §9 agreed).

### 2.1 How the control lane bypasses a held FIFO head without reordering normal traffic

A new column, `inter_bot_deliveries.lane VARCHAR(16) NOT NULL DEFAULT 'normal'`, added through the store's
existing `ADD COLUMN IF NOT EXISTS` path, plus a partial index `(target_bot_id, ordinal) WHERE lane='control' AND
status='QUEUED'`.

In `claim_next`, under the same `lock_target(target)` and before the user-Stop bypass:

1. `load_control_head(conn, target)` (new, in `inter_bot_control_lane.py`) returns the oldest ready QUEUED
   `lane='control'` row, locked, or none. Unlike `load_user_bypass` it doesn't wait for other claimed rows to
   finish: a control steer reserves nothing and only adds a `next`-priority message to a run that is already
   executing.
2. For a control head, the gate check is `admit_control(conn, row)` instead of `gate_blocks`. It resolves the
   authorizing member **at claim time** (§3: any live member whose `control_delivery_id` is this row and whose
   pinned run equals the row's pin). No stored gate/member ref is authority. If none exists, it settles
   `CANCELLED` / `not_needed:control_revoked`. It never falls through to normal handling.
3. Normal heads: `load_claim_head` and `load_user_bypass` gain `AND lane='normal'` on the candidate **and** on
   `load_user_bypass`'s NOT EXISTS fence over STEERING/DISPATCHING rows (r2 S3), so a STEERING control row never
   blocks the user-Stop bypass. Control rows therefore never head the normal FIFO, never fence it, and are never
   counted by the TASK-1067 accepted-dispatch rule. A normal row's position relative to other normal rows is
   unchanged, because both queries still order by `ordinal` over the same set minus control rows.
4. One control row per target per claim. When `claim_next` settles a control head at claim and returns None, it
   sets the dispatcher wake (r2 N1) so the next row doesn't wait for the 15 s sweep.

Normal steers issued while the gate is live are still refused by `check_steer` (no grant). So the run gets only
the control message, and held normal traffic resumes in its original order after release.

### 2.2 Pinned run (`target_run`)

`target_run = {turn_id, agent_request_id, backend, session_key}` is stored **immutably** in
`metadata_json.control.target_run`, written once at enqueue (r2 S2). It is never derived from `target_turn_id`,
which `requeue_claim(clear_delivery_mode=True)` NULLs (inter_bot_delivery.py:746). `target_turn_id` stays the
dispatcher's steer bookkeeping only. No requeue/settle path writes `metadata_json.control`.

- Claim judges **only the pinned turn's own `turn_logs` row**, as the pump does (r2 S5), never bot-level
  `classify_target`, which fails closed on ambiguous or multiple open rows.
  - Pinned row open, with request id and backend/session matching the pin → steer.
  - Open but not yet executing → leave QUEUED with the not-ready backoff.
  - Ended or gone → settle `not_needed:run_ended`.
  - A different turn executing doesn't count: the pin is the only address.
- The RPC address is `ControlTarget` from `control_target(member)` for the member resolved at claim (§3). A pin
  never falls back to "the bot's current turn".
- Invariant kept from Byte r1 claim 5: no control-lane enqueue, claim or settle writes or locks a `turn_logs`
  row. Turn rows are read only.

### 2.3 Authorization (one check, not a second path)

`lane='control'` and `target_run` are `SendRequest` fields that `DeliverySender.submit` admits only for
`principal.kind == 'system'`. Bot and user principals get 403 `control_lane_forbidden` before any side effect.
A control send must also be `mode=steer_only` and carry `target_run` plus its `run_key` (§3), else 422. It
carries no gate/member reference: authority is resolved at claim and at grant (r2 B1).

Until TASK-1420 adds service auth, **no HTTP field can carry `lane` or `target_run`**. The pydantic HTTP models
don't declare them (`extra='forbid'` on the batch model, so they 422). The only system principal is an
in-process caller. This is the same fence as `ControlGrant` today ("the public steer route has no field that can
carry one"). Hold-bypass for Nick from the UI therefore waits for TASK-1420's user principal. The UI ships with
target and mode pickers only.

`ControlGrant` is re-keyed from the member's claim token to the delivery's claim:
`ControlGrant(bot_id, turn_id, delivery_id, claim_token)`. There's no stored gate id: the gate is whichever live
member links the delivery at the moment of admission (§3). `_admit_grant`, under `lock_target`, admits only when
all of these hold (r2 S6):
- the delivery row exists with `lane='control'`, `status='STEERING'`, `claim_token` = the grant's, and
  `lease_expires_at > clock_timestamp()`;
- its `target_bot_id` = grant bot, and `metadata_json.control.target_run.turn_id` = grant turn;
- at least one member passes `LIVE_MEMBER_SQL` with `m.control_delivery_id = delivery_id`, `m.bot_id` = grant
  bot, `m.turn_id` = grant turn, and a pinned request equal to the delivery's pin.

The `ControlTarget` returned is the one from that member's `control_target`.

The dispatcher's `_steer` passes `control=grant` for control rows. That's its only branch on lane.

## 3. Exactly once per pinned run, across jobs and generations (r2, rewritten for B1)

The contract is the pump's: **a run that has been asked (or may have been) is never asked again; a run that
hasn't been asked and still needs asking always gets asked.** Delivery rows are per run. Authority and lifecycle
are per member, so the two are joined only through `gate_members.control_delivery_id`, never through a ref
stored on the delivery.

**Run key.** `run_key = control_request_id(turn_id, agent_request_id)` = `steer_quiesce_<turn>_<request>`, stored in
`metadata_json.control.run_key`, with a partial index `(target_bot_id, (metadata_json->'control'->>'run_key'))
WHERE lane='control'`. The idempotency key for successor k is `run_key` (k=0) or `run_key:r<k>` (k≥1).

**Classifying a run's existing rows.** A row is:
- *absorbing* if it is live (`QUEUED`/`STEERING`), or asked: `DELIVERED`, or `transport_accepted_at IS NOT
  NULL`, or a TASK-1300 fence marker for its current generation (`steer_started`, `steer_uncertain`). An uncertain
  steer counts as asked, so it is never re-sent; the pump's `unconfirmed` rule is the same. A DELIVERED row the
  fence later reopens on proven rejection is QUEUED again, i.e. live, and still absorbs.
- *dead* if it is terminal (`CANCELLED`/`FAILED`) with none of the asked markers. Every path that clears
  `transport_accepted_at` (requeue, fence reopen) does so only on proven non-execution, so "terminal and not
  asked" means provably never ran.

**Link step (producer, under `lock_target(bot)`, which serializes every control enqueue for that bot).** For a
member that needs asking:
1. Read that run's rows by `run_key`, newest successor first.
2. If an absorbing row exists, link the member to it. Its outcome is the member's, via §5.
3. Otherwise (none, or only dead rows), submit successor `k = max(k)+1`, or 0 if none, and link.
4. A `duplicate=true` result (race lost on the unique index) is linked only after checking that the returned row
   is `lane='control'`, its `run_key` and `target_run` equal the member's, and the row is absorbing (r2 S8).
   Otherwise the member goes `unconfirmed` with `control_error='control_key_collision'` and is not linked.

So the G1→G2 rebuild path Byte traced is closed: if D was settled `control_revoked` (dead) before G2's member
linked, G2's member gets successor `:r1` and the run is asked. If D is still live, G2's member links to D (see
carry-forward below), and D's claim resolves its authority from G2's member.

**Claim-time authority.** `admit_control` and `_admit_grant` (§2.1, §2.3) look for *any* live member with
`control_delivery_id = D` and the matching pin. A job's gate lapsing, or one of two jobs releasing, therefore
never strands D while another live member still needs it. D is settled `control_revoked` only when no live
member links it, and then it is dead, so the next member that needs that run gets a successor.

**Carry-forward keeps the link.** `turn_gates.carry_forward` extends its existing ack-carry UPDATE to copy
`prior.control_delivery_id` onto the fresh row for the same `turn_id`, together with the matching
`control_state='steering'`, when the fresh row has no link yet, and only while D is absorbing, read `FOR SHARE`
(r2 R1b). A rebuilt generation inherits its predecessor's in-flight ask instead of racing it. If the copy loses
(fresh row already linked, or the predecessor's link was dead), the link step above still applies.

**A link to a dead row is re-linked (r2 R1a, required).** `carry_forward` runs inside `activate_hold`, which
doesn't hold `lock_target`. So a claim can still settle D `control_revoked` (no live member: G1 lapsed, G2 not yet
committed) while the carry copies D's link onto G2's member, and both commit. The producer's selector therefore
covers live members with no link **and** live members whose linked row is dead (§3 classification). Under
`lock_target(bot)` it re-runs the link step for them, which yields an absorbing row or a successor. Nothing relies on
the carry being race-free; the FOR SHARE read only narrows the window.

**Release is reference-counted.** `cancel_pending_controls(job)` sets the job's members `cancelled`. For each
distinct linked delivery it then CASes `QUEUED → CANCELLED` (`not_needed:control_released`) only if no other live
member links it. All of this runs under `lock_target(bot)`, so it is serialized against the link step and claim.
A `STEERING` row in flight is left to settle, as today.

**Steer identity.** One delivery row has one steer RPC identity sequence: `steer_delivery_<id>` with TASK-1251
generations. The generation advances only on a definitive pre-accept rejection and freezes once accepted.
TASK-1300's proven-non-execution reopen applies unchanged. A successor row is a new delivery only after its
predecessor is provably un-accepted, so there is never a second steer of an accepted ask. The timeline message id
is the delivery's `user_message_id`.

Required PG tests (S2/S3 slices):
- (e) a lapsed-gate rebuild with D still QUEUED: G2 member links D, D steers under G2's authority, one ask;
- (f) a rebuild after D is settled `control_revoked`: successor `:r1` is created and steered, one ask;
- (g) two jobs on one run, job A releases: D survives for job B and steers once; then B releases: D cancelled;
- (h) D accepted, then a new job on the same run: links D, no second steer;
- (i) a duplicate-key collision with a mismatched row: not linked, member `unconfirmed`;
- (j) the R1 race: carry_forward links G2's member to D while a concurrent claim settles D `control_revoked`, both
  commit; the next producer pass re-links that member to successor `:r1`, which steers once.

## 4. The pump becomes a caller

`service/quiesce_control.py` is replaced by `QuiesceControlProducer`, about 100 lines, same file name. It has no
claim, lease, retry, backoff, attempts or CAS.

Per live member with its run pinned and either (`control_state='pending'` and `control_delivery_id IS NULL`) or
(`control_state='steering'` and its linked row dead, per §3; r2 R1a), in one
transaction:
1. `lock_target(bot)`, then the member `FOR UPDATE SKIP LOCKED`, re-validated by `LIVE_MEMBER_SQL`. Turn rows
   are read only, never locked, preserving the Caid df30f61 rule against bind deadlock.
2. Member-local decisions stay where they are: revoked, turn gone or ended → `not_needed`; caller's own turn →
   `self`; non-steerable pinned backend → `unsupported`; unaddressable → `unconfirmed`; not pinned → skip,
   retried next pass.
3. Otherwise run the §3 link step. A submit is `DeliverySender.submit(... principal=system, lane=control,
   mode=steer_only, target_run=pinned, run_key, idempotency_key=<run_key or successor>,
   message=resolve(ops.quiesce_request, bot), priority next, author system, user=<pinned turn's
   turn_user_id> (r2 S4), connection=conn)`. Then `UPDATE gate_members SET control_delivery_id=:d,
   control_state='steering', control_changed_at=now()`. The payload `user` must be the pinned turn's user,
   because `steer_active_turn` refuses an ownership mismatch, and the producer never falls back to a literal. A
   pinned turn without a user goes `unconfirmed` (`control_error='turn_user_missing'`). The enqueue's
   `lock_target` is re-entrant (xact advisory lock, same session). The producer must run on the store's own
   engine, since `enqueue(connection=…)` requires `connection.engine is store.engine` (r2 N2).

Triggering: the producer runs from `LeaderSidePumps` on every dispatcher pass, as the runner does now.
`activate_hold` and `pin_member_request` additionally set the dispatcher wake after commit, so a new gate or
newly pinned run is enqueued and steered at once, not on the 15 s sweep. `QuiesceControlRunner`'s 2 s drain
loop is deleted. Once a row is enqueued, the dispatcher's own loop owns retry.

Release: reference-counted, per §3. A `STEERING` row in flight is left to settle. That's no change: today's
`cancelled` member doesn't stop an RPC already sent either. Once no live member links it, `_admit_grant` refuses
any later attempt.

## 5. How the gate drain maps onto delivery states

**It doesn't. The drain predicate stays evidence-based and never reads delivery or control state.** A DELIVERED
control row means "the run was asked", not "the run stopped". Drain still waits for `execution_exit` (or
revoked / never-bound) exactly as `_drain_status` does today. Delivery state only feeds the operator view and
the ack prompt.

The operator view (`quiesce_status`) keeps its API field `control_state`, computed as:

| stored member `control_state` | linked delivery | reported `control_state` |
|---|---|---|
| `pending` | none | `pending` |
| `steering` | QUEUED / STEERING | `steering` |
| `steering` | DELIVERED | `steered` |
| `steering` | CANCELLED `not_needed:*` | `not_needed` |
| `steering` | CANCELLED (other) | `cancelled` |
| `steering` | FAILED | `unconfirmed` |
| `self` / `unsupported` / `unconfirmed` / `not_needed` / `cancelled` | none | as stored |

`control_attempts` / `control_error` in the view come from the delivery (`attempt_count` / `last_error`).
`ack_status` / `record_ack` are unchanged.

## 6. Migration of `gate_members` control columns (expand, then contract)

- **Expand (this change):** add `control_delivery_id VARCHAR(64)` via `_MEMBER_ADDED_COLUMNS`, plus a partial
  index `WHERE control_delivery_id IS NOT NULL`. `control_state`, its CHECK and `control_changed_at` stay (their
  meaning narrows to the table above). New code stops reading and writing `control_request_id`,
  `control_claim_token`, `control_lease_expires_at`, `control_attempts`, `control_next_attempt_at` and
  `control_generation`. No data migration: no live job has ever written members (TASK-1430 fence), and a
  non-terminal member found at startup with no delivery link is simply `pending`, so the producer enqueues it.
- **Contract (separate commit, after the expand build has been restarted and verified):** drop those six
  columns from `_GATE_RELATIONS` / `_MEMBER_ADDED_COLUMNS` and run `ALTER TABLE gate_members DROP COLUMN IF
  EXISTS ...` once. Dropping in the same build would break the currently loaded 4df476c pump if an app restart
  were rolled back.
- `inter_bot_deliveries.lane` is additive. Old code ignores it.
- **Rollback (r2 S7).** Old code (4df476c) would claim a surviving `QUEUED`/`STEERING` `lane='control'` row as
  ordinary FIFO, which could steer the quiesce prompt or start a system-authored fallback turn. Two fences:
  1. **Runbook step, before any rollback restart:** `UPDATE inter_bot_deliveries SET status='CANCELLED',
     last_error='not_needed:rollback', claim_token=NULL, claim_owner=NULL, lease_expires_at=NULL,
     next_retry_at=NULL WHERE lane='control' AND status IN ('QUEUED','STEERING')`. Document it in
     `reference/ops-actions.md` next to the quiesce rollout and in the S2 commit body.
  2. **Startup self-fence in the new code:** the leader's startup recovery (TASK-1232 check) refuses to start
     the control producer when `quiesce` is off. That keeps the window in which control rows exist bounded to
     quiesce-on operation, so a rollback normally finds none.

  The DB can't fence old code by itself without a CHECK that old code would also trip, so the runbook step is
  the guarantee and the self-fence narrows it.

## 7. `bots_send_message` backward compatibility

- Every existing parameter keeps its name, default and meaning. `target_bot_id` becomes optional (default
  `""`), which widens the schema and doesn't break any caller. New `targets` (selector or list) is mutually
  exclusive with `target_bot_id`: both or neither → `dispatched:false` error before side effects, in
  `resolve_send_mode`.
- New `delivery="steer_only"` joins the existing values. `queue_if_busy` / `fire_and_forget` /
  `wait_for_reply` / `force` resolve as today. `steer_only` + waited is refused.
- Single target (`target_bot_id`, or `targets` resolving to exactly one explicit bot): the response is today's
  flat receipt, unchanged, still from `POST /v1/inter-bot-deliveries`. Multi-target: the batch shape (1.2) from
  `POST /v1/inter-bot-deliveries/batch`.
- Waited-reply reverse check, inactive refusal and sender resolution (TASK-1241) run per target inside
  `_admit_one`, so they hold for fan-out.
- No `lane`, `target_run`, `control` or `include_private` parameter exists on the tool. The hold-bypass is
  unreachable from bots by construction, not by a runtime check alone.

## 8. Slices and order

1. **S1:** `DeliverySender` extraction (pure refactor), plus fan-out selector, batch route and MCP `targets` /
   `steer_only`. Normal lane only.
2. **S2:** `lane` / `target_run` / control head in claim; `ControlGrant` re-keyed to delivery; dispatcher
   passes the grant. PG tests:
   - (a) held FIFO head + control row: control steers, normal order is preserved after release;
   - (b) two jobs on one run make one delivery;
   - (c) pinned run ended → `not_needed`, no reservation ever written;
   - (d) a bot or user principal setting control fields → 403 with no row.
3. **S3:** `QuiesceControlProducer` replaces the pump; `control_delivery_id`; `quiesce_status` derivation; the
   pump's claim/lease/settle code and tests deleted, with their behavioural cases ported onto S2.
4. **S4 (contract):** drop the six member columns after S3 is loaded and verified.
5. **S5 (BawtHub, frontend bot):** "Message bots" control with target and mode pickers on the batch route. No
   hold-bypass until TASK-1420.

S2 and S3 must land before the TASK-1430 (d) rollout makes quiesce live. S1 can land first.

## 9. Settled in r1 review

- `CANCELLED` with a machine-readable `not_needed:*` reason, not a new `SKIPPED` status.
- Bot fan-out: 3 batches per 10 min, key required.
- One enqueue site (producer) plus wake. No enqueue inside `activate_hold`, which would add a lock path next to
  turn-row locks.
