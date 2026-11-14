---
name: bawthub-ops
description: Run and verify BawtHub or llm-bawt service actions and one-time maintenance. Covers the live ops catalog, delegated approval, recovery and durable receipts.
---

# Service operations

Discover the live catalog with `ops_list_operations` and use `ops_run` for a supported enabled action. Retain the job ID and verify the terminal receipt plus intended behavior. An approval or queued job is not completion.

Approval cards are a coordination check, not a security boundary: someone who can see the whole system (in-flight turns, other restarts, holds, pending activations) confirms the timing and scope fit. Approvers judge that, not threat models. Use the existing delegated approver where the chosen action requires a card; no self-approval. Explicit user authorization controls the operation's scope; do not ask again for the same authorized scope. A missing catalog action is not a veto: execute the smallest concrete manual alternative under applicable authorization, recording scope, recovery and verification, rather than requiring new tooling or another consent loop. Actual denials, explicit disabled/protected boundaries and unknown outcomes are not permission to bypass them. Private DEV authority is defined in local-infrastructure (not included in this install).

For one-time maintenance, compare a short maintenance window with an online implementation before commissioning infrastructure. Identify affected data/writers, recovery and verification; reuse an existing applicable technical review rather than reviewing the same scope again. Consequential unreviewed operations get one technical review, then execution and a result receipt. Reusable tooling can follow afterward.

Architecture (live since TASK-1173): the ops reconciler is the only executor and the only first-party Docker-socket holder; the llm-bawt app is admission-only, frontends and Traefik have no socket. Reference: `llm-bawt/docs/OPS_SERVICE_ARCHITECTURE.md`. A `reconcile-*-config` refusal `compose_source_invalid (config files label)` means the container was created with a compose file that no longer exists; one manual `docker compose up -d --no-deps --no-build <svc>` from the in-repo files clears it.

Check recoverable backups before irreversible data changes. A service stop needs a working recovery/start path. Scope Docker actions to named services and use `--no-deps` for recreation. Bridge/Redis actions can end active turns: coordinate a bounded safe point for the affected workers and writers, not indefinite global fleet idleness; retain checkpoints/callbacks and dispatch the interrupting action last. Find who a bridge restart cuts off, and who can run it, from `bots_list_available` `state.location` (or `location=<bridge>`), not from a bot's model or provider. Respect explicit workload holds and required action cards. If quiesce instructs you to end the turn, do so; let the durable receipt wake you.

When a quiesce holds your turn, answer with `ops_quiesce_ack`. A "stopping" bot is woken with its note when the operation ends; a "finished" bot is not. "finished" requires a non-empty note naming what wakes you next for your task (e.g. "gate receipt", "Caid's reply", "none: task closed").
- "finished" = if this bot were never woken again, nothing would be lost, half-applied or forgotten. ALL of these hold: no edit unsaved or uncommitted that this turn meant to commit; no tool call, command, build, gate or background job this turn started and still owes a result to act on; no step of the current plan that was going to run in THIS turn; no reply, handoff or card update owed that hasn't been sent/written. Remaining task work is fine ONLY if something durable already wakes the bot for it (a delivery, ops receipt, gate callback, scheduled wake-up), and the note names it.
- "stopping" = anything else. You are pausing mid-work; the note says exactly where you stopped and what's next. You will be resumed.
- When in doubt: "stopping". A wasted resume turn is cheap; silently dropped work is not.
- Never ack "finished" to avoid being resumed, and never stop the turn earlier than needed to make a "finished" ack true.

Do not poll approval-gated jobs; the continuation returns the result. Inspect a job once when its receipt is missing or uncertain. Reconcile an unknown outcome before retrying.

Implementation and specific operation contracts: llm-bawt `docs/OPERATIONS_RUNNER.md`, `docs/OPS_CONFIG_RECONCILE.md`, [service notes](../llm-bawt/reference/ops-actions.md), and production releases (not included in this install).
