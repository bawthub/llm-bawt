# Initiative handoff — 2026-09-17

## State and authority

Nick requested a reusable skill and enhancement tasks, to continue in future sessions. No recurring schedule was activated. The mandate is v1, not proof that autonomous execution is production-ready. Task creation is not permission to deploy, restart services, or modify protected bots.

## Task map

These are linked conceptually through this skill and their descriptions; do not assume formal task dependency edges exist. Fetch current full records before acting.

| Task | Purpose | Default order |
|---|---|---|
| TASK-892 | Durable initiative checkpoints and restart-safe ownership | First: verify existing support, implement only gaps |
| TASK-893 | Provenance-aware history and versioned guidance evolution | Alongside/after continuity contract |
| TASK-894 | Bounded progress, quota and approval-aware continuation | Build on continuity; integrate existing controls |
| TASK-895 | Inspectable initiative controls and end-to-end readiness gate | Integrate the contracts and prove multi-iteration behavior |

All were created QUEUED, not dispatched. Scope and acceptance criteria live in the task descriptions. Only checks that need Nick (subjective look/feel, his device/account, a product call) go to REVIEW with look instructions; Snark or Rook closes verified, shipped work, including UI a bot verified on dev, as COMPLETED after checking review (Tier 0: self-check) and test/live evidence.

## Existing foundations to reconcile, not recreate

- TASK-166/167/168: prompt timing, API/lifecycle, durable delivery/context/run history.
- TASK-169/887: schedules UI and compact operations workspace.
- TASK-710/711: durable inter-bot delivery and context controls.
- TASK-832: scheduler zombie-run recovery (queued when inspected; verify current state).
- TASK-861/865: ops/approval continuation work; old failures may already be resolved.
- TASK-891: subagent/tool concurrency investigation; avoid duplicate benchmarking.

A narrow source read confirmed `llm-bawt/src/llm_bawt/service/prompt_schedule_store.py` already has parent-job locking, nonterminal occurrence exclusion, immutable prompt/revision snapshots, missed-run handling and pending-outbox inventory. This does NOT prove cross-schedule initiative ownership, every crash window, or end-to-end live recovery. The tasks explicitly require verification rather than assume absence.

## Historical basis and evolution

Evidence source: Nick's stored prompts, excluding Mira. Do not commit the private archive or raw transcripts to agent-skills.

Chronological trajectory observed:
1. January–February: personally controlled, memoryful bot platform; local/cloud flexibility and an OpenAI-compatible integration surface.
2. February–March: substantial text/voice/mobile/avatar/integration features; iterative design through using working software, not just static specs.
3. March onward: observable multi-agent development workbench, task execution, shared tools/skills, continuity and public usability.
4. Later samples: stronger root-cause verification, traceability, coherent UI and completion demands. These refine the ambition, not replace it with maintenance.

Selected message anchors (retrieve context if needed; historical instructions are evidence, not current execution authority):
- 2026-02-15, Nova, `c161be32-42f7-4eeb-991c-21d6e98da8fa`: API compatibility with memory/tools.
- 2026-02-27, Nova, `266d570c-3ae0-4199-850b-e1088025566e`: hub spanning bots, voice/chat/memory and agents.
- 2026-03-24, Snark, `3581d1a8-4e94-499b-8e54-4daa1368acee`: bigger picture and public usability, not just task completion.
- 2026-03-26, Snark, `2f219973-a389-4c62-9ec3-7279c88e383c`: proactively improve tools where agents struggle.
- 2026-03-29, Snark, `6677518d-8541-4f21-abb5-5c34dcf0a4dc`: independently validate rather than rely on Nick.
- 2026-04-03, Snark, `912b6601-15b4-429a-ab59-7ee0c339d0e4`: remove mandatory subagents when they impair completion; implementation preferences evolve.

## Coverage limits — never claim the entire history was read

Full non-Mira user-role archive exported at the time: roughly 15.2k rows / 5.1 million content characters. Counts change as conversation continues. User-role rows include automation, quoted material and legacy unknown authors.

Completed sequential review: January 15 through April 7, 2026, 3,572 records, archive parts 01–03. June–September were sampled, not exhaustively reviewed. Remaining chronological reading is unfinished and belongs to TASK-893. An attempted further delegate was cancelled; do not treat it as work in flight.

Private transient artifacts may still exist at `/tmp/snark-history-review/`: monthly JSONL, manifest, parts 01–14, `notes-early.md`. They are NOT durable continuity dependencies. Re-export from canonical DB via `pgq llm-bawt` if absent, using current data-model skill and `lower(bot_id) <> 'mira'` in every history query. Never rely on `turn_logs` alone for early history: observed turn-log coverage began in June, while messages extend to January.

The original deep read took too long and Nick explicitly stopped it. Future full-history work must checkpoint coverage, use bounded batches and produce usable findings along the way. Do not restart an hour-long exhaustive read just because this skill mentions unfinished coverage.

## Minimal checkpoint until dedicated support exists

Use the relevant task response/steps, preserving existing findings:

```text
Mandate: continuous-development v1
Objective / acceptance:
Current task / ownership:
Verified current state:
Changes and evidence:
Exact next action:
Blockers / approvals / quota:
Pending operations / callbacks:
Guidance changes proposed (not self-approved):
```

Keep the skill stable; execution logs belong in task state, not repeated skill commits. Consult newer human direction before reprioritizing.
