---
name: llm-bawt
description: Work in llm-bawt, the orchestration backend for bots, memory, tools and agent bridges. Routes to relevant architecture, testing, deployment and runtime references.
---

# llm-bawt

LLM orchestration, bots, memory, tools and agent bridges. In the llm-bawt checkout, the default ports are app port 8642, MCP 8001. Canonical branch `master`.

Use [git-commits](../git-commits/SKILL.md) for isolated worktrees and publication, [agent-system](../agent-system/SKILL.md) for tasks, and [bawthub-ops](../bawthub-ops/SKILL.md) for service changes. These are their authoritative homes; do not add parallel workflow rules here.

Configuration lives in the DB. Bots resolve pinned endpoint IDs and tags/capabilities. Agent prompts are pinned per session; edits take effect at the next supported boundary. The app owns OAuth refresh; do not add another refresher. Read the specific subsystem reference before changing a sensitive boundary.

Prompt DB edits are tenant overrides, not automatic product releases. For default inheritance, reset, selected approved fixture export and drift, use [product defaults](reference/product-defaults.md); never hand-maintain a second prompt body.

Keep changes bounded and verify the actual behavior. Routine design details belong to the owner. Refactoring is justified by the change, not a file-size threshold.

## Stopping jobs safely

On the host or inside the bridge, never stop jobs by name/pattern with `pkill`, `killall`, or `pkill -f`: bot processes carry skill words in their argv and can match. Use the PID or process group recorded at launch, or the queue's own cancel command. If the PID is unknown, list candidates with `pgrep -a` and read their command lines before selecting a target.

## Long-running work

The Claude bridge aborts a turn after 300 s with no SDK output, or when a background subagent is still running after 900 s. Don't block a turn on a scrub, VM push, bootstrap or gate: launch it as a durable host job (`systemd-run`/`nohup` with a log and receipt), read its state once, record it on the card, and set a wake-up or callback for the result. An SDK background shell dies with an aborted turn, so it can't carry such work. After an abort, check with `pgrep -a` for a surviving run before starting another.

## Technical references

| If your task touches… | Read first |
|---|---|
| Layout, directory map, API routes, request pipeline, request extensions, bot types, memory/tool systems, config names, task dispatch, "Claude bridge bots not responding" | `reference/architecture.md` |
| Houses, assigning a bot to one, where a given test or UI check runs | `reference/houses.md` |
| Running tests, `make gate`/`test-snapshot` procedure, timing-test rules | `reference/testing.md` |
| Shipping a code change, what to restart vs rebuild, worktree-to-shared integration, release tags, self-deploy | `reference/deploying.md` |
| Ops catalog, executor, ops approvals, any service mutation | `../bawthub-ops/SKILL.md` + `reference/ops-actions.md` |
| Anthropic-compat proxy (`claude_code_bridge/proxy/`), adapters, ChatGPT-OAuth 401/400s | `reference/anthropic-proxy.md` |
| Claude credentials, `/v1/usage`, Reconnect | `reference/claude-credentials-usage.md` |
| Personas / `prompt_override_id` | `reference/personas.md` |
| Prompt product defaults vs tenant overrides, promotion, seeding and drift | `reference/product-defaults.md` |
| Task capability, non-activating sessions, MCP task tools, `/v1/turn-changed-files*` | `reference/task-conversation.md` (cross-stack: `agent-system/task-conversation-architecture.md`) |
| `bots_send_message`, deliveries, `agent_context_*` | `reference/inter-bot-deliveries.md` |
| Agent persona loss, prompt change not taking effect, cache cost, `agents.global_prompt` edits | `reference/agent-system-prompt.md` |
| Prompt assembly order, every bot flag's prompt impact | `reference/system-prompt-generation.md` |
| History, summaries, session seeding | `reference/history-context-flow.md` |
| Conversation storage, session scoping, tool-call rendering, model catalog | `reference/data-model.md` |
| Anything that runs on a timer, "is X running" | `reference/background-jobs.md` |
| Security findings | `reference/security-model.md` |
| "This action requires user approval" | `reference/approval-policies.md` |
| OOP/ABC, monolith split, slug-literal decisions | `reference/code-standards.md` |
