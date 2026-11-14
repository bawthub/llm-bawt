---
name: llm-bawt
description: Work in llm-bawt, the orchestration backend for bots, memory, tools and agent bridges. Routes to relevant architecture, testing, deployment and runtime references.
---

# llm-bawt

LLM orchestration, bots, memory, tools and agent bridges. Repo `~/dev/llm-bawt`; app port 8642, MCP 8001. Canonical branch `master`.

Use [git-commits](../git-commits/SKILL.md) for isolated worktrees and publication, [agent-system](../agent-system/SKILL.md) for tasks, and [bawthub-ops](../bawthub-ops/SKILL.md) for service changes. These are their authoritative homes; do not add parallel workflow rules here.

Configuration lives in the DB. Bots resolve pinned endpoint IDs and tags/capabilities. Agent prompts are pinned per session; edits take effect at the next supported boundary. The app owns OAuth refresh; do not add another refresher. Read the specific subsystem reference before changing a sensitive boundary.

Prompt DB edits are tenant overrides, not automatic product releases. For default inheritance, reset, selected approved fixture export and drift, use [product defaults](reference/product-defaults.md); never hand-maintain a second prompt body.

Keep changes bounded and verify the actual behavior. Routine design details belong to the owner. Refactoring is justified by the change, not a file-size threshold.

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
