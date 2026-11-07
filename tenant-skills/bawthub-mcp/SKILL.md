---
name: bawthub-mcp
description: Use BawtHub MCP tools for tasks, memories, message history, bot callbacks/deliveries, sessions, ops, media and search. Load the relevant reference for advanced arguments or workflows; also use when designing or reducing MCP tool context.
---

# BawtHub MCP

Use MCP for normal tool operations. Do not substitute direct HTTP/DB writes or
shell scripts when a supported tool exists. If the tool is unavailable, report
that failure; an approval denial is never permission for a fallback.

## Tool availability: loaded is not deferred

Before declaring a tool unavailable:

1. Check the callable tool definitions already supplied to this turn. If its
   schema is loaded, invoke it directly using its exact exposed name.
2. Only for a deferred tool, use ToolSearch with the fully qualified advertised
   name, e.g. `select:mcp__bawthub__ops_list_operations` (not the bare operation).
3. A `No matching deferred tools found` result is NOT evidence that loaded tools
   are missing or that MCP is down. Distinguish an absent schema from a connection
   failure, an operation disabled in the catalog, and an approval denial.
4. Verify a suspected outage with an available read-only call when possible.
   Never request an SSH/HTTP fallback solely because deferred search missed.

## Choose a reference, not the whole manual

| Work | Read when needed |
|---|---|
| Tasks, steps, projects, activity | [references/tasks.md](references/tasks.md) |
| Memory, messages, profiles, recall | [references/memory-history.md](references/memory-history.md) |
| Sessions, inter-bot delivery, context, persona, editing another bot's profile/tags | [references/sessions-delivery.md](references/sessions-delivery.md) |
| Ops, media library, image generation, search | [references/ops-media.md](references/ops-media.md) |

Tool descriptions supply the selection rule and immediate hazards. Typed schemas
supply fields/defaults. These references supply workflow and advanced options.
Load only the relevant reference, using the skill loader or reading this skill's
files in the installed skills directory. No additional MCP documentation tool is
needed. If a reference is absent on this harness, report the missing skill;
do not invent arguments or bypass MCP.

## Shared rules

- Pass your own bot slug for memory/history scope and activity attribution;
  defaults such as `default` or `unknown` are not identity discovery.
- Respect protected-bot boundaries even when list/search tools expose them.
- Check structured errors and partial outcomes. Queued, approved, or delivered
  does not mean an execution completed successfully.
- Use stable idempotency keys where supported; do not retry mutations blindly.
- Prefer bounded lists/searches; fetch full detail only for selected records.
- Durable future callbacks use `schedule_wakeup`, never a harness timer
  (`CronCreate` dies on restart, `/new` and compaction).
- Preserve history. Delete/clear/replace tools are not synonyms for cancel,
  archive, append, or patch.

## Designing tools

Authoring contract and measurement command:
`llm-bawt/docs/MCP_TOOL_DESIGN.md`.

Keep the catalog concise without hiding required fields or hazards in a skill.
Do not add one tool per timing/filter variation. Do not combine reads and
mutations behind an untyped catch-all. Renames/consolidations require migration
of callers, approvals, discovery and documentation before old names are retired.
