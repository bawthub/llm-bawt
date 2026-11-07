---
name: git-commits
description: Commit and publish coherent changes using isolated worktrees, proportionate review and tested revisions. Use for staging, commits, integration and pushes.
---

# Git workflow

Use an isolated task worktree; the canonical checkout feeds live services. Start with `bawt-wt --dev-root <shared-root> --bot <bot> start <repo> TASK-N --slug <words>`. The task-qualified helper creates `<shared-root>/.worktrees/.wt-<repo>-<bot>-TASK-N` only after the target repo's hook compatibility is published and active; it resumes registered legacy task trees in place and never switches a sibling task. Keep unfinished work separate, preserve other authors' changes, and use the installed hooks. Mutable environments stay task-local. `bawt-wt done` removes a merged clean tree (branch kept), and eligible clean trees of completed/archived tasks are reaped automatically. See [task worktrees](reference/task-worktrees.md) for rollout and legacy handling.

When a task closes, its owner removes the clean, published task worktree with `bawt-wt done`; preserve and report dirty or unmerged work instead.

Keep bot-generated logs, durable artifacts and scratch under `<shared-root>/.agent-work/{logs,artifacts,scratch}/TASK-N/<bot>/`, not dev/repository roots. This replaces `<shared-root>/task-artifacts/` for new output; existing referenced evidence stays there until owner-cleared relocation. Retain test/review receipts as artifacts, not disposable temporary files. Existing held paths stay in place until their owner releases them; cleanup must preserve a path map and use Git-aware moves for registered worktrees.

Review your diff, run relevant checks, and commit a coherent change with `commit-slice --bot <bot> -F <message-file> -- <paths>`. Use a task-linked subject and your bot's Co-Authored-By trailer. On an account with `bawthub.requireTurnIdentity=true`, the helper verifies the signed turn capability and scopes the bot author plus configured committer to this command; `--bot` must match. Do not toggle shared Git identities. Use the helper's bounded `--ssh` envelope for remote commits, not a token in shell arguments. Accounts without that opt-in retain their existing behavior. See [authorship and SSH](reference/audit-and-slice.md) and [GitHub App credentials](reference/github-app-credentials.md) for setup. Attribution tools are helpful evidence, not prerequisites to an otherwise understood commit.

Obtain one independent review for consequential changes; simple low-risk changes can use a recorded self-check. Reuse valid review and test receipts for unchanged scope; staging, handoff and publication are not new review/approval rounds. Send the tested SHA and receipts to the integrator. Routine author checks, installed hooks and commits need no fresh arbitration or central slot grant; coordinate only actual shared writers/heavy-job conflicts and respect their locks.

For llm-bawt/BawtHub, the integrator batches ready work and runs one full gate on the exact final tip; reuse its passing receipt rather than rerunning the same candidate. Integrate and push that pinned SHA when green, verify the remote, and follow its CI. The integrator may finish publication. Assign service activation and usable verification with the service owner as part of requested delivery; a source-only handoff does not finish a requested deployed outcome. For agent-skills, run relevant script/frontmatter checks and `skills-sync <SHA>`; instruction-only changes need no application gate or app-test queue reservation.

Do not force-push, discard another author's work, bypass hooks, or rewrite published history without explicit authorization. A push is separate from a service deployment; respect active user holds.

Details when needed: [audit](reference/audit-and-slice.md), [messages](reference/message-style.md), [traceability](reference/task-traceability.md), [integration](reference/push-and-ci.md).
