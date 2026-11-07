# Committing a coherent change

In your task worktree, inspect status and diff, including new files and staged content. Confirm ownership and that the change is complete. `commit-recon` can help, but missing capture is not a reason for an exhaustive investigation when the diff and owner are known.

Use `commit-slice --bot <bot> -F <message-file> -- <paths>` for an attributable commit with an explicit scope. Use `--patch` when selecting hunks. Do not sweep another author's unfinished work into your commit.

## Signed bot identity

An account that sets `git config bawthub.requireTurnIdentity true` (with `bawthub.committerName`/`bawthub.committerEmail` for the shared App committer) gets per-command attribution: commit-slice verifies the turn's `LLM_BAWT_TASK_TURN_CONTEXT` at llm-bawt `GET /v1/git/identity`, authors the commit as `<bot> <<bot>@bots.bawt.local>`, sets both only for its own Git and hook processes, and checks the new commit object before moving HEAD. `--bot` must match the signed identity; a missing or rejected context is refused. Accounts without the setting are unchanged. The context is a six-hour bearer capability: it attributes the commit to the turn that minted it, not proof that turn is still running. Raw `git commit` keeps the account's own identity.

To commit as another account, add `--ssh <account>@<host> --remote-helper <absolute commit-slice path> --repo <remote repo>`; paths are relative to that repo. The context, argv, message and patch contents travel in a bounded stdin envelope, never on a command line.

An account's post-commit hook can call `python3 <agent-skills>/scripts/bawt_git_identity.py warn-head` to flag raw Git commits made with a turn context but a non-bot author. Install it without replacing existing hooks. It is a local advisory check, not signature validation or a same-user plumbing barrier; manual commits with no turn context are not flagged.

Keep unrelated tasks separate. A routine local correction does not need arbitration; involve the affected owner or integrator for genuine overlapping ownership. Preserve work rather than discarding it to make a checkout clean.

Tests, review and integration follow [git-commits](../SKILL.md). Reuse valid evidence instead of repeating the implementation review merely to stage files.
