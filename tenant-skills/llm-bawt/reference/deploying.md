# Deploying llm-bawt

The canonical source is mounted into live services; author in an isolated worktree. Publish reviewed/tested work through the [git workflow](../../git-commits/SKILL.md), and coordinate activation with the service owner. Source can be lazily imported before a restart, so incompatible changes need a compatible rollout or a quiet cutover.

Use the matching enabled ops action and its required approval:

| Change | Activation |
| --- | --- |
| Python source | Matching service restart |
| Compose env, mounts or ports | Named-service recreation with `--no-deps` |
| Dependencies or Dockerfile | Build and recreate the named service |

The harness bridges share `llm-bawt-agent-bridge`; local-model-bridge has its own image. A source-only edit usually needs no rebuild. llm-bawt releases do not create version tags; `bawthub.release-prod` is the frontend release.

Use the exact-tip passing gate for code publication and verify live behavior after activation. Bridge/Redis changes interrupt agent work: coordinate a quiet point, obtain the required exact-run approval, dispatch last, and verify on the durable receipt. Do not restart the whole stack as a shortcut.

Activate promptly; never park finished work for a "natural" restart. An app-only change: the owner runs the app restart at a quiet moment right after publication, then verifies. A bridge change: the coordinator schedules a shared window the same day for everything waiting on it. Before picking a restart operator, check where the bot runs, not its model.

Catalog and maintenance guidance: [bawthub-ops](../../bawthub-ops/SKILL.md). Technical contracts: repo `docs/OPERATIONS_RUNNER.md` and `docs/OPS_CONFIG_RECONCILE.md`.
