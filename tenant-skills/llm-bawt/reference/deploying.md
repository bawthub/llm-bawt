# Deploying llm-bawt

The canonical source is mounted into live services; author in an isolated worktree. Publish reviewed/tested work through the [git workflow](../../git-commits/SKILL.md), and coordinate activation with the service owner. Source can be lazily imported before a restart, so incompatible changes need a compatible rollout or a quiet cutover.

Use the matching enabled ops action and its required approval:

| Change | Activation |
| --- | --- |
| Python source | Matching service restart |
| Compose env, mounts or ports | Named-service recreation with `--no-deps` |
| Dependencies or Dockerfile | Build and recreate the named service |

The harness bridges share `llm-bawt-agent-bridge`; local-model-bridge has its own image. A source-only edit usually needs no rebuild. llm-bawt releases do not create version tags; `bawthub.release-prod` is the frontend release.

Restart each service whose loaded code changed, and only those. The `ops-reconciler` Compose service runs the executor, the GPU reconcile fence and the release coordinator. When `src/llm_bawt/ops/` or `src/llm_bawt/media/gpu_handoff*` changed, a window includes `container.restart` of the deployed reconciler container (it runs as an isolated self-restart), or those changes stay dark after an otherwise complete window. Process-mode ops jobs (image deploy/rollback) run from the pinned ops worker image, which bundles its own `llm_bawt.ops`. When that code changed, also rebuild the worker image from the published source and set `LLM_BAWT_OPS_WORKER_IMAGE` to it, then recreate the app and reconciler. This is an env change, so a restart does not pick it up.

A window starts with `canonical-check --repo "$LLM_BAWT_REPO"` (on the host) (exit 0 and its HEAD equal to the expected SHA, recorded in the receipt) before any restart trusts the shared checkout. A window that builds or recreates the app image ends with `python3 "$LLM_BAWT_REPO/scripts/check_running_lock.py" --lock "$LLM_BAWT_REPO/uv.lock"` (on the host), and its exit code and any listed differences go in the receipt. A nonzero exit means the running app differs from the lock the gate tested; the window owner rolls back or names the difference before calling the window done.

Use the exact-tip passing gate for code publication and verify live behavior after activation. Bridge/Redis changes interrupt agent work: coordinate a quiet point, obtain the required exact-run approval, dispatch last, and verify on the durable receipt. Do not restart the whole stack as a shortcut.

Create or recreate llm-bawt containers only through an ops action, a Make target, or `scripts/compose-secrets -- <compose args>`. Plain `docker compose up` silently renders an empty `LLM_BAWT_REDIS_URL` (and other secrets kept out of `.env`), so the bridge and app crash-loop. Check a hand-written command with `scripts/compose-secrets -- config <service>` first. A scheduled recreate is a one-shot `systemd-run --on-active`, never a recurring `OnCalendar` timer; remove the unit once its receipt is in.

Activate promptly; never park finished work for a "natural" restart. An app-only change: the owner runs the app restart at a quiet moment right after publication, then verifies. A bridge change: the coordinator schedules a shared window the same day for everything waiting on it. Before picking a restart operator, check where the bot runs, not its model.

Catalog and maintenance guidance: [bawthub-ops](../../bawthub-ops/SKILL.md). Technical contracts: repo `docs/OPERATIONS_RUNNER.md` and `docs/OPS_CONFIG_RECONCILE.md`.
