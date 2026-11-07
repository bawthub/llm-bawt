# Integration and publication

Authors send the integrator a ready revision, scope, review receipt and relevant checks. The integrator checks outstanding blockers, ownership, conflicts and secrets, then batches compatible work. Resolve genuine conflicts with the author; avoid new reviews for unchanged patches.

For llm-bawt/BawtHub, run one `make gate REV=<final-tip>` on the host for the reviewed batch and require its exact-SHA PASSED receipt. Reuse that receipt; do not full-gate each local iteration or pure rebase. A changed final tip needs its own gate. Attribute a failing check before choosing a repair or retry.

For llm-bawt/BawtHub, publish the gated SHA with `gate-push [-C <repo>] <SHA>`. It requires that receipt and pushes the branch and a `bawthub-gate PASSED <gate id>` note (`refs/notes/bawthub-gate`) atomically. CI skips the suites the gate already ran for a noted master/main SHA, and runs them for anything else. A plain `git push` is still correct, but it runs full CI. Verify the remote and CI. Tenant images (app, agent-bridge, pocket-tts, voice-backend) no longer build per push; dispatch the repo's tenant publish workflow for an exact SHA when a deploy needs one. The integrator may finish publication instead of bouncing it back to the author. Coordinate the clean shared checkout's fast-forward with activation: it is mounted into running services, so incompatible source/schema changes need their planned rollout. Do not move a dirty live tree over uncommitted work.

For agent-skills, run relevant script checks and publish with `skills-sync <SHA>`; it verifies the pinned push before updating the live checkout. No application full-suite gate is needed for instruction-only changes.

A long-running check needs a durable callback/wakeup if the turn ends. Read the actual receipt before claiming success. A push alone does not prove service deployment. Use the repo's release guidance for activation and honor explicit user holds.
