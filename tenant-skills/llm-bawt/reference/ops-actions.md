# Operations and maintenance

The [operations skill](../../bawthub-ops/SKILL.md) owns authorization, dispatch, receipts and the one-time maintenance path. Discover live slugs and schemas instead of relying on examples.

Typical operations: `llm-bawt.restart-app`, `llm-bawt.restart-bridge`, `llm-bawt.restart-aux`, `container.restart`, and the service-specific config-reconcile actions. A restart reloads source; it does not apply image, mount or environment changes. Reconcile actions keep the current image; do not substitute them for an upgrade.

Use a named service and `--no-deps` for an authorized manual recreation. A stopped service needs a verified start/recovery path. For data-loss risk, check backup freshness with infra-backups (not included in this install).

Approval-gated jobs deliver durable completion receipts. If a quiesce result asks you to end the turn, end it without polling. Verify receipt state, exit code and target behavior. Inspect an uncertain existing job before retrying.

For ordered multi-service work, discover whether `llm-bawt.operation-sequence` is enabled. Its bounded `steps: [{operation, args}]` uses existing allowed catalog actions under one approval, one whole-window all-agent quiesce and one terminal callback. Children retain their individual admission checks and require bounded health verification. The parent receipt's structured `sequence.steps` carries every step's state, exit, health and rollback independently of text tails. It stops on failure; earlier successful restarts are not undone. A worker interruption preserves completed steps, reports an in-flight step as lost and skips the rest—never automatically resume or replay it.

The sequence seed is disabled until its capable worker and eligible live child rows are configured. Do not bypass that refusal or manufacture a multi-service smoke-test run; admission/tests precede the first genuine authorized use, which supplies live acceptance. Catalog grammar, supported kinds, fencing and activation are documented in `docs/OPERATIONS_RUNNER.md`.

Consult repo `docs/OPERATIONS_RUNNER.md`, `docs/OPS_CONFIG_RECONCILE.md`, and production release details (not included in this install) for implementation-specific activation or recovery. Historical incidents belong in task records, not new universal approval stages.
