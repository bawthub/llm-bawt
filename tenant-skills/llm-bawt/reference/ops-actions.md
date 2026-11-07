# Operations and maintenance

The [operations skill](../../bawthub-ops/SKILL.md) owns authorization, dispatch, receipts and the one-time maintenance path. Discover live slugs and schemas instead of relying on examples.

Typical operations: `llm-bawt.restart-app`, `llm-bawt.restart-bridge`, `llm-bawt.restart-aux`, `container.restart`, and the service-specific config-reconcile actions. A restart reloads source; it does not apply image, mount or environment changes. Reconcile actions keep the current image; do not substitute them for an upgrade.

Use a named service and `--no-deps` for an authorized manual recreation. A stopped service needs a verified start/recovery path. For data-loss risk, check backup freshness with infra-backups (not included in this install).

Approval-gated jobs deliver durable completion receipts. If a quiesce result asks you to end the turn, end it without polling. Verify receipt state, exit code and target behavior. Inspect an uncertain existing job before retrying.

Consult repo `docs/OPERATIONS_RUNNER.md`, `docs/OPS_CONFIG_RECONCILE.md`, and production release details (not included in this install) for implementation-specific activation or recovery. Historical incidents belong in task records, not new universal approval stages.
