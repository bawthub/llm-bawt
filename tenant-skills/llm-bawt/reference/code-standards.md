# Code judgment

Follow existing interfaces and style. Reuse an established provider/client abstraction when it fits; avoid needless duplication. Choose functions, composition, or classes according to the behavior. No mandatory hierarchy for a simple helper.

Refactor before a change when doing so materially improves correctness, testability, or scope. A module over 1,000 lines invites inspection, not an automatic split. Keep unrelated cleanup out of the critical path and record any meaningful tradeoff briefly.

Product behavior comes from configuration and tags/capabilities, not hardcoded bot identities or deployment details. Resolve prompts through the registry and model choices through the configured catalog/settings. Tests and generic seeds can use explicit fixtures.

Detect change with events, in the browser and on the server. When there is no event for a change you own, publish one; lacking one today doesn't justify polling. **No polling without the user's explicit approval** (the user, 11-06), recorded on the task, and that includes reconcile sweeps, safety-net timers, dev/draft builds and "wait until done" loops. Reconcile after a missed event or reconnect with a one-shot, event-triggered resync. If you believe a source genuinely can't push (for example an external API with no webhook), bring the case to the coordinator for the user instead of building the timer; an approved timer's code must cite the approval. Reconcile timers should be slow, and browser ones should pause in hidden tabs. Timers whose purpose is a deadline, scheduled action, retry delay or local display update are outside change detection. Polling for an operation to complete is still change detection and must meet the rule above, however short or bounded.

Run relevant checks and get independent review for consequential changes. Review blockers must apply to the submitted scope; technical preferences and pre-existing debt do not hold an otherwise correct change.
