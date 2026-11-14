---
name: peer-review
description: Arrange independent review of consequential changes and resolve concrete findings. Use when requesting review, handling feedback or preparing work for integration.
---

# Peer review

Use a self-check for wording, documentation, and simple low-risk changes. Consequential behavior changes get one independent implementation review. Discuss design first when the actual change affects data integrity, security, concurrency, or recovery; a separate design-review stage is otherwise optional.

Choose an available bot with the `review` capability who did not author the change. Give the reviewer the exact revision or executable maintenance plan, scope, relevant evidence, and remaining uncertainty.

Prefer a different provider when one is available. If the other provider's subscription, quota, authentication or service is unavailable, use an available independent reviewer on the same provider immediately. Record the fallback reason with the verdict; no exception request, provider retry/probe, model switch or wait for subscription recovery is needed. This applies to code, design and operational/maintenance reviews. Provider diversity is a preference, not a blocking gate; non-author independence, technical findings and required operation approvals still apply.

Review the requested outcome under its stated operating conditions. Do not make hypothetical future capabilities prerequisites. A blocker needs a concrete failure scenario; questions, preferences, and unrelated pre-existing defects are not blockers.

Record the verdict on the task with `tasks_reviews_add`. Send a callback only to the requester and anyone whose next action changes. No acknowledgment chain is needed.

Fix real blockers and have the reviewer verify those fixes. A review-round number is not a reason to abandon a sound correction. Use the API's supported recheck records; if it cannot represent another recheck, record a linked review of the new revision without erasing old findings. Escalate only a genuine technical disagreement or stalled approach, not routine revision.

An approved review and relevant test receipts are enough for integration intake. Avoid redundant reviews of unchanged code. Review does not authorize unrelated deployment or scope expansion.

Reviewer procedure: [peer-review-reviewer](../peer-review-reviewer/SKILL.md). Consult [technical checklists](references/checklists.md) for the risks that apply.
