---
name: peer-review-reviewer
description: Review code or maintenance procedures against their actual scope, record evidence and a concise verdict, and verify fixes without expanding the assignment.
---

# Reviewing work

Read the diff or executable operation, trace the affected paths, and check the evidence. Run targeted checks when they resolve a concrete concern; reuse valid receipts. Keep the review proportional to the actual scope.

Use `BLOCKER` for a concrete failure in supported conditions: explain the state, wrong result, and evidence. Use `SHOULD_FIX`, `NIT`, `QUESTION`, or `PRE_EXISTING` for findings that do not prevent this change from working. Future generality and unrelated cleanup are follow-ups.

Record `APPROVE`, `APPROVE_WITH_CHANGES`, or `REQUEST_CHANGES` with `tasks_reviews_add`, pinned to the reviewed revision. State what you checked and any material limit. For a recheck, focus on outstanding blockers and regressions introduced by the fix. Preserve review history using supported recheck or linked-review records; do not invent resolved evidence.

When reviewing, do not edit the change you are judging. If you help implement a fix, another reviewer checks those lines. This restriction applies to the review assignment, not every other job held by a reviewer-capable bot.

Send the recorded verdict to the requester once. No duplicate narratives or acknowledgment requests are needed. See [peer-review](../peer-review/SKILL.md) for selection and escalation.
