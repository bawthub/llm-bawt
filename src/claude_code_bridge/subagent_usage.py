"""Measured subagent usage, separate from the SDK's mutable progress estimate.

TaskProgress.total_tokens is latest input + accumulated output + a streaming
estimate in CLI 2.1.280; TaskNotification can instead contain last-request usage.
Neither is a cumulative consumption counter. Account provider usage by response
ID, scoped to the spawning tool (a Workflow may have several child tasks).
"""
from __future__ import annotations

from typing import Any


class SubagentUsage:
    """Request-local ledger. Never sum progress snapshots or infer missing usage."""

    TOKEN_FIELDS = (
        "input_tokens", "output_tokens", "cache_creation_input_tokens",
        "cache_read_input_tokens",
    )

    def __init__(self) -> None:
        self._responses: dict[str, dict[str, dict[str, int]]] = {}

    @staticmethod
    def _count(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    def observe(self, message: Any) -> None:
        parent = getattr(message, "parent_tool_use_id", None)
        response_id = getattr(message, "message_id", None)
        usage = getattr(message, "usage", None)
        if not parent or not response_id or not isinstance(usage, dict):
            return
        counters = {
            key: value for key in self.TOKEN_FIELDS
            if self._count(value := usage.get(key))
        }
        if not counters:
            return
        responses = self._responses.setdefault(parent, {})
        snapshot = responses.setdefault(response_id, {})
        # SDK echoes multiple blocks with the same response ID, sometimes with
        # preliminary zero/partial usage. The counters are per-response snapshots,
        # not deltas. A repeated block or replay must never double-charge it.
        for key, value in counters.items():
            snapshot[key] = max(snapshot.get(key, 0), value)

    def for_event(self, parent: str | None, usage: Any) -> dict | None:
        result = {
            key: value for key in ("total_tokens", "tool_uses", "duration_ms")
            if isinstance(usage, dict) and self._count(value := usage.get(key))
        }
        responses = self._responses.get(parent or "", {})
        complete = [u for u in responses.values() if "input_tokens" in u and "output_tokens" in u]
        if complete:
            # Explicitly parent-scoped: sibling Workflow task events carry the
            # SAME aggregate. Consumers take it once per spawning tool, not once
            # per task. Cached input is included, not added twice to input_tokens.
            result["cumulative_tokens"] = sum(sum(u.values()) for u in complete)
        return result or None
