"""Compact agent-facing receipts for ``bots_send_message`` (TASK-1151).

The durable delivery receipt carries ~40 fields, most of them null or
internal on a fresh send. Agents only need the delivery identity, its state
and anything that changes their next action. Nothing is lost: the full
record stays available through ``bots_delivery_get(delivery_id)``.
"""

from __future__ import annotations

from typing import Any

# Internal or redundant on the send path: the caller is the sender, ``mode``
# timing/retry bookkeeping lives in the
# full receipt.
_DROP = frozenset({
    "author", "metadata", "revision", "updated_at",
    "available_at", "overflow_recovery_count",
})
# Kept only when they say something beyond the default.
_DEFAULTS: dict[str, Any] = {
    "attempt_count": 0, "max_attempts": 5, "session_policy": "continue",
    "duplicate": False, "fire_and_forget": False,
}


def compact_send_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """Drop null, internal and default-valued fields; keep everything else."""
    if not receipt.get("delivery_id"):
        return receipt  # refusals and errors pass through untouched
    out = {
        k: v for k, v in receipt.items()
        if v is not None and k not in _DROP
        and not (k in _DEFAULTS and v == _DEFAULTS[k])
    }
    # Pure duplicates: ``mode`` repeats ``reply_mode``; ``queued`` repeats a
    # QUEUED ``status``.
    if out.get("mode") == out.get("reply_mode"):
        out.pop("mode", None)
    if out.get("queued") is True and out.get("status") == "QUEUED":
        out.pop("queued")
    out["full_receipt"] = "bots_delivery_get(delivery_id)"
    return out
