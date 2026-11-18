"""Spec of the operator freeze/thaw operations (TASK-1489).

Only ``executor_kind="freeze"`` may carry this spec, and that executor runs no
other. Kept free of executor imports to avoid a cycle, like ``release_spec``.
"""
from __future__ import annotations

FREEZE_EXECUTOR_KIND = "freeze"
FREEZE_ACTION = "operator_freeze"
FREEZE_OPS = ("freeze", "thaw")


def is_freeze_spec(spec: object) -> bool:
    return isinstance(spec, dict) and spec.get("action") == FREEZE_ACTION


def validate_freeze_spec(spec: dict) -> dict:
    if set(spec) != {"action", "op"} or spec["op"] not in FREEZE_OPS:
        raise ValueError("operator_freeze spec is exactly {action, op} with op freeze or thaw")
    return spec


__all__ = ["FREEZE_ACTION", "FREEZE_EXECUTOR_KIND", "FREEZE_OPS", "is_freeze_spec", "validate_freeze_spec"]
