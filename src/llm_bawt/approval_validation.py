"""Write/preview validation shared by every approval policy entry point."""
from __future__ import annotations

import re
from typing import Any

from agent_bridge.approval import ApprovalPolicy, MatcherType, PolicyAction, Severity


def validate_policy(data: dict[str, Any]) -> dict[str, Any]:
    """Validate a complete candidate, including disabled rules, without coercion loss."""
    out = dict(data)
    if "order_index" in out:
        if "order" in out and out["order"] != out["order_index"]:
            raise ValueError("order and order_index disagree")
        out["order"] = out.pop("order_index")
    for key, enum, default in (
        ("matcher_type", MatcherType, "always"),
        ("action", PolicyAction, "require_approval"),
        ("severity", Severity, "medium"),
    ):
        value = out.get(key, default)
        if not isinstance(value, str) or value.strip().lower() not in {e.value for e in enum}:
            raise ValueError(f"Invalid {key}: {value!r}")
        out[key] = value.strip().lower()
    for key, default, maximum in (
        ("pattern", "", 8192), ("backend_scope", "*", 64),
        ("tool_name", "*", 128), ("field", "", 128),
    ):
        value = out.get(key, default)
        if not isinstance(value, str) or len(value) > maximum or "\x00" in value:
            raise ValueError(f"{key} must be a string of at most {maximum} characters without NUL")
        if key in ("tool_name", "backend_scope") and not value.strip():
            raise ValueError(f"{key} cannot be blank")
        out[key] = value
    category = out.get("category")
    if category is not None and (not isinstance(category, str) or len(category) > 64 or "\x00" in category):
        raise ValueError("category must be null or a string of at most 64 characters without NUL")
    if out["matcher_type"] != "always" and not out["pattern"]:
        raise ValueError("pattern is required unless matcher_type is always")
    if out["matcher_type"] == "regex":
        try:
            re.compile(out["pattern"])
        except re.error as exc:
            raise ValueError(f"Invalid regular expression: {exc}") from exc
    if "enabled" in out and not isinstance(out["enabled"], bool):
        raise ValueError("enabled must be a boolean")
    if "order" in out and (type(out["order"]) is not int or not -2147483648 <= out["order"] <= 2147483647):
        raise ValueError("order must be a 32-bit integer")
    return out


def candidate_policy(data: dict[str, Any], index: int = 0) -> ApprovalPolicy:
    value = validate_policy(data)
    value.setdefault("id", f"draft-{index:06d}")
    return ApprovalPolicy.from_dict(value)
