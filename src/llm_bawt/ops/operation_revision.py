"""An approval's catalog revision must still be current when admitted/claimed."""
from __future__ import annotations


class OperationChanged(ValueError):
    """No new job/action may use the obsolete catalog revision."""


def changed_reason(approved, current):
    """Both inputs are catalog API dictionaries; metadata edits invalidate too."""
    if current is None or any(
        approved.get(key) != current.get(key)
        for key in ("id", "slug", "version", "script_hash")
    ):
        return ("[operation_changed] Settings changed since this request; "
                "no action submitted. Deny and resubmit for fresh approval.")
    return None


def with_safety_lines(prompt, snapshot):
    operation = snapshot.get("operation") or {}
    if "version" not in operation:
        return prompt
    quiesce = snapshot.get("quiesce")
    lines = [
        f"Operation version: v{operation['version']}",
        f"Quiesce: {'on' if quiesce else 'off'}",
        f"Execution timeout: {snapshot['execution']['timeout_seconds']}s",
    ]
    if quiesce:
        import json

        lines.extend([
            f"Quiesce scope: {json.dumps(quiesce['scope'], sort_keys=True)}",
            f"Quiesce timeout: {quiesce['timeout_seconds']}s",
        ])
    lines.append("Changed operation settings invalidate this approval; request a fresh one.")
    return "\n".join([*([prompt] if prompt else []), *lines])
