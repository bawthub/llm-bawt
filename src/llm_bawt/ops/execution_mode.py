"""Where an operation's worker runs (TASK-1169 S4).

``execution_mode`` is a logical capability on the versioned operation row,
copied into the approval snapshot and shown on the approval card; the worker
never chooses it.

- ``process``: a bounded child process of the ops service (the default for
  operations created from S4 on).
- ``isolated``: a worker separated from the ops service's own lifetime. On the
  Docker executor it is today's one-shot worker container; another executor maps
  it to its own isolation unit (a k8s Job, say). Existing operations were
  backfilled to it, which is what they already did.

Snapshots approved before S4 carry no mode. They are the ``legacy`` shape and run
as their executor kind already did (a Docker job isolated), the semantics they
were approved under, without rewriting their hash.
"""
from __future__ import annotations

from typing import Any, Mapping

EXECUTION_PROCESS = "process"
EXECUTION_ISOLATED = "isolated"
EXECUTION_MODES = (EXECUTION_PROCESS, EXECUTION_ISOLATED)

# What each executor kind can run. Release orchestration never starts a worker,
# so the field is inert there and only the default is accepted.
EXECUTOR_MODES = {"docker": frozenset(EXECUTION_MODES), "release": frozenset({EXECUTION_PROCESS})}

# The column backfill for rows that predate S4, and the default for new rows.
# A Docker job ran in its own worker container; a release job never started one.
LEGACY_EXECUTION_MODES = {"docker": EXECUTION_ISOLATED, "release": EXECUTION_PROCESS}
DEFAULT_EXECUTION_MODE = EXECUTION_PROCESS


def legacy_execution_mode(executor_kind: str) -> str:
    """What a row or snapshot from before S4 already did on ``executor_kind``."""
    try:
        return LEGACY_EXECUTION_MODES[executor_kind]
    except KeyError:
        raise ValueError(f"unknown executor_kind {executor_kind!r}") from None

_CARD_LINES = {
    EXECUTION_PROCESS: "Runs: inside the ops service (child process)",
    EXECUTION_ISOLATED: "Runs: isolated worker",
}
SELF_TARGET_CARD_LINE = "Target is the ops service itself"


def validate_execution_mode(value: Any, executor_kind: str) -> str:
    if value not in EXECUTION_MODES:
        raise ValueError(f"execution_mode must be one of {', '.join(EXECUTION_MODES)}")
    if value not in EXECUTOR_MODES.get(executor_kind, frozenset()):
        raise ValueError(f"executor_kind {executor_kind!r} does not support execution_mode {value!r}")
    return value


def is_legacy_snapshot(snapshot: Mapping[str, Any]) -> bool:
    """A pre-S4 snapshot: no mode in either the execution or the operation copy."""
    execution = snapshot.get("execution") or {}
    operation = snapshot.get("operation") or {}
    return "execution_mode" not in execution and "execution_mode" not in operation


def snapshot_execution_mode(snapshot: Mapping[str, Any], executor_kind: str | None = None) -> str:
    """The mode a snapshot runs in. Raises ``ValueError`` for an incoherent shape.

    Legacy shape → what its executor kind already did. Otherwise the mode must be present, valid and equal
    in the execution and in the operation copy that produced it. An executor
    reading its own job passes its ``executor_kind``.
    """
    execution = snapshot.get("execution") or {}
    operation = snapshot.get("operation") or {}
    if is_legacy_snapshot(snapshot):
        return legacy_execution_mode(executor_kind or execution.get("executor_kind"))
    if "execution_mode" not in execution or "execution_mode" not in operation:
        raise ValueError("snapshot execution_mode present in only one of execution/operation")
    mode = execution["execution_mode"]
    if mode not in EXECUTION_MODES:
        raise ValueError("snapshot execution_mode is unknown")
    if operation["execution_mode"] != mode:
        raise ValueError("snapshot execution execution_mode mismatch")
    return mode


def card_lines(snapshot: Mapping[str, Any], *, self_target: bool = False) -> list[str]:
    """The approval card's lines for where the job runs."""
    lines = [_CARD_LINES[snapshot_execution_mode(snapshot)]]
    if self_target:
        lines.append(SELF_TARGET_CARD_LINE)
    return lines


def with_card_lines(prompt: str | None, snapshot: Mapping[str, Any]) -> str | None:
    """Append where the job runs to its approval card (Docker jobs only).

    A release job never starts a worker, so the line would say nothing true.
    """
    if (snapshot.get("execution") or {}).get("executor_kind") != "docker":
        return prompt
    lines = "\n".join(card_lines(snapshot, self_target=bool(snapshot.get("self_target"))))
    return f"{prompt}\n{lines}" if prompt else lines
