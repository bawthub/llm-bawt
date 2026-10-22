"""Contain a stalled resumed sub-agent to its own SendMessage call (TASK-1148).

A ``SendMessage`` that resumes a stopped agent runs that worker inline; the
CLI serialises sibling resumes behind it. When the worker goes silent past
its inactivity limit, the bridge stops only that worker's task
(``ClaudeSDKClient.stop_task``). The CLI then hands ``SendMessage`` an
unsuccessful tool result, the parent model continues, and the next queued
resume starts. The parent turn ends only on a genuine parent-level failure.

This ledger is request-local state shared by the watchdog (which decides and
issues the stop), the sub-agent lifecycle emitter (which labels the terminal
event ``timed_out`` rather than a user stop), the tool-result emitter (which
marks the card failed), and the PostToolUse hook (which tells the parent model
why). Diagnostics are ids, durations, and limits only; no prompts or payloads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("claude_code_bridge.bridge")

# Tool calls whose hosted worker the bridge may stop without ending the parent.
# Agent spawns keep the parent-deadline behaviour: the CLI reports a stopped
# spawn as a user interrupt and drops the explanatory hook context.
CONTAINABLE_HOST_TOOLS = frozenset({"SendMessage"})
TERMINAL_TASK_STATUSES = frozenset({"completed", "failed", "stopped", "killed"})
TIMED_OUT_STATUS = "timed_out"


@dataclass(frozen=True)
class WorkerStop:
    """Why the bridge stopped one worker; safe to log and publish."""

    tool_use_id: str
    task_id: str
    tool_name: str
    idle_seconds: float
    limit_seconds: float
    requested_at: float
    # The call whose limit was binding (TASK-1148 R2): the host itself when
    # its inactivity window ran out, or a descendant (e.g. a Bash call with a
    # declared timeout) that outlived it. None = the host itself.
    limit_tool_name: str | None = None
    limit_tool_use_id: str | None = None

    @property
    def _limit_is_descendant(self) -> bool:
        return bool(self.limit_tool_use_id) and self.limit_tool_use_id != self.tool_use_id

    @property
    def reason(self) -> str:
        if self._limit_is_descendant:
            return (
                f"had no sub-agent activity for {self.idle_seconds:.0f}s and its "
                f"{self.limit_tool_name} call passed its {self.limit_seconds:.0f}s "
                "limit; stopped by the bridge"
            )
        return (
            f"had no sub-agent activity for {self.idle_seconds:.0f}s "
            f"(liveness limit {self.limit_seconds:.0f}s); stopped by the bridge"
        )

    def diagnostic(self) -> dict:
        return {
            "status": TIMED_OUT_STATUS,
            "stopped_by": "bridge_liveness",
            "reason": self.reason,
            "tool_use_id": self.tool_use_id,
            "task_id": self.task_id,
            "idle_seconds": round(self.idle_seconds, 1),
            "limit_seconds": round(self.limit_seconds, 1),
            "limit_tool_name": self.limit_tool_name or self.tool_name,
            "limit_tool_use_id": self.limit_tool_use_id or self.tool_use_id,
        }

    def parent_note(self) -> str:
        return (
            f"[Bridge worker liveness] The resumed agent (task {self.task_id}) "
            f"{self.reason}. This SendMessage did not complete: treat the worker "
            "as timed out, not as finished. Its partial work may or may not have "
            "been applied; verify before relying on it or retrying."
        )


class WorkerLiveness:
    """Request-local map of host tool call → live task, plus bridge stops.

    A stop belongs to one host-call *attempt*, not to the task: the same task
    can be resumed again by a later ``SendMessage``. Task-only events resolve
    against the task's current attempt; events that name their host call
    resolve against that call's own stop (late notifications for an old
    attempt stay attributed to it).
    """

    def __init__(self) -> None:
        self._live_task_by_tool: dict[str, str] = {}
        self._task_by_attempt: dict[str, str] = {}  # every attempt ever started
        self._attempt_by_task: dict[str, str] = {}  # the task's current attempt
        self._stops_by_tool: dict[str, WorkerStop] = {}

    def task_started(self, tool_use_id: str | None, task_id: str | None) -> None:
        if tool_use_id and task_id:
            self._live_task_by_tool[tool_use_id] = task_id
            self._task_by_attempt[tool_use_id] = task_id
            self._attempt_by_task[task_id] = tool_use_id

    def attempt_for(self, task_id: str | None, tool_use_id: str | None = None) -> str | None:
        """The host call a task event belongs to.

        An event naming a known attempt of this task belongs to that attempt
        (a late notification for an old resume must not touch the current
        one). Only an event without a usable host id (task_updated) falls back
        to the task's current attempt.
        """
        if tool_use_id and task_id and self._task_by_attempt.get(tool_use_id) == task_id:
            return tool_use_id
        return self._attempt_by_task.get(task_id or "")

    def task_status(
        self, task_id: str | None, status: str | None, tool_use_id: str | None = None,
    ) -> WorkerStop | None:
        """Track a terminal status for one attempt; return the stop it acknowledges."""
        if not task_id or status not in TERMINAL_TASK_STATUSES:
            return None
        attempt = self.attempt_for(task_id, tool_use_id)
        if attempt is None:
            return None
        if self._live_task_by_tool.get(attempt) == task_id:
            del self._live_task_by_tool[attempt]
        stop = self._stop_of(attempt, task_id)
        if stop is not None:
            logger.info(
                "worker_liveness_stop_ack tool_use_id=%s task_id=%s status=%s",
                stop.tool_use_id, task_id, status,
            )
        return stop

    def live_task(self, tool_use_id: str) -> str | None:
        return self._live_task_by_tool.get(tool_use_id)

    def containable(self, tool_use_id: str, tool_name: str) -> str | None:
        """The task to stop for an expired tool, or None if the parent must fail."""
        if tool_name not in CONTAINABLE_HOST_TOOLS or tool_use_id in self._stops_by_tool:
            return None
        return self._live_task_by_tool.get(tool_use_id)

    def record_stop(self, stop: WorkerStop) -> None:
        self._stops_by_tool[stop.tool_use_id] = stop
        self._task_by_attempt.setdefault(stop.tool_use_id, stop.task_id)
        self._attempt_by_task.setdefault(stop.task_id, stop.tool_use_id)

    def _stop_of(self, attempt: str | None, task_id: str | None) -> WorkerStop | None:
        stop = self._stops_by_tool.get(attempt or "")
        return stop if stop is not None and stop.task_id == task_id else None

    def stop_for_task(self, task_id: str | None) -> WorkerStop | None:
        """The bridge stop of the task's *current* attempt, if any."""
        return self._stop_of(self.attempt_for(task_id), task_id)

    def stop_for_tool(self, tool_use_id: str | None) -> WorkerStop | None:
        return self._stops_by_tool.get(tool_use_id or "")

    def stop_status(
        self, task_id: str | None, status: str | None, tool_use_id: str | None = None,
    ) -> tuple[str | None, dict | None]:
        """Relabel a bridge-caused stop as ``timed_out`` with its diagnostic."""
        if status not in {"stopped", "killed"}:
            return status, None
        stop = self._stop_of(self.attempt_for(task_id, tool_use_id), task_id)
        if stop is None:
            return status, None
        return TIMED_OUT_STATUS, stop.diagnostic()
