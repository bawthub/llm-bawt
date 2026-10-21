"""State-aware liveness watchdog for Claude Agent SDK turns.

The SDK emits a tool-use message before executing a tool and its tool-result
message only after execution finishes. A fixed per-message timeout therefore
cannot distinguish a wedged CLI from a legitimate long-running tool. This
module tracks that SDK-visible lifecycle: ordinary tools use their declared
runtime limit, while Agent workers use the same limit as an inactivity window
that refreshes whenever their nested SDK activity arrives.

TASK-1148: a resumed worker (``SendMessage``) that exceeds that inactivity
window is stopped on its own; its call returns an unsuccessful tool result and
the parent continues. Only an unacknowledged stop, an SDK-idle CLI, or a
non-containable tool still ends the parent turn.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

from claude_agent_sdk.types import (
    AssistantMessage,
    TaskProgressMessage,
    TaskNotificationMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from .worker_liveness import WorkerLiveness, WorkerStop

HealthCallback = Callable[[dict[str, Any]], None]
StopTask = Callable[[str], Awaitable[None]]

logger = logging.getLogger("claude_code_bridge.bridge")

DEFAULT_HEALTH_INTERVAL_SECONDS = 30.0
DEFAULT_TOOL_GRACE_SECONDS = 30.0
MAX_TOOL_TIMEOUT_SECONDS = 600.0


class TurnWatchdogTimeout(TimeoutError):
    """A phase-aware timeout from :class:`TurnWatchdog`."""

    def __init__(self, message: str, *, phase: str) -> None:
        super().__init__(message)
        self.phase = phase

    @property
    def retry_fresh_session(self) -> bool:
        """Only an idle SDK can safely use the existing fresh-session retry."""
        return self.phase == "sdk_idle"


@dataclass(frozen=True)
class ActiveTool:
    tool_use_id: str
    name: str
    started_at: float
    timeout_seconds: float
    last_activity_at: float
    # TASK-1095: any tool that hosts a sub-agent run (Agent, Workflow, or a
    # SendMessage resuming a stopped agent) is judged on inactivity.
    hosts_subagent: bool = False
    # TASK-1148: set when the bridge has asked the SDK to stop this tool's
    # worker; the tool then has one grace window to deliver its result.
    stop_requested_at: float | None = None
    # The CLI reported the stopped task terminal; its tool result may still be
    # buffered behind siblings like any other result.
    stop_acknowledged: bool = False

    @property
    def stop_outstanding(self) -> bool:
        return self.stop_requested_at is not None and not self.stop_acknowledged

    @property
    def progress_aware(self) -> bool:
        """Long-running orchestrators expose nested SDK activity."""
        return self.hosts_subagent or self.name in {"Agent", "Workflow"}

    @property
    def deadline_base(self) -> float:
        """Orchestrators time out on inactivity; other tools on total runtime."""
        return self.last_activity_at if self.progress_aware else self.started_at

    def deadline(self, tool_grace: float) -> float:
        if self.stop_requested_at is not None:
            return self.stop_requested_at + tool_grace
        return self.deadline_base + self.timeout_seconds + tool_grace


class TurnWatchdog:
    """Wait for SDK messages without misclassifying active tools as hangs."""

    def __init__(
        self,
        idle_timeout: float,
        *,
        health_interval: float = DEFAULT_HEALTH_INTERVAL_SECONDS,
        tool_grace: float = DEFAULT_TOOL_GRACE_SECONDS,
        max_tool_timeout: float = MAX_TOOL_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        on_health: HealthCallback | None = None,
        resolve_parent: Callable[[str], str | None] | None = None,
        liveness: WorkerLiveness | None = None,
        stop_task: StopTask | None = None,
    ) -> None:
        if idle_timeout <= 0:
            raise ValueError("idle_timeout must be positive")
        if health_interval <= 0:
            raise ValueError("health_interval must be positive")
        if tool_grace < 0:
            raise ValueError("tool_grace cannot be negative")
        if max_tool_timeout <= 0:
            raise ValueError("max_tool_timeout must be positive")
        self._idle_timeout = float(idle_timeout)
        self._health_interval = float(health_interval)
        self._tool_grace = float(tool_grace)
        self._max_tool_timeout = float(max_tool_timeout)
        self._clock = clock
        self._on_health = on_health
        # Shared spawn → live-host resolution (SubagentUsage.host_parent), so
        # resumed-agent activity keeps the RUNNING host alive, not its old spawn.
        self._resolve_parent = resolve_parent
        self._active_tools: dict[str, ActiveTool] = {}
        # Worker containment needs both the ledger and a way to stop a task;
        # without either, every expiry stays parent-terminal (old behaviour).
        self._liveness = liveness
        self._stop_task = stop_task if liveness is not None else None

    @property
    def active_tools(self) -> tuple[ActiveTool, ...]:
        return tuple(self._active_tools.values())

    def observe(self, message: Any) -> None:
        """Update tool state and refresh active Agent workers on child activity."""
        now = self._clock()
        if self._liveness is not None:
            if isinstance(message, TaskStartedMessage):
                self._liveness.task_started(message.tool_use_id, message.task_id)
            elif isinstance(message, (TaskUpdatedMessage, TaskNotificationMessage)):
                acked = self._liveness.task_status(
                    message.task_id, message.status, getattr(message, "tool_use_id", None),
                )
                tool = self._active_tools.get(acked.tool_use_id) if acked else None
                if tool is not None and tool.stop_requested_at is not None:
                    self._active_tools[tool.tool_use_id] = replace(tool, stop_acknowledged=True)
        parent_tool_use_id = getattr(message, "parent_tool_use_id", None)
        if not parent_tool_use_id and isinstance(
            message, (TaskStartedMessage, TaskProgressMessage)
        ):
            parent_tool_use_id = message.tool_use_id
        if parent_tool_use_id and self._resolve_parent is not None:
            parent_tool_use_id = self._resolve_parent(parent_tool_use_id)
        if parent_tool_use_id:
            parent = self._active_tools.get(parent_tool_use_id)
            if parent is not None and isinstance(message, TaskStartedMessage):
                parent = replace(parent, hosts_subagent=True)
                self._active_tools[parent_tool_use_id] = parent
            if parent is not None and parent.progress_aware:
                self._active_tools[parent_tool_use_id] = replace(
                    parent, last_activity_at=now
                )

        if isinstance(message, AssistantMessage):
            for block in message.content:
                if not isinstance(block, ToolUseBlock):
                    continue
                self._active_tools[block.id] = ActiveTool(
                    tool_use_id=block.id,
                    name=block.name,
                    started_at=now,
                    timeout_seconds=self._tool_timeout(block.name, block.input),
                    last_activity_at=now,
                )
        elif isinstance(message, UserMessage) and isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    self._active_tools.pop(block.tool_use_id, None)

    async def receive_next(
        self,
        message_stream: AsyncIterator[Any],
        *,
        request_id: str,
        session_key: str,
    ) -> Any:
        """Return the next SDK message or raise when the current phase wedges.

        One persistent ``__anext__`` task is shielded across periodic health
        checks. This is important: repeatedly timing out ``__anext__`` directly
        would cancel the SDK stream read and could corrupt the iterator.
        """
        receive_task = asyncio.create_task(message_stream.__anext__())
        wait_started_at = self._clock()
        try:
            while True:
                await self._contain_expired_workers(
                    request_id=request_id, session_key=session_key,
                )
                now = self._clock()
                deadline, phase = self._deadline(wait_started_at)
                remaining = deadline - now
                if remaining <= 0:
                    logger.warning(
                        "turn_watchdog_failure request_id=%s session=%s phase=%s "
                        "wait_elapsed_ms=%.0f active_tools=%s",
                        request_id, session_key, phase,
                        (now - wait_started_at) * 1000,
                        ",".join(tool.name for tool in self._active_tools.values()) or "none",
                    )
                    raise TurnWatchdogTimeout(
                        self._timeout_message(phase),
                        phase=phase,
                    )

                done, _ = await asyncio.wait(
                    {receive_task},
                    timeout=min(
                        self._health_interval,
                        remaining,
                        self._next_containment_in(now),
                    ),
                )
                if receive_task in done:
                    message = receive_task.result()
                    self.observe(message)
                    return message

                self._log_health(
                    request_id=request_id,
                    session_key=session_key,
                    phase=phase,
                    remaining=max(0.0, deadline - self._clock()),
                    wait_elapsed=self._clock() - wait_started_at,
                )
        finally:
            if not receive_task.done():
                receive_task.cancel()
                await asyncio.gather(receive_task, return_exceptions=True)

    def _containable(self, tool: ActiveTool) -> str | None:
        if self._stop_task is None or tool.stop_requested_at is not None:
            return None
        return self._liveness.containable(tool.tool_use_id, tool.name)

    def _next_containment_in(self, now: float) -> float:
        """Seconds until the earliest containable worker expires."""
        pending = [
            tool.deadline(self._tool_grace) - now
            for tool in self._active_tools.values()
            if self._containable(tool)
        ]
        return max(0.0, min(pending)) if pending else float("inf")

    async def _contain_expired_workers(self, *, request_id: str, session_key: str) -> None:
        """Stop each expired resumed worker instead of failing the parent turn."""
        for tool in list(self._active_tools.values()):
            task_id = self._containable(tool)
            now = self._clock()
            if task_id is None or now < tool.deadline(self._tool_grace):
                continue
            stop = WorkerStop(
                tool_use_id=tool.tool_use_id,
                task_id=task_id,
                tool_name=tool.name,
                idle_seconds=now - tool.last_activity_at,
                limit_seconds=tool.timeout_seconds,
                requested_at=now,
            )
            # Record first: the PostToolUse hook and task events triggered by
            # the stop must already see why the worker ended.
            self._liveness.record_stop(stop)
            self._active_tools[tool.tool_use_id] = replace(tool, stop_requested_at=now)
            logger.warning(
                "worker_liveness_stop request_id=%s session=%s tool=%s tool_use_id=%s "
                "task_id=%s idle_s=%.0f limit_s=%.0f runtime_s=%.0f",
                request_id, session_key, tool.name, tool.tool_use_id, task_id,
                stop.idle_seconds, stop.limit_seconds, now - tool.started_at,
            )
            try:
                await asyncio.wait_for(
                    self._stop_task(task_id), timeout=max(self._tool_grace, 1.0)
                )
            except Exception as exc:
                logger.warning(
                    "worker_liveness_stop_failed request_id=%s session=%s "
                    "tool_use_id=%s task_id=%s error=%s",
                    request_id, session_key, tool.tool_use_id, task_id,
                    type(exc).__name__,
                )
                raise TurnWatchdogTimeout(
                    f"Could not stop timed-out worker {task_id} for "
                    f"{tool.name}({tool.tool_use_id}): {type(exc).__name__}",
                    phase="tool_running",
                ) from exc

    def _deadline(self, wait_started_at: float) -> tuple[float, str]:
        if not self._active_tools:
            return wait_started_at + self._idle_timeout, "sdk_idle"
        deadlines = [
            tool.deadline(self._tool_grace) for tool in self._active_tools.values()
        ]
        # The CLI can buffer sibling results behind a long-running Agent.
        # When that Agent completes, an already-expired sibling must not stop
        # us from reading its queued result. Allow one bounded drain window
        # per SDK read; a genuinely silent tool still expires after this grace.
        deadline = max(max(deadlines), wait_started_at + self._tool_grace)
        # TASK-1148: an unacknowledged worker stop gets one absolute grace
        # window; sibling deadlines and unrelated activity cannot extend it.
        outstanding = [
            tool.stop_requested_at + self._tool_grace
            for tool in self._active_tools.values() if tool.stop_outstanding
        ]
        if outstanding:
            deadline = min(deadline, min(outstanding))
        return deadline, "tool_running"

    def _timeout_message(self, phase: str) -> str:
        if phase == "sdk_idle":
            return f"No SDK messages for {self._idle_timeout}s — CLI may be hung"
        tools = ", ".join(
            f"{tool.name}({tool.tool_use_id})"
            + (" [worker stop unacknowledged]" if tool.stop_outstanding else "")
            for tool in self._active_tools.values()
        )
        return (
            "No SDK tool result before the active-tool deadline "
            f"(tools: {tools or 'unknown'})"
        )

    def _log_health(
        self,
        *,
        request_id: str,
        session_key: str,
        phase: str,
        remaining: float,
        wait_elapsed: float = 0.0,
    ) -> None:
        if phase == "tool_running":
            now = self._clock()
            details = ",".join(
                f"{tool.name}:runtime={now - tool.started_at:.0f}s/"
                f"idle={now - tool.last_activity_at:.0f}s/"
                f"limit={tool.timeout_seconds:.0f}s"
                for tool in self._active_tools.values()
            )
        else:
            details = "none"
        logger.info(
            "Turn health: request_id=%s session=%s phase=%s "
            "active_tools=%s deadline_in=%.1fs",
            request_id,
            session_key,
            phase,
            details,
            remaining,
        )
        if self._on_health is not None:
            try:
                self._on_health(
                    {
                        "phase": phase,
                        "active_tools": [
                            {
                                "tool_use_id": tool.tool_use_id,
                                "name": tool.name,
                                "elapsed_seconds": round(
                                    self._clock() - tool.started_at, 1
                                ),
                                "idle_seconds": round(
                                    self._clock() - tool.last_activity_at, 1
                                ),
                                "timeout_seconds": tool.timeout_seconds,
                            }
                            for tool in self._active_tools.values()
                        ],
                        "deadline_in_seconds": round(remaining, 1),
                        "wait_elapsed_seconds": round(wait_elapsed, 1),
                    }
                )
            except Exception:
                logger.warning(
                    "Failed to publish turn health: request_id=%s session=%s",
                    request_id,
                    session_key,
                    exc_info=True,
                )

    def _tool_timeout(self, tool_name: str, tool_input: dict[str, Any]) -> float:
        if tool_name != "Bash":
            return self._idle_timeout
        raw_timeout = tool_input.get("timeout")
        if isinstance(raw_timeout, bool) or not isinstance(raw_timeout, (int, float)):
            return self._idle_timeout
        timeout_seconds = float(raw_timeout) / 1000.0
        if timeout_seconds <= 0:
            return self._idle_timeout
        return min(timeout_seconds, self._max_tool_timeout)
