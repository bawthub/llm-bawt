"""Shared bridge hooks for changed-file attribution."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from .changed_files import ToolChangedFileCapture
from .events import AgentEvent, AgentEventKind
from .turn_snapshots import TurnSnapshotObserver, file_identity

logger = logging.getLogger(__name__)

# A start snapshot slower than this lets the tool run anyway; the tree's start
# then may include that tool's effects (under-attribution, never misattribution).
TURN_SNAPSHOT_HOOK_TIMEOUT_SECONDS = 10.0
_TERMINAL_KINDS = frozenset({AgentEventKind.ASSISTANT_DONE, AgentEventKind.ERROR})
# Trees a session's previous turn touched, pre-observed at the next turn's
# start by bridges whose tool notifications race execution (Codex).
CARRIED_TREES_PER_SESSION = 4


def turn_snapshots_enabled() -> bool:
    return os.getenv("AGENT_TURN_SNAPSHOTS", "1").strip().lower() not in ("0", "false", "off", "no")


class ChangedFileLifecycleMixin:
    """Attach changed-file capture to standardized bridge events.

    Edit/Write-style tools are captured per call: the declared ``file_path`` is
    snapshotted and a FILE_CHANGED event is emitted as soon as the tool ends.
    Shell-command edits (TASK-1218) come from per-turn Git snapshots of the
    trees a turn touched; bridges that call ``_observe_tool_for_turn_snapshot``
    from a pre-execution hook get them as FILE_CHANGED events published just
    before the turn's terminal event.

    Bridges without a blocking pre-execution hook (Codex runs with
    ``approvalPolicy=never``; its ``item/started`` notification races the
    command) also call ``_begin_turn_snapshot`` before the turn starts.  That
    pre-observes the trees the session's previous turn touched, so only the
    first command in a newly touched tree can race its own start snapshot,
    which loses that command's effects (under-attribution) but never
    misattributes.
    """

    async def _begin_turn_snapshot(self, request_id: str, session_key: str | None) -> None:
        """Turn-start hook: take start snapshots of the session's carried trees."""
        carried = getattr(self, "_turn_snapshot_carried", {}).get(session_key or "")
        for root in list(carried or ()):
            await self._observe_tool_for_turn_snapshot(
                request_id, session_key, {"tool_name": "turn_start", "tool_input": {}, "cwd": root},
            )

    def _carry_turn_snapshot_trees(self, session_key: str | None, receipts: list[dict[str, Any]]) -> None:
        if not session_key:
            return
        # Tree receipts only (timing entries carry no status); skip trees that
        # failed or were too large to snapshot.
        roots = [
            str(r["tree"]) for r in receipts
            if r.get("tree") and r.get("status")
            and r["status"] != "error" and not str(r["status"]).startswith("skipped")
        ]
        if not roots:
            return
        carried = getattr(self, "_turn_snapshot_carried", None)
        if carried is None:
            carried = {}
            self._turn_snapshot_carried = carried
        merged = list(dict.fromkeys([*roots, *carried.get(session_key, [])]))
        carried[session_key] = merged[:CARRIED_TREES_PER_SESSION]

    def _turn_snapshot_observer(self) -> TurnSnapshotObserver:
        observer = getattr(self, "_turn_snapshots", None)
        if observer is None:
            observer = TurnSnapshotObserver()
            self._turn_snapshots = observer
        return observer

    async def _observe_tool_for_turn_snapshot(
        self, request_id: str, session_key: str | None, input_data: Any,
    ) -> None:
        """Pre-execution hook: make the trees this tool references candidates."""
        if not request_id or not turn_snapshots_enabled() or not isinstance(input_data, dict):
            return
        tool_name = str(input_data.get("tool_name") or "")
        if tool_name.startswith("mcp__"):
            return
        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    self._turn_snapshot_observer().observe_tool,
                    request_id=request_id,
                    # Session keys are "<bot>:<user>..."; the bot slug is the
                    # commit author a HEAD advance must match.
                    bot_id=(session_key or "").split(":", 1)[0] or None,
                    tool_input=input_data.get("tool_input"),
                    cwd=input_data.get("cwd"),
                ),
                timeout=TURN_SNAPSHOT_HOOK_TIMEOUT_SECONDS,
            )
        except Exception:
            logger.debug("turn snapshot observe failed request_id=%s", request_id, exc_info=True)

    def _command_file_events(self, request_id: str, terminal: AgentEvent) -> list[AgentEvent]:
        """End-of-turn snapshots as FILE_CHANGED events (before the terminal)."""
        try:
            covered = [
                identity
                for path in self._changed_file_capture().captured_paths(request_id)
                if (identity := file_identity(path)) is not None
            ]
            files, receipts = self._turn_snapshot_observer().finish(request_id, covered=covered)
            self._carry_turn_snapshot_trees(terminal.session_key, receipts)
        except Exception:
            logger.warning("turn snapshot finish failed request_id=%s", request_id, exc_info=True)
            return []
        return [
            AgentEvent(
                event_id=f"{terminal.event_id}:command-file:{index}",
                session_key=terminal.session_key,
                run_id=terminal.run_id,
                kind=AgentEventKind.FILE_CHANGED,
                origin="system",
                tool_name="Bash",
                tool_arguments={},
                tool_use_id=None,
                seq=terminal.seq,
                timestamp=terminal.timestamp,
                raw={"file": file_data},
                provider=terminal.provider,
                trigger_message_id=terminal.trigger_message_id,
            )
            for index, file_data in enumerate(files)
        ]

    def _changed_file_capture(self) -> ToolChangedFileCapture:
        capture = getattr(self, "_tool_changed_file_capture", None)
        if capture is None:
            capture = ToolChangedFileCapture(cwd=getattr(self, "_cwd", "/app"))
            self._tool_changed_file_capture = capture
        return capture

    def _capture_changed_file_event(
        self,
        event: AgentEvent,
        *,
        request_id: str | None = None,
    ) -> AgentEvent | None:
        try:
            capture = self._changed_file_capture()
            capture_request_id = request_id or event.run_id or ""
            if event.kind == AgentEventKind.TOOL_START:
                capture.start(
                    request_id=capture_request_id,
                    tool_use_id=event.tool_use_id,
                    tool_name=event.tool_name,
                    arguments=event.tool_arguments,
                )
                return None
            if event.kind != AgentEventKind.TOOL_END:
                return None
            finished = capture.finish(
                request_id=capture_request_id,
                tool_use_id=event.tool_use_id,
                tool_name=event.tool_name,
            )
            if finished is None:
                return None
            tool_name, arguments, file_data = finished
            return AgentEvent(
                event_id=f"{event.event_id}:file-changed",
                session_key=event.session_key,
                run_id=event.run_id,
                kind=AgentEventKind.FILE_CHANGED,
                origin="system",
                tool_name=tool_name,
                tool_arguments=arguments,
                tool_use_id=event.tool_use_id,
                parent_tool_use_id=event.parent_tool_use_id,
                seq=event.seq,
                timestamp=event.timestamp,
                raw={"file": file_data},
                provider=event.provider,
                trigger_message_id=event.trigger_message_id,
            )
        except Exception:
            logger.debug("changed-file tool attribution failed", exc_info=True)
            return None

    def _publish_run_event_with_changed_file(
        self,
        request_id: str,
        event: AgentEvent,
    ) -> None:
        """Capture before TOOL_START; publish FILE_CHANGED after TOOL_END."""
        if event.kind == AgentEventKind.TOOL_START:
            self._capture_changed_file_event(event, request_id=request_id)
            self._publisher.publish_run_event(request_id, event)
            return
        if event.kind in _TERMINAL_KINDS:
            for file_event in self._command_file_events(request_id, event):
                self._publisher.publish_run_event(request_id, file_event)
        self._publisher.publish_run_event(request_id, event)
        file_event = self._capture_changed_file_event(event, request_id=request_id)
        if file_event is not None:
            self._publisher.publish_run_event(request_id, file_event)

    def _discard_changed_file_request(self, request_id: str) -> None:
        try:
            self._changed_file_capture().discard_request(request_id)
            self._turn_snapshot_observer().discard(request_id)
        except Exception:
            logger.debug("changed-file request cleanup failed", exc_info=True)
