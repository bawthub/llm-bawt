"""Tool execution-start stamps for one streamed turn (TASK-1095 #3).

A tool card used to read "running" from the moment the model emitted the call,
so queue wait behind a foreground sub-agent counted as run time. The bridge's
PreToolUse hook now reports when a call really starts executing; this class turns
that report into a ``tool_executing`` tool_event keyed to the call's ``call_id``.

The hook can outrun its own TOOL_START over the event stream, so a stamp for a
call the turn has not announced yet is held (bounded) and released by the start.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from .pending_tool_calls import PendingToolCallCorrelator


class ToolExecutionStamps:
    MAX_HELD = 256

    def __init__(
        self,
        calls: PendingToolCallCorrelator,
        publish: Callable[[dict], Any],
        envelope: dict[str, Any],
    ) -> None:
        self._calls = calls
        self._publish = publish
        self._envelope = envelope
        self._held: dict[str, dict] = {}

    def on_executing(self, item: dict) -> None:
        tool_use_id = item.get("tool_use_id")
        call_id = self._calls.call_id_for(tool_use_id)
        if call_id:
            self._emit(item, call_id)
        elif tool_use_id and len(self._held) < self.MAX_HELD:
            self._held[tool_use_id] = item

    def on_tool_started(self, tool_use_id: str | None, call_id: str) -> None:
        held = self._held.pop(tool_use_id or "", None)
        if held is not None:
            self._emit(held, call_id)

    def _emit(self, item: dict, call_id: str) -> None:
        self._publish({
            "_type": "tool_event",
            "event": "tool_executing",
            **self._envelope,
            "tool_name": item.get("name") or "",
            "call_id": call_id,
            "tool_use_id": item.get("tool_use_id"),
            "ts": item.get("ts") or time.time(),
        })
