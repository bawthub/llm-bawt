from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PendingToolCall:
    call_id: str
    tool_name: str
    tool_use_id: str | None


class PendingToolCallCorrelator:
    """Pair tool results with starts without confusing parallel same-name calls."""

    def __init__(self) -> None:
        self._calls: list[PendingToolCall] = []

    def start(
        self,
        *,
        call_id: str,
        tool_name: str,
        tool_use_id: str | None,
    ) -> None:
        self._calls.append(PendingToolCall(call_id, tool_name, tool_use_id))

    def call_id_for(self, tool_use_id: str | None) -> str:
        """Pending call's call_id by SDK id without consuming it (TASK-1095)."""
        if tool_use_id:
            for call in reversed(self._calls):
                if call.tool_use_id == tool_use_id:
                    return call.call_id
        return ""

    def finish(self, *, tool_name: str, tool_use_id: str | None) -> str:
        if tool_use_id:
            for index in range(len(self._calls) - 1, -1, -1):
                if self._calls[index].tool_use_id == tool_use_id:
                    return self._calls.pop(index).call_id
            return ""

        if tool_name:
            for index in range(len(self._calls) - 1, -1, -1):
                if self._calls[index].tool_name == tool_name:
                    return self._calls.pop(index).call_id

        return self._calls.pop().call_id if self._calls else ""
