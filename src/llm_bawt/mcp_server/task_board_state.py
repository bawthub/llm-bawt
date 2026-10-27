"""Focus and archive request helpers for the task MCP tools (TASK-1380).

Split out of task_tools.py (monolith gate); task_tools re-imports every name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import httpx

from .task_api import http_error as _http_error


# TASK-1380: the user's focus list. BawtHub owns the ranks; a caller names a
# placement and the server rewrites ranks 1..n in the same transaction as the
# rest of the PATCH. Any actor may focus or reorder; BawtHub records who did.
FocusAction = Literal["on", "off"]
FocusPosition = Literal["top", "bottom", "before", "after"]
_ANCHORED_POSITIONS = frozenset({"before", "after"})


def _focus_patch(
    focus: FocusAction | None,
    position: FocusPosition | None,
    anchor: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """BawtHub ``focus`` PATCH object for the tool args, or an argument error.

    A position alone means focus on. The server owns every rule that depends on
    task state or request shape it can see (anchor is self, before/after needs
    an anchor, anchor not focused); this only rejects arg combinations the tool
    itself makes contradictory.
    """
    if focus is None and position is None and anchor is None:
        return None, None
    if focus == "off":
        if position is not None or anchor is not None:
            return None, "focus='off' takes no focus_position or focus_anchor"
        return {"on": False}, None
    if anchor is not None and position not in _ANCHORED_POSITIONS:
        return None, "focus_anchor needs focus_position='before' or 'after'"
    patch: dict[str, Any] = {"on": True}
    if position is not None:
        patch["placement"] = position
    if anchor is not None:
        patch["anchorId"] = anchor
    return patch, None


def _patch_error(error: httpx.HTTPStatusError) -> dict:
    """Task PATCH error dict; a focus 409 keeps the server's current order."""
    result = _http_error(error)
    if error.response.status_code == 409:
        try:
            body = error.response.json()
        except ValueError:
            body = None
        if isinstance(body, dict) and isinstance(body.get("focused"), list):
            result["focused"] = body["focused"]
    return result


@dataclass(frozen=True)
class _BoardState:
    """A board state BawtHub added after this tool: focus or archive.

    A BawtHub that predates the state silently ignores its PATCH field and list
    filter instead of rejecting them, so success needs proof from the response:
    each task row always carries ``row_key`` (null when the state is off), and
    the list envelope echoes the applied filter as ``echo_key`` (also on an
    empty list, where row checks prove nothing).
    """

    name: str
    patch_key: str
    row_key: str
    echo_key: str

    @property
    def unsupported(self) -> str:
        return f"The BawtHub task API does not support {self.name} yet (no {self.row_key} in its response)"

    def is_on(self, task: dict) -> bool:
        return task[self.row_key] is not None

    def filter_applied(self, payload: Any, value: bool) -> bool:
        """True only when BawtHub confirms it applied the filter we sent."""
        if not isinstance(payload, dict) or payload.get(self.echo_key) is not value:
            return False
        tasks = payload.get("tasks")
        return isinstance(tasks, list) and all(
            isinstance(task, dict) and self.row_key in task and self.is_on(task) is value
            for task in tasks
        )


_FOCUS = _BoardState(name="focus", patch_key="focus", row_key="focusRank", echo_key="focusFilter")
# TASK-1380: archiving takes a finished task off the board. BawtHub allows it
# only for a COMPLETED/FAILED/CANCELLED (resulting) status and clears it on any
# other status, DELETED included; the default list still includes archived tasks.
_ARCHIVE = _BoardState(name="archive", patch_key="archived", row_key="archivedAt", echo_key="archiveFilter")
