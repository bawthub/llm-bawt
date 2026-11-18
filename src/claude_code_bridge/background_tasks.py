"""Hold a subagent's "done" while work it launched is still running (TASK-2367).

The CLI sends ``task_notification`` (status ``completed``) when a background
subagent ends a turn to wait on its own background work (a backgrounded shell,
monitor or nested agent), then resumes and notifies again when it truly ends.
The frontend treats done as terminal, so the first notification would leave the
card "Complete" while the agent still works.

``system/background_tasks_changed`` is a level signal: ``tasks`` is the full set
of live background tasks, each with ``task_id`` and an optional
``parent_task_id`` (the subagent task that launched it). A notifying subagent
with live children is deferred; its done is released by its next notification
or, if none comes, once neither it nor any of its children is still live.
"""
from __future__ import annotations

from typing import Any

LEVEL_SUBTYPE = "background_tasks_changed"


class BackgroundTaskSet:
    """Request-local view of the CLI's live background tasks."""

    def __init__(self) -> None:
        self._live: set[str] = set()
        self._children: dict[str, set[str]] = {}
        self._deferred: dict[str, Any] = {}
        self._provisional: set[str] = set()

    def observe_level(self, data: Any) -> list[Any]:
        """Replace the live set; return deferred notifications now released."""
        tasks = data.get("tasks") if isinstance(data, dict) else None
        if not isinstance(tasks, list):
            return []
        live: set[str] = set()
        children: dict[str, set[str]] = {}
        for entry in tasks:
            if not isinstance(entry, dict) or not entry.get("task_id"):
                continue
            task_id = str(entry["task_id"])
            live.add(task_id)
            parent = entry.get("parent_task_id")
            if parent:
                children.setdefault(str(parent), set()).add(task_id)
        self._live, self._children = live, children
        released = [
            task_id for task_id in self._deferred
            if task_id not in live and not children.get(task_id)
        ]
        return [self._deferred.pop(task_id) for task_id in released]

    def discard(self, task_id: str) -> None:
        """Drop a held completion: a failed/stopped notification is final."""
        self._deferred.pop(task_id, None)

    def hold_update(self, task_id: str) -> None:
        """Hold a ``task_updated`` completed until it can be told apart.

        The CLI also sends one when a subagent pauses on its own background
        work, possibly before the level frame names the child.
        """
        if task_id:
            self._provisional.add(task_id)

    def settle(self, task_id: str) -> None:
        """A notification decides the task's outcome; drop its held update."""
        self._provisional.discard(task_id)

    def release_updates(self) -> list[str]:
        """Held updates whose task and children have left the live set."""
        released = [
            task_id for task_id in self._provisional
            if task_id not in self._live and not self._children.get(task_id)
            and task_id not in self._deferred
        ]
        self._provisional.difference_update(released)
        return released

    def defer_done(self, task_id: str, notification: Any) -> bool:
        """Hold ``notification`` while ``task_id`` has live children."""
        if not task_id or not self._children.get(task_id):
            self._deferred.pop(task_id, None)
            return False
        self._deferred[task_id] = notification
        return True
