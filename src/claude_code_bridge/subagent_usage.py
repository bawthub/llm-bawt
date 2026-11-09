"""Measured subagent usage, separate from the SDK's mutable progress estimate.

TaskProgress.total_tokens is latest input + accumulated output + a streaming
estimate in CLI 2.1.280; TaskNotification can instead contain last-request usage.
Neither is a cumulative consumption counter. Account provider usage by response
ID, scoped to the spawning tool (a Workflow may have several child tasks).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


class SubagentUsage:
    """Request-local ledger. Never sum progress snapshots or infer missing usage."""

    TOKEN_FIELDS = (
        "input_tokens", "output_tokens", "cache_creation_input_tokens",
        "cache_read_input_tokens",
    )

    def __init__(self, spawn_lookup: Callable[[str], str | None] | None = None) -> None:
        # TASK-1148 R2: durable task → original-spawn lookup (the CLI session
        # transcript) for agents spawned in an earlier turn or bridge process.
        self._spawn_lookup = spawn_lookup
        self._single_agent_spawns: set[str] = set()  # Agent spawns found in the transcript
        self._responses: dict[str, dict[str, dict[str, int]]] = {}
        self._tasks: dict[str, dict[str, str]] = {}
        self._last_live_total: dict[str, tuple[int, bool]] = {}
        # TASK-2169: stream-event accounting (see observe_stream_event).
        self._open_responses: dict[str, list[str]] = {}
        self._partial: set[str] = set()
        self._open_count: dict[str, int] = {}
        self._delta_seen: set[tuple[str, str]] = set()
        # TASK-1095: the SDK stamps a resumed sub-agent's messages with the
        # tool_use_id of its ORIGINAL spawn (Agent call), even when a later
        # SendMessage call is what is now running it. Track spawn → current
        # host so every consumer attributes child activity to the live call.
        self._spawn_of_task: dict[str, str] = {}
        self._host_of_spawn: dict[str, str] = {}

    def register_task(self, parent: str | None, task_id: str, description: str) -> None:
        if not (parent and task_id):
            return
        self._tasks.setdefault(parent, {})[task_id] = description
        earlier = (
            self._spawn_lookup(task_id)
            if task_id not in self._spawn_of_task and self._spawn_lookup is not None
            else None
        )
        if earlier and earlier != parent:
            # An Agent result names exactly one agentId, so a transcript spawn
            # is unambiguous (unlike a multi-task Workflow seen this turn).
            self._spawn_of_task[task_id] = earlier
            self._single_agent_spawns.add(earlier)
        spawn = self._spawn_of_task.setdefault(task_id, parent)
        if spawn == parent:
            return
        # A Workflow spawn hosts several tasks under ONE parent id; children of
        # different siblings are indistinguishable by parent, so re-homing the
        # whole spawn would mis-file the siblings. Only remap unambiguous spawns.
        if spawn in self._single_agent_spawns or len(self._tasks.get(spawn, {})) == 1:
            self._host_of_spawn[spawn] = parent

    def host_parent(self, parent: str | None) -> str | None:
        """The live host tool call for an SDK ``parent_tool_use_id``."""
        if not parent:
            return parent
        return self._host_of_spawn.get(parent, parent)

    def live_updates(self, message: Any) -> list[tuple[str, str, str, dict]]:
        """Report a measured change immediately, without waiting for TaskProgress."""
        changed = self.observe(message)
        parent = self.host_parent(getattr(message, "parent_tool_use_id", None))
        return self._changed_updates(parent, changed)

    def live_stream_updates(
        self, event: Any, parent_tool_use_id: str | None,
    ) -> list[tuple[str, str, str, dict]]:
        """Same as ``live_updates`` for a subagent's raw stream event."""
        changed = self.observe_stream_event(event, parent_tool_use_id)
        return self._changed_updates(self.host_parent(parent_tool_use_id), changed)

    def observe_stream_event(self, event: Any, parent_tool_use_id: str | None) -> bool:
        """Account a subagent's message_start/message_delta usage (TASK-2169).

        Responses-backed proxies only know usage at completion: message_start
        carries zeros and the real counts arrive in message_delta, after the SDK
        has already yielded the per-block AssistantMessages (which keep the
        zeros). Deltas name no response ID, so they are attributed to the one
        open response under this parent; concurrent Workflow siblings share a
        parent, and an ambiguous delta marks the aggregate partial instead of
        being guessed onto a sibling's response.
        """
        parent = self.host_parent(parent_tool_use_id)
        if not parent or not isinstance(event, dict):
            return False
        kind = event.get("type")
        open_ids = self._open_responses.setdefault(parent, [])
        if kind == "message_start":
            message = event.get("message")
            response_id = message.get("id") if isinstance(message, dict) else None
            if not isinstance(response_id, str) or not response_id:
                return False
            if open_ids and response_id not in open_ids and len(self._tasks.get(parent, {})) <= 1:
                # One worker's requests are sequential, so a new start proves
                # every open response ended without message_stop (interrupted
                # or aborted). Retire them; one that never reported usage
                # leaves a known gap, so the aggregate becomes a lower bound.
                # Only a message_delta carries final output; message_start can
                # already hold real (native) input, so a non-zero sum is no proof.
                if any((parent, rid) not in self._delta_seen for rid in open_ids):
                    self._partial.add(parent)
                open_ids.clear()
                self._open_count[parent] = 0
            if response_id not in open_ids:
                open_ids.append(response_id)
                self._open_count[parent] = self._open_count.get(parent, 0) + 1
            return self._record(parent, response_id, message.get("usage"))
        if kind == "message_delta":
            if len(open_ids) != 1:
                if open_ids and parent not in self._partial:
                    self._partial.add(parent)
                    return True
                return False
            self._delta_seen.add((parent, open_ids[0]))
            return self._record(parent, open_ids[0], event.get("usage"))
        if kind == "message_stop":
            # A stop names no response either. With one open it is that one;
            # during an overlap only the count is known, so resynchronise once
            # every overlapping response has stopped (later deltas measure
            # again; the partial flag on the earlier gap stays).
            remaining = max(0, self._open_count.get(parent, 0) - 1)
            self._open_count[parent] = remaining
            if len(open_ids) == 1 or remaining == 0:
                open_ids.clear()
                self._open_count[parent] = 0
        return False

    def _changed_updates(
        self, parent: str | None, changed: bool,
    ) -> list[tuple[str, str, str, dict]]:
        if not changed or not parent or parent not in self._tasks:
            return []
        usage = self.for_event(parent, None)
        total = usage.get("cumulative_tokens") if usage else None
        key = (total, parent in self._partial)
        if total is None or key == self._last_live_total.get(parent):
            return []
        self._last_live_total[parent] = key
        return [
            (parent, task_id, description, usage)
            for task_id, description in self._tasks[parent].items()
        ]

    @staticmethod
    def _count(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    def observe(self, message: Any) -> bool:
        parent = self.host_parent(getattr(message, "parent_tool_use_id", None))
        response_id = getattr(message, "message_id", None)
        usage = getattr(message, "usage", None)
        if not parent or not response_id:
            return False
        return self._record(parent, response_id, usage)

    def _record(self, parent: str, response_id: str, usage: Any) -> bool:
        if not isinstance(usage, dict):
            return False
        counters = {
            key: value for key in self.TOKEN_FIELDS
            if self._count(value := usage.get(key))
        }
        if not counters:
            return False
        responses = self._responses.setdefault(parent, {})
        snapshot = responses.setdefault(response_id, {})
        # SDK echoes multiple blocks with the same response ID, sometimes with
        # preliminary zero/partial usage. The counters are per-response snapshots,
        # not deltas. A repeated block or replay must never double-charge it.
        changed = False
        for key, value in counters.items():
            if key not in snapshot or value > snapshot[key]:
                snapshot[key] = value
                changed = True
        return changed

    def for_event(self, parent: str | None, usage: Any) -> dict | None:
        result = {
            key: value for key in ("total_tokens", "tool_uses", "duration_ms")
            if isinstance(usage, dict) and self._count(value := usage.get(key))
        }
        responses = self._responses.get(parent or "", {})
        complete = [u for u in responses.values() if "input_tokens" in u and "output_tokens" in u]
        if complete:
            # Explicitly parent-scoped: sibling Workflow task events carry the
            # SAME aggregate. Consumers take it once per spawning tool, not once
            # per task. Cached input is included, not added twice to input_tokens.
            result["cumulative_tokens"] = sum(sum(u.values()) for u in complete)
            if parent in self._partial:
                # Some sibling response's counts could not be attributed:
                # the total is a measured lower bound, not the full amount.
                result["cumulative_partial"] = True
        return result or None
