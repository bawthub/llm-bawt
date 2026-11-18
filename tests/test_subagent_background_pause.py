"""TASK-2367: a subagent paused on its own background work is not done."""
from types import SimpleNamespace
from unittest.mock import Mock

from agent_bridge.events import AgentEventKind
from claude_agent_sdk.types import SystemMessage, TaskNotificationMessage

from claude_code_bridge.send_stream import ClaudeStreamMixin
from claude_code_bridge.subagent_usage import SubagentUsage

SPAWN = "toolu_spawn"


def notification(status="completed", summary="paused"):
    return TaskNotificationMessage(
        subtype="task_notification", data={}, task_id="agent1", status=status,
        output_file="", summary=summary, uuid="u", session_id="s", tool_use_id=SPAWN,
    )


def level(*tasks):
    data = {"type": "system", "subtype": "background_tasks_changed", "tasks": list(tasks)}
    return SystemMessage(subtype="background_tasks_changed", data=data)


AGENT = {"task_id": "agent1", "task_type": "local_agent", "description": "probe"}
SHELL = {"task_id": "shell1", "task_type": "local_bash", "description": "sleep",
         "parent_task_id": "agent1"}


def run(*messages):
    emitter = SimpleNamespace(_publish_event=Mock())
    ledger = SubagentUsage()
    for msg in messages:
        ClaudeStreamMixin._emit_subagent_task_events(
            emitter, msg, request_id="r", session_key="s", seq=0, usage_ledger=ledger,
        )
    return [(c.kwargs["kind"], c.kwargs["extra_raw"].get("status"), c.kwargs.get("text"))
            for c in emitter._publish_event.call_args_list]


def test_completed_with_live_child_is_progress_then_done_on_final_notification():
    events = run(level(AGENT, SHELL), notification(), level(AGENT),
                 notification(summary="final"))
    assert events == [
        (AgentEventKind.SUBAGENT_PROGRESS, "running", None),
        (AgentEventKind.SUBAGENT_DONE, "completed", "final"),
    ]


def test_deferred_done_is_released_when_agent_and_children_leave_live_set():
    events = run(level(AGENT, SHELL), notification(summary="held"), level())
    assert events == [
        (AgentEventKind.SUBAGENT_PROGRESS, "running", None),
        (AgentEventKind.SUBAGENT_DONE, "completed", "held"),
    ]


def test_no_live_children_is_done_immediately():
    assert run(level(AGENT), notification(summary="x")) == [
        (AgentEventKind.SUBAGENT_DONE, "completed", "x"),
    ]


def test_failure_is_never_deferred():
    assert run(level(AGENT, SHELL), notification(status="failed", summary="x")) == [
        (AgentEventKind.SUBAGENT_DONE, "failed", "x"),
    ]


def test_failed_or_stopped_after_held_completion_stays_final():
    for status in ("failed", "stopped"):
        events = run(level(AGENT, SHELL), notification(summary="interim"),
                     notification(status=status, summary="terminal"), level())
        assert events == [
            (AgentEventKind.SUBAGENT_PROGRESS, "running", None),
            (AgentEventKind.SUBAGENT_DONE, status, "terminal"),
        ]


def updated(status="completed"):
    from claude_agent_sdk.types import TaskUpdatedMessage
    return TaskUpdatedMessage(subtype="task_updated", data={}, task_id="agent1",
                              patch={"status": status}, status=status)


RUNNING = (AgentEventKind.SUBAGENT_PROGRESS, "running", None)


def test_task_updated_completed_while_waiting_on_children_reports_running():
    events = run(level(AGENT, SHELL), updated(), notification(), updated(), level(AGENT),
                 notification(summary="final"))
    assert events == [RUNNING, RUNNING, RUNNING,
                      (AgentEventKind.SUBAGENT_DONE, "completed", "final")]


def test_update_before_level_frame_is_never_terminal():
    # Codex's reordering: the update lands before the frame naming the child.
    events = run(updated(), level(AGENT, SHELL), notification(), level(AGENT),
                 notification(summary="final"), level())
    assert events == [RUNNING, RUNNING,
                      (AgentEventKind.SUBAGENT_DONE, "completed", "final")]


def test_update_only_completion_is_released_when_task_leaves_live_set():
    events = run(level(AGENT), updated(), level(AGENT), level())
    assert events == [RUNNING, (AgentEventKind.SUBAGENT_PROGRESS, "completed", None)]


def test_held_update_after_failed_notification_stays_failed():
    events = run(level(AGENT, SHELL), updated(),
                 notification(status="failed", summary="x"), level())
    assert events == [RUNNING, (AgentEventKind.SUBAGENT_DONE, "failed", "x")]


def test_reordered_frames_persist_as_running_until_final_done():
    from llm_bawt.service.worker_lifecycle_store import _advance, event_status

    kinds = {AgentEventKind.SUBAGENT_PROGRESS: "subagent_progress",
             AgentEventKind.SUBAGENT_DONE: "subagent_done"}
    stored, history = "started", []
    for kind, status, _ in run(updated(), level(AGENT, SHELL), notification(),
                               level(AGENT), notification(summary="final")):
        stored = _advance(stored, event_status(kinds[kind], status))
        history.append(stored)
    assert history == ["running", "running", "completed"]
