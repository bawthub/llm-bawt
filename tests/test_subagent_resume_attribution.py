"""TASK-1095 #2: a SendMessage-resumed sub-agent nests under the live resume call.

Replays the SDK message sequence from Snark's turn ec62…: Agent call spawns task
a57c, the agent fails, a later SendMessage resumes it. The SDK keeps stamping the
resumed agent's messages with the ORIGINAL Agent call id.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

from claude_agent_sdk import AssistantMessage, UserMessage
from claude_agent_sdk.types import (
    TaskNotificationMessage,
    TaskStartedMessage,
    ToolResultBlock,
    ToolUseBlock,
)

from claude_code_bridge.send_stream import ClaudeStreamMixin
from claude_code_bridge.subagent_usage import SubagentUsage
from claude_code_bridge.turn_watchdog import TurnWatchdog

SPAWN = "call_9a65_agent"
RESUME = "call_852e_sendmessage"
TASK = "a57c7440c8d12bc72"


def started(tool_use_id, task_id=TASK):
    return TaskStartedMessage(subtype="task_started", data={}, task_id=task_id, description="navigation",
                              uuid=f"u-{tool_use_id}", session_id="s", tool_use_id=tool_use_id)


def notified(tool_use_id, status, task_id=TASK):
    return TaskNotificationMessage(subtype="task_notification", data={}, task_id=task_id, status=status,
                                   output_file="", summary="", uuid=f"n-{tool_use_id}", session_id="s",
                                   tool_use_id=tool_use_id, usage=None)


def child_call(tool_id, message_id="m1", **usage):
    return AssistantMessage(content=[ToolUseBlock(id=tool_id, name="Read", input={"file_path": "/x"})],
                            model="test", parent_tool_use_id=SPAWN, message_id=message_id, usage=usage or None)


def child_result(tool_id):
    return UserMessage(content=[ToolResultBlock(tool_use_id=tool_id, content="ok", is_error=False)],
                       parent_tool_use_id=SPAWN)


class _Emitter(SimpleNamespace):
    _host_parent = staticmethod(ClaudeStreamMixin._host_parent)
    _is_ask_user_question = staticmethod(lambda name: False)
    _is_image_result_tool = staticmethod(lambda name: False)


def _parents(emitter, kind):
    return [(c.kwargs["tool_use_id"], c.kwargs["parent_tool_use_id"])
            for c in emitter._publish_event.call_args_list if c.kwargs.get("kind") == kind]


def _publish(emitter, ledger, msg):
    common = dict(request_id="r", session_key="s", seq=0, tool_names_by_id={}, tool_arguments_by_id={})
    if isinstance(msg, (TaskStartedMessage, TaskNotificationMessage)):
        ClaudeStreamMixin._emit_subagent_task_events(emitter, msg, request_id="r", session_key="s",
                                                     seq=0, usage_ledger=ledger)
    elif isinstance(msg, AssistantMessage):
        ClaudeStreamMixin._on_assistant_message(emitter, msg, latest_assistant_usage=None,
                                                assistant_snapshot_text="", usage_ledger=ledger, **common)
    else:
        asyncio.run(ClaudeStreamMixin._on_user_message_tool_results(
            emitter, msg, turn_screenshot_assets=[], screenshot_artifacts_by_tool_use_id={},
            usage_ledger=ledger, **common))


def test_resumed_agent_children_nest_under_the_resume_call_not_the_closed_spawn():
    from claude_agent_sdk.types import ToolUseBlock as _T  # noqa: F401 — sdk import sanity
    from agent_bridge.events import AgentEventKind

    ledger = SubagentUsage()
    emitter = _Emitter(_publish_event=Mock())
    sequence = [
        started(SPAWN), child_call("t_before"), child_result("t_before"), notified(SPAWN, "failed"),
        started(RESUME), child_call("t_after", "m2"), child_result("t_after"), notified(RESUME, "completed"),
    ]
    for msg in sequence:
        _publish(emitter, ledger, msg)

    assert _parents(emitter, AgentEventKind.TOOL_START) == [("t_before", SPAWN), ("t_after", RESUME)]
    assert _parents(emitter, AgentEventKind.TOOL_END) == [("t_before", SPAWN), ("t_after", RESUME)]


def test_resume_usage_is_credited_to_the_resume_run():
    ledger = SubagentUsage()
    ledger.register_task(SPAWN, TASK, "navigation")
    ledger.observe(child_call("a", "m1", input_tokens=100, output_tokens=10))
    ledger.register_task(RESUME, TASK, "navigation")
    ledger.observe(child_call("b", "m2", input_tokens=7, output_tokens=3))
    assert ledger.for_event(SPAWN, None) == {"cumulative_tokens": 110}
    assert ledger.for_event(RESUME, None) == {"cumulative_tokens": 10}


def test_workflow_spawn_with_several_tasks_is_never_rehomed():
    ledger = SubagentUsage()
    ledger.register_task("workflow", "t1", "a")
    ledger.register_task("workflow", "t2", "b")
    ledger.register_task("resume", "t1", "a")
    assert ledger.host_parent("workflow") == "workflow"
    assert ledger.host_parent(None) is None
    assert ledger.host_parent("unknown") == "unknown"


def test_watchdog_keeps_a_long_resume_alive_on_child_activity():
    now = [0.0]
    ledger = SubagentUsage()
    watchdog = TurnWatchdog(idle_timeout=300, tool_grace=30, clock=lambda: now[0],
                            resolve_parent=ledger.host_parent)
    ledger.register_task(SPAWN, TASK, "navigation")
    watchdog.observe(AssistantMessage(
        content=[ToolUseBlock(id=RESUME, name="SendMessage", input={"to": TASK})], model="test"))
    watchdog.observe(started(RESUME))
    ledger.register_task(RESUME, TASK, "navigation")  # handler runs after the watchdog
    host = next(t for t in watchdog.active_tools if t.tool_use_id == RESUME)
    assert host.progress_aware

    now[0] = 290.0
    watchdog.observe(child_call("t_after"))  # SDK stamps the OLD spawn id
    now[0] = 500.0  # > 300s since the resume started, but only 210s idle
    deadline, phase = watchdog._deadline(wait_started_at=500.0)
    assert phase == "tool_running"
    assert deadline == 290.0 + 300 + 30
