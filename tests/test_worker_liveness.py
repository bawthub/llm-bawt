"""TASK-1148: a stalled resumed worker fails its own SendMessage, not the turn.

Replays the CLI 2.1.283 sequence captured from a real-CLI probe: serial
SendMessage resumes, the first worker goes silent, ``stop_task`` yields
``task_updated: killed`` + ``task_notification: stopped`` + an unflagged
"interrupted" tool result, then the queued sibling starts and completes.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from claude_agent_sdk.types import (
    AssistantMessage,
    TaskNotificationMessage,
    TaskProgressMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from agent_bridge.events import AgentEventKind
from claude_code_bridge.send_stream import ClaudeStreamMixin
from claude_code_bridge.subagent_usage import SubagentUsage
from claude_code_bridge.turn_watchdog import TurnWatchdog, TurnWatchdogTimeout
from claude_code_bridge.worker_liveness import WorkerLiveness, WorkerStop
from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher

SEND_A, SEND_B = "toolu_sendA", "toolu_sendB"
TASK_A, TASK_B = "task_a", "task_b"
INTERRUPTED = '{"success":false,"message":"Agent \\"a\\" was resumed but was interrupted: "}'


def calls(*pairs):
    return AssistantMessage(
        content=[ToolUseBlock(id=i, name=n, input={}) for i, n in pairs], model="m",
    )


def started(tool_use_id, task_id):
    return TaskStartedMessage(subtype="task_started", data={}, task_id=task_id,
                              description="worker", uuid="u", session_id="s",
                              tool_use_id=tool_use_id)


def progress(tool_use_id, task_id):
    return TaskProgressMessage(subtype="task_progress", data={}, task_id=task_id,
                               description="worker", usage={}, uuid="u", session_id="s",
                               tool_use_id=tool_use_id)


def updated(task_id, status):
    return TaskUpdatedMessage(subtype="task_updated", data={}, task_id=task_id,
                              patch={"status": status}, status=status)


def notified(tool_use_id, task_id, status):
    return TaskNotificationMessage(subtype="task_notification", data={}, task_id=task_id,
                                   status=status, output_file="", summary="", uuid="n",
                                   session_id="s", tool_use_id=tool_use_id)


def result(tool_use_id, content="ok", is_error=None):
    return UserMessage(content=[ToolResultBlock(tool_use_id=tool_use_id, content=content,
                                                is_error=is_error)])


class FakeCli:
    """An SDK stream whose first resumed worker stalls until it is stopped."""

    def __init__(self, *, ack_stop=True, first=(), after_stop=None):
        self.stopped: list[str] = []
        self._stop = asyncio.Event()
        self._ack = ack_stop
        self._first = list(first)
        self._after = after_stop

    async def stop_task(self, task_id):
        self.stopped.append(task_id)
        if self._ack:
            self._stop.set()

    async def stream(self):
        for msg in self._first:
            yield msg
        await self._stop.wait()
        for msg in self._after or ():
            yield msg


def _watchdog(liveness, cli, **kw):
    return TurnWatchdog(idle_timeout=0.05, health_interval=0.01, tool_grace=0.02,
                        liveness=liveness, stop_task=cli.stop_task, **kw)


async def _drain(watchdog, stream):
    seen = []
    while True:
        try:
            seen.append(await watchdog.receive_next(stream, request_id="r", session_key="snark:nick"))
        except StopAsyncIteration:
            return seen


def test_stalled_resume_is_stopped_and_queued_sibling_advances(caplog):
    liveness = WorkerLiveness()
    cli = FakeCli(
        first=[calls((SEND_A, "SendMessage"), (SEND_B, "SendMessage")), started(SEND_A, TASK_A)],
        after_stop=[
            updated(TASK_A, "killed"), notified(SEND_A, TASK_A, "stopped"),
            result(SEND_A, INTERRUPTED),
            started(SEND_B, TASK_B), progress(SEND_B, TASK_B),
            result(SEND_B, '{"success":true}'),
        ],
    )

    async def run():
        watchdog = _watchdog(liveness, cli)
        seen = await _drain(watchdog, cli.stream())
        assert watchdog.active_tools == ()
        return seen

    caplog.set_level("INFO", logger="claude_code_bridge.bridge")
    seen = asyncio.run(run())

    # Only the worker that was actually running is stopped; never the queued one.
    assert cli.stopped == [TASK_A]
    assert len(seen) == 8
    stop = liveness.stop_for_tool(SEND_A)
    assert stop is not None and stop.task_id == TASK_A
    assert liveness.stop_for_tool(SEND_B) is None
    assert liveness.stop_status(TASK_A, "stopped")[0] == "timed_out"
    assert liveness.stop_status(TASK_B, "completed") == ("completed", None)
    assert "worker_liveness_stop request_id=r" in caplog.text
    assert "worker_liveness_stop_ack tool_use_id=toolu_sendA" in caplog.text


def test_unacknowledged_stop_is_a_genuine_parent_failure():
    liveness = WorkerLiveness()
    cli = FakeCli(ack_stop=False, first=[calls((SEND_A, "SendMessage")), started(SEND_A, TASK_A)])

    async def run():
        with pytest.raises(TurnWatchdogTimeout, match="worker stop unacknowledged") as raised:
            await _drain(_watchdog(liveness, cli), cli.stream())
        assert raised.value.phase == "tool_running"
        assert raised.value.retry_fresh_session is False

    asyncio.run(run())
    assert cli.stopped == [TASK_A]


def test_failed_stop_request_ends_the_parent():
    liveness = WorkerLiveness()
    cli = FakeCli(first=[calls((SEND_A, "SendMessage")), started(SEND_A, TASK_A)])

    async def broken_stop(task_id):
        raise RuntimeError("control channel closed")

    async def run():
        watchdog = TurnWatchdog(idle_timeout=0.05, health_interval=0.01, tool_grace=0.02,
                                liveness=liveness, stop_task=broken_stop)
        with pytest.raises(TurnWatchdogTimeout, match="Could not stop timed-out worker task_a"):
            await _drain(watchdog, cli.stream())

    asyncio.run(run())


def test_parallel_agent_spawn_keeps_parent_deadline_behaviour():
    liveness = WorkerLiveness()
    cli = FakeCli(first=[calls(("toolu_spawnA", "Agent"), ("toolu_spawnB", "Agent")),
                         started("toolu_spawnA", TASK_A), started("toolu_spawnB", TASK_B)])

    async def run():
        with pytest.raises(TurnWatchdogTimeout, match="active-tool deadline"):
            await _drain(_watchdog(liveness, cli), cli.stream())

    asyncio.run(run())
    assert cli.stopped == []


def test_queued_resume_without_a_running_task_is_never_stopped():
    liveness = WorkerLiveness()
    cli = FakeCli(first=[calls((SEND_A, "SendMessage"))])

    async def run():
        with pytest.raises(TurnWatchdogTimeout):
            await _drain(_watchdog(liveness, cli), cli.stream())

    asyncio.run(run())
    assert cli.stopped == []


def test_without_liveness_the_watchdog_is_unchanged():
    cli = FakeCli(first=[calls((SEND_A, "SendMessage")), started(SEND_A, TASK_A)])

    async def run():
        watchdog = TurnWatchdog(idle_timeout=0.05, health_interval=0.01, tool_grace=0.02,
                                stop_task=cli.stop_task)
        with pytest.raises(TurnWatchdogTimeout):
            await _drain(watchdog, cli.stream())

    asyncio.run(run())
    assert cli.stopped == []


def _stop():
    return WorkerStop(tool_use_id=SEND_A, task_id=TASK_A, tool_name="SendMessage",
                      idle_seconds=331.2, limit_seconds=300.0, requested_at=0.0)


class _Emitter(SimpleNamespace):
    _host_parent = staticmethod(ClaudeStreamMixin._host_parent)
    _is_image_result_tool = staticmethod(lambda name: False)


def _events(emitter, kind):
    return [c.kwargs for c in emitter._publish_event.call_args_list if c.kwargs.get("kind") == kind]


def test_bridge_stop_publishes_task_scoped_timed_out_lifecycle():
    liveness = WorkerLiveness()
    liveness.task_started(SEND_A, TASK_A)
    liveness.record_stop(_stop())
    emitter = _Emitter(_publish_event=Mock())
    for msg in (updated(TASK_A, "killed"), notified(SEND_A, TASK_A, "stopped"),
                notified(SEND_B, TASK_B, "stopped")):
        ClaudeStreamMixin._emit_subagent_task_events(
            emitter, msg, request_id="r", session_key="s", seq=0, liveness=liveness,
        )

    progress_raw = _events(emitter, AgentEventKind.SUBAGENT_PROGRESS)[0]["extra_raw"]
    assert progress_raw["status"] == "timed_out"
    done_a, done_b = _events(emitter, AgentEventKind.SUBAGENT_DONE)
    assert done_a["tool_use_id"] == SEND_A
    assert done_a["extra_raw"]["status"] == "timed_out"
    failure = done_a["extra_raw"]["failure"]
    assert failure["task_id"] == TASK_A and failure["tool_use_id"] == SEND_A
    assert failure["stopped_by"] == "bridge_liveness"
    assert failure["idle_seconds"] == 331.2 and failure["limit_seconds"] == 300.0
    # A stop the bridge did not cause (e.g. the user's) keeps its own status.
    assert done_b["extra_raw"]["status"] == "stopped"
    assert "failure" not in done_b["extra_raw"]


def test_stopped_resume_tool_result_is_failed_and_explains_why():
    liveness = WorkerLiveness()
    liveness.record_stop(_stop())
    emitter = _Emitter(_publish_event=Mock())
    asyncio.run(ClaudeStreamMixin._on_user_message_tool_results(
        emitter, UserMessage(content=[
            ToolResultBlock(tool_use_id=SEND_A, content=INTERRUPTED),
            ToolResultBlock(tool_use_id=SEND_B, content='{"success":true}'),
        ]),
        request_id="r", session_key="s", seq=0,
        tool_names_by_id={SEND_A: "SendMessage", SEND_B: "SendMessage"},
        tool_arguments_by_id={}, turn_screenshot_assets=[],
        screenshot_artifacts_by_tool_use_id={}, liveness=liveness,
    ))
    end_a, end_b = _events(emitter, AgentEventKind.TOOL_END)
    assert end_a["tool_error"] is True
    assert "timed out" in end_a["tool_result"] and "331s" in end_a["tool_result"]
    assert end_b["tool_error"] is False
    assert "Bridge worker liveness" not in end_b["tool_result"]


def test_post_tool_use_hook_tells_the_parent_model():
    from claude_code_bridge.event_ops import ClaudeEventMixin

    liveness = WorkerLiveness()
    liveness.record_stop(_stop())
    hook = ClaudeEventMixin._make_post_tool_use_hook(
        SimpleNamespace(), session_key="s", screenshot_artifacts_by_tool_use_id={},
        worker_liveness=liveness,
    )
    out = asyncio.run(hook({"tool_name": "SendMessage"}, SEND_A, None))
    context = out["hookSpecificOutput"]["additionalContext"]
    assert out["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    assert TASK_A in context and "timed out" in context
    assert asyncio.run(hook({"tool_name": "SendMessage"}, SEND_B, None)) == {}


def test_unfinalized_durable_turn_names_the_recorded_cause():
    describe = InterBotDeliveryDispatcher._unfinalized_turn_error
    assert describe("turn-x", None) == (
        "streamed durable turn did not finalize: no turn log for turn-x"
    )
    turn = SimpleNamespace(status="error", error_text="claude-code error: boom")
    assert describe("turn-x", turn) == (
        "streamed durable turn did not finalize (status=error; error: claude-code error: boom)"
    )


def test_same_task_resume_after_timeout_keeps_a_later_user_stop_a_user_stop():
    """Snark r1 #1: the stop cause belongs to the attempt, not the task."""
    resume2 = "toolu_send2"
    liveness = WorkerLiveness()
    liveness.task_started(SEND_A, TASK_A)
    liveness.record_stop(_stop())
    assert liveness.task_status(TASK_A, "killed").tool_use_id == SEND_A
    liveness.task_started(resume2, TASK_A)  # the parent resumes the same worker
    # Task-only event (task_updated) resolves against the current attempt.
    assert liveness.stop_status(TASK_A, "killed") == ("killed", None)
    assert liveness.stop_status(TASK_A, "stopped", resume2) == ("stopped", None)
    assert liveness.task_status(TASK_A, "stopped") is None
    # A late notification for the old attempt stays attributed to it.
    status, failure = liveness.stop_status(TASK_A, "stopped", SEND_A)
    assert status == "timed_out" and failure["tool_use_id"] == SEND_A
    assert liveness.stop_for_tool(SEND_A) is not None
    assert liveness.stop_for_tool(resume2) is None
    assert liveness.containable(resume2, "SendMessage") is None  # no longer live


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_stop_acknowledgement_deadline_is_absolute():
    """Snark r1 #2: a queued long Bash or unrelated progress cannot extend it."""
    clock = Clock()
    liveness = WorkerLiveness()
    stopped = []

    async def stop_task(task_id):
        stopped.append(task_id)

    watchdog = TurnWatchdog(idle_timeout=300, health_interval=1, tool_grace=30,
                            clock=clock, liveness=liveness, stop_task=stop_task)
    bash = AssistantMessage(content=[ToolUseBlock(id="toolu_bash", name="Bash",
                                                  input={"timeout": 600_000})], model="m")
    watchdog.observe(calls((SEND_A, "SendMessage")))
    watchdog.observe(bash)
    watchdog.observe(started(SEND_A, TASK_A))
    clock.now = 330.0
    asyncio.run(watchdog._contain_expired_workers(request_id="r", session_key="s"))
    assert stopped == [TASK_A]
    # Bash alone would allow 630s; unrelated activity at 350 renews the drain floor.
    clock.now = 350.0
    watchdog.observe(progress("toolu_other", "task_other"))
    assert watchdog._deadline(wait_started_at=350.0) == (360.0, "tool_running")
    # Once the CLI acknowledges the stop, ordinary sibling buffering resumes.
    watchdog.observe(updated(TASK_A, "killed"))
    assert watchdog._deadline(wait_started_at=350.0) == (630.0, "tool_running")


def test_late_old_notification_leaves_the_live_resume_containable():
    """Snark r2 #1: a named old attempt never mutates the current one."""
    resume2 = "toolu_send2"
    clock = Clock()
    liveness = WorkerLiveness()
    watchdog = TurnWatchdog(idle_timeout=300, health_interval=1, tool_grace=30, clock=clock,
                            liveness=liveness, stop_task=lambda task_id: asyncio.sleep(0))
    watchdog.observe(calls((SEND_A, "SendMessage")))
    watchdog.observe(started(SEND_A, TASK_A))
    clock.now = 330.0
    asyncio.run(watchdog._contain_expired_workers(request_id="r", session_key="s"))
    watchdog.observe(updated(TASK_A, "killed"))
    watchdog.observe(result(SEND_A, INTERRUPTED))
    watchdog.observe(calls((resume2, "SendMessage")))
    watchdog.observe(started(resume2, TASK_A))
    assert liveness.containable(resume2, "SendMessage") == TASK_A
    watchdog.observe(notified(SEND_A, TASK_A, "stopped"))  # delayed, names resume1
    assert liveness.containable(resume2, "SendMessage") == TASK_A
    assert liveness.stop_status(TASK_A, "stopped", SEND_A)[1]["tool_use_id"] == SEND_A
    # The new attempt's own stop is acknowledged only by its own events.
    clock.now = 700.0
    asyncio.run(watchdog._contain_expired_workers(request_id="r", session_key="s"))
    watchdog.observe(notified(SEND_A, TASK_A, "stopped"))
    assert next(t for t in watchdog.active_tools if t.tool_use_id == resume2).stop_outstanding
    watchdog.observe(notified(resume2, TASK_A, "stopped"))
    assert not next(t for t in watchdog.active_tools if t.tool_use_id == resume2).stop_outstanding


def test_named_old_user_stop_never_borrows_the_current_attempt_timeout():
    resume2 = "toolu_send2"
    liveness = WorkerLiveness()
    liveness.task_started(SEND_A, TASK_A)
    assert liveness.task_status(TASK_A, "stopped", SEND_A) is None  # user stop of resume1
    liveness.task_started(resume2, TASK_A)
    liveness.record_stop(WorkerStop(tool_use_id=resume2, task_id=TASK_A, tool_name="SendMessage",
                                    idle_seconds=301, limit_seconds=300, requested_at=0))
    assert liveness.stop_status(TASK_A, "stopped", SEND_A) == ("stopped", None)
    assert liveness.stop_status(TASK_A, "stopped", resume2)[1]["tool_use_id"] == resume2
    assert liveness.stop_status(TASK_A, "killed")[1]["tool_use_id"] == resume2  # task-only event
    # An unknown host id (not an attempt of this task) resolves like a task-only event.
    assert liveness.stop_status(TASK_A, "stopped", "toolu_unknown")[1]["tool_use_id"] == resume2


# ---------------------------------------------------------------------------
# TASK-1148 R2: one liveness rule at every depth.
# A resumed helper spawned in an EARLIER turn: the CLI stamps its children with
# the original spawn id, which only the transcript lookup can re-home.
# ---------------------------------------------------------------------------


OLD_SPAWN = "toolu_spawn_prev_turn"
CHILD = "toolu_child_bash"


def _bash(tool_id, timeout_ms=None, parent=None, name="Bash"):
    tool_input = {"command": "sleep"} if timeout_ms is None else {"command": "sleep",
                                                                   "timeout": timeout_ms}
    return AssistantMessage(content=[ToolUseBlock(id=tool_id, name=name, input=tool_input)],
                            model="m", parent_tool_use_id=parent)


def _tree(depth, timeout_ms):
    """A Bash call at parent level, or inside a cross-turn resumed helper."""
    clock = Clock()
    liveness = WorkerLiveness()
    usage = SubagentUsage(spawn_lookup=lambda task_id: OLD_SPAWN if task_id == TASK_A else None)
    stopped: list[str] = []

    async def stop_task(task_id):
        stopped.append(task_id)

    watchdog = TurnWatchdog(idle_timeout=300, health_interval=1, tool_grace=30,
                            max_tool_timeout=600, clock=clock, resolve_parent=usage.host_parent,
                            liveness=liveness, stop_task=stop_task)
    if depth == "parent":
        watchdog.observe(_bash(CHILD, timeout_ms))
    else:
        watchdog.observe(calls((SEND_A, "SendMessage")))
        watchdog.observe(started(SEND_A, TASK_A))
        usage.register_task(SEND_A, TASK_A, "worker")  # send_stream order: after observe
        watchdog.observe(_bash(CHILD, timeout_ms, parent=OLD_SPAWN))

    def alive(at):
        clock.now = at
        if depth == "parent":
            return at < watchdog._deadline(wait_started_at=0.0)[0]
        asyncio.run(watchdog._contain_expired_workers(request_id="r", session_key="s"))
        return not stopped

    return SimpleNamespace(watchdog=watchdog, liveness=liveness, usage=usage, clock=clock,
                           stopped=stopped, alive=alive)


@pytest.mark.parametrize("depth", ["parent", "helper"])
@pytest.mark.parametrize(("timeout_ms", "at", "expected_alive"), [
    (None, 329, True),          # undeclared: the ordinary 300s + grace
    (None, 331, False),
    (600_000, 420, True),       # the live-check case: sleep 420 with timeout=600000
    (600_000, 629, True),
    (600_000, 631, False),
    (3_600_000, 629, True),     # declared beyond the cap: still 600s + grace
    (3_600_000, 631, False),
])
def test_same_liveness_rule_at_parent_level_and_inside_a_helper(depth, timeout_ms, at,
                                                                 expected_alive):
    assert _tree(depth, timeout_ms).alive(at) is expected_alive


def test_cross_turn_resume_children_link_to_the_resuming_call():
    tree = _tree("helper", 600_000)
    child = next(t for t in tree.watchdog.active_tools if t.tool_use_id == CHILD)
    assert child.parent_tool_use_id == SEND_A
    assert tree.usage.host_parent(OLD_SPAWN) == SEND_A


def test_containment_wakeup_uses_the_same_rule_no_hot_spin():
    tree = _tree("helper", 600_000)
    tree.clock.now = 420.0
    assert tree.watchdog._next_containment_in(420.0) == pytest.approx(210.0)


def test_stop_names_the_binding_child_limit():
    tree = _tree("helper", 600_000)
    assert tree.alive(631) is False
    stop = tree.liveness.stop_for_tool(SEND_A)
    assert (stop.limit_tool_name, stop.limit_tool_use_id, stop.limit_seconds) == ("Bash", CHILD, 600)
    assert "its Bash call passed its 600s limit" in stop.reason
    assert stop.diagnostic()["limit_tool_use_id"] == CHILD
    assert "The resumed agent (task task_a) had no sub-agent activity" in stop.parent_note()


def test_host_inactivity_stop_names_the_host_limit():
    tree = _tree("helper", None)
    assert tree.alive(331) is False
    stop = tree.liveness.stop_for_tool(SEND_A)
    assert stop.diagnostic()["limit_tool_name"] == "SendMessage"
    assert "(liveness limit 300s)" in stop.reason and stop.reason.startswith("had no")


def test_finished_child_stops_extending_its_host():
    tree = _tree("helper", 600_000)
    tree.clock.now = 10.0
    tree.watchdog.observe(UserMessage(content=[ToolResultBlock(tool_use_id=CHILD, content="ok")],
                                      parent_tool_use_id=OLD_SPAWN))
    assert tree.alive(339) is True   # refreshed by the child's result at t=10
    assert tree.alive(341) is False


def test_stop_purges_descendants_and_late_children_stay_with_the_stopped_host():
    tree = _tree("helper", None)
    assert tree.alive(331) is False
    assert [t.tool_use_id for t in tree.watchdog.active_tools] == [SEND_A]
    tree.watchdog.observe(_bash("toolu_late", 600_000, parent=OLD_SPAWN))
    late = next(t for t in tree.watchdog.active_tools if t.tool_use_id == "toolu_late")
    assert (late.parent_tool_use_id, late.nested) == (SEND_A, True)
    # The stop window stays absolute: the late call cannot extend it.
    assert tree.watchdog._deadline(wait_started_at=331.0)[0] == pytest.approx(361.0)
    tree.watchdog.observe(result(SEND_A, "stopped"))
    assert tree.watchdog.active_tools == ()


def test_late_nested_resume_after_host_stop_is_never_contained_and_purged_with_host():
    """Loopy r2 blocker: a nested SendMessage arriving after its host's stop."""
    tree = _tree("helper", None)
    assert tree.alive(331) is False and tree.stopped == [TASK_A]
    tree.watchdog.observe(_bash("toolu_late_send", parent=OLD_SPAWN, name="SendMessage"))
    tree.watchdog.observe(started("toolu_late_send", TASK_B))
    tree.watchdog.observe(notified(SEND_A, TASK_A, "stopped"))
    assert tree.alive(1000) is False and tree.stopped == [TASK_A]  # TASK_B never stopped
    tree.watchdog.observe(result(SEND_A, "stopped"))
    assert tree.watchdog.active_tools == ()


def test_orphaned_nested_resume_is_never_contained():
    """A nested call whose host is already gone is still not top-level."""
    tree = _tree("helper", None)
    tree.watchdog.observe(result(SEND_A, '{"success":true}'))
    tree.watchdog.observe(_bash("toolu_orphan_send", parent=OLD_SPAWN, name="SendMessage"))
    tree.watchdog.observe(started("toolu_orphan_send", TASK_B))
    orphan = next(t for t in tree.watchdog.active_tools if t.tool_use_id == "toolu_orphan_send")
    assert (orphan.parent_tool_use_id, orphan.nested) == (None, True)
    assert tree.alive(1000) is True and tree.stopped == []


def test_host_result_purges_its_descendants():
    tree = _tree("helper", 600_000)
    tree.watchdog.observe(result(SEND_A, '{"success":true}'))
    assert tree.watchdog.active_tools == ()


def test_nested_resume_is_never_contained_only_the_top_level_host():
    tree = _tree("helper", None)
    tree.watchdog.observe(_bash("toolu_nested_send", parent=OLD_SPAWN, name="SendMessage"))
    tree.watchdog.observe(started("toolu_nested_send", TASK_B))
    assert tree.alive(331) is False
    assert tree.stopped == [TASK_A]
