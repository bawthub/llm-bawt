from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from claude_agent_sdk import AssistantMessage
from claude_agent_sdk.types import TaskNotificationMessage, TaskProgressMessage, TaskStartedMessage

from claude_code_bridge.send_stream import ClaudeStreamMixin
from claude_code_bridge.subagent_usage import SubagentUsage


def response(message_id="m1", parent="agent1", **usage):
    return AssistantMessage(content=[], model="test", message_id=message_id,
                            parent_tool_use_id=parent, usage=usage)


def test_counts_requests_not_sdk_estimates_or_duplicate_content_blocks():
    ledger = SubagentUsage()
    first = response(input_tokens=100, output_tokens=20, cache_read_input_tokens=900)
    ledger.observe(first)
    ledger.observe(first)
    ledger.observe(response("m2", input_tokens=10, output_tokens=5, cache_read_input_tokens=20))
    # A smaller context/new request adds its usage; it cannot replace the total.
    assert ledger.for_event("agent1", {"total_tokens": 225}) == {
        "total_tokens": 225, "cumulative_tokens": 1055,
    }


def test_sparse_echoes_and_preliminary_zeros_do_not_erase_usage():
    ledger = SubagentUsage()
    ledger.observe(response(input_tokens=100))
    assert ledger.for_event("agent1", None) is None
    ledger.observe(response(output_tokens=25, cache_creation_input_tokens=50))
    ledger.observe(response(input_tokens=0, output_tokens=0))
    assert ledger.for_event("agent1", None) == {"cumulative_tokens": 175}


def test_parents_are_isolated_and_workflow_siblings_share_one_parent_aggregate():
    ledger = SubagentUsage()
    ledger.observe(response(input_tokens=100, output_tokens=10))
    ledger.observe(response(parent="agent2", input_tokens=20, output_tokens=5))
    ledger.observe(response("m2", input_tokens=300, output_tokens=30))
    assert ledger.for_event("agent1", None)["cumulative_tokens"] == 440
    assert ledger.for_event("agent2", None)["cumulative_tokens"] == 25
    assert SubagentUsage().for_event("agent1", None) is None


@pytest.mark.parametrize("usage", [None, {}, {"tool_uses": 3}, {"total_tokens": True},
                                  {"total_tokens": -1}, {"duration_ms": float("nan")}])
def test_absent_invalid_usage_is_not_zero(usage):
    ledger = SubagentUsage()
    expected = {"tool_uses": 3} if usage == {"tool_uses": 3} else None
    assert ledger.for_event("agent1", usage) == expected


def test_unidentifiable_or_parent_usage_is_never_charged_to_a_child():
    ledger = SubagentUsage()
    ledger.observe(response(message_id=None, input_tokens=10, output_tokens=10))
    ledger.observe(response(parent=None, input_tokens=10, output_tokens=10))
    ledger.observe(response(input_tokens=True, output_tokens=-5))
    assert ledger.for_event("agent1", None) is None


def test_zero_reported_usage_is_distinct_from_unavailable():
    ledger = SubagentUsage()
    ledger.observe(response(input_tokens=0, output_tokens=0))
    assert ledger.for_event("agent1", None) == {"cumulative_tokens": 0}


def test_live_usage_publishes_on_each_measured_response_without_waiting_for_progress():
    ledger = SubagentUsage()
    emitter = SimpleNamespace(_publish_event=Mock())
    common = dict(data={}, uuid="event1", session_id="session1", tool_use_id="agent1")
    for task_id in ("task1", "task2"):
        started = TaskStartedMessage(
            subtype="task_started", task_id=task_id, description="working", **common,
        )
        ClaudeStreamMixin._emit_subagent_task_events(
            emitter, started, request_id="r", session_key="s", seq=0,
            usage_ledger=ledger,
        )
    emitter._publish_event.reset_mock()
    seq = ClaudeStreamMixin._publish_subagent_live_usage(
        emitter, response(input_tokens=100, output_tokens=20),
        request_id="r", session_key="s", seq=2, usage_ledger=ledger,
    )
    assert seq == 4
    assert {c.kwargs["extra_raw"]["task_id"] for c in emitter._publish_event.call_args_list} == {"task1", "task2"}
    assert all(c.kwargs["extra_raw"]["usage"] == {"cumulative_tokens": 120}
               for c in emitter._publish_event.call_args_list)
    emitter._publish_event.reset_mock()
    assert ClaudeStreamMixin._publish_subagent_live_usage(
        emitter, response(input_tokens=100, output_tokens=20),
        request_id="r", session_key="s", seq=seq, usage_ledger=ledger,
    ) == seq
    emitter._publish_event.assert_not_called()
    assert ClaudeStreamMixin._publish_subagent_live_usage(
        emitter, response("m2", parent="agent2", input_tokens=300, output_tokens=30),
        request_id="r", session_key="s", seq=seq, usage_ledger=ledger,
    ) == seq
    emitter._publish_event.assert_not_called()
    assert ClaudeStreamMixin._publish_subagent_live_usage(
        emitter, response("m2", input_tokens=30, output_tokens=5),
        request_id="r", session_key="s", seq=seq, usage_ledger=ledger,
    ) == seq + 2
    assert emitter._publish_event.call_args.kwargs["extra_raw"]["usage"] == {"cumulative_tokens": 155}


def test_lifecycle_publishes_measured_total_even_when_final_usage_is_missing():
    ledger = SubagentUsage()
    ledger.observe(response(input_tokens=120, output_tokens=25))
    emitter = SimpleNamespace(_publish_event=Mock())
    common = dict(data={}, task_id="task1", uuid="event1", session_id="session1", tool_use_id="agent1")
    progress = TaskProgressMessage(subtype="task_progress", description="working", usage={"tool_uses": 3}, **common)
    done = TaskNotificationMessage(subtype="task_notification", status="completed", output_file="", summary="done", usage=None, **common)
    for message in (progress, done):
        ClaudeStreamMixin._emit_subagent_task_events(
            emitter, message, request_id="r", session_key="s", seq=0, usage_ledger=ledger,
        )
        assert emitter._publish_event.call_args.kwargs["extra_raw"]["usage"]["cumulative_tokens"] == 145
    assert "total_tokens" not in emitter._publish_event.call_args.kwargs["extra_raw"]["usage"]


def test_earlier_turn_spawn_is_rehomed_via_lookup_once():
    """TASK-1148 R2: a spawn from an earlier turn is found in the transcript."""
    calls = []

    def lookup(task_id):
        calls.append(task_id)
        return "toolu_old_spawn" if task_id == "agent_a" else None

    usage = SubagentUsage(spawn_lookup=lookup)
    usage.register_task("toolu_resume", "agent_a", "worker")
    assert usage.host_parent("toolu_old_spawn") == "toolu_resume"
    usage.register_task("toolu_resume2", "agent_a", "worker")  # known now: no second read
    assert calls == ["agent_a"]
    assert usage.host_parent("toolu_old_spawn") == "toolu_resume2"
    # A fresh spawn this turn (lookup finds nothing) keeps today's behaviour.
    usage.register_task("toolu_spawn_b", "agent_b", "worker")
    assert usage.host_parent("toolu_spawn_b") == "toolu_spawn_b"


def test_agent_spawn_index_scans_the_transcript_once_per_request(tmp_path, monkeypatch):
    import json

    from claude_code_bridge import _bridge_helpers
    from claude_code_bridge._bridge_helpers import AgentSpawnIndex

    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / ".claude" / "projects" / "-work"
    project.mkdir(parents=True)

    def tool_result(tool_use_id, tool_use_result):
        return json.dumps({"type": "user", "toolUseResult": tool_use_result, "message": {
            "role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use_id}]}})

    (project / "sess.jsonl").write_text("\n".join([
        'not json "agentId" "toolUseResult"',
        tool_result("toolu_spawn", {"status": "completed", "agentId": "agent_a"}),
        tool_result("toolu_resume", {"success": True, "message": "agent_a resumed"}),
        tool_result("toolu_respawn", {"status": "completed", "agentId": "agent_a"}),
        tool_result("toolu_spawn_c", {"status": "completed", "agentId": "agent_c"}),
    ]) + "\n")
    reads = []
    real = _bridge_helpers._read_agent_spawns
    monkeypatch.setattr(_bridge_helpers, "_read_agent_spawns",
                        lambda sid: reads.append(sid) or real(sid))
    session = ["sess"]
    index = AgentSpawnIndex(lambda: session[0])
    usage = SubagentUsage(spawn_lookup=index)
    # Fresh spawns, an earlier-turn resume and an unknown task: ONE scan.
    usage.register_task("toolu_new_1", "agent_new_1", "w")
    usage.register_task("toolu_new_2", "agent_new_2", "w")
    usage.register_task("toolu_resume_now", "agent_a", "w")
    usage.register_task("toolu_resume_c", "agent_c", "w")
    assert reads == ["sess"] and index.scans == 1
    assert usage.host_parent("toolu_spawn") == "toolu_resume_now"  # first spawn wins
    assert usage.host_parent("toolu_spawn_c") == "toolu_resume_c"
    assert index("agent_zzz") is None and index.scans == 1
    session[0] = None  # fresh-session retry: nothing earlier, no scan
    assert index("agent_a") is None and reads == ["sess"]
    assert AgentSpawnIndex(lambda: "missing")("agent_a") is None
