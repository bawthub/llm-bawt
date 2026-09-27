from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from claude_agent_sdk import AssistantMessage
from claude_agent_sdk.types import TaskNotificationMessage, TaskProgressMessage

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
