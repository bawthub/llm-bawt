"""Headless content folding cannot trigger any live execution side effects."""
from copy import deepcopy
import uuid

import pytest

from agent_bridge.events import AgentEvent, AgentEventKind as Kind
from agent_bridge.run_contract import RunTerminal, TeardownReport
from agent_bridge.terminal_replay import ReplayDisposition, RetainedRun
from llm_bawt.service.turn_terminal_fold import IncompleteReplay, fold_retained_run


def event(kind, **kwargs):
    return AgentEvent(event_id=uuid.uuid4().hex, session_key="test:user", run_id="run",
                      kind=kind, origin="system", **kwargs)


def replay(*events):
    return RetainedRun(ReplayDisposition.COMPLETE, "test", events,
                       RunTerminal("request", "claude-code", "boot", TeardownReport(exited=True)))


@pytest.mark.parametrize("deltas", [[], ["truncated"], ["Hello ", "world"], ["wrong draft"]])
def test_full_aggregate_not_delta_sum_is_content_authority(deltas):
    result = fold_retained_run(replay(
        *(event(Kind.ASSISTANT_DELTA, text=delta) for delta in deltas),
        event(Kind.ASSISTANT_DONE, text="Hello world", token_usage={"output_tokens": 3}),
    ))
    assert result.response_text == "Hello world"
    assert result.token_usage == {"output_tokens": 3}
    assert result.assistant_done and not result.cancelled


@pytest.mark.parametrize("events", [
    [], [event(Kind.ASSISTANT_DELTA, text="cannot prove this is all")],
    [event(Kind.ERROR, text="failed")], [event(Kind.ASSISTANT_DONE, text=None)],
    [event(Kind.ASSISTANT_DONE, text="one"), event(Kind.ASSISTANT_DONE, text="two")],
])
def test_missing_or_ambiguous_aggregate_requires_durable_partial_fallback(events):
    with pytest.raises(IncompleteReplay):
        fold_retained_run(replay(*events))


@pytest.mark.parametrize("disposition", [d for d in ReplayDisposition if d is not ReplayDisposition.COMPLETE])
def test_unproven_replay_never_folds(disposition):
    with pytest.raises(IncompleteReplay):
        fold_retained_run(RetainedRun(disposition, "test"))


def test_interaction_and_reset_evidence_is_retained_without_effects():
    proof = replay(
        event(Kind.REASONING_DELTA, text="thinking"),
        event(Kind.APPROVAL_REQUIRED, tool_name="Bash", tool_arguments={"command": "never execute"}),
        event(Kind.AWAIT_TOOL_RESULT, tool_use_id="question", tool_arguments={"questions": []}),
        event(Kind.ASSISTANT_DONE, text="", raw={"new_session_ack": True, "end_reason": "aborted"}),
    )
    original = deepcopy(proof)
    result = fold_retained_run(proof)
    assert result.cancelled and result.reset_ack
    assert result.reasoning == "thinking" and result.response_text == ""
    assert len(result.interactions) == 2
    # Repeating or mutating the result cannot mutate input/another fold.
    result.interactions[0].tool_arguments["command"] = "changed"
    assert proof == original
    assert fold_retained_run(proof).interactions[0].tool_arguments["command"] == "never execute"


@pytest.mark.parametrize("aborted", [True, False])
def test_abort_cause_is_preserved_only_on_cancelled_aggregate(aborted):
    proof = replay(event(Kind.ASSISTANT_DONE, text="partial", raw={
        "end_reason": "aborted" if aborted else "stop", "abort_cause": "host_shutdown",
    }))
    result = fold_retained_run(proof)
    assert result.abort_cause == ("host_shutdown" if aborted else None)
    assert result.cancelled is aborted


def test_tool_only_attachments_stay_with_tool_not_reply():
    proof = replay(
        event(Kind.TOOL_END, tool_use_id="a", attachments=[{"asset_id": "tool-image", "kind": "image"}]),
        event(Kind.ASSISTANT_DONE, text="Answer", attachments=[{"asset_id": "reply-image", "kind": "image"}]),
    )
    result = fold_retained_run(proof)
    assert result.attachments == ({"asset_id": "reply-image", "kind": "image"},)
    assert result.tools[0]["attachments"] == [{"asset_id": "tool-image", "kind": "image"}]


def test_parallel_tool_merge_is_by_identity_not_lifo_and_missing_result_not_fabricated():
    proof = replay(
        event(Kind.TOOL_START, tool_use_id="a", tool_name="Read", tool_arguments={"file": "a"}),
        event(Kind.TOOL_START, tool_use_id="b", tool_name="Read", tool_arguments={"file": "b"}),
        event(Kind.TOOL_END, tool_use_id="a", tool_result="actual a", tool_error=False),
        event(Kind.ASSISTANT_DONE, text="Answer"),
    )
    result = fold_retained_run(proof)
    assert result.tools[0]["result"] == "actual a"
    assert result.tools[1]["tool_use_id"] == "b" and "result" not in result.tools[1]
    assert fold_retained_run(proof) == result


def test_usage_enrichment_and_attachments_are_snapshot_data_not_double_counted():
    proof = replay(
        event(Kind.USAGE_UPDATE, token_usage={"output_tokens": 1}),
        event(Kind.TOOL_END, tool_use_id="a", attachments=[{"asset_id": "image", "kind": "image"}]),
        event(Kind.ASSISTANT_DONE, text="Answer", token_usage={"output_tokens": 2},
              attachments=[{"asset_id": "image", "kind": "image"}]),
        event(Kind.USAGE_UPDATE, token_usage={"output_tokens": 3, "cost": 0.01}),
        event(Kind.ERROR, text="provider failed"),
    )
    original = deepcopy(proof)
    result = fold_retained_run(proof)
    assert result.token_usage == {"output_tokens": 3, "cost": 0.01}
    assert len(result.attachments) == 1
    assert result.error_text == "provider failed"
    result.token_usage["output_tokens"] = 999
    assert proof == original
