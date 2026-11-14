"""TASK-737: characterization of the extracted streaming tool-event interceptor."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_bawt.service.pending_tool_calls import PendingToolCallCorrelator
from llm_bawt.service.tool_event_interceptor import ToolEventInterceptor, js_len
from llm_bawt.service.turn_stream_context import _TEXT_DELTA_FLUSH_CHARS


class _Tts:
    def __init__(self):
        self.offset = 0
        self.fed = []

    def feed(self, s):
        self.fed.append(s)
        start = self.offset
        self.offset += len(s)
        return s if start == 0 else ""


def _make(*, agent=True, reaped=(), changed=None, usage=None, tts=None, redis=None):
    published: list[dict] = []
    ctx = SimpleNamespace(
        turn_log_id="turn-1", trigger_message_id="trig-1", bot_id="bot-a", user_id="u1",
        is_agent_backend=agent, cancel_event=threading.Event(), tts_scrubber=tts,
        stream_text_offset=None, _upstream_model=[None], token_usage_holder=[None],
        agent_attachments_holder=[], reasoning_holder=[""], tool_call_details_holder=[],
        model_alias="m@path", _redis_sub=redis, loop=Mock(),
    )
    svc = SimpleNamespace(
        _turn_log_store=SimpleNamespace(reap_other_open_turns=Mock(return_value=list(reaped))),
        _update_turn_log=Mock(), config=None,
    )
    changed_files = SimpleNamespace(persist=Mock(side_effect=changed or (lambda e: None)))
    turn_usage = SimpleNamespace(
        capture=Mock(side_effect=usage or (lambda **kw: None)),
        http_chunk=Mock(side_effect=lambda e: {"token_usage": e["token_usage"]}),
    )
    tool_calls = PendingToolCallCorrelator()
    pin = SimpleNamespace(observe=Mock())
    notices = SimpleNamespace(observe=Mock())
    stamps = SimpleNamespace(on_executing=Mock(), on_tool_started=Mock())
    approval, await_ = Mock(), Mock()
    interceptor = ToolEventInterceptor(
        ctx, svc=svc, publish_event_direct=lambda e: published.append(e),
        persist_publish_approval=approval, persist_publish_await=await_,
        changed_files=changed_files, turn_usage=turn_usage, pin_confirmer=pin,
        scheduled_notices=notices, tool_calls=tool_calls, exec_stamps=stamps,
        call_index=[0], confirmed_start_reaped=[False],
    )
    return SimpleNamespace(
        it=interceptor, ctx=ctx, svc=svc, published=published, pin=pin,
        notices=notices, stamps=stamps, approval=approval, await_=await_,
        changed=changed_files, usage=turn_usage,
    )


def _types(published):
    return [(e["_type"], e.get("event")) for e in published]


def test_text_tool_text_ordering_offsets_and_paragraph_break():
    h = _make()
    out = list(h.it.run([
        "Hi 🙂.",
        {"event": "tool_call", "name": "Bash", "arguments": {"c": 1}, "tool_use_id": "tu1",
         "provider": "claude"},
        {"event": "tool_result", "name": "Bash", "tool_use_id": "tu1", "result": "ok"},
        "Done",
    ]))
    pre = js_len("Hi 🙂.")
    assert pre == len("Hi 🙂.") + 1
    # Paragraph break injected before text that follows a tool.
    assert out[0] == "Hi 🙂." and out[-2:] == ["\n\n", "Done"]
    call = out[1]
    assert call["_text_offset"] == pre and call["_call_id"].startswith("call_")
    assert out[2]["_call_id"] == call["_call_id"]  # tool_use_id correlation
    # Text flushed before the tool event; tail flushed at end.
    assert _types(h.published) == [
        ("text_delta", None), ("tool_event", "tool_start"),
        ("tool_event", "tool_end"), ("text_delta", None),
    ]
    first, start, end, last = h.published
    assert first["text_offset"] == 0 and first["delta"] == "Hi 🙂."
    assert start["text_offset"] == pre and start["tool_use_id"] == "tu1"
    assert end["call_id"] == start["call_id"] and end["result"] == "ok"
    assert last["delta"] == "\n\nDone" and last["text_offset"] == pre
    assert h.ctx.stream_text_offset == pre + 2 + 4
    details = h.ctx.tool_call_details_holder
    assert details[0]["text_offset"] == pre and details[0]["result"] == "ok"
    h.stamps.on_tool_started.assert_called_once_with("tu1", call["_call_id"])


def test_protocol_checkpoints_complete_text_and_reasoning_before_events():
    h = _make()
    h.ctx.terminal_follower = Mock()
    checkpoints = []

    def checkpoint(**fields):
        checkpoints.append((fields, len(h.published)))

    h.ctx.terminal_follower.write_partial.side_effect = checkpoint
    out = list(h.it.run([
        {"event": "reasoning", "text": "thinking"}, "Hi 🙂.",
        {"event": "tool_call", "name": "Read", "arguments": {}, "tool_use_id": "call"},
        {"event": "tool_result", "name": "Read", "result": "done", "tool_use_id": "call"},
        "Answer",
    ]))
    assert out[-2:] == ["\n\n", "Answer"]
    assert [fields for fields, _ in checkpoints] == [
        {"text_value": "", "reasoning": "thinking"},
        {"text_value": "Hi 🙂.", "reasoning": "thinking"},
        {"text_value": "Hi 🙂.\n\nAnswer", "reasoning": "thinking"},
    ]
    assert [seen for _, seen in checkpoints] == [0, 1, 4]
    assert h.published[-1]["delta"] == "\n\nAnswer"
    h.svc._update_turn_log.assert_not_called()


def test_protocol_reasoning_batches_are_checkpointed_before_publish():
    h = _make()
    h.ctx.terminal_follower = Mock()
    checkpoints = []
    h.ctx.terminal_follower.write_partial.side_effect = lambda **fields: checkpoints.append(
        (fields, len(h.published))
    )
    chunks = [{"event": "reasoning", "text": "r"} for _ in range(1024)]
    out = list(h.it.run(chunks + ["Answer"]))
    reasoning_events = [event for event in h.published if event["_type"] == "reasoning_delta"]
    assert "".join(event["delta"] for event in reasoning_events) == "r" * 1024
    assert [event["reasoning_offset"] for event in reasoning_events] == [0, 512]
    assert len(checkpoints) == 3  # two reasoning batches, one answer
    assert checkpoints[0][0]["reasoning"] == "r" * 512
    assert checkpoints[1][0]["reasoning"] == "r" * 1024
    assert checkpoints[2][0]["text_value"] == "Answer"
    assert out[-1] == "Answer"


def test_protocol_fence_loss_stops_before_any_partial_event():
    from llm_bawt.service.turn_execution import TurnExecutionDetached
    from llm_bawt.service.turn_terminal_protocol import FollowerFenceLost
    from tests.test_terminal_follower import follower

    h = _make()
    guard = follower()
    guard.protocol.write_partial.side_effect = FollowerFenceLost("lost to another owner")
    h.ctx.terminal_follower = guard
    with pytest.raises(TurnExecutionDetached):
        list(h.it.run(["Never publish"] ))
    assert guard.execution.detach_requested()
    assert h.published == []
    guard.protocol.claim.assert_not_called()


def test_protocol_bridge_cancel_is_classified_without_generic_terminal_write():
    h = _make()
    h.ctx.terminal_follower = Mock()
    h.ctx.cancelled_holder = [False]
    assert list(h.it.run([{"event": "turn_cancelled", "abort_cause": "bridge_stop"}])) == []
    assert h.ctx.cancelled_holder[0]
    h.svc._update_turn_log.assert_not_called()


def test_text_buffer_flushes_at_threshold_and_feeds_tts():
    tts = _Tts()
    h = _make(tts=tts)
    chunk = "x" * _TEXT_DELTA_FLUSH_CHARS
    list(h.it.run(["ab", chunk]))
    assert [e["_type"] for e in h.published] == ["text_delta", "tts_delta"]
    assert h.published[0]["delta"] == "ab" + chunk and h.published[0]["text_offset"] == 0
    assert h.published[1]["tts_offset"] == 0 and tts.fed == ["ab" + chunk]


def test_reasoning_is_published_and_kept_out_of_text_offsets():
    h = _make()
    out = list(h.it.run([
        {"event": "reasoning", "text": "think🙂"}, {"event": "reasoning", "text": "more"},
        {"event": "reasoning", "text": ""}, "answer",
    ]))
    assert out == [
        {"_type": "reasoning_delta", "delta": "think🙂"},
        {"_type": "reasoning_delta", "delta": "more"}, "answer",
    ]
    r = [e for e in h.published if e["_type"] == "reasoning_delta"]
    assert [e["reasoning_offset"] for e in r] == [0, js_len("think🙂")]
    assert h.ctx.reasoning_holder[0] == "think🙂more"
    assert h.published[-1]["text_offset"] == 0


def test_confirmed_start_reaps_once_and_steer_frames_do_not_confirm():
    redis = SimpleNamespace(publish_tool_event=Mock(return_value="coro"))
    h = _make(reaped=[{"id": "old", "user_id": "u2"}], redis=redis)
    import llm_bawt.service.tool_event_interceptor as mod
    orig = mod.asyncio.run_coroutine_threadsafe
    mod.asyncio.run_coroutine_threadsafe = Mock()
    try:
        list(h.it.run([{"event": "steer_lifecycle", "phase": "x"}, "a", "b"]))
    finally:
        mod.asyncio.run_coroutine_threadsafe = orig
    assert h.published[0]["_type"] == "steer_lifecycle" and "event" not in h.published[0]
    h.svc._turn_log_store.reap_other_open_turns.assert_called_once_with(
        bot_id="bot-a", current_turn_id="turn-1")
    bot, uid, evt = redis.publish_tool_event.call_args.args
    assert (bot, uid, evt["turn_id"], evt["status"]) == ("bot-a", "u2", "old", "timeout")
    assert h.pin.observe.call_count == 2 and h.notices.observe.call_count == 2


def test_native_cancel_stops_before_publishing():
    h = _make(agent=False)
    h.ctx.cancel_event.set()
    assert list(h.it.run(["a"])) == []
    assert h.published == []
    h.svc._turn_log_store.reap_other_open_turns.assert_not_called()


def test_control_events_are_consumed_not_yielded():
    usage_evt = {"_type": "turn_usage", "token_usage": {"in": 1}}
    h = _make(usage=lambda **kw: usage_evt)
    out = list(h.it.run([
        {"event": "turn_cancelled", "abort_cause": "user"},
        {"event": "metadata", "upstream_model": "up"},
        {"event": "live_usage", "token_usage": {"in": 1}},
        {"event": "token_usage", "token_usage": {"in": 2}},
        {"event": "attachments", "attachments": [{"a": 1}, "bad"]},
        {"event": "tool_executing", "tool_use_id": "t"},
        {"event": "subagent_progress", "task_id": "k", "tool_use_id": None},
        {"event": "llm_response", "response_id": "r1"},
        {"event": "upstream_status", "status": "retrying"},
    ]))
    assert out == [{"_type": "turn_usage", "token_usage": {"in": 1}}]
    h.svc._update_turn_log.assert_called_once_with(
        turn_id="turn-1", status="aborted", end_reason="aborted", abort_cause="user")
    assert h.ctx._upstream_model[0] == "up"
    assert h.ctx.token_usage_holder[0] == {"in": 2}
    assert h.ctx.agent_attachments_holder == [{"a": 1}]
    assert h.usage.capture.call_args.kwargs["model"] == "m@path"
    types = [e["_type"] for e in h.published]
    assert types[:3] == ["turn_usage", "subagent_progress", "llm_response_observed"]
    assert h.published[1]["task_id"] == "k" and h.published[1]["tool_use_id"] is None


def test_approvals_and_questions_persist_and_still_yield():
    h = _make()
    a = {"event": "approval_required", "id": 1}
    q = {"event": "await_tool_result", "id": 2}
    assert list(h.it.run([a, q])) == [a, q]
    h.approval.assert_called_once_with(a)
    h.await_.assert_called_once_with(q)


def test_file_changed_persisted_before_publish_and_tool_end_metadata_order():
    def persist(e):
        return {"file": e.get("file") or "f.py", "summary": "s"}

    h = _make(changed=persist)
    list(h.it.run([
        {"event": "file_changed", "file": "a.py", "tool_use_id": "t0", "tool_name": "Edit"},
        {"event": "tool_call", "name": "Edit", "tool_use_id": "t1"},
        {"event": "tool_result", "name": "Edit", "tool_use_id": "t1", "result": "x"},
        {"event": "tool_call", "name": "Edit", "tool_use_id": "t2"},
        {"event": "tool_result", "name": "Edit", "tool_use_id": "t2", "is_error": True},
    ]))
    assert _types(h.published) == [
        ("file_changed", None), ("tool_event", "tool_start"), ("tool_event", "tool_end"),
        ("file_changed", None), ("tool_event", "tool_start"), ("tool_event", "tool_end"),
    ]
    assert h.published[0]["file"] == "a.py"
    # Errored tool_end is not persisted as a change.
    assert h.changed.persist.call_count == 2


@pytest.mark.parametrize("name", ["Read", ""])
def test_tool_result_without_id_falls_back_to_name_or_lifo(name):
    h = _make()
    out = list(h.it.run([
        {"event": "tool_call", "name": "Read"},
        {"event": "tool_result", "name": name, "result": "r"},
    ]))
    assert out[1]["_call_id"] == out[0]["_call_id"]
