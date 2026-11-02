"""Actual worker/publisher boundaries behind the voice-step observer."""
import asyncio
from types import SimpleNamespace

from llm_bawt.service.turn_stream_publish import TurnStreamPublishMixin
from tests import test_turn_detach

turn_store = test_turn_detach.turn_store


def test_worker_flushes_pre_injection_text_before_barrier_and_tool_boundary(turn_store, monkeypatch):
    lifecycle = {"event": "steer_lifecycle", "message_id": "voice", "state": "started",
                 "run_request_id": "run", "steer_rpc_id": "rpc", "origin": "user", "proof": None}

    def stream(_owner):
        yield "Old 🙂"
        yield lifecycle
        yield "Fresh reply"
        yield {"event": "tool_call", "name": "Read", "arguments": {}, "tool_use_id": "tool"}
        yield {"event": "tool_result", "name": "Read", "result": "not spoken", "tool_use_id": "tool"}

    worker, ctx, _, _, events = test_turn_detach._agent_worker(turn_store, monkeypatch, stream)
    worker._stream_to_queue()
    relevant = [e for e in events if e.get("_type") in {"text_delta", "steer_lifecycle", "tool_event", "turn_complete"}]
    assert [e["_type"] for e in relevant] == [
        "text_delta", "steer_lifecycle", "text_delta", "tool_event", "tool_event", "turn_complete"]
    assert relevant[0]["delta"] == "Old 🙂" and relevant[0]["text_offset"] == 0
    barrier = relevant[1]
    assert barrier["text_offset"] == 6  # UTF-16, not Python code points.
    assert {k: barrier[k] for k in lifecycle if k != "event"} == {
        k: v for k, v in lifecycle.items() if k != "event"}
    assert barrier["turn_id"] == "A" and barrier["bot_id"] == "test"
    assert relevant[2]["delta"] == "Fresh reply" and relevant[2]["text_offset"] == 6
    assert relevant[3]["event"] == "tool_start" and relevant[3]["text_offset"] == 17
    assert relevant[-1]["text_offset"] == ctx.stream_text_offset == 17


def test_lifecycle_control_frame_does_not_confirm_model_output_or_reap(turn_store, monkeypatch):
    def stream(_owner):
        yield {"event": "steer_lifecycle", "message_id": "voice", "state": "started"}

    worker, ctx, _, _, events = test_turn_detach._agent_worker(turn_store, monkeypatch, stream)
    from unittest.mock import Mock
    ctx.svc._turn_log_store.reap_other_open_turns = Mock(return_value=[])
    observed = Mock()
    monkeypatch.setattr("llm_bawt.service.turn_stream_worker.SessionPinConfirmer.observe", observed)
    worker._stream_to_queue()
    observed.assert_not_called()
    ctx.svc._turn_log_store.reap_other_open_turns.assert_not_called()
    assert any(e.get("_type") == "steer_lifecycle" for e in events)


def test_async_publish_lock_preserves_text_barrier_tool_and_terminal_order():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        writes = []

        async def publish(_bot, _user, event):
            if event["_type"] == "text_delta":
                entered.set()
                await release.wait()
            writes.append(event["_type"])

        publisher = TurnStreamPublishMixin()
        publisher.ctx = SimpleNamespace(_redis_sub=SimpleNamespace(publish_tool_event=publish),
                                       bot_id="test", user_id="user", publish_lock=asyncio.Lock())
        first = asyncio.create_task(publisher._publish_unified({"_type": "text_delta"}))
        await entered.wait()
        rest = [asyncio.create_task(publisher._publish_unified({"_type": kind}))
                for kind in ["steer_lifecycle", "tool_event", "turn_complete"]]
        # Lock contenders enter in task creation order; the release happens on
        # this same loop, not via a timing sleep or a wall-clock assertion.
        release.set()
        await asyncio.gather(first, *rest)
        assert writes == ["text_delta", "steer_lifecycle", "tool_event", "turn_complete"]

    asyncio.run(run())
