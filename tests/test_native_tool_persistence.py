"""Native model-facing output stays exact while timeline writes are best-effort."""
from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from llm_bawt.service import tool_call_store, turn_stream_publish
from llm_bawt.service.tool_call_store import ToolCallRecord, ToolCallResultPayloadRecord
from llm_bawt.service.tool_event_coordinator import ToolEventCoordinator
from llm_bawt.service.turn_logs import TurnLog
from llm_bawt.service.turn_stream_publish import TurnStreamPublishMixin
from llm_bawt.tools.parser import format_tool_result
from tests import test_native_tool_stream

native_rig = test_native_tool_stream.rig
tool_result = test_native_tool_stream.tool_result


@pytest.fixture
def durable_rig(native_rig, monkeypatch):
    rig = native_rig
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine, tables=[
        TurnLog.__table__, ToolCallRecord.__table__, ToolCallResultPayloadRecord.__table__,
    ])
    monkeypatch.setattr(tool_call_store, "_get_tool_blob_backend", lambda: None)
    coordinator = ToolEventCoordinator(engine)
    rig.ctx._tool_event_coordinator = coordinator
    rig.ctx.tool_start_persist_failures = 0
    rig.ctx.publish_lock = asyncio.Lock()
    async def publish(_bot, _user, event):
        rig.events.append(event)
    rig.ctx._redis_sub = SimpleNamespace(publish_tool_event=publish)
    publisher = TurnStreamPublishMixin()
    publisher.ctx = rig.ctx
    monkeypatch.setattr(
        turn_stream_publish.asyncio, "run_coroutine_threadsafe",
        lambda coroutine, _loop: asyncio.run(coroutine),
    )
    rig.stream.publish_event_direct = publisher._publish_event_direct
    rig.engine, rig.coordinator = engine, coordinator
    yield rig
    engine.dispose()


def _requests(rig, outputs):
    requests = []
    rig.executor.execute.side_effect = outputs
    calls = tool_result()
    if len(outputs) > 1:
        second = tool_result()["tool_calls"][0]
        second["id"] = "call-second"
        calls["tool_calls"].append(second)
    def stream(messages, **_kwargs):
        requests.append(list(messages))
        return iter([calls] if len(requests) == 1 else ["Completed"])
    rig.client.stream_with_tools.side_effect = stream
    return requests


def _rows(rig):
    with Session(rig.engine) as session:
        return session.exec(select(ToolCallRecord).order_by(ToolCallRecord.id)).all()


def test_long_unicode_native_results_remain_exact_in_store_and_continuation(durable_rig):
    rig = durable_rig
    first = "\t" + "café 雪 😀\n" * 400 + " first tail \n"
    second = "tool failed: " + "résultat 🧪\n" * 320 + "second tail"
    requests = _requests(rig, [first, format_tool_result("GetLiveContext", None, error=second)])
    assert list(rig.stream.stream(rig.messages, {})) == ["\n\n", "Completed"]
    assert [m.content for m in requests[1] if m.role == "tool"] == [first, second]
    rows = _rows(rig)
    assert len(rows) == 2
    assert [row.is_error for row in rows] == [False, True]
    ends = [event for event in rig.events if event["event"] == "tool_end"]
    for row, event, expected in zip(rows, ends, [first, second]):
        assert len(expected) > 2_001
        assert row.result_text == event["result"] == expected[:2_000]
        assert "tool_result_payload" not in event
        assert expected not in json.dumps(event, ensure_ascii=False)
        digest = hashlib.sha256(expected.encode("utf-8")).hexdigest()
        assert event["result_meta"]["sha256"] == row.result_sha256 == digest
        assert event["result_meta"]["available"] is True
        assert row.result_total_bytes == len(expected.encode("utf-8"))
        store = rig.coordinator.store
        text, offset = "", 0
        while True:
            page = store.page(row.id, bot_id=rig.ctx.bot_id, user_id=rig.ctx.user_id,
                              offset=offset, limit=701)
            assert page is not None
            text += page.content
            if page.next_offset is None:
                break
            offset = page.next_offset
        assert text == expected
        raw = store.raw_range(row.id, bot_id=rig.ctx.bot_id, user_id=rig.ctx.user_id)
        assert raw is not None and raw.data == expected.encode("utf-8")

    from llm_bawt.service.routes.turn_logs import _records_to_calls
    history = _records_to_calls(rows)
    assert [call["result"] for call in history] == [first[:2_000], second[:2_000]]
    assert first not in json.dumps(history, ensure_ascii=False)
    assert second not in json.dumps(history, ensure_ascii=False)


@pytest.mark.parametrize("redis_enabled", [True, False])
def test_start_persistence_failure_is_logged_counted_and_never_blocks_tools(
    durable_rig, monkeypatch, redis_enabled,
):
    rig = durable_rig
    if not redis_enabled:
        rig.ctx._redis_sub = None
    failure = Mock(side_effect=RuntimeError("timeline unavailable"))
    monkeypatch.setattr(rig.coordinator.store, "save_start", failure)
    warning = Mock()
    monkeypatch.setattr(turn_stream_publish.log, "warning", warning)
    outputs = ["雪 one\n" * 400, "😀 two\n" * 400]
    requests = _requests(rig, outputs)
    assert list(rig.stream.stream(rig.messages, {})) == ["\n\n", "Completed"]
    assert rig.executor.execute.call_count == failure.call_count == 2
    assert rig.ctx.tool_start_persist_failures == warning.call_count == 2
    assert [call.args[-1] for call in warning.call_args_list] == [1, 2]
    assert all(call.kwargs["exc_info"] for call in warning.call_args_list)
    assert [m.content for m in requests[1] if m.role == "tool"] == outputs
    # End persistence recovers both rows without needing a successful start.
    assert len(_rows(rig)) == 2
    if redis_enabled:
        starts = [event for event in rig.events if event["event"] == "tool_start"]
        assert len(starts) == 2
        assert all("record_id" not in event and "_tool_persisted" not in event for event in starts)
        assert all(event["result_meta"]["available"] for event in rig.events
                   if event["event"] == "tool_end")


def test_end_storage_failure_keeps_exact_continuation_and_bounded_unavailable_event(
    durable_rig, monkeypatch,
):
    rig = durable_rig
    output = "long résultat 🧪\n" * 400
    requests = _requests(rig, [output])
    def execute(_call):
        # Start succeeded. The actual end-store guard must tolerate a DB outage.
        monkeypatch.setattr(tool_call_store, "Session", Mock(side_effect=RuntimeError("DB offline")))
        return output
    rig.executor.execute.side_effect = execute
    assert list(rig.stream.stream(rig.messages, {})) == ["\n\n", "Completed"]
    assert [m.content for m in requests[1] if m.role == "tool"] == [output]
    (end,) = [event for event in rig.events if event["event"] == "tool_end"]
    assert end["result"] == output[:2_000]
    assert end["result_meta"]["available"] is False
    assert end["result_meta"]["record_id"] is None
    assert "tool_result_payload" not in end
    assert rig.ctx.tool_start_persist_failures == 0


def test_nonstream_replay_uses_the_same_lossless_store(durable_rig):
    rig = durable_rig
    output = "\t" + "unicode Ω 雪\n" * 400 + " tail \n"
    rig.coordinator.record_completed(
        [{"tool": "GetLiveContext", "call_id": "replayed", "result": output}],
        turn_id=rig.ctx.turn_log_id, trigger_message_id=rig.ctx.trigger_message_id,
        bot_id=rig.ctx.bot_id, user_id=rig.ctx.user_id,
    )
    (row,) = _rows(rig)
    assert row.result_text == output[:2_000]
    page = rig.coordinator.store.page(
        row.id, bot_id=rig.ctx.bot_id, user_id=rig.ctx.user_id, offset=0, limit=len(output),
    )
    assert page is not None and page.content == output
