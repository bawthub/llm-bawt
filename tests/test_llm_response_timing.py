"""TASK-2114: bridge-measured per-response wait/stream timing."""
from __future__ import annotations

from types import SimpleNamespace

from sqlmodel import Session, SQLModel, create_engine

from claude_code_bridge.response_tracker import ResponseTracker
from llm_bawt.service.llm_response_store import LlmResponseRecord, LlmResponseStore
from llm_bawt.service.tool_event_coordinator import ToolEventCoordinator
from llm_bawt.service.turn_logs import TurnLog
from llm_bawt.service.tool_call_store import ToolCallRecord, ToolCallResultPayloadRecord


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _start(rid):
    return {"type": "message_start", "message": {"id": rid, "model": "m", "usage": {"input_tokens": 1}}}


STOP = {"type": "message_stop"}


def _results(parent=None):
    return SimpleNamespace(parent_tool_use_id=parent,
                           content=[SimpleNamespace(type="tool_result", tool_use_id="t")])


def test_wait_runs_from_turn_start_then_from_the_last_tool_result():
    clock = Clock()
    t = ResponseTracker(clock=clock)
    t.on_turn_start()
    clock.now = 103.25
    rec = t.on_stream_event(_start("r1"), source=None, seq=1)
    assert rec.wait_ms == 3250 and rec.stream_ms is None
    clock.now = 105.0
    assert t.on_stream_event(STOP, source=None, seq=2).stream_ms == 1750
    # Two parallel tool results: the wait starts at the last one.
    clock.now = 110.0
    t.on_user_message(_results())
    clock.now = 112.0
    t.on_user_message(_results())
    clock.now = 112.5
    assert t.on_stream_event(_start("r2"), source=None, seq=3).wait_ms == 500
    assert t.record("r1").payload()["wait_ms"] == 3250


def test_no_fresh_boundary_means_no_wait_never_zero():
    clock = Clock()
    t = ResponseTracker(clock=clock)
    # No turn start recorded: unknown.
    assert t.on_stream_event(_start("r1"), source=None, seq=1).wait_ms is None
    t.on_stream_event(STOP, source=None, seq=2)
    # A boundary is consumed once: a second response without tool results has none.
    t.on_turn_start()
    t.on_stream_event(_start("r2"), source=None, seq=3)
    t.on_stream_event(STOP, source=None, seq=4)
    assert t.on_stream_event(_start("r3"), source=None, seq=5).wait_ms is None
    # Steer replacement drops the boundary.
    t.on_user_message(_results())
    t.on_query_replaced()
    assert t.on_stream_event(_start("r4"), source=None, seq=6).wait_ms is None


def test_subagent_and_non_tool_messages_set_no_boundary_and_get_no_timing():
    clock = Clock()
    t = ResponseTracker(clock=clock)
    t.on_turn_start()
    clock.now = 101.0
    t.on_stream_event(_start("r1"), source=None, seq=1)
    clock.now = 150.0
    t.on_user_message(_results(parent="toolu_agent"))          # subagent's tool result
    t.on_user_message(SimpleNamespace(parent_tool_use_id=None, content="hello"))
    assert t.on_stream_event(_start("r2"), source=None, seq=2).wait_ms is None
    # Subagent stream frames and snapshots never carry timing.
    assert t.on_stream_event(_start("s1"), source="toolu_agent", seq=3) is None
    sub = t.on_assistant_message(SimpleNamespace(message_id="s2", usage={}, model="m"),
                                 host_parent="toolu_agent", seq=4)
    assert sub.wait_ms is None and sub.stream_ms is None


def test_repeated_start_does_not_restamp_and_timing_change_republishes():
    clock = Clock()
    t = ResponseTracker(clock=clock)
    t.on_turn_start()
    clock.now = 102.0
    assert t.on_stream_event(_start("r1"), source=None, seq=1).wait_ms == 2000
    clock.now = 109.0
    t.on_turn_start()
    rec = t.on_stream_event(_start("r1"), source=None, seq=2)
    assert rec is None and t.record("r1").wait_ms == 2000
    # stream_ms is a change, so the stop republishes even with unchanged usage.
    assert t.on_stream_event(STOP, source=None, seq=3).stream_ms == 7000


def _engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine, tables=[
        TurnLog.__table__, ToolCallRecord.__table__, ToolCallResultPayloadRecord.__table__,
        LlmResponseRecord.__table__,
    ])
    with Session(engine) as session:
        session.add(TurnLog(id="turn", bot_id="al", user_id="nick", status="streaming", trigger_message_id="m"))
        session.commit()
    return engine


def test_store_sets_timing_once_and_rejects_bad_values():
    store = LlmResponseStore(_engine())
    up = lambda **kw: store.upsert(turn_id="turn", response_id="r1", first_seen_seq=1, **kw)  # noqa: E731
    rec = up()
    assert rec["wait_ms"] is None and rec["stream_ms"] is None
    assert up(wait_ms=-5, stream_ms=True)["wait_ms"] is None
    assert up(wait_ms=10**9)["wait_ms"] is None
    rec = up(wait_ms=3200)
    assert rec["wait_ms"] == 3200 and rec["stream_ms"] is None
    rec = up(wait_ms=None, stream_ms=900)    # replay without wait keeps it
    assert (rec["wait_ms"], rec["stream_ms"]) == (3200, 900)
    assert up(wait_ms=1, stream_ms=1)["wait_ms"] == 3200  # never rewritten


def test_coordinator_carries_timing_to_the_public_event():
    coord = ToolEventCoordinator(_engine())
    public = coord.response({"turn_id": "turn", "response_id": "r1", "first_seen_seq": 1,
                             "wait_ms": 4100, "stream_ms": 800, "usage_state": "final"})
    assert public["wait_ms"] == 4100 and public["stream_ms"] == 800
