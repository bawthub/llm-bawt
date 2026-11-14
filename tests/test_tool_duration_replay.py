"""Replayed non-stream tool calls have no measured duration (TASK-2292)."""
from __future__ import annotations

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from llm_bawt.service.tool_call_store import ToolCallRecord, ToolCallResultPayloadRecord, ToolCallStore
from llm_bawt.service.tool_event_coordinator import ToolEventCoordinator
from llm_bawt.service.turn_logs import TurnLog


def _engine():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng, tables=[
        TurnLog.__table__, ToolCallRecord.__table__, ToolCallResultPayloadRecord.__table__,
    ])
    return eng


def _rows(engine):
    with Session(engine) as session:
        return {r.call_id: r for r in session.exec(select(ToolCallRecord)).all()}


def test_replayed_call_stores_null_duration_not_zero():
    engine = _engine()
    ToolEventCoordinator(engine).record_completed(
        [{"tool": "time", "call_id": "replayed", "parameters": {}, "result": "ok"}],
        turn_id="turn-1", trigger_message_id="m", bot_id="b", user_id="u",
    )
    row = _rows(engine)["replayed"]
    assert row.ended_at is not None and row.duration_ms is None


def test_measured_call_keeps_its_duration():
    engine = _engine()
    store = ToolCallStore(engine)
    store.save_start(turn_id="turn-1", bot_id="b", user_id="u", call_id="live", tool_name="time",
                     arguments={}, iteration=1, started_at=10.0)
    coordinator = ToolEventCoordinator(engine)
    coordinator.end({"tool_name": "time", "turn_id": "turn-1", "call_id": "live", "bot_id": "b",
                     "user_id": "u", "result": "ok", "ts": 10.25})
    assert _rows(engine)["live"].duration_ms == 250.0
