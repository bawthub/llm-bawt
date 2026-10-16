"""TASK-1095 #3: tool duration measures execution, not queue wait.

The model may emit a tool call long before the CLI runs it (serial unsafe tools,
a foreground sub-agent ahead in the batch, an approval wait). The bridge's
PreToolUse hook reports the real start; the app stamps it on the canonical row
and the UI flips the card from queued to running.
"""
from __future__ import annotations

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from llm_bawt.service.pending_tool_calls import PendingToolCallCorrelator
from llm_bawt.service.tool_call_store import ToolCallRecord, ToolCallResultPayloadRecord, ToolCallStore
from llm_bawt.service.tool_event_coordinator import ToolEventCoordinator
from llm_bawt.service.tool_execution_stamps import ToolExecutionStamps
from llm_bawt.service.turn_logs import TurnLog

ENVELOPE = {"turn_id": "turn", "trigger_message_id": "m", "bot_id": "al", "user_id": "nick"}


@pytest.fixture
def engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(
        engine, tables=[TurnLog.__table__, ToolCallRecord.__table__, ToolCallResultPayloadRecord.__table__],
    )
    with Session(engine) as session:
        session.add(TurnLog(id="turn", bot_id="al", user_id="nick", status="ok", trigger_message_id="m"))
        session.commit()
    return engine


def _start(coord, call_id, ts=100.0):
    coord.start({"turn_id": "turn", "bot_id": "al", "user_id": "nick", "call_id": call_id,
                 "tool_use_id": f"tu-{call_id}", "tool_name": "Bash", "arguments": {}, "ts": ts})


def _end(coord, call_id, ts):
    coord.end({"turn_id": "turn", "bot_id": "al", "user_id": "nick", "call_id": call_id,
               "tool_use_id": f"tu-{call_id}", "tool_name": "Bash", "result": "ok", "ts": ts,
               "is_error": False})


def _row(engine, call_id) -> ToolCallRecord:
    with Session(engine) as session:
        return session.exec(select(ToolCallRecord).where(ToolCallRecord.call_id == call_id)).one()


def test_duration_excludes_queue_wait_and_first_stamp_wins(engine):
    coord = ToolEventCoordinator(engine)
    _start(coord, "c1", ts=100.0)
    public = coord.executing({**ENVELOPE, "event": "tool_executing", "call_id": "c1",
                              "tool_use_id": "tu-c1", "ts": 160.0})
    assert public["_tool_persisted"] is True
    coord.executing({**ENVELOPE, "call_id": "c1", "tool_use_id": "tu-c1", "ts": 170.0})
    _end(coord, "c1", 165.0)
    row = _row(engine, "c1")
    assert row.exec_started_at == 160.0
    assert row.duration_ms == pytest.approx(5000.0)


def test_untracked_call_keeps_emit_time_duration_and_late_stamp_is_ignored(engine):
    coord = ToolEventCoordinator(engine)
    _start(coord, "c2", ts=100.0)
    _end(coord, "c2", 103.0)
    coord.executing({**ENVELOPE, "call_id": "c2", "tool_use_id": "tu-c2", "ts": 104.0})
    row = _row(engine, "c2")
    assert row.exec_started_at is None
    assert row.duration_ms == pytest.approx(3000.0)


def test_stamp_never_creates_a_row(engine):
    ToolCallStore(engine).mark_executing(turn_id="turn", call_id="ghost", tool_use_id="tu-ghost",
                                         exec_started_at=1.0)
    with Session(engine) as session:
        assert session.exec(select(ToolCallRecord)).all() == []


def test_stamp_that_outruns_its_tool_start_is_held_then_released():
    calls = PendingToolCallCorrelator()
    published: list[dict] = []
    stamps = ToolExecutionStamps(calls, published.append, ENVELOPE)

    stamps.on_executing({"tool_use_id": "tu-1", "name": "Read", "ts": 5.0})
    assert published == []
    calls.start(call_id="c-1", tool_name="Read", tool_use_id="tu-1")
    stamps.on_tool_started("tu-1", "c-1")
    stamps.on_tool_started("tu-1", "c-1")  # released once
    assert published == [{"_type": "tool_event", "event": "tool_executing", **ENVELOPE,
                          "tool_name": "Read", "call_id": "c-1", "tool_use_id": "tu-1", "ts": 5.0}]

    calls.start(call_id="c-2", tool_name="Bash", tool_use_id="tu-2")
    stamps.on_executing({"tool_use_id": "tu-2", "name": "Bash", "ts": 6.0})
    assert published[-1]["call_id"] == "c-2"


def test_held_stamps_are_bounded_and_idless_stamps_dropped():
    published: list[dict] = []
    stamps = ToolExecutionStamps(PendingToolCallCorrelator(), published.append, ENVELOPE)
    stamps.on_executing({"name": "Read"})
    for i in range(ToolExecutionStamps.MAX_HELD + 50):
        stamps.on_executing({"tool_use_id": f"tu-{i}"})
    assert len(stamps._held) == ToolExecutionStamps.MAX_HELD
    assert published == []
