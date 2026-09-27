"""Tool history must distinguish persisted starts from terminal turns (TASK-949)."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlmodel import SQLModel, Session, create_engine

from llm_bawt.service.routes import turn_logs as routes
from llm_bawt.service.tool_call_store import ToolCallRecord


@pytest.mark.parametrize("ended", [False, True])
def test_history_exposes_parent_terminal_signal_without_completing_tool(monkeypatch, ended):
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine, tables=[ToolCallRecord.__table__])
    timestamp = datetime(2026, 9, 27, tzinfo=timezone.utc)
    row = SimpleNamespace(
        id="turn-history", created_at=timestamp, request_id=None,
        model=None, bot_id="snark", user_id="nick",
        trigger_message_id="user-history", ended_at=timestamp if ended else None,
    )
    with Session(engine) as session:
        session.add(ToolCallRecord(
            turn_id=row.id, bot_id="snark", user_id="nick", tool_name="Bash",
            call_id="call-history", started_at=timestamp.timestamp(), result_text="",
        ))
        session.commit()
    store = SimpleNamespace(engine=engine, list_turns=lambda **kwargs: ([row], 1))
    monkeypatch.setattr(routes, "get_turn_log_store", lambda: store)
    response = routes.get_tool_call_events(
        bot_id="snark", user_id="nick", message_id=None, message_ids=["user-history"],
        after=None, before=None, since_hours=168, limit=200,
    )
    event = response.events[0]
    assert event.turn_ended_at == row.ended_at
    assert event.tool_calls[0]["status"] == "pending"
    assert event.tool_calls[0]["result"] is None
    assert event.tool_calls[0]["started_at"] == timestamp.timestamp()
    assert "turn_ended_at" in response.model_dump(mode="json")["events"][0]
    engine.dispose()
