from __future__ import annotations

import json

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from agent_bridge.tool_results import ToolResultPayload
from llm_bawt.service.tool_call_store import (
    ToolCallRecord,
    ToolCallResultPayloadRecord,
    ToolCallStore,
)


def test_mcp_resolution_replaces_placeholder_on_original_tool_row():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    from llm_bawt.service.turn_logs import TurnLog
    SQLModel.metadata.create_all(
        bind=engine,
        tables=[TurnLog.__table__, ToolCallRecord.__table__, ToolCallResultPayloadRecord.__table__],
    )
    calls = ToolCallStore(engine)
    calls.save_start(
        turn_id="turn-1",
        bot_id="loopy",
        user_id="nick",
        call_id="call-record-1",
        tool_name="mcp__bawthub__ops_run",
        arguments={"operation": "llm-bawt.restart-bridge"},
        started_at=100.0,
        tool_use_id="call-original",
    )
    calls.save_result(
        turn_id="turn-1",
        call_id="call-record-1",
        tool_use_id="call-original",
        tool_name="mcp__bawthub__ops_run",
        bot_id="loopy",
        user_id="nick",
        payload=ToolResultPayload.from_value({"status": "approval_required"}),
        ended_at=100.2,
        is_error=False,
    )

    assert calls.resolve_approval_result(
        tool_use_id="call-original",
        approval_request_id="mcp-appr-1",
        approval_status="approved",
        result={"operation": "llm-bawt.restart-bridge", "state": "accepted"},
        is_error=False,
    )

    with Session(engine) as session:
        rows = list(session.exec(select(ToolCallRecord)).all())
    assert len(rows) == 1
    row = rows[0]
    assert row.tool_use_id == "call-original"
    assert row.approval_request_id == "mcp-appr-1"
    assert row.approval_status == "approved"
    assert row.is_error is False
    assert json.loads(row.result_text) == {
        "operation": "llm-bawt.restart-bridge",
        "state": "accepted",
    }
