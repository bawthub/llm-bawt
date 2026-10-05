"""Tests for the path-agnostic turn-completion signal (turn_logs.ended_at).

The `ended_at` stamp is the single source of truth for "is this turn still
running?" — set exactly once on the first terminal transition by
save_turn/update_turn.  Both writers delegate the decision to `_is_terminal`,
so testing that pure function (plus the terminal-status set) validates the
stamping behavior without needing a live Postgres engine.
"""

from datetime import datetime, timedelta, timezone
import json

from sqlmodel import Session, SQLModel, create_engine, select

from llm_bawt.service.turn_logs import (
    TERMINAL_TURN_STATUSES,
    TurnLog,
    TurnLogStore,
    _is_terminal,
)


def _store():
    engine = create_engine("sqlite://")
    from llm_bawt.service.tool_call_store import ToolCallRecord
    SQLModel.metadata.create_all(engine, tables=[TurnLog.__table__, ToolCallRecord.__table__])
    store = TurnLogStore.__new__(TurnLogStore)
    store.engine = engine
    return store


def _row(store, turn_id):
    with Session(store.engine) as s:
        return s.exec(select(TurnLog).where(TurnLog.id == turn_id)).first()


def _naive(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None)


def test_in_progress_statuses_are_not_terminal():
    # The two non-terminal statuses: streaming (streaming path) and pending
    # (non-streaming / bot-to-bot path). Neither should stamp ended_at.
    assert _is_terminal("streaming", None) is False
    assert _is_terminal("pending", None) is False
    assert "streaming" not in TERMINAL_TURN_STATUSES
    assert "pending" not in TERMINAL_TURN_STATUSES


def test_all_terminal_statuses_stamp():
    for status in ("ok", "completed", "error", "timeout", "cancelled", "aborted"):
        assert _is_terminal(status, None) is True, status


def test_end_reason_alone_is_terminal():
    # Streaming finalize stamps end_reason without a terminal status — that
    # path must still count as done (covers end_reason="stop"/"question"/etc).
    assert _is_terminal(None, "stop") is True
    assert _is_terminal(None, "question") is True
    assert _is_terminal("streaming", "stop") is True


def test_no_signal_is_not_terminal():
    assert _is_terminal(None, None) is False
    assert _is_terminal("", None) is False


def test_unknown_status_without_end_reason_is_not_terminal():
    # Defensive: an unrecognized status shouldn't accidentally mark done.
    assert _is_terminal("weird_new_status", None) is False


def test_update_turn_stamps_ended_at_from_created_at_plus_latency():
    store = _store()
    created_at = datetime(2026, 7, 10, 16, 37, 43, tzinfo=timezone.utc)
    with Session(store.engine) as s:
        s.add(TurnLog(
            id="turn",
            bot_id="caid",
            user_id="nick",
            status="streaming",
            created_at=created_at,
        ))
        s.commit()

    store.update_turn(turn_id="turn", status="ok", end_reason="stop", latency_ms=1250.0)

    row = _row(store, "turn")
    assert row.status == "ok"
    assert row.end_reason == "stop"
    assert row.ended_at == _naive(created_at + timedelta(milliseconds=1250))


def test_timeout_transition_persists_partial_token_usage():
    store = _store()
    with Session(store.engine) as session:
        session.add(TurnLog(
            id="turn-partial-usage",
            bot_id="snark",
            user_id="nick",
            status="streaming",
        ))
        session.commit()

    usage = {
        "resident_tokens": 139613,
        "resident_source": "interrupted_iteration_total_input",
        "usage_status": "partial",
    }
    store.update_turn(
        turn_id="turn-partial-usage",
        status="timeout",
        end_reason="error",
        token_usage=usage,
    )

    row = _row(store, "turn-partial-usage")
    assert row.status == "timeout"
    assert row.end_reason == "error"
    assert json.loads(row.token_usage_json) == usage
    assert row.ended_at is not None


def test_active_turns_are_not_hidden_by_the_history_age_window():
    store = _store()
    store.ttl_hours = None
    old = datetime.now(timezone.utc) - timedelta(hours=8)
    with Session(store.engine) as session:
        session.add_all([
            TurnLog(id="old-active", bot_id="al", status="streaming", created_at=old),
            TurnLog(id="old-finished", bot_id="al", status="ok", created_at=old, ended_at=old),
        ])
        session.commit()

    active, total = store.list_turns(bot_id="al", active_only=True, since_hours=1)
    assert total == 1
    assert [turn.id for turn in active] == ["old-active"]
    historical, total = store.list_turns(bot_id="al", since_hours=1)
    assert total == 0
    assert historical == []


def test_successful_late_completion_repairs_stale_timeout_ended_at():
    store = _store()
    created_at = datetime(2026, 7, 10, 16, 37, 43, tzinfo=timezone.utc)
    stale_ended_at = created_at + timedelta(milliseconds=164)
    with Session(store.engine) as s:
        s.add(TurnLog(
            id="turn",
            bot_id="caid",
            user_id="nick",
            status="timeout",
            end_reason="timeout",
            created_at=created_at,
            ended_at=stale_ended_at,
        ))
        s.commit()

    store.update_turn(
        turn_id="turn",
        status="ok",
        end_reason="stop",
        latency_ms=528_841.177,
        response_text="Here - I'm with you.",
    )

    row = _row(store, "turn")
    assert row.status == "ok"
    assert row.end_reason == "stop"
    assert row.response_text == "Here - I'm with you."
    assert row.ended_at == _naive(created_at + timedelta(milliseconds=528_841.177))
