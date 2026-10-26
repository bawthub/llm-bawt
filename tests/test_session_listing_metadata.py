"""Conversation-list metadata contract for the sessions API."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from llm_bawt.memory.postgresql_short_term import PostgreSQLShortTermManager


def _session_row(**overrides):
    values = {
        "id": "session-1",
        "bot_id": "snark",
        "user_id": "nick",
        "started_at": datetime(2026, 8, 4, 20, 0, 0),
        "ended_at": None,
        "archived_at": None,
        "status": "active",
        "session_metadata": {"title": "Picker work"},
        "message_count": 7,
        "turn_count": 4,
        "first_message_at": 1_775_500_000.0,
        "last_message_at": 1_775_500_600.5,
        "first_user_message": "please improve this",
        "last_user_message": "add the useful metadata",
        "last_assistant_message": "Added it; tests pass.",
        "last_assistant_at": 1_775_500_590.0,
        "summary": "Picker redesign and metadata work.",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_session_row_exposes_picker_metadata():
    row = PostgreSQLShortTermManager._row_to_session_dict(_session_row())

    assert row["message_count"] == 7
    assert row["turn_count"] == 4
    assert row["first_user_message"] == "please improve this"
    assert row["last_user_message"] == "add the useful metadata"
    assert row["summary"] == "Picker redesign and metadata work."
    assert row["first_message_at"] == datetime.fromtimestamp(
        1_775_500_000.0, tz=timezone.utc
    ).isoformat()
    assert row["last_activity_at"] == datetime.fromtimestamp(
        1_775_500_600.5, tz=timezone.utc
    ).isoformat()
    assert row["last_assistant_message"] == "Added it; tests pass."
    assert row["last_assistant_at"] == datetime.fromtimestamp(
        1_775_500_590.0, tz=timezone.utc
    ).isoformat()


def test_session_row_without_a_reply_reports_none():
    row = PostgreSQLShortTermManager._row_to_session_dict(
        _session_row(last_assistant_message=None, last_assistant_at=None)
    )

    assert row["last_assistant_message"] is None
    assert row["last_assistant_at"] is None


def test_session_info_carries_latest_reply():
    from llm_bawt.service.routes.sessions import _to_info

    info = _to_info(PostgreSQLShortTermManager._row_to_session_dict(_session_row()))

    assert info.last_assistant_message == "Added it; tests pass."
    assert info.last_assistant_at == datetime.fromtimestamp(
        1_775_500_590.0, tz=timezone.utc
    ).isoformat()


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _Connection:
    def __init__(self, rows, calls):
        self._rows = rows
        self._calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, statement, params):
        self._calls.append((str(statement), params))
        return _Result(self._rows)


class _Engine:
    def __init__(self, rows, calls):
        self._rows = rows
        self._calls = calls

    def connect(self):
        return _Connection(self._rows, self._calls)


def test_list_sessions_aggregates_bot_partition_and_sorts_by_activity():
    calls = []
    manager = PostgreSQLShortTermManager.__new__(PostgreSQLShortTermManager)
    manager.bot_id = "snark"
    manager.user_id = "nick"
    manager._backend = SimpleNamespace(
        _messages_table_name="messages_p_snark",
        engine=_Engine([_session_row()], calls),
    )

    rows = manager.list_sessions(bot_id="snark", user_id="nick", limit=25)

    assert rows[0]["last_user_message"] == "add the useful metadata"
    sql, params = calls[0]
    assert "FROM messages_p_snark" in sql
    assert "COUNT(*) FILTER" in sql
    assert "FILTER (WHERE role = 'assistant'))[1] AS last_assistant_message" in sql
    assert "ms.last_assistant_at" in sql
    assert "COALESCE(ms.last_message_at" in sql
    assert "CASE WHEN s.status = 'active' THEN 0 ELSE 1 END" in sql
    assert params == {"limit": 25, "bot_id": "snark", "user_id": "nick"}
