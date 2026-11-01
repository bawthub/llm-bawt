"""Stable-id native-upsert contract, shared with real partitioned PG tests."""
from __future__ import annotations

from contextlib import contextmanager

import pytest
from sqlalchemy import event, null, select, update

from llm_bawt.message_authorship import AuthorReference
from tests.test_message_row_timestamp import rows

__all__ = ["rows"]


class SQLReceipt(list):
    """Application statements plus separately visible driver pre-ping reads."""

    def __init__(self, dialect):
        super().__init__()
        self.dialect = dialect
        self.pool_ping_sql = []
        self.all_statements = []


@contextmanager
def capture_sql(engine):
    statements = SQLReceipt(engine.dialect.name)

    def capture(_conn, _cursor, statement, *_args):
        statements.append(statement)
        statements.all_statements.append(statement)

    original_ping = engine.dialect.do_ping

    def ping(connection):
        # psycopg2 do_ping calls cursor.execute directly, bypassing SQLAlchemy's
        # before_cursor_execute. Delegate the REAL check; do not suppress it or
        # claim zero total SELECTs from the application-event list alone.
        sql = engine.dialect._dialect_specific_select_one
        statements.pool_ping_sql.append(sql)
        statements.all_statements.append(sql)
        return original_ping(connection)

    event.listen(engine, "before_cursor_execute", capture)
    if statements.dialect == "postgresql":
        engine.dialect.do_ping = ping
    try:
        yield statements
    finally:
        engine.dialect.do_ping = original_ping
        event.remove(engine, "before_cursor_execute", capture)


def one_upsert(statements):
    assert len(statements) == 1, statements
    assert statements[0].lstrip().startswith("INSERT INTO"), statements
    assert "ON CONFLICT" in statements[0], statements
    expected_ping = ["SELECT 1"] if statements.dialect == "postgresql" else []
    assert statements.pool_ping_sql == expected_ping, statements.all_statements
    assert len(statements.all_statements) == 1 + len(expected_ping)


def row(store, message_id="stable"):
    with store.engine.connect() as conn:
        return dict(conn.execute(select(store.table).where(store.table.c.id == message_id)).mappings().one())


def check_retry_contract(store, stored_role, first_session="first-session"):
    with capture_sql(store.engine) as sql:
        store.upsert(message_id="stable", role=stored_role, content="first", timestamp=10,
                     session_id=first_session, attachments=[{"asset_id": "old"}], reasoning="thought",
                     author=AuthorReference.user("first-author"))
    one_upsert(sql)
    first = row(store)
    with capture_sql(store.engine) as sql:
        store.upsert(message_id="stable", role="assistant" if stored_role == "user" else "user",
                     content="retry", timestamp=20, session_id="other-session")
    one_upsert(sql)
    retried = row(store)
    assert retried == {**first, "content": "retry", "timestamp": 10 if stored_role == "user" else 20}
    with capture_sql(store.engine) as sql:
        store.upsert(message_id="stable", role="assistant", content="final", timestamp=30,
                     attachments=[], reasoning="", author=AuthorReference.unknown())
    one_upsert(sql)
    final = row(store)
    assert final == {**retried, "content": "final", "timestamp": 10 if stored_role == "user" else 30,
                     "attachments": [], "reasoning": "", "author_entity_type": None,
                     "author_entity_id": None}


@pytest.mark.parametrize("stored_role", ["user", "assistant", "system", "summary"])
def test_retry_preserves_immutable_and_absent_optional_fields(rows, stored_role):
    check_retry_contract(rows, stored_role)


FALSY_METADATA = [None, False, 0, 0.0, "", [], {}]
INVALID_METADATA = [True, 1, "invalid", ["invalid"]]


def check_metadata_contract(store, metadata):
    store.upsert(message_id="stable", role="assistant", content="first", timestamp=10)
    with store.engine.begin() as conn:
        conn.execute(update(store.table).values(summary_metadata=metadata))
    first = row(store)
    with capture_sql(store.engine) as sql:
        if metadata and not isinstance(metadata, dict):
            with pytest.raises(TypeError):
                store.upsert(message_id="stable", role="user", content="must rollback", timestamp=20,
                             attachments=[], reasoning="changed", extract_memory=False)
        else:
            store.upsert(message_id="stable", role="user", content="excluded", timestamp=20,
                         extract_memory=False)
    one_upsert(sql)
    if metadata and not isinstance(metadata, dict):
        assert row(store) == first
    else:
        actual = row(store)
        assert actual["processed"] is True
        assert actual["summary_metadata"] == {**(metadata or {}), "extract_memory": False}
        assert actual["summary_metadata"]["extract_memory"] is False
        # An ordinary retry cannot restore extraction or erase existing metadata.
        store.upsert(message_id="stable", role="assistant", content="ordinary retry", timestamp=30)
        assert row(store)["summary_metadata"] == actual["summary_metadata"]
        assert row(store)["processed"] is True


@pytest.mark.parametrize("metadata", FALSY_METADATA + INVALID_METADATA + [
    {"nested": {"keep": True}, "extract_memory": True, "other": None},
])
def test_exclusion_preserves_merge_falsy_and_rollback_contract(rows, metadata):
    check_metadata_contract(rows, metadata)


def check_new_exclusion(store):
    with capture_sql(store.engine) as sql:
        store.upsert(message_id="stable", role="assistant", content="excluded", timestamp=10,
                     extract_memory=False)
    one_upsert(sql)
    result = row(store)
    assert result["processed"] is True
    assert result["summary_metadata"] == {"extract_memory": False}
    assert result["summary_metadata"]["extract_memory"] is False
    assert result["attachments"] == []
    assert result["session_id"] is None


def test_new_exclusion_has_real_boolean_and_defaults(rows):
    check_new_exclusion(rows)


def check_sql_null_metadata(store):
    store.upsert(message_id="stable", role="assistant", content="first", timestamp=10)
    with store.engine.begin() as conn:
        conn.execute(update(store.table).values(summary_metadata=null()))
    with capture_sql(store.engine) as sql:
        store.upsert(message_id="stable", role="assistant", content="excluded", timestamp=20,
                     extract_memory=False)
    one_upsert(sql)
    assert row(store)["summary_metadata"] == {"extract_memory": False}


def test_sql_null_is_distinct_from_json_null_and_normalized(rows):
    check_sql_null_metadata(rows)


def test_chat_request_validates_roles_but_raw_history_store_uses_strings():
    from pydantic import ValidationError
    from llm_bawt.service.schemas import ChatMessage

    with pytest.raises(ValidationError):
        ChatMessage(role="x" * 21, content="invalid")


@pytest.mark.parametrize("content", ["", " ", "\n\t"])
def test_blank_content_does_not_issue_sql(rows, content):
    with capture_sql(rows.engine) as sql:
        rows.upsert(message_id="stable", role="user", content=content, timestamp=10)
    assert sql == []
