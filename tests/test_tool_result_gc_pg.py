"""TASK-2216: legacy payload pruning and availability share a DB transaction."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, event, text
from sqlmodel import Session, SQLModel

from llm_bawt.service.jobs.tool_result_gc import run_tool_result_gc
from llm_bawt.service.tool_call_store import ToolCallRecord, ToolCallResultPayloadRecord
from tests.private_postgres import private_url

pytestmark = pytest.mark.integration


@pytest.fixture
def gc_db(private_pg, private_module_db, monkeypatch):
    from llm_bawt.service import tool_call_store
    from llm_bawt.utils import db

    engine = create_engine(private_url(private_pg, private_module_db))
    tables = [ToolCallRecord.__table__, ToolCallResultPayloadRecord.__table__]
    SQLModel.metadata.create_all(engine, tables=tables)
    monkeypatch.setattr(db, "get_shared_engine", lambda config: engine)
    monkeypatch.setattr(tool_call_store, "_get_tool_blob_backend", lambda: None)
    try:
        yield engine
    finally:
        SQLModel.metadata.drop_all(engine, tables=tables)
        engine.dispose()


def add_record(engine, *, days=30, blob=None, payload=True, available=True):
    with Session(engine) as session:
        row = ToolCallRecord(
            created_at=datetime.now(timezone.utc) - timedelta(days=days),
            tool_name="fixture", result_text="preview stays", result_blob_key=blob,
            result_payload_available=available,
        )
        session.add(row)
        session.flush()
        record_id = row.id
        if payload:
            session.add(ToolCallResultPayloadRecord(
                tool_call_record_id=record_id, content_text="full result", content_type="text/plain",
                total_chars=11, total_bytes=11, sha256="0" * 64, complete=True,
            ))
        session.commit()
    return record_id


def state(engine, record_id):
    with Session(engine) as session:
        row = session.get(ToolCallRecord, record_id)
        return (row.result_payload_available, row.result_blob_key, row.result_text,
                session.get(ToolCallResultPayloadRecord, record_id) is not None)


def test_legacy_pruning_clears_only_expired_payload_availability(gc_db):
    old = add_record(gc_db)
    recent = add_record(gc_db, days=1)
    already_false = add_record(gc_db, available=False)
    untouched = add_record(gc_db, payload=False)
    result = run_tool_result_gc(None)
    assert result["legacy_rows_pruned"] == 2
    assert result["records_cleared"] == 1
    assert state(gc_db, old) == (False, None, "preview stays", False)
    assert state(gc_db, recent) == (True, None, "preview stays", True)
    assert state(gc_db, already_false) == (False, None, "preview stays", False)
    assert state(gc_db, untouched) == (True, None, "preview stays", False)
    replay = run_tool_result_gc(None)
    assert replay["legacy_rows_pruned"] == replay["records_cleared"] == 0


@pytest.mark.parametrize("shared", [False, True])
def test_pruning_preserves_surviving_blob_availability(gc_db, shared):
    old = add_record(gc_db, blob="shared-key")
    if shared:
        add_record(gc_db, days=1, blob="shared-key", payload=False)
    deleted = []
    # Shared blobs survive the keep-set; unshared blobs survive a missing backend.
    backend = SimpleNamespace(delete=deleted.append) if shared else None
    result = run_tool_result_gc(None, blob_backend=backend)
    assert result["legacy_rows_pruned"] == 1
    assert result["records_cleared"] == 0
    assert state(gc_db, old) == (True, "shared-key", "preview stays", False)
    assert deleted == []


def test_blob_and_legacy_pruning_count_record_once(gc_db):
    old = add_record(gc_db, blob="expired-key")
    deleted = []
    result = run_tool_result_gc(None, blob_backend=SimpleNamespace(delete=deleted.append))
    assert result["legacy_rows_pruned"] == result["records_cleared"] == 1
    assert deleted == ["expired-key"]
    assert state(gc_db, old) == (False, None, "preview stays", False)


def test_dry_run_preserves_legacy_payload_and_availability(gc_db):
    old = add_record(gc_db)
    result = run_tool_result_gc(None, dry_run=True)
    assert result["legacy_rows_pruned"] == 1
    assert result["records_cleared"] == 0
    assert state(gc_db, old) == (True, None, "preview stays", True)


def test_flag_update_failure_rolls_back_postgres_payload_pruning(gc_db):
    old = add_record(gc_db)

    def fail_update(conn, cursor, statement, parameters, context, executemany):
        if "UPDATE tool_call_records" in statement:
            # The DELETE already ran on this connection, but has not committed.
            assert conn.execute(text("SELECT count(*) FROM tool_call_result_payloads")).scalar_one() == 0
            raise RuntimeError("injected flag update failure")

    event.listen(gc_db, "before_cursor_execute", fail_update)
    try:
        with pytest.raises(RuntimeError, match="injected flag update failure"):
            run_tool_result_gc(None)
    finally:
        event.remove(gc_db, "before_cursor_execute", fail_update)
    # No object-store deletes in this regression: only DB rollback is asserted.
    assert state(gc_db, old) == (True, None, "preview stays", True)
