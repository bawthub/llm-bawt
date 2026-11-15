"""Tests for confirmed approval-request persistence (TASK-306 Section A).

Exercises ``ToolApprovalPolicyStore.record_request`` against a real in-memory
SQLite engine — commit, idempotency, and the two hard-failure paths that now
raise ``ApprovalPersistError`` instead of silently returning ``None``.

No Redis, no live SDK turn, no HTTP. Skips cleanly if the llm_bawt package /
its deps aren't importable (i.e. when run outside the app container).

Runnable under pytest, or standalone: ``python tests/test_approval_persist.py``.
"""

from __future__ import annotations

try:
    from sqlmodel import Session, create_engine, select

    from llm_bawt.approval_policies import (
        ApprovalPersistError,
        REQ_PENDING,
        ToolApprovalPolicyStore,
        ToolApprovalRequest,
    )

    _OK = True
    _SKIP_REASON = ""
except Exception as exc:  # noqa: BLE001
    _OK = False
    _SKIP_REASON = f"llm_bawt deps unavailable ({exc}); run in app container"


def _store(*, with_tables: bool = True, engine_none: bool = False) -> "ToolApprovalPolicyStore":
    """Build a store with an isolated in-memory engine, bypassing __init__
    (which needs a Config + live DB credentials)."""
    store = object.__new__(ToolApprovalPolicyStore)
    if engine_none:
        store.engine = None
        return store
    # A fresh in-memory DB per store; StaticPool keeps it alive across sessions.
    from sqlalchemy.pool import StaticPool

    store.engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    if with_tables:
        store._ensure_tables_exist()
    return store


def _req(**over):
    base = dict(
        request_id="tuid-1",
        bot_id="snark",
        user_id="operator",
        turn_id="turn-1",
        backend="claude-code",
        tool_name="make rebuild-prod",
        tool_arguments={"target": "prod"},
        subject="rebuild prod",
        grant_key="grant-1",
        policy_id="pol-1",
        severity="high",
        prompt="Approve rebuild?",
    )
    base.update(over)
    return base


def test_commit_returns_pending_row():
    store = _store()
    row = store.record_request(**_req())
    assert row is not None
    assert row.id == "tuid-1"
    assert row.status == REQ_PENDING
    assert row.tool_name == "make rebuild-prod"
    # And it is actually durable in the DB, not just returned.
    with Session(store.engine) as s:
        found = s.exec(select(ToolApprovalRequest)).all()
    assert len(found) == 1
    assert found[0].id == "tuid-1"


def test_idempotent_on_request_id():
    store = _store()
    first = store.record_request(**_req())
    # Same request_id, different incidental fields — must NOT insert a 2nd row
    # and must return the pre-existing record.
    second = store.record_request(**_req(tool_name="something-else"))
    assert first.id == second.id == "tuid-1"
    assert second.tool_name == "make rebuild-prod"  # original preserved
    with Session(store.engine) as s:
        count = len(s.exec(select(ToolApprovalRequest)).all())
    assert count == 1


def test_none_engine_raises():
    store = _store(engine_none=True)
    raised = False
    try:
        store.record_request(**_req())
    except ApprovalPersistError as exc:
        raised = True
        assert "no DB engine" in str(exc)
    assert raised, "expected ApprovalPersistError when engine is None"


def test_insert_failure_raises():
    # Engine present but tables never created → the INSERT fails. The store must
    # surface ApprovalPersistError, NOT swallow it to None (the old bug).
    store = _store(with_tables=False)
    raised = False
    try:
        store.record_request(**_req())
    except ApprovalPersistError as exc:
        raised = True
        assert "tuid-1" in str(exc)
    assert raised, "expected ApprovalPersistError when the insert fails"


if __name__ == "__main__":
    import sys
    import traceback

    if not _OK:
        print(f"SKIP test_approval_persist: {_SKIP_REASON}")
        sys.exit(0)

    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed ({len(fns)} total)")
    sys.exit(1 if failed else 0)
