"""Strict workload identity, bounded legacy migration and indexed attempt lookup."""
from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine, event, inspect, text

from llm_bawt.ops.gate_history import (
    GATE_HISTORY_INDEX, _BACKFILL_PAGE_SIZE, gate_plan_key, has_prior_attempt,
    migrate_gate_history, plan_key,
)
from llm_bawt.ops.gate_plan import GatePlan
from llm_bawt.ops.validation import canonical_json
from tests.test_ops_store import _op_data, _store

PLAN = GatePlan("llm-bawt", "a" * 40)


def test_key_is_bounded_layout_independent_and_not_caller_supplied():
    variants = [canonical_json(PLAN.to_dict()), json.dumps(PLAN.to_dict(), indent=2),
                json.dumps(dict(reversed(list(PLAN.to_dict().items())))).replace("llm", "\\u006clm")]
    assert len(plan_key(PLAN)) <= 128
    assert {gate_plan_key(raw) for raw in variants} == {plan_key(PLAN)}
    assert plan_key(GatePlan("llm-bawt", "b" * 40)) != plan_key(PLAN)
    store = _store()
    op = store.create_operation(_op_data())
    for index, raw in enumerate(variants):
        job = store.create_job(operation=op, args_json=raw, display_args_json="{}", idempotency_key=str(index))
        assert job.gate_plan_key == plan_key(PLAN) and job.args_json == raw
        assert "gate_plan_key" not in job.to_api()  # Internal projection, not a public grant.
    with pytest.raises(TypeError):
        store.create_job(operation=op, args_json="{}", display_args_json="{}", idempotency_key="bad",
                         gate_plan_key=plan_key(PLAN))


@pytest.mark.parametrize("raw", ["{}", "not-json", "[]", "null", "1", None,
    canonical_json({**PLAN.to_dict(), "attempt": 2}),
    canonical_json({**PLAN.to_dict(), "required_tag": "fixture"}),
    canonical_json({**PLAN.to_dict(), "repository": "bawthub"}),
    canonical_json({**PLAN.to_dict(), "source_sha": "HEAD"}),
    canonical_json({**PLAN.to_dict(), "schema": "unknown"})])
def test_nongate_or_invalid_args_have_no_gate_key(raw):
    assert gate_plan_key(raw) is None


def test_additive_legacy_migration_pages_once_preserves_rows_and_leaves_nongates_null():
    engine = create_engine("sqlite://")
    count = _BACKFILL_PAGE_SIZE * 2 + 3
    raw = json.dumps(dict(reversed(list(PLAN.to_dict().items()))), indent=2)
    statements = []
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE ops_jobs (id TEXT PRIMARY KEY, args_json TEXT NOT NULL, "
                          "state TEXT NOT NULL, dispatched_at TIMESTAMP)"))
        conn.execute(text("INSERT INTO ops_jobs (id,args_json,state) VALUES (:id,:args,:state)"),
                     [{"id": f"{n:04}", "args": raw if n % 2 == 0 else "not-json", "state": "lost"}
                      for n in range(count)])
        before = conn.execute(text("SELECT * FROM ops_jobs ORDER BY id")).all()
        event.listen(engine, "before_cursor_execute",
                     lambda conn, cur, statement, params, context, many: statements.append(statement))
        migrate_gate_history(conn)
        pages = [s for s in statements if s.startswith("SELECT id, args_json")]
        assert len(pages) == 4 and all("LIMIT" in s and "ORDER BY id" in s for s in pages)
        assert conn.execute(text("SELECT id,args_json,state,dispatched_at FROM ops_jobs ORDER BY id")).all() == before
        keys = conn.execute(text("SELECT gate_plan_key FROM ops_jobs ORDER BY id")).scalars().all()
        assert keys == [plan_key(PLAN) if n % 2 == 0 else None for n in range(count)]
        column = next(c for c in inspect(conn).get_columns("ops_jobs") if c["name"] == "gate_plan_key")
        assert column["nullable"] and column["default"] is None
        assert GATE_HISTORY_INDEX in {i["name"] for i in inspect(conn).get_indexes("ops_jobs")}
        statements.clear()
        migrate_gate_history(conn)
        assert not any("SELECT id, args_json" in s or s.startswith(("UPDATE ops_jobs", "CREATE INDEX"))
                       for s in statements)
        assert has_prior_attempt(conn, job_id="other", plan=PLAN)
    engine.dispose()


@pytest.mark.parametrize("state,dispatched,blocks", [
    ("queued", False, False), ("queued", True, True), ("dispatching", False, True),
    ("accepted", False, True), ("running", False, True), ("quiescing", False, True),
    ("cancelled", False, True), ("lost", False, True), ("timed_out", False, True),
    ("succeeded", False, True), ("failed", False, True)])
def test_partial_index_lookup_preserves_original_first_attempt_predicate(state, dispatched, blocks):
    store = _store()
    op = store.create_operation(_op_data())
    job = store.create_job(operation=op, args_json=canonical_json(PLAN.to_dict()),
                           display_args_json="{}", idempotency_key="history")
    with store.engine.begin() as conn:
        conn.execute(text("UPDATE ops_jobs SET state=:state, dispatched_at=:at WHERE id=:id"),
                     {"state": state, "at": "2026-01-01 00:00:00" if dispatched else None, "id": job.id})
        assert has_prior_attempt(conn, job_id="other", plan=PLAN) is blocks
        assert not has_prior_attempt(conn, job_id=job.id, plan=PLAN)
        assert not has_prior_attempt(conn, job_id="other", plan=GatePlan("llm-bawt", "b" * 40))


def test_normal_store_additive_migration_preserves_snapshot_payload_and_state():
    store = _store()
    op = store.create_operation(_op_data())
    job = store.create_job(operation=op, args_json=json.dumps(PLAN.to_dict(), indent=2),
                           invocation_snapshot_json="{}", display_args_json="{}", idempotency_key="legacy")
    store.mark_terminal(job.id, state="cancelled")
    before = store.get_job(job.id).model_dump()
    with store.engine.begin() as conn:
        conn.execute(text(f"DROP INDEX {GATE_HISTORY_INDEX}"))
        conn.execute(text("ALTER TABLE ops_jobs DROP COLUMN gate_plan_key"))
        store._migrate_add_columns(conn)
        store._migrate_add_columns(conn)
    assert store.get_job(job.id).model_dump() == before
