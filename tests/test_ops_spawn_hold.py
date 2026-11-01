"""TASK-1169: the durable self-config hold's ledger/claim prerequisite only.

No enabled action takes a hold yet. Spawn-side flocks and Docker drain probes
are a later slice, not simulated as completed here.
"""
from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import inspect, text

from llm_bawt.ops import JOB_ACCEPTED, JOB_DISPATCHING, JOB_QUEUED, JOB_RUNNING, OpsService
from llm_bawt.ops.models import JOB_TERMINAL_STATES
from llm_bawt.ops.store import OpsStore
from llm_bawt.ops.worker_slots import WORKER_SLOT_STATES, process_spawn_held
from tests.test_ops_service import _make, _op_data, _store


def _admit(store, *, mode="isolated", **over):
    slug = f"test.spawn-hold-{uuid.uuid4().hex}"
    store.create_operation(_op_data(slug=slug, execution_mode=mode, **over))
    service, _ = _make(store)
    admission = OpsService(store, admission_only=True,
                           preparer=SimpleNamespace(prepare=service.prepare_invocation))
    return admission.dispatch_job(operation_slug=slug, args={"target": "a"},
                                  idempotency_key=uuid.uuid4().hex)["id"]


def _held(store):
    with store.engine.connect() as conn:
        return process_spawn_held(conn)


def _hold(store, job_id, **over):
    return store.claim_job(job_id, hold_process_spawns=True, **{"process_child_count": 0, **over})


def test_hold_and_dispatch_claim_are_one_commit_and_survive_store_reconstruction():
    store = _store()
    job_id = _admit(store)
    assert _hold(store, job_id)
    row = store.get_job(job_id)
    assert row.state == JOB_DISPATCHING and row.worker_slot_at is None
    assert row.process_spawn_hold_at == row.dispatched_at
    restarted = object.__new__(OpsStore)
    restarted.engine = store.engine
    restarted._ensure_tables_exist()
    assert _held(restarted)
    assert restarted.get_job(job_id).process_spawn_hold_at == row.process_spawn_hold_at
    # Duplicate claims cannot re-arm or change a hold.
    assert not _hold(restarted, job_id)
    assert restarted.get_job(job_id).process_spawn_hold_at == row.process_spawn_hold_at


@pytest.mark.parametrize("state", [JOB_DISPATCHING, JOB_ACCEPTED, JOB_RUNNING, "unknown-open-state"])
def test_every_nonterminal_holder_fences_process_claims_but_not_isolated_work(state):
    store = _store()
    job_id = _admit(store)
    assert _hold(store, job_id)
    with store.engine.begin() as conn:
        conn.execute(text("UPDATE ops_jobs SET state=:state WHERE id=:id"), {"state": state, "id": job_id})
    process, isolated, contender = _admit(store, mode="process"), _admit(store), _admit(store)
    assert not store.claim_job(process)
    row = store.get_job(process)
    assert (row.state, row.error_text, row.worker_slot_at, row.dispatched_at) == (JOB_QUEUED, None, None, None)
    assert store.claim_job(isolated)
    assert not _hold(store, contender)
    assert store.get_job(contender).process_spawn_hold_at is None
    assert _held(store)


@pytest.mark.parametrize("state", sorted(JOB_TERMINAL_STATES))
def test_terminal_transition_releases_hold_atomically_but_retains_forensic_timestamp(state):
    store = _store()
    job_id = _admit(store)
    assert _hold(store, job_id)
    stamp = store.get_job(job_id).process_spawn_hold_at
    store.mark_terminal(job_id, state=state)
    assert not _held(store)
    assert store.get_job(job_id).process_spawn_hold_at == stamp
    assert store.claim_job(_admit(store, mode="process"))
    assert not _hold(store, job_id)  # Terminal jobs never replay.


@pytest.mark.parametrize("count", [None, 1, -1, False, True, 0.0, "0"])
def test_unknown_nonzero_or_noninteger_child_evidence_never_takes_a_hold(count):
    store = _store()
    job_id = _admit(store)
    assert not _hold(store, job_id, process_child_count=count)
    row = store.get_job(job_id)
    assert (row.state, row.process_spawn_hold_at, row.dispatched_at) == (JOB_QUEUED, None, None)
    assert not _held(store)


@pytest.mark.parametrize("state", WORKER_SLOT_STATES)
def test_existing_process_slot_refuses_hold_even_with_zero_reported_children(state):
    store = _store()
    process = _admit(store, mode="process")
    assert store.claim_job(process)
    with store.engine.begin() as conn:
        conn.execute(text("UPDATE ops_jobs SET state=:state WHERE id=:id"), {"state": state, "id": process})
    isolated = _admit(store)
    assert not _hold(store, isolated)
    assert store.get_job(isolated).state == JOB_QUEUED
    assert not _held(store)


def test_process_mode_cannot_take_its_own_spawn_hold():
    store = _store()
    job_id = _admit(store, mode="process")
    assert not _hold(store, job_id)
    assert store.get_job(job_id).worker_slot_at is None
    assert not _held(store)


def test_quiesce_snapshot_cannot_take_a_spawn_hold():
    store = _store()
    job_id = _admit(store)
    # Admission of real quiesce jobs is PostgreSQL-only. Here exercise the
    # store's snapshot guard; the real driver case lives in the PG suite.
    snapshot = json.loads(store.get_job(job_id).invocation_snapshot_json)
    snapshot["quiesce"] = {"enabled": True}
    with store.engine.begin() as conn:
        conn.execute(text("UPDATE ops_jobs SET invocation_snapshot_json=:raw WHERE id=:id"),
                     {"raw": json.dumps(snapshot), "id": job_id})
    assert not _hold(store, job_id)
    assert not _held(store)


def test_failed_claim_transaction_cannot_leak_a_hold(monkeypatch):
    store = _store()
    job_id = _admit(store)

    def fail(*_args, **_kwargs):
        raise RuntimeError("claim transaction interrupted")

    monkeypatch.setattr("llm_bawt.ops.store.append_job_transition", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        _hold(store, job_id)
    row = store.get_job(job_id)
    assert (row.state, row.process_spawn_hold_at, row.dispatched_at) == (JOB_QUEUED, None, None)
    assert not _held(store)


def test_additive_migration_preserves_old_jobs_and_is_idempotent():
    store = _store()
    job_id = _admit(store)
    with store.engine.begin() as conn:
        conn.execute(text("ALTER TABLE ops_jobs DROP COLUMN process_spawn_hold_at"))
        store._migrate_add_columns(conn)
        store._migrate_add_columns(conn)
        assert "process_spawn_hold_at" in {col["name"] for col in inspect(conn).get_columns("ops_jobs")}
    assert store.get_job(job_id).state == JOB_QUEUED
    assert store.get_job(job_id).process_spawn_hold_at is None
    assert not _held(store)


def test_spawn_side_hold_read_has_no_advisory_or_row_lock():
    statements = []

    class Connection:
        def execute(self, statement, params):
            statements.append(str(statement))
            return SimpleNamespace(first=lambda: None)

    assert not process_spawn_held(Connection())
    assert len(statements) == 1
    assert statements[0].startswith("SELECT 1 FROM ops_jobs")
    assert "pg_advisory" not in statements[0] and "FOR UPDATE" not in statements[0]
