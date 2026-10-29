"""Tests for :mod:`llm_bawt.ops.store` (TASK-639).

Exercises OpsStore + OpsOperation + OpsJob against a real in-memory SQLite
engine — CRUD, per-slug seeding, job state machine, and the seed bootstrap.

No Redis, no HTTP, no live executor. Bypasses ``OpsStore.__init__`` (which
wants a Config + shared engine) with the same pattern as
test_approval_persist.py.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

from sqlmodel import Session, create_engine, select
from sqlalchemy.pool import StaticPool

from llm_bawt.ops import (
    JOB_CANCELLED,
    JOB_DISPATCHING,
    JOB_FAILED,
    JOB_QUEUED,
    JOB_RUNNING,
    JOB_SUCCEEDED,
    JOB_TERMINAL_STATES,
    OpsJob,
    OpsOperation,
    OpsStore,
)
from llm_bawt.ops.seeds import SEEDS, seed_all
from llm_bawt.ops.store import OpsStoreUnavailable

_OK = True


def _store():
    store = object.__new__(OpsStore)
    store.engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    store._ensure_tables_exist()
    return store


def _op_data(**over):
    base = dict(
        slug="test.restart-thing",
        title="Restart the thing",
        description="A test operation",
        enabled=True,
        target_host="",
        command_script='{"action":"restart","container_name":"test"}',
        args_schema_json='{"type":"object","additionalProperties":false,"properties":{}}',
        args_defaults_json="{}",
        timeout_seconds=60,
        start_delay_seconds=0,
        risk_level="low",
        category="restart",
    )
    base.update(over)
    return base


# ---- CRUD ------------------------------------------------------------------

def test_create_operation_writes_row_with_script_hash():
    store = _store()
    row = store.create_operation(_op_data(), actor="nick")
    assert row.slug == "test.restart-thing"
    assert row.version == 1
    expected = hashlib.sha256(row.command_script.encode("utf-8")).hexdigest()
    assert row.script_hash == expected
    assert row.created_by == "nick"


def test_create_operation_rejects_duplicate_slug():
    store = _store()
    store.create_operation(_op_data())
    raised = False
    try:
        store.create_operation(_op_data())
    except ValueError as exc:
        raised = "already exists" in str(exc)
    assert raised


def test_get_operation_by_slug_and_by_id():
    store = _store()
    row = store.create_operation(_op_data())
    assert store.get_operation_by_slug("test.restart-thing").id == row.id
    assert store.get_operation(row.id).id == row.id
    # get_operation also accepts slug for convenience.
    assert store.get_operation("test.restart-thing").id == row.id
    assert store.get_operation("nope") is None


def test_update_operation_bumps_version_and_rehashes_script():
    store = _store()
    row = store.create_operation(_op_data())
    orig_hash = row.script_hash
    updated = store.update_operation(
        row.id,
        {"command_script": '{"action":"start","container_name":"test"}'},
        actor="nick",
    )
    assert updated.version == 2
    assert updated.script_hash != orig_hash
    assert updated.updated_by == "nick"


def test_soft_delete_operation_disables_and_marks():
    store = _store()
    row = store.create_operation(_op_data())
    assert store.soft_delete_operation(row.id) is True
    fresh = store.get_operation(row.id)
    assert fresh.soft_deleted_at is not None
    assert fresh.enabled is False
    # Idempotent.
    assert store.soft_delete_operation(row.id) is True


def test_list_operations_filters_enabled_and_soft_deleted():
    store = _store()
    store.create_operation(_op_data(slug="a.enabled", enabled=True))
    store.create_operation(_op_data(slug="b.disabled", enabled=False))
    c = store.create_operation(_op_data(slug="c.deleted", enabled=True))
    store.soft_delete_operation(c.id)
    # default: enabled + not-soft-deleted
    live = [r.slug for r in store.list_operations()]
    assert live == ["a.enabled"]
    # include disabled
    with_disabled = [r.slug for r in store.list_operations(include_disabled=True)]
    assert set(with_disabled) == {"a.enabled", "b.disabled"}
    # include soft-deleted
    all_including_deleted = [r.slug for r in store.list_operations(
        include_disabled=True, include_soft_deleted=True,
    )]
    assert set(all_including_deleted) == {"a.enabled", "b.disabled", "c.deleted"}


def test_create_operation_rejects_invalid_json():
    store = _store()
    raised = False
    try:
        store.create_operation(_op_data(args_schema_json="not-json"))
    except ValueError:
        raised = True
    assert raised


def test_none_engine_raises_on_write():
    store = object.__new__(OpsStore)
    store.engine = None
    raised = False
    try:
        store.create_operation(_op_data())
    except OpsStoreUnavailable:
        raised = True
    assert raised


# ---- Seeding ---------------------------------------------------------------

def test_seed_is_idempotent_per_slug():
    store = _store()
    seed = _op_data(slug="seedy", enabled=True)
    row = store.seed_operation_if_missing(seed)
    assert row is not None
    # Second call — existing row present — returns None (no insert).
    assert store.seed_operation_if_missing(seed) is None
    # And the row is unchanged, even if the seed data drifts.
    drifted = dict(seed); drifted["title"] = "different"
    assert store.seed_operation_if_missing(drifted) is None
    fresh = store.get_operation_by_slug("seedy")
    assert fresh.title == "Restart the thing"  # original preserved


def test_seed_all_bootstraps_canonical_catalog():
    store = _store()
    inserted, skipped = seed_all(store)
    assert set(inserted) == {seed["slug"] for seed in SEEDS}
    assert skipped == []
    # Second run: everything is a skip.
    inserted2, skipped2 = seed_all(store)
    assert inserted2 == []
    assert set(skipped2) == {seed["slug"] for seed in SEEDS}
    # And every seeded op ships DISABLED by default (safety).
    for slug in inserted:
        assert store.get_operation_by_slug(slug).enabled is False


# ---- Job lifecycle ---------------------------------------------------------

def test_create_job_snapshots_operation_version_and_hash():
    store = _store()
    op = store.create_operation(_op_data())
    job = store.create_job(
        operation=op,
        args_json="{}",
        display_args_json="{}",
        idempotency_key="idem-1",
        caller_bot_id="snark",
    )
    assert job.state == JOB_QUEUED
    assert job.operation_slug == op.slug
    assert job.operation_version == op.version
    assert job.operation_script_hash == op.script_hash
    assert job.caller_bot_id == "snark"


def test_create_job_idempotent_on_idempotency_key():
    store = _store()
    op = store.create_operation(_op_data())
    first = store.create_job(
        operation=op, args_json="{}", display_args_json="{}",
        idempotency_key="idem-1",
    )
    second = store.create_job(
        operation=op, args_json='{}', display_args_json='{}',
        idempotency_key="idem-1",
    )
    assert first.id == second.id
    assert second.args_json == "{}"  # first-write wins


def test_job_state_transitions_forward_only():
    store = _store()
    op = store.create_operation(_op_data())
    job = store.create_job(
        operation=op, args_json="{}", display_args_json="{}", idempotency_key="j1",
    )
    dispatching = store.mark_dispatching(
        job.id, host_unit_name="llm-bawt-ops-j1",
        status_file_path="/host/.logs/ops/j1/status.json",
    )
    assert dispatching.state == JOB_DISPATCHING
    assert dispatching.host_unit_name == "llm-bawt-ops-j1"
    running = store.mark_running(job.id)
    assert running.state == JOB_RUNNING
    assert running.started_at is not None
    terminal = store.mark_terminal(
        job.id, state=JOB_SUCCEEDED, exit_code=0, output_tail="ok\n",
    )
    assert terminal.state == JOB_SUCCEEDED
    assert terminal.exit_code == 0
    assert terminal.output_tail == "ok\n"


def test_mark_terminal_idempotent_and_does_not_regress():
    store = _store()
    op = store.create_operation(_op_data())
    job = store.create_job(
        operation=op, args_json="{}", display_args_json="{}", idempotency_key="j1",
    )
    store.mark_terminal(job.id, state=JOB_SUCCEEDED, exit_code=0)
    # Second terminal call with FAILED must NOT overwrite the SUCCEEDED state.
    second = store.mark_terminal(job.id, state=JOB_FAILED, exit_code=1)
    assert second.state == JOB_SUCCEEDED
    assert second.exit_code == 0


def test_mark_terminal_refuses_non_terminal_state():
    store = _store()
    op = store.create_operation(_op_data())
    job = store.create_job(
        operation=op, args_json="{}", display_args_json="{}", idempotency_key="j1",
    )
    raised = False
    try:
        store.mark_terminal(job.id, state=JOB_RUNNING)
    except ValueError:
        raised = True
    assert raised


def test_find_active_jobs_excludes_terminal():
    store = _store()
    op = store.create_operation(_op_data())
    j1 = store.create_job(
        operation=op, args_json="{}", display_args_json="{}", idempotency_key="j1",
    )
    j2 = store.create_job(
        operation=op, args_json="{}", display_args_json="{}", idempotency_key="j2",
    )
    store.mark_terminal(j2.id, state=JOB_CANCELLED)
    active = [j.id for j in store.find_active_jobs()]
    assert active == [j1.id]


def test_quiesce_api_exposes_scope_for_idle_bots_not_in_turn_roster():
    job = OpsJob(operation_slug="test.restart-thing", args_json="{}", display_args_json="{}",
                 invocation_snapshot_json=json.dumps({"quiesce": {"scope": {"all_agents": True}}}),
                 state="quiescing")
    assert job.to_api()["quiesce"]["scope"] == {"all_agents": True}
    job.invocation_snapshot_json = "bad snapshot"
    assert job.to_api()["quiesce"]["scope"] is None


def test_live_jobs_include_old_active_and_only_fresh_outcomes():
    store = _store()
    op = store.create_operation(_op_data())
    active = store.create_job(operation=op, args_json="{}", display_args_json="{}", idempotency_key="active")
    fresh = store.create_job(operation=op, args_json="{}", display_args_json="{}", idempotency_key="fresh")
    old = store.create_job(operation=op, args_json="{}", display_args_json="{}", idempotency_key="old")
    now = datetime.now(timezone.utc)
    store.mark_terminal(fresh.id, state=JOB_SUCCEEDED, finished_at=now)
    store.mark_terminal(old.id, state=JOB_FAILED, finished_at=now - timedelta(minutes=1))
    live = store.find_active_jobs(limit=None, recent_terminal_since=now - timedelta(seconds=15))
    assert {row.id for row in live} == {active.id, fresh.id}
    assert [row.id for row in store.find_active_jobs()] == [active.id]


def test_list_jobs_filters():
    store = _store()
    op = store.create_operation(_op_data())
    op2 = store.create_operation(_op_data(slug="other.thing"))
    j1 = store.create_job(
        operation=op, args_json="{}", display_args_json="{}", idempotency_key="j1",
    )
    j2 = store.create_job(
        operation=op2, args_json="{}", display_args_json="{}", idempotency_key="j2",
    )
    store.mark_terminal(j1.id, state=JOB_FAILED, exit_code=2)
    only_op = [j.id for j in store.list_jobs(operation_slug=op.slug)]
    assert only_op == [j1.id]
    only_failed = [j.id for j in store.list_jobs(state=JOB_FAILED)]
    assert only_failed == [j1.id]


# ---- Seed validation -------------------------------------------------------

def test_every_seed_has_valid_args_schema():
    """Every canonical seed's args_schema_json must be well-formed JSON with
    ``additionalProperties: false`` so unknown args are rejected at runtime."""
    for seed in SEEDS:
        schema = json.loads(seed["args_schema_json"])
        assert schema.get("type") == "object", seed["slug"]
        assert schema.get("additionalProperties") is False, seed["slug"]


def test_every_seed_ships_disabled():
    """Safety: no operator-runnable ops without a human enable click."""
    for seed in SEEDS:
        assert seed["enabled"] is False, seed["slug"]


if __name__ == "__main__":
    import sys
    import traceback

    if not _OK:
        print(f"SKIP test_ops_store: {_SKIP_REASON}")
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
