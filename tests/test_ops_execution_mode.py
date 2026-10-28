"""TASK-1169 S4: ``execution_mode`` on the operation row, snapshot and card.

The mode is pinned by the approved snapshot like every other execution field.
Rows from before S4 are backfilled to what they already did, and snapshots
from before S4 (no mode anywhere) keep running as approved, only while the
current row still agrees.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import inspect, text
from sqlmodel import Session, select

from llm_bawt.ops import OpsDispatchError
from llm_bawt.ops.models import OpsOperationRevision
from llm_bawt.ops.execution_mode import (
    SELF_TARGET_CARD_LINE,
    card_lines,
    is_legacy_snapshot,
    snapshot_execution_mode,
    validate_execution_mode,
    with_card_lines,
)
from llm_bawt.ops.executor import DockerExecutor, ExecutorError
from llm_bawt.ops.release_executor import ReleaseExecutor
from llm_bawt.ops.validation import canonical_json
from llm_bawt.ops.worker_launch import ContainerLauncher
from tests.test_ops_service import _make, _op_data, _store

SLUG = "test.restart-thing"
ARGS = {"target": "a"}


def _rehash(snapshot):
    body = {k: v for k, v in snapshot.items() if k != "snapshot_hash"}
    body["snapshot_hash"] = hashlib.sha256(canonical_json(body).encode()).hexdigest()
    return json.loads(canonical_json(body))


def _legacy(snapshot):
    """The same snapshot as a pre-S4 prepare would have produced it."""
    stripped = json.loads(canonical_json(snapshot))
    stripped["execution"].pop("execution_mode")
    stripped["operation"].pop("execution_mode")
    return _rehash(stripped)


# ---- row, migration, validation ------------------------------------------

def test_a_new_row_defaults_to_process_and_reports_it():
    store = _store()
    row = store.create_operation(_op_data())
    assert row.execution_mode == "process"
    assert row.to_api()["execution_mode"] == "process"


def test_the_migration_backfills_existing_rows_to_what_they_already_did():
    store = _store()
    store.create_operation(_op_data(slug="old.docker"))
    store.create_operation(_op_data(slug="old.release"))
    with store.engine.begin() as conn:
        # A release row; its spec is irrelevant to the backfill.
        conn.execute(text("UPDATE ops_operations SET executor_kind='release' WHERE slug='old.release'"))
        conn.execute(text("ALTER TABLE ops_operations DROP COLUMN execution_mode"))
        assert "execution_mode" not in {c["name"] for c in inspect(conn).get_columns("ops_operations")}
        store._migrate_add_columns(conn)
        modes = dict(conn.execute(text("SELECT slug, execution_mode FROM ops_operations")).all())
    assert modes == {"old.docker": "isolated", "old.release": "process"}


def test_the_migration_never_rewrites_an_admin_choice():
    store = _store()
    store.create_operation(_op_data(execution_mode="process"))
    with store.engine.begin() as conn:
        store._migrate_add_columns(conn)
        assert conn.execute(text("SELECT execution_mode FROM ops_operations")).scalar_one() == "process"


def test_a_patch_to_the_mode_is_versioned_and_audited():
    store = _store()
    row = store.create_operation(_op_data(execution_mode="isolated"))
    updated = store.update_operation(SLUG, {"execution_mode": "process"}, actor="admin")
    assert (updated.version, updated.execution_mode) == (row.version + 1, "process")
    with Session(store.engine) as session:
        revisions = session.exec(select(OpsOperationRevision).where(
            OpsOperationRevision.operation_id == row.id).order_by(OpsOperationRevision.version)).all()
    assert [json.loads(r.snapshot_json)["execution_mode"] for r in revisions] == ["isolated", "process"]


@pytest.mark.parametrize("kind, mode, ok", [
    ("docker", "process", True), ("docker", "isolated", True),
    ("release", "process", True), ("release", "isolated", False),
    ("docker", "remote", False), ("docker", None, False),
])
def test_each_executor_kind_accepts_only_its_modes(kind, mode, ok):
    if ok:
        assert validate_execution_mode(mode, kind) == mode
    else:
        with pytest.raises(ValueError):
            validate_execution_mode(mode, kind)


def test_the_store_refuses_an_unknown_mode():
    store = _store()
    with pytest.raises(ValueError, match="execution_mode"):
        store.create_operation(_op_data(execution_mode="remote"))


# ---- snapshot -------------------------------------------------------------

def test_prepare_pins_the_rows_mode_in_both_copies():
    store = _store()
    store.create_operation(_op_data(execution_mode="isolated"))
    service, _ = _make(store)
    snapshot = service.prepare_invocation(SLUG, ARGS)
    assert snapshot["execution"]["execution_mode"] == snapshot["operation"]["execution_mode"] == "isolated"
    assert not is_legacy_snapshot(snapshot)
    assert snapshot_execution_mode(snapshot) == "isolated"


def test_an_approval_is_refused_once_the_row_moved_to_another_mode():
    store = _store()
    store.create_operation(_op_data(execution_mode="isolated"))
    service, _ = _make(store)
    snapshot = service.prepare_invocation(SLUG, ARGS)
    store.update_operation(SLUG, {"execution_mode": "process"})
    with pytest.raises(OpsDispatchError) as exc:
        service.dispatch_job(operation_slug=SLUG, args=ARGS, approved_snapshot=snapshot)
    assert exc.value.code == "snapshot_invalid"
    assert "execution_mode mismatch" in str(exc.value)
    assert not store.list_jobs()


def test_an_approval_dispatches_in_its_mode_while_the_row_agrees():
    store = _store()
    store.create_operation(_op_data(execution_mode="isolated"))
    service, _ = _make(store)
    snapshot = service.prepare_invocation(SLUG, ARGS)
    job = service.dispatch_job(operation_slug=SLUG, args=ARGS, approved_snapshot=snapshot)
    assert json.loads(store.get_job(job["id"]).invocation_snapshot_json)["execution"]["execution_mode"] == "isolated"


def test_a_legacy_approval_dispatches_while_the_row_is_still_isolated():
    store = _store()
    store.create_operation(_op_data(execution_mode="isolated"))
    service, _ = _make(store)
    legacy = _legacy(service.prepare_invocation(SLUG, ARGS))
    assert is_legacy_snapshot(legacy) and snapshot_execution_mode(legacy) == "isolated"
    job = service.dispatch_job(operation_slug=SLUG, args=ARGS, approved_snapshot=legacy)
    assert job["id"]


def test_a_legacy_approval_is_refused_once_the_row_moved_to_process():
    store = _store()
    store.create_operation(_op_data(execution_mode="isolated"))
    service, _ = _make(store)
    legacy = _legacy(service.prepare_invocation(SLUG, ARGS))
    store.update_operation(SLUG, {"execution_mode": "process"})
    with pytest.raises(OpsDispatchError) as exc:
        service.dispatch_job(operation_slug=SLUG, args=ARGS, approved_snapshot=legacy)
    assert exc.value.code == "snapshot_invalid"
    assert "execution_mode mismatch" in str(exc.value)


@pytest.mark.parametrize("damage", ["execution_only", "operation_only", "disagree", "unknown"])
def test_an_incoherent_mode_shape_is_snapshot_invalid(damage):
    store = _store()
    store.create_operation(_op_data(execution_mode="isolated"))
    service, _ = _make(store)
    snapshot = json.loads(canonical_json(service.prepare_invocation(SLUG, ARGS)))
    if damage == "execution_only":
        snapshot["operation"].pop("execution_mode")
    elif damage == "operation_only":
        snapshot["execution"].pop("execution_mode")
    elif damage == "disagree":
        snapshot["execution"]["execution_mode"] = "process"
    else:
        snapshot["execution"]["execution_mode"] = snapshot["operation"]["execution_mode"] = "remote"
    with pytest.raises(OpsDispatchError) as exc:
        service.dispatch_job(operation_slug=SLUG, args=ARGS, approved_snapshot=_rehash(snapshot))
    assert exc.value.code == "snapshot_invalid"


# ---- executors ------------------------------------------------------------

def _docker():
    return DockerExecutor(client_factory=lambda: None, worker_image="sha256:" + "0" * 64,
                          receipt_volume="v", receipt_root="/nonexistent")


def _snap(kind, mode=None):
    snapshot = {"execution": {"executor_kind": kind}, "operation": {}}
    if mode:
        snapshot["execution"]["execution_mode"] = snapshot["operation"]["execution_mode"] = mode
    return snapshot


def test_docker_runs_isolated_and_legacy_jobs_in_the_container_launcher():
    executor = _docker()
    assert isinstance(executor._launcher_for(_snap("docker", "isolated")), ContainerLauncher)
    # An executor reads its own legacy job by its own kind.
    assert isinstance(executor._launcher_for({"execution": {}}), ContainerLauncher)


def test_docker_refuses_a_mode_it_has_no_launcher_for():
    executor = _docker()
    with pytest.raises(ExecutorError, match="not available"):
        executor._launcher_for(_snap("docker", "process"))


def test_release_runs_only_the_inert_process_mode():
    executor = ReleaseExecutor(SimpleNamespace(releases=SimpleNamespace(engine=object())))
    assert executor.supported_execution_modes() == {"process"}
    executor.preflight(_snap("release", "process"))
    executor.preflight(_snap("release"))  # legacy release job
    with pytest.raises(ExecutorError, match="not available"):
        executor.preflight(_snap("release", "isolated"))


# ---- approval card --------------------------------------------------------

def test_the_card_says_where_a_docker_job_runs():
    assert with_card_lines("Restart?", _snap("docker", "process")) == (
        "Restart?\nRuns: inside the ops service (child process)")
    assert with_card_lines(None, _snap("docker", "isolated")) == "Runs: isolated worker"
    assert with_card_lines("Restart?", _snap("docker")) == "Restart?\nRuns: isolated worker"


def test_the_card_names_a_self_target():
    snapshot = {**_snap("docker", "isolated"), "self_target": True}
    assert card_lines(snapshot, self_target=True)[-1] == SELF_TARGET_CARD_LINE
    assert with_card_lines("Restart?", snapshot).endswith("\n" + SELF_TARGET_CARD_LINE)


def test_a_release_card_is_unchanged():
    assert with_card_lines("Release?", _snap("release", "process")) == "Release?"
