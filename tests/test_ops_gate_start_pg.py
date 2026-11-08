"""Committed admission capacity + short ledger guard; no runner or live writes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import os
from types import SimpleNamespace

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from llm_bawt.ops import gate_custody as custody
from llm_bawt.ops import gate_start as start
from llm_bawt.ops.models import JOB_TERMINAL_STATES
from llm_bawt.ops.service import OpsService
from llm_bawt.ops.validation import canonical_json
from llm_bawt.ops.worker_slots import (WORKER_CONCURRENCY_ENV, _lock_worker_slots,
    live_worker_slots, needs_worker_slot)
from tests import test_ops_gate_custody_pg as helpers
from tests import test_ops_gate_trust_pg as trust_helpers
from tests.pg_lock_wait import wait_blocked_by
from tests.test_ops_service import FakeExecutor

rig = helpers.rig
configured = helpers.configured
claimed = helpers.claimed
HANG_GUARD = helpers.HANG_GUARD
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def ready(claimed, monkeypatch):
    monkeypatch.setenv(WORKER_CONCURRENCY_ENV, "1")
    with helpers.reserve(claimed):
        pass
    return claimed


def reserve(c, *, store=None):
    return start.reserve_gate_start(store or c.rig.store, c.job.id, receipt_root=c.root)


def directory(c):
    return c.root / c.job.id


def process_job(c):
    op = c.rig.operation(execution_mode="process")
    snapshot = OpsService(c.rig.store, executor=FakeExecutor()).prepare_invocation(op.slug, {"target": "a"})
    return c.rig.job(op=op, snapshot=snapshot, args={"target": "a"})


def assert_unreserved(c):
    job = c.rig.store.get_job(c.job.id)
    assert job.worker_slot_at is None and job.log_file_path is None
    assert not (directory(c) / start._START).exists()


def test_commit_before_yield_holds_shared_budget_without_running_or_events(ready):
    c = ready
    before_events = c.rig.events(c.job.id)
    peer = process_job(c)
    assert needs_worker_slot(json.loads(peer.invocation_snapshot_json))
    assert not needs_worker_slot(json.loads(c.job.invocation_snapshot_json))
    with reserve(c) as boundary:
        fresh = c.rig.other.get_job(c.job.id)  # A different connection sees phase-one commit.
        raw = (directory(c) / start._START).read_bytes()
        record = json.loads(raw)
        assert fresh.worker_slot_at is not None and fresh.state == "dispatching"
        assert fresh.log_file_path == str(directory(c) / start._START)
        assert boundary.job_id == c.job.id and boundary.owner_nonce == record["owner_nonce"]
        assert boundary.start_sha256 == hashlib.sha256(raw).hexdigest()
        assert record["custody_sha256"] == fresh.host_unit_name.removeprefix(custody._MARKER)
        assert fresh.started_at is None and fresh.finished_at is None and fresh.exit_code is None
        assert fresh.receipt_callback_json is None
        assert not c.rig.other.claim_job(peer.id)  # Existing ordinary claim sees this gate's slot.
        with c.rig.engine.begin() as conn:
            assert live_worker_slots(conn) == 1
        with pytest.raises(start.GateStartError, match="^gate_start_busy$"):
            with reserve(c, store=c.rig.other):
                pytest.fail("second consumer entered")
    assert c.rig.events(c.job.id) == before_events
    assert c.rig.store.get_job(c.job.id).worker_slot_at is not None  # No implicit release.
    with pytest.raises(start.GateStartError, match="already_reserved"):
        with reserve(c):
            pytest.fail("completed boundary reissued")


def test_existing_process_reservation_denies_gate_without_consuming_identity(ready):
    c = ready
    peer = process_job(c)
    assert c.rig.store.claim_job(peer.id)
    with pytest.raises(start.GateStartError, match="capacity_unavailable"):
        with reserve(c):
            pytest.fail("overbooked")
    assert_unreserved(c)
    c.rig.store.mark_terminal(peer.id, state="succeeded")
    with reserve(c):
        pass  # A definite capacity refusal did not publish a start marker.


@pytest.mark.parametrize("terminal", [False, True])
def test_process_spawn_holds_include_unreleased_prepared_terminal_holds(ready, terminal):
    c = ready
    peer = c.rig.job(op=c.rig.operation())
    c.rig.sql("UPDATE ops_jobs SET process_spawn_hold_at=now(),process_spawn_hold_target_json=:target,"
        "state=:state WHERE id=:id", id=peer.id, state="lost" if terminal else "queued",
        target='{"fixture":"prepared"}' if terminal else None)
    with pytest.raises(start.GateStartError, match="capacity_unavailable"):
        with reserve(c):
            pytest.fail("ignored spawn hold")
    assert_unreserved(c)


@pytest.mark.parametrize("change", ["disabled", "deleted", "quiesce", "operation-missing"])
def test_unavailable_operation_denies_before_capacity_or_start_evidence(ready, change):
    c = ready
    if change == "operation-missing":
        c.rig.sql("DELETE FROM ops_operations WHERE slug=:slug", slug=c.job.operation_slug)
    else:
        assignment = {"disabled": "enabled=false", "deleted": "soft_deleted_at=now()",
            "quiesce": "quiesce_enabled=true"}[change]
        c.rig.sql(f"UPDATE ops_operations SET {assignment} WHERE slug=:slug", slug=c.job.operation_slug)
    with pytest.raises(start.GateStartError, match="operation_unavailable"):
        with reserve(c):
            pytest.fail("unavailable operation yielded")
    assert_unreserved(c)


@pytest.mark.parametrize("cap_source", ["current", "holder"])
def test_strictest_live_or_holder_operation_cap_is_preserved(ready, cap_source):
    c = ready
    peer = c.rig.job()
    raw = json.loads(peer.invocation_snapshot_json)
    if cap_source == "current":
        c.rig.sql("UPDATE ops_operations SET max_concurrent=1 WHERE slug=:slug", slug=c.job.operation_slug)
    else:
        raw["execution"]["max_concurrent"] = 1
    c.rig.sql("UPDATE ops_jobs SET state='dispatching',dispatched_at=now(),invocation_snapshot_json=:raw "
        "WHERE id=:id", id=peer.id, raw=canonical_json(raw))
    with pytest.raises(start.GateStartError, match="capacity_unavailable"):
        with reserve(c):
            pytest.fail("ignored stricter operation cap")
    assert_unreserved(c)


def test_admitted_cap_cannot_be_relaxed_by_later_operation_edit(configured, tmp_path, monkeypatch):
    rig = configured.rig
    rig.op = rig.store.update_operation(rig.op.id, {"max_concurrent": 1})
    job = trust_helpers._create(configured)
    assert helpers.claim_trusted_gate_job(rig.store, job.id)
    root = tmp_path / "receipts"
    root.mkdir(mode=0o700)
    c = SimpleNamespace(rig=rig, job=rig.store.get_job(job.id), root=root)
    with helpers.reserve(c):
        pass
    rig.op = rig.store.update_operation(rig.op.id, {"max_concurrent": None})
    peer = rig.job(op=rig.op)
    rig.sql("UPDATE ops_jobs SET state='dispatching',dispatched_at=now() WHERE id=:id", id=peer.id)
    monkeypatch.setenv(WORKER_CONCURRENCY_ENV, "2")
    with pytest.raises(start.GateStartError, match="capacity_unavailable"):
        with reserve(c):
            pytest.fail("relaxed admitted cap")
    assert_unreserved(c)


@pytest.mark.parametrize("lock", ["operation", "slots", "job"])
def test_fresh_terminal_after_observed_lock_wait_denies_before_reservation(ready, lock):
    c = ready

    def attempt():
        with reserve(c, store=c.rig.other):
            pytest.fail("stale job yielded")

    with ThreadPoolExecutor(1) as pool, c.rig.engine.connect() as holder:
        holder.begin()
        pid = holder.execute(text("SELECT pg_backend_pid()")).scalar_one()
        if lock == "operation":
            holder.execute(text("SELECT 1 FROM ops_operations WHERE slug=:s FOR UPDATE"),
                {"s": c.job.operation_slug})
        elif lock == "slots":
            _lock_worker_slots(holder)
        else:
            holder.execute(text("SELECT 1 FROM ops_jobs WHERE id=:id FOR UPDATE"), {"id": c.job.id})
        future = pool.submit(attempt)
        try:
            wait_blocked_by(c.rig.engine, pid, future)
            # This transaction either owns the job lock already or can acquire it
            # because the candidate has not reached its fresh job lock yet.
            holder.execute(text("UPDATE ops_jobs SET state='cancelled',finished_at=now() WHERE id=:id"),
                {"id": c.job.id})
        finally:
            holder.commit()
        with pytest.raises(start.GateStartError, match="claim_invalid"):
            future.result(timeout=HANG_GUARD)
    assert_unreserved(c)


def test_gate_winning_last_shared_slot_denies_waiting_process_claim(ready, monkeypatch):
    c = ready
    peer = process_job(c)
    original = start._publish
    original_reserve = start._reserve
    holder = []

    def record_pid(conn, *args):
        holder.append(conn.execute(text("SELECT pg_backend_pid()")).scalar_one())
        return original_reserve(conn, *args)

    monkeypatch.setattr(start, "_reserve", record_pid)
    with ThreadPoolExecutor(1) as pool:
        pending = []

        def publish(directory, raw):
            original(directory, raw)
            pending.append(pool.submit(c.rig.other.claim_job, peer.id))
            wait_blocked_by(c.rig.engine, holder[0], pending[0])

        monkeypatch.setattr(start, "_publish", publish)
        with reserve(c):
            assert pending[0].result(timeout=HANG_GUARD) is False
    assert c.rig.store.get_job(peer.id).state == "queued"


@pytest.mark.parametrize("state", sorted(JOB_TERMINAL_STATES))
def test_terminal_between_committed_reservation_and_guard_refuses_and_never_reissues(ready, monkeypatch, state):
    c = ready
    original = start._guard

    def terminalize(conn, job_id, directory, expected):
        assert c.rig.other.get_job(job_id).worker_slot_at is not None
        c.rig.other.mark_terminal(job_id, state=state)
        return original(conn, job_id, directory, expected)

    monkeypatch.setattr(start, "_guard", terminalize)
    with pytest.raises(start.GateStartError, match="claim_invalid"):
        with reserve(c):
            pytest.fail("terminal job yielded")
    assert (directory(c) / start._START).exists()
    with pytest.raises(start.GateStartError):
        with reserve(c):
            pytest.fail("terminal reservation replayed")


def test_terminal_after_guard_waits_only_for_surviving_transaction(ready, monkeypatch):
    c = ready
    original = start._guard
    holder = []

    def guard(conn, *args):
        result = original(conn, *args)
        holder.append(conn.execute(text("SELECT pg_backend_pid()")).scalar_one())
        return result

    monkeypatch.setattr(start, "_guard", guard)
    with ThreadPoolExecutor(1) as pool:
        with reserve(c):
            future = pool.submit(c.rig.other.mark_terminal, c.job.id, state="lost")
            wait_blocked_by(c.rig.engine, holder[0], future)
            assert c.rig.store.get_job(c.job.id).state == "dispatching"
        assert future.result(timeout=HANG_GUARD).state == "lost"
    # A terminal row is NOT retirement: the reserved gate keeps its shared
    # slot until gate_retire stamps positive evidence (TASK-1865).
    with c.rig.engine.begin() as conn:
        assert live_worker_slots(conn) == 1
    with pytest.raises(start.GateStartError):
        with reserve(c):
            pytest.fail("terminal marker was reusable")


def test_published_orphan_on_rollback_is_not_adopted(ready, monkeypatch):
    c = ready
    original = start._publish

    def publish_then_interrupt(directory, raw):
        original(directory, raw)
        raise RuntimeError("fixture interrupted before commit")

    with monkeypatch.context() as patch:
        patch.setattr(start, "_publish", publish_then_interrupt)
        with pytest.raises(RuntimeError, match="before commit"):
            with reserve(c):
                pytest.fail("uncommitted boundary yielded")
    assert c.rig.store.get_job(c.job.id).worker_slot_at is None
    assert (directory(c) / start._START).exists()
    with pytest.raises(start.GateStartError, match="journal_occupied"):
        with reserve(c):
            pytest.fail("orphan adopted")


@pytest.mark.parametrize("phase,committed", [(1, False), (1, True), (2, False)])
def test_uncertain_transaction_cannot_yield_or_reissue(ready, phase, committed):
    c = ready
    real = c.rig.store
    count = 0

    @contextmanager
    def uncertain():
        nonlocal count
        count += 1
        with real.engine.begin() as conn:
            yield conn
            if count == phase and not committed:
                raise SQLAlchemyError("fixture private DB detail")
        if count == phase and committed:
            raise SQLAlchemyError("fixture private commit detail")

    store = SimpleNamespace(_require=real._require, get_job=real.get_job,
        engine=SimpleNamespace(dialect=real.engine.dialect, begin=uncertain))
    with pytest.raises(start.GateStartError, match="^gate_start_ledger_uncertain$"):
        with reserve(c, store=store):
            if phase == 1:
                pytest.fail("unconfirmed capacity commit yielded")
    assert (directory(c) / start._START).exists()
    with pytest.raises(start.GateStartError):
        with reserve(c):
            pytest.fail("uncertain reservation reissued")


@pytest.mark.parametrize("kind", ["raw", "mode", "symlink", "hardlink", "fifo", "missing", "extra",
    "slot-time", "nonce", "claim-time"])
def test_changed_evidence_or_ledger_before_guard_never_yields(ready, monkeypatch, kind):
    c = ready
    original = start._guard

    def tamper(conn, job_id, path, expected):
        marker = path / start._START
        if kind == "raw":
            marker.write_bytes(b"{partial")
        elif kind == "mode":
            marker.chmod(0o644)
        elif kind == "hardlink":
            os.link(marker, c.root / "other-link")
        elif kind in {"symlink", "fifo", "missing"}:
            marker.unlink()
            if kind == "symlink":
                marker.symlink_to(path / custody._INTENT)
            elif kind == "fifo":
                os.mkfifo(marker, mode=0o600)
        elif kind == "extra":
            (path / "unexpected").write_bytes(b"")
        elif kind in {"slot-time", "claim-time"}:
            column = "worker_slot_at" if kind == "slot-time" else "dispatched_at"
            conn.execute(text(f"UPDATE ops_jobs SET {column}={column}+interval '1 second' WHERE id=:id"),
                {"id": job_id})
        else:
            raw = (path / custody._INTENT).read_bytes()
            record = json.loads(raw)
            record["owner_nonce"] = "f" * 32
            raw = canonical_json(record).encode()
            (path / custody._INTENT).write_bytes(raw)
            conn.execute(text("UPDATE ops_jobs SET host_unit_name=:name WHERE id=:id"),
                {"id": job_id, "name": custody._MARKER + hashlib.sha256(raw).hexdigest()})
        return original(conn, job_id, path, expected)

    monkeypatch.setattr(start, "_guard", tamper)
    with pytest.raises(start.GateStartError):
        with reserve(c):
            pytest.fail("changed start evidence yielded")
    assert c.rig.store.get_job(c.job.id).worker_slot_at is not None


@pytest.mark.parametrize("loss", ["marker", "all"])
def test_committed_reservation_never_reissues_after_storage_loss(ready, loss):
    c = ready
    with pytest.raises(KeyboardInterrupt):
        with reserve(c):
            raise KeyboardInterrupt()
    (directory(c) / start._START).unlink()
    if loss == "all":
        (directory(c) / custody._INTENT).unlink()
        (directory(c) / "submission.lock").unlink()
        directory(c).rmdir()
    with pytest.raises(start.GateStartError):
        with reserve(c):
            pytest.fail("lost journal allowed reissue")
    assert not (directory(c) / start._START).exists()
