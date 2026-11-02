"""One-shot gate custody on disposable PG/files only; fake GET-only Docker."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import multiprocessing
import os
from threading import Event
from types import SimpleNamespace

import pytest

from llm_bawt.ops import gate_custody as custody
from llm_bawt.ops.gate_runner import gate_container_labels, gate_container_names
from llm_bawt.ops.gate_trust import read_trusted_gate_request
from llm_bawt.ops.gate_trusted_claim import claim_trusted_gate_job
from tests import test_ops_gate_trust_pg as helpers

rig = helpers.rig
configured = helpers.configured
HANG_GUARD = helpers.HANG_GUARD
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


class Docker:
    def __init__(self, containers=None):
        self.containers, self.calls = containers or {}, []

    def request(self, method, path):
        assert method == "GET" and path.endswith("/json")
        self.calls.append((method, path))
        name = path.split("/")[2]
        if name not in self.containers:
            raise RuntimeError("Docker HTTP 404: fixture missing")
        return self.containers[name]


@pytest.fixture
def claimed(configured, tmp_path):
    job = helpers._create(configured)
    assert claim_trusted_gate_job(configured.rig.store, job.id)
    root = tmp_path / "receipts"
    root.mkdir(mode=0o700)
    return SimpleNamespace(rig=configured.rig, job=configured.rig.store.get_job(job.id), root=root)


def reserve(claimed, *, store=None):
    return custody.reserve_gate_launch(store or claimed.rig.store, claimed.job.id, receipt_root=claimed.root)


def observe(claimed, api):
    return custody.observe_gate_launch(claimed.rig.store, claimed.job.id, receipt_root=claimed.root, api=api)


def files(root):
    return {str(p.relative_to(root)): (p.lstat().st_mode, p.read_bytes() if p.is_file() else None)
        for p in root.rglob("*")}


def test_reservation_binds_ledger_and_nonce_without_launch_or_lifecycle_change(claimed):
    c = claimed
    before_events = c.rig.events(c.job.id)
    with reserve(c) as ticket:
        assert ticket.job_id == c.job.id and len(ticket.owner_nonce) == 32
        path = c.root / c.job.id / custody._INTENT
        raw = path.read_bytes()
        assert ticket.intent_sha256 == hashlib.sha256(raw).hexdigest()
        record = json.loads(raw)
        request = read_trusted_gate_request(c.job)
        assert record["owner_nonce"] == ticket.owner_nonce
        assert record["gate_plan"] == request.plan.to_dict()
        assert record["environment_sha256"] == request.environment_sha256
        fresh = c.rig.store.get_job(c.job.id)
        assert fresh.state == "dispatching" and fresh.worker_slot_at is None
        assert fresh.started_at is None and fresh.finished_at is None and fresh.exit_code is None
        assert fresh.receipt_callback_json is None
        assert fresh.host_unit_name == "gate-custody:" + ticket.intent_sha256
        assert fresh.status_file_path == str(path)
        assert observe(c, Docker()) == custody.GateCustodyObservation(True)
    assert c.rig.events(c.job.id) == before_events  # Metadata only, not accepted/running.
    before = files(c.root)
    api = Docker()
    result = observe(c, api)
    assert not result.busy and result.runner.status == result.postgres.status == "missing"
    assert len(api.calls) == 2 and files(c.root) == before
    with pytest.raises(custody.GateCustodyError, match="already_reserved_or_closed"):
        with reserve(c):
            pytest.fail("reservation replayed")


@pytest.mark.parametrize("state", ["succeeded", "failed", "lost", "timed_out", "cancelled"])
def test_terminal_row_is_not_retirement_and_missing_journal_never_reissues(claimed, state):
    c = claimed
    with reserve(c):
        pass
    c.rig.store.mark_terminal(c.job.id, state=state)
    result = observe(c, Docker())  # Read-only observation remains possible, never success.
    assert result.runner.status == "missing"
    (c.root / c.job.id / custody._INTENT).unlink()
    api = Docker()
    with pytest.raises(custody.GateCustodyError):
        observe(c, api)
    assert not api.calls
    with pytest.raises(custody.GateCustodyError):
        with reserve(c):
            pytest.fail("terminal/missing intent replayed")


@pytest.mark.parametrize("remove", ["intent", "directory"])
def test_lost_storage_with_nonterminal_db_marker_does_not_reissue(claimed, remove):
    c = claimed
    with reserve(c):
        pass
    directory = c.root / c.job.id
    (directory / custody._INTENT).unlink()
    if remove == "directory":
        (directory / "submission.lock").unlink()
        directory.rmdir()
    assert c.rig.store.get_job(c.job.id).state == "dispatching"
    with pytest.raises(custody.GateCustodyError, match="already_reserved_or_closed"):
        with reserve(c):
            pytest.fail("missing journal replayed")
    if remove == "directory":
        assert not directory.exists()  # DB denial precedes all filesystem mutations.


def test_interrupted_yield_keeps_both_durable_fences(claimed):
    with pytest.raises(KeyboardInterrupt):
        with reserve(claimed):
            raise KeyboardInterrupt()
    assert (claimed.root / claimed.job.id / custody._INTENT).exists()
    with pytest.raises(custody.GateCustodyError):
        with reserve(claimed):
            pytest.fail("interrupted consumer replayed")


def test_orphan_intent_after_rollback_is_never_adopted(claimed, monkeypatch):
    original = custody._publish

    def publish_then_interrupt(directory, raw):
        original(directory, raw)
        raise RuntimeError("fixture interrupted before DB commit")

    with monkeypatch.context() as patch:
        patch.setattr(custody, "_publish", publish_then_interrupt)
        with pytest.raises(RuntimeError, match="interrupted before DB commit"):
            with reserve(claimed):
                pytest.fail("uncommitted ticket yielded")
    assert claimed.rig.store.get_job(claimed.job.id).host_unit_name is None
    before = files(claimed.root)
    with pytest.raises(custody.GateCustodyError, match="journal_occupied"):
        with reserve(claimed):
            pytest.fail("orphan adopted")
    assert files(claimed.root) == before
    api = Docker()
    with pytest.raises(custody.GateCustodyError, match="reservation_missing"):
        observe(claimed, api)
    assert not api.calls


def test_commit_response_lost_cannot_return_or_reissue_ticket(claimed):
    real = claimed.rig.store

    @contextmanager
    def lost_commit_response():
        with real.engine.begin() as conn:
            yield conn
        raise RuntimeError("fixture commit acknowledgement lost")

    store = SimpleNamespace(_require=real._require, get_job=real.get_job,
        engine=SimpleNamespace(dialect=real.engine.dialect, begin=lost_commit_response))
    with pytest.raises(RuntimeError, match="acknowledgement lost"):
        with reserve(claimed, store=store):
            pytest.fail("uncertain commit returned a ticket")
    assert real.get_job(claimed.job.id).host_unit_name is not None
    with pytest.raises(custody.GateCustodyError):
        with reserve(claimed):
            pytest.fail("uncertain commit replayed")


def test_concurrent_reservation_never_yields_twice(claimed, monkeypatch):
    entered, release = Event(), Event()
    original = custody._publish

    def hold(directory, raw):
        entered.set()
        assert release.wait(timeout=HANG_GUARD)
        original(directory, raw)

    def first():
        with reserve(claimed) as ticket:
            return ticket

    monkeypatch.setattr(custody, "_publish", hold)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(first)
        try:
            assert entered.wait(timeout=HANG_GUARD)
            with pytest.raises(custody.GateCustodyError, match="gate_custody_busy"):
                with reserve(claimed, store=claimed.rig.other):
                    pytest.fail("second reservation yielded")
            api = Docker()
            assert observe(claimed, api).busy and not api.calls
        finally:
            release.set()
        assert future.result(timeout=HANG_GUARD).job_id == claimed.job.id


@pytest.mark.parametrize("kind", ["partial", "oversize", "nonce", "nonce-replaced", "job", "snapshot", "claim-time",
    "extra", "duplicate", "symlink", "hardlink", "mode", "fifo"])
def test_invalid_or_transplanted_intent_refuses_before_docker(claimed, kind):
    c = claimed
    with reserve(c):
        pass
    path = c.root / c.job.id / custody._INTENT
    raw = path.read_bytes()
    data = json.loads(raw)
    if kind == "partial":
        path.write_bytes(b"{")
    elif kind == "oversize":
        path.write_bytes(b"x" * (custody._BOUND + 1))
    elif kind in {"symlink", "fifo"}:
        path.unlink()
        if kind == "symlink":
            path.symlink_to(c.root / "no-follow")
        else:
            os.mkfifo(path, mode=0o600)
    elif kind == "hardlink":
        os.link(path, path.with_suffix(".other"))
    elif kind == "mode":
        path.chmod(0o644)
    elif kind == "duplicate":
        path.write_bytes(raw[:-1] + b',"owner_nonce":"' + data["owner_nonce"].encode() + b'"}')
    elif kind == "nonce-replaced":
        data["owner_nonce"] = "f" * 32
        path.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")))
    else:
        key = {"nonce": "owner_nonce", "job": "job_id", "snapshot": "invocation_sha256",
            "claim-time": "dispatched_at", "extra": "extra"}[kind]
        data[key] = "fixture-invalid"
        path.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")))
    api = Docker()
    with pytest.raises(custody.GateCustodyError):
        observe(c, api)
    assert not api.calls
    with pytest.raises(custody.GateCustodyError):
        with reserve(c):
            pytest.fail("invalid intent repaired or replayed")


@pytest.mark.parametrize("kind", ["root-mode", "root-link", "directory-link", "lock-link", "lock-fifo",
    "missing-lock", "request-drift", "claim-time-drift", "metadata-drift"])
def test_unsafe_storage_or_ledger_drift_refuses_without_inspection(claimed, kind):
    c = claimed
    with reserve(c):
        pass
    directory = c.root / c.job.id
    if kind == "root-mode":
        c.root.chmod(0o755)
    elif kind == "root-link":
        link = c.root.parent / "link"
        link.symlink_to(c.root, target_is_directory=True)
        c.root = link
    elif kind == "directory-link":
        moved = c.root / "retained"
        directory.rename(moved)
        directory.symlink_to(moved, target_is_directory=True)
    elif kind in {"lock-link", "lock-fifo", "missing-lock"}:
        lock = directory / "submission.lock"
        lock.unlink()
        if kind == "lock-link":
            lock.symlink_to(directory / custody._INTENT)
        elif kind == "lock-fifo":
            os.mkfifo(lock, mode=0o600)
    elif kind == "request-drift":
        c.rig.sql("UPDATE ops_jobs SET request_payload_json='{}' WHERE id=:id", id=c.job.id)
    elif kind == "claim-time-drift":
        c.rig.sql("UPDATE ops_jobs SET dispatched_at=dispatched_at + interval '1 second' WHERE id=:id", id=c.job.id)
    else:
        c.rig.sql("UPDATE ops_jobs SET host_unit_name='foreign' WHERE id=:id", id=c.job.id)
    api = Docker()
    with pytest.raises(custody.GateCustodyError):
        observe(c, api)
    assert not api.calls


def test_observed_owned_running_or_exit_never_changes_ledger_or_removes(claimed):
    c = claimed
    with reserve(c) as ticket:
        pass
    request = read_trusted_gate_request(c.job)
    labels = gate_container_labels(request.plan, c.job.id, ticket.owner_nonce, request.environment_sha256)
    containers = {name: {"Id": ("a" if role == "pg" else "b") * 64, "Name": "/" + name,
        "Config": {"Labels": {**labels, "com.docker.compose.service": f"gate-{role}"}},
        "State": {"Running": role == "pg", "Status": "running" if role == "pg" else "exited",
            "ExitCode": 0}} for role, name in gate_container_names(c.job.id).items()}
    api = Docker(containers)
    before, events, inventory = c.rig.store.get_job(c.job.id).model_dump(), c.rig.events(c.job.id), files(c.root)
    result = observe(c, api)
    assert result.postgres.status == "running" and result.runner.status == "exited"
    assert c.rig.store.get_job(c.job.id).model_dump() == before
    assert c.rig.events(c.job.id) == events and files(c.root) == inventory
    assert len(api.calls) == 2 and len(api.containers) == 2
    assert not hasattr(result, "cleanup_confirmed") and not hasattr(result, "exit_code")


def test_unclaimed_or_unauthenticated_job_cannot_create_custody(configured, tmp_path):
    rig = configured.rig
    job = helpers._create(configured)
    root = tmp_path / "absent"
    with pytest.raises(custody.GateCustodyError, match="claim_required"):
        with custody.reserve_gate_launch(rig.store, job.id, receipt_root=root):
            pytest.fail("unclaimed reservation yielded")
    assert not root.exists()
    assert claim_trusted_gate_job(rig.store, job.id)
    rig.sql("UPDATE ops_jobs SET invocation_snapshot_json='{}' WHERE id=:id", id=job.id)
    with pytest.raises(custody.GateCustodyError, match="request_invalid"):
        with custody.reserve_gate_launch(rig.store, job.id, receipt_root=root):
            pytest.fail("forged reservation yielded")
    assert not root.exists()


def test_fresh_locked_row_not_stale_preread_controls_reservation(claimed, monkeypatch):
    c = claimed
    original = custody._reservable
    first = True

    def terminalize_after_preread(job):
        nonlocal first
        request = original(job)
        if first:
            first = False
            c.rig.store.mark_terminal(c.job.id, state="cancelled")
        return request

    monkeypatch.setattr(custody, "_reservable", terminalize_after_preread)
    with pytest.raises(custody.GateCustodyError, match="already_reserved_or_closed"):
        with reserve(c):
            pytest.fail("stale preread returned a ticket")
    assert c.rig.store.get_job(c.job.id).host_unit_name is None
    assert not (c.root / c.job.id / custody._INTENT).exists()


def _hold_submission_in_child(directory, entered, release):
    # Child touches only an existing filesystem lock, never an inherited DB pool.
    with custody._submission(directory, create=False):
        entered.set()
        release.wait(timeout=HANG_GUARD)


def test_dead_lock_holder_allows_observation_but_never_new_reservation(claimed):
    c = claimed
    with reserve(c):
        pass
    ctx = multiprocessing.get_context("fork")
    entered, release = ctx.Event(), ctx.Event()
    child = ctx.Process(target=_hold_submission_in_child,
        args=(c.root / c.job.id, entered, release), daemon=True)
    child.start()
    try:
        assert entered.wait(timeout=HANG_GUARD)
        api = Docker()
        assert observe(c, api).busy and not api.calls
        child.kill()
        child.join(timeout=HANG_GUARD)
        assert not child.is_alive()
        assert observe(c, api).runner.status == "missing"
        with pytest.raises(custody.GateCustodyError, match="already_reserved_or_closed"):
            with reserve(c):
                pytest.fail("dead local holder allowed replay")
    finally:
        # Never touch the waiter's Event after SIGKILL: its condition lock may
        # have died with it. This fixture always kills/joins, never releases it.
        if child.is_alive():
            child.kill()
        child.join(timeout=HANG_GUARD)
