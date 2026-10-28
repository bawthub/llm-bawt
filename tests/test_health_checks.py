"""TASK-1157: internal capability checks report the real failure reason."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from llm_bawt.ops.executor import DockerExecutor
from llm_bawt.ops.models import JOB_QUEUED, JOB_RUNNING
from llm_bawt.ops.service import OpsDispatchError, OpsService
from llm_bawt.service.health_checks import (
    DEGRADED, DOWN, OK, check_ops_executors, check_ops_queue, overall,
)

IMAGE = "sha256:" + "b" * 64
NOW = datetime(2026, 10, 18, 14, 0, tzinfo=timezone.utc)


def _executor(tmp_path, monkeypatch, *, root=None, mounts=None):
    monkeypatch.setenv("LLM_BAWT_OPS_APP_CONTAINER", "fake-app")
    root = root or str(tmp_path)
    app = SimpleNamespace(attrs={"Mounts": mounts if mounts is not None else [
        {"Type": "volume", "Name": "receipts", "Destination": root, "RW": True}]})
    client = SimpleNamespace(
        containers=SimpleNamespace(get=lambda _name: app),
        images=SimpleNamespace(get=lambda _name: object()),
        volumes=SimpleNamespace(get=lambda _name: object()), ping=lambda: True)
    return DockerExecutor(client_factory=lambda: client, worker_image=IMAGE,
                          receipt_volume="receipts", receipt_root=root)


def test_readiness_ok_and_available_agree(tmp_path, monkeypatch):
    executor = _executor(tmp_path, monkeypatch)
    assert executor.readiness() is None
    assert executor.available() is True


def test_readiness_names_missing_receipt_mount(tmp_path, monkeypatch):
    # The TASK-1145 outage: app recreated without the ops overlay.
    executor = _executor(tmp_path, monkeypatch, root=str(tmp_path / "absent"))
    assert executor.readiness() == "receipt volume is not mounted at configured app path"
    assert executor.available() is False
    service = OpsService(SimpleNamespace(engine=None), executor=executor,
                         release_store=SimpleNamespace(), release_verifier=SimpleNamespace())
    [row] = check_ops_executors(service)
    assert row == {"name": "ops.executor.docker", "status": DOWN,
                   "reason": "receipt volume is not mounted at configured app path"}


def test_ops_executor_rows_come_from_the_ops_service_readiness():
    # TASK-1169 S3c: the app's admission-only service asks the ops service.
    ready = SimpleNamespace(executor_readiness=lambda: {"release": None, "docker": "socket gone"})
    assert check_ops_executors(ready) == [
        {"name": "ops.executor.docker", "status": DOWN, "reason": "socket gone"},
        {"name": "ops.executor.release", "status": "ok", "reason": None},
    ]


def test_unreachable_ops_service_is_one_down_control_row():
    def unreachable():
        raise OpsDispatchError("ops_unavailable", "ops_unavailable: cannot connect")

    assert check_ops_executors(SimpleNamespace(executor_readiness=unreachable)) == [
        {"name": "ops.control", "status": DOWN, "reason": "ops_unavailable: cannot connect"}]


def test_readiness_wraps_daemon_errors(tmp_path, monkeypatch):
    executor = _executor(tmp_path, monkeypatch)
    def boom():
        raise ConnectionError("socket gone")
    executor.client.ping = boom
    assert executor.readiness() == "Docker worker prerequisites unavailable: socket gone"


def _job(slug, *, error=None, submitted=0, reconciled=None):
    return SimpleNamespace(operation_slug=slug, error_text=error,
                           submitted_at=NOW - timedelta(seconds=submitted),
                           dispatched_at=None,
                           last_reconcile_at=None if reconciled is None else NOW - timedelta(seconds=reconciled))


def _store(by_state):
    return SimpleNamespace(engine=object(),
                           list_jobs=lambda *, state, limit: by_state.get(state, []))


def test_queue_ok_when_fresh():
    row = check_ops_queue(_store({JOB_QUEUED: [_job("a", submitted=5)]}), NOW)
    assert row["status"] == OK and row["reason"] is None


def test_queue_blocked_preflight_is_down_with_reason():
    row = check_ops_queue(_store({JOB_QUEUED: [
        _job("bawthub.restart-backend", error="receipt volume is not mounted at configured app path")]}), NOW)
    assert row["status"] == DOWN
    assert "bawthub.restart-backend blocked: receipt volume" in row["reason"]


def test_queue_stale_and_stranded_are_degraded():
    row = check_ops_queue(_store({
        JOB_QUEUED: [_job("a", submitted=600)],
        JOB_RUNNING: [_job("b", submitted=900, reconciled=300)],
    }), NOW)
    assert row["status"] == DEGRADED
    assert "a queued but unclaimed" in row["reason"]
    assert "reconciler" in row["reason"]


def test_overall_takes_worst():
    assert overall([{"status": OK}, {"status": DEGRADED}]) == DEGRADED
    assert overall([{"status": DEGRADED}, {"status": DOWN}]) == DOWN
    assert overall([]) == OK
