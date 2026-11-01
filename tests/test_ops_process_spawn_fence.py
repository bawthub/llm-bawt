"""Spawn fencing with real flocks/children and authoritative ledger reads."""
from __future__ import annotations

import fcntl
import json
import selectors
import subprocess
import sys
from contextlib import ExitStack, contextmanager

import pytest

from llm_bawt.ops import JOB_SUCCEEDED, OpsService
from llm_bawt.ops.executor import DockerExecutor
from llm_bawt.ops.executor_types import SpawnDeferred
from llm_bawt.ops.process_spawn_fence import PROCESS_SPAWN_LOCK, process_spawn_guard
from llm_bawt.ops.self_target import SELF_REQUIRES_ISOLATED, SelfTargetError
from llm_bawt.ops.worker_launch import Liveness, RecoveryRefused
from tests.test_ops_process_launch import HANG_GUARD, IMAGE_ID, JOB, _fresh, _job, _wait, rig as _process_rig
from tests.test_ops_service import _store
from tests.test_ops_spawn_hold import _admit, _hold

# Reuse the fixture without shadowing an imported name in each test signature.
rig = _process_rig


@contextmanager
def _exclusive_holder(root):
    # A distinct process owns EX, signals only after acquisition, and exits on
    # explicit input. No sleeps or assertions on how fast the spawn returns.
    code = """import fcntl, sys
with open(sys.argv[1], 'a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    print('locked', flush=True)
    sys.stdin.buffer.read(1)
"""
    holder = subprocess.Popen([sys.executable, "-u", "-c", code, str(root / PROCESS_SPAWN_LOCK)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(holder.stdout, selectors.EVENT_READ)
            assert selector.select(HANG_GUARD), "exclusive-holder handshake missing"
        assert holder.stdout.readline() == b"locked\n"
        yield
    finally:
        try:
            holder.communicate(input=b"x", timeout=HANG_GUARD)
        except subprocess.TimeoutExpired:
            holder.kill()
            holder.communicate(timeout=HANG_GUARD)
            raise
        assert holder.returncode == 0


def _no_attempt_artifacts(directory, intent):
    if intent is not None:
        assert (directory / "spawn-intent.json").read_bytes() == intent
    else:
        assert not (directory / "spawn-intent.json").exists()
    for name in ("spawn.json", "receipt.json", "abandoned.json"):
        assert not (directory / name).exists(), name


@pytest.mark.parametrize("recovery", [False, True])
@pytest.mark.parametrize("fence", ["exclusive", "ledger", "read_error"])
def test_fenced_initial_and_recovery_spawns_consume_no_attempt_and_retry_after_release(rig, recovery, fence):
    launcher, settings, root = _fresh(rig) if recovery else rig[:3]
    directory, digest = _job(root, "ok")
    intent = b'{"attempt": 1, "intended_at": "prior"}' if recovery else None
    if intent:
        (directory / "spawn-intent.json").write_bytes(intent)
    checks, spawned = [], []
    real_popen = launcher._popen
    launcher._popen = lambda *_a, **_kw: spawned.append("unexpected spawn")
    with ExitStack() as stack:
        if fence == "exclusive":
            stack.enter_context(_exclusive_holder(root))
        elif fence == "ledger":
            store = _store()
            holder = _admit(store)
            assert _hold(store, holder)
            launcher._spawn_hold_reader = store.process_spawns_held
            stack.callback(lambda: store.mark_terminal(holder, state=JOB_SUCCEEDED))
        else:
            def unavailable():
                raise OSError("ledger offline")
            launcher._spawn_hold_reader = unavailable
        for _ in range(2):
            if recovery:
                assert launcher.probe(job_id=JOB, settings=settings, directory=directory, digest=digest,
                                      revalidate=lambda: checks.append("checked")) is Liveness.DEFERRED
            else:
                with pytest.raises(SpawnDeferred):
                    launcher.launch(job_id=JOB, settings=settings, directory=directory, digest=digest)
            _no_attempt_artifacts(directory, intent)
        assert not spawned and not checks and not launcher._children
        # A held fence is not unhealthy interpreter/process readiness.
        assert launcher.interpreter_pinned() and launcher.child_stats() == (0, None)
    if fence == "read_error":
        launcher._spawn_hold_reader = lambda: False
    launcher._popen = real_popen
    if recovery:
        assert launcher.probe(job_id=JOB, settings=settings, directory=directory, digest=digest,
                              revalidate=lambda: checks.append("checked")) is Liveness.STARTED
        assert checks == ["checked"]
    else:
        launcher.launch(job_id=JOB, settings=settings, directory=directory, digest=digest)
    _wait(launcher)
    assert json.loads((directory / "spawn-intent.json").read_text())["attempt"] == (2 if recovery else 1)
    assert json.loads((directory / "receipt.json").read_text())["state"] == "succeeded"


def _shared_is_held(root):
    with (root / PROCESS_SPAWN_LOCK).open("a") as contender:
        with pytest.raises(BlockingIOError):
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (root / PROCESS_SPAWN_LOCK).open("a") as peer:
        fcntl.flock(peer, fcntl.LOCK_SH | fcntl.LOCK_NB)


def test_shared_flock_covers_plain_read_revalidation_and_popen_then_releases(rig):
    launcher, settings, root = _fresh(rig)
    directory, digest = _job(root, "ok")
    stages = []

    def reader():
        _shared_is_held(root)
        stages.append("read")
        return False

    def revalidate():
        _shared_is_held(root)
        stages.append("revalidate")

    real_popen = launcher._popen

    def popen(*args, **kwargs):
        _shared_is_held(root)
        stages.append("popen")
        return real_popen(*args, **kwargs)

    launcher._spawn_hold_reader, launcher._popen = reader, popen
    assert launcher.probe(job_id=JOB, settings=settings, directory=directory, digest=digest,
                          revalidate=revalidate) is Liveness.STARTED
    _wait(launcher)
    assert stages == ["read", "revalidate", "popen"]
    with (root / PROCESS_SPAWN_LOCK).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_guard_acquires_shared_nonblocking_and_releases_on_read_failure(tmp_path, monkeypatch):
    real_flock, flags = fcntl.flock, []

    def recorded(lock, operation):
        flags.append(operation)
        return real_flock(lock, operation)

    def unavailable():
        raise OSError("offline")

    monkeypatch.setattr(fcntl, "flock", recorded)
    with pytest.raises(SpawnDeferred, match="read unavailable"):
        with process_spawn_guard(tmp_path, unavailable):
            pytest.fail("unavailable read was treated as open")
    assert flags == [fcntl.LOCK_SH | fcntl.LOCK_NB]
    with (tmp_path / PROCESS_SPAWN_LOCK).open("a") as lock:
        real_flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize("value", [None, 0, "false", True])
def test_unknown_reader_results_fail_closed(tmp_path, value):
    with pytest.raises(SpawnDeferred):
        with process_spawn_guard(tmp_path, lambda: value):
            pytest.fail("only an actual False is open")


def test_missing_reader_and_unavailable_lock_fail_closed(tmp_path):
    with pytest.raises(SpawnDeferred, match="reader unavailable"):
        with process_spawn_guard(tmp_path, None):
            pytest.fail("reader missing")
    with pytest.raises(SpawnDeferred, match="fence unavailable"):
        with process_spawn_guard(tmp_path / "unmounted", lambda: False):
            pytest.fail("root missing")


@pytest.mark.parametrize("prior_attempt", [False, True])
def test_real_executor_maps_fence_to_nonterminal_without_lost_and_later_runs(rig, monkeypatch, prior_attempt):
    launcher, settings, root = _fresh(rig)
    executor = DockerExecutor(client_factory=lambda: None, worker_image=IMAGE_ID,
                              receipt_volume="v", receipt_root=str(root))
    executor._launchers["process"] = launcher
    # Skip unrelated Docker identity/mount checks; exercise the real request
    # publication, launcher, recovery and receipt-to-state mapping.
    monkeypatch.setattr(executor, "preflight", lambda _s: None)
    monkeypatch.setattr(executor, "_check_settings", lambda _s: None)
    monkeypatch.setattr(executor, "self_target", lambda _s: None)
    snapshot = {"execution": {**settings, "execution_mode": "process"},
                "operation": {"execution_mode": "process"}}
    directory = root / JOB
    intent = b'{"attempt": 1}' if prior_attempt else None
    if intent:
        directory.mkdir()
        (directory / "spawn-intent.json").write_bytes(intent)
    with _exclusive_holder(root):
        if not prior_attempt:
            with pytest.raises(SpawnDeferred):
                executor.dispatch(job_id=JOB, snapshot=snapshot)
        for _ in range(2):
            assert executor.reconcile(job_id=JOB, snapshot=snapshot).state is None
        _no_attempt_artifacts(directory, intent)
    if prior_attempt:
        # Recovery adopts the exact same already-published request identity.
        from llm_bawt.ops.executor import worker_request
        from llm_bawt.ops.validation import canonical_json
        (directory / "request.json").write_text(canonical_json(worker_request(JOB, snapshot)))
    assert executor.reconcile(job_id=JOB, snapshot=snapshot).state == "accepted"
    _wait(launcher)
    assert executor.reconcile(job_id=JOB, snapshot=snapshot).state == "succeeded"
    assert not (directory / "abandoned.json").exists()


def test_observations_stay_available_while_spawn_fence_is_closed(rig):
    launcher, settings, root = _fresh(rig)
    launcher._spawn_hold_reader = lambda: True
    directory, digest = _job(root, "ok")
    with (directory / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert launcher.probe(job_id=JOB, settings=settings, directory=directory, digest=digest) is Liveness.ALIVE
    (directory / "receipt.json").write_text(json.dumps({"state": "succeeded", "started_at": "t"}))
    assert launcher.probe(job_id=JOB, settings=settings, directory=directory, digest=digest) is Liveness.EXITED


def test_deferred_initial_attempt_revalidation_refusal_is_certain_not_prior_attempt(rig):
    launcher, settings, root = _fresh(rig)
    directory, digest = _job(root, "ok")

    def refuse():
        raise SelfTargetError(SELF_REQUIRES_ISOLATED, "now the ops service")

    with pytest.raises(RecoveryRefused) as caught:
        launcher.probe(job_id=JOB, settings=settings, directory=directory, digest=digest, revalidate=refuse)
    assert not caught.value.prior_attempt
    _no_attempt_artifacts(directory, None)


def test_default_service_wires_authoritative_plain_store_reader():
    store = _store()
    service = OpsService(store)
    launcher = service.executors()["docker"]._launchers["process"]
    assert launcher._spawn_hold_reader() is False
    holder = _admit(store)
    assert _hold(store, holder)
    assert launcher._spawn_hold_reader() is True
    store.mark_terminal(holder, state=JOB_SUCCEEDED)
    assert launcher._spawn_hold_reader() is False
