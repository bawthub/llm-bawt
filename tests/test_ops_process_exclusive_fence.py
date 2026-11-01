"""Dormant exclusive-acquisition seam: real processes, no sleeps or lease expiry."""
from __future__ import annotations

import fcntl
import selectors
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest

from llm_bawt.ops.executor_types import SpawnDeferred
from llm_bawt.ops.process_spawn_fence import (PROCESS_SPAWN_LOCK, SpawnFenceDeadlineExceeded,
                                             exclusive_process_spawn_guard, process_spawn_guard)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
DEADLINE = NOW + timedelta(seconds=30)
HANG_GUARD = 10


@contextmanager
def _shared_holder(root):
    code = """import fcntl, sys
with open(sys.argv[1], 'a') as lock:
    fcntl.flock(lock, fcntl.LOCK_SH)
    print('locked', flush=True)
    sys.stdin.buffer.read(1)
"""
    holder = subprocess.Popen([sys.executable, "-u", "-c", code, str(root / PROCESS_SPAWN_LOCK)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(holder.stdout, selectors.EVENT_READ)
            assert selector.select(HANG_GUARD), "shared-holder handshake missing"
        assert holder.stdout.readline() == b"locked\n"
        yield holder
    finally:
        try:
            holder.communicate(input=b"x", timeout=HANG_GUARD)
        except subprocess.TimeoutExpired:
            holder.kill()
            holder.communicate(timeout=HANG_GUARD)
            raise
        assert holder.returncode == 0


def _guard(root, *, deadline=DEADLINE, now=NOW):
    return exclusive_process_spawn_guard(root, acquire_deadline=deadline, clock=lambda: now)


def _assert_exclusive_available(root):
    with (root / PROCESS_SPAWN_LOCK).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_distinct_shared_holder_defers_repeated_attempts_then_retry_succeeds(tmp_path):
    with _shared_holder(tmp_path):
        # Spawn observations remain available and SH peers remain compatible.
        with process_spawn_guard(tmp_path, lambda: False):
            pass
        for _ in range(3):
            with pytest.raises(SpawnDeferred, match="busy or unavailable"):
                with _guard(tmp_path):
                    pytest.fail("exclusive guard entered despite a distinct SH holder")
    with _guard(tmp_path):
        with pytest.raises(SpawnDeferred):
            with process_spawn_guard(tmp_path, lambda: False):
                pytest.fail("spawn guard entered while exclusive owner was active")
    _assert_exclusive_available(tmp_path)
    assert [entry.name for entry in tmp_path.iterdir()] == [PROCESS_SPAWN_LOCK]


def test_acquisition_deadline_is_not_renewed_by_retries_or_holder_release(tmp_path):
    now = NOW
    with _shared_holder(tmp_path) as holder:
        with pytest.raises(SpawnDeferred):
            with exclusive_process_spawn_guard(tmp_path, acquire_deadline=DEADLINE, clock=lambda: now):
                pytest.fail("busy")
        now = DEADLINE
        with pytest.raises(SpawnFenceDeadlineExceeded):
            with exclusive_process_spawn_guard(tmp_path, acquire_deadline=DEADLINE, clock=lambda: now):
                pytest.fail("expired")
        assert holder.poll() is None  # Deadline never stops/revokes the actual owner.
    # A new invocation after owner exit still uses the original expired budget.
    with pytest.raises(SpawnFenceDeadlineExceeded):
        with exclusive_process_spawn_guard(tmp_path, acquire_deadline=DEADLINE, clock=lambda: now):
            pytest.fail("owner exit must not reset an absolute deadline")
    _assert_exclusive_available(tmp_path)


def test_guard_uses_single_nonblocking_exclusive_attempt_and_releases_on_body_error(tmp_path, monkeypatch):
    real_flock, flags = fcntl.flock, []

    def recorded(lock, operation):
        flags.append(operation)
        return real_flock(lock, operation)

    monkeypatch.setattr(fcntl, "flock", recorded)
    with pytest.raises(RuntimeError, match="critical section failed"):
        with _guard(tmp_path):
            assert flags == [fcntl.LOCK_EX | fcntl.LOCK_NB]
            raise RuntimeError("critical section failed")
    _assert_exclusive_available(tmp_path)


def test_exclusive_owner_blocks_shared_and_exclusive_contenders_in_another_process(tmp_path):
    code = """import fcntl, sys
for operation in (fcntl.LOCK_SH, fcntl.LOCK_EX):
    with open(sys.argv[1], 'a') as lock:
        try:
            fcntl.flock(lock, operation | fcntl.LOCK_NB)
        except BlockingIOError:
            continue
        raise SystemExit('unexpected acquisition')
print('both blocked')
"""
    with _guard(tmp_path):
        result = subprocess.run([sys.executable, "-c", code, str(tmp_path / PROCESS_SPAWN_LOCK)],
                                capture_output=True, text=True, timeout=HANG_GUARD)
        assert result.returncode == 0, result.stderr
        assert result.stdout == "both blocked\n"
    _assert_exclusive_available(tmp_path)


def test_expiry_during_acquire_never_enters_body_and_releases_lock(tmp_path):
    ticks = iter([NOW, DEADLINE])
    with pytest.raises(SpawnFenceDeadlineExceeded):
        with exclusive_process_spawn_guard(tmp_path, acquire_deadline=DEADLINE, clock=lambda: next(ticks)):
            pytest.fail("deadline expired while acquiring")
    _assert_exclusive_available(tmp_path)


def test_busy_result_after_deadline_is_expired_not_transient(tmp_path):
    ticks = iter([NOW, DEADLINE])
    with _shared_holder(tmp_path):
        with pytest.raises(SpawnFenceDeadlineExceeded):
            with exclusive_process_spawn_guard(tmp_path, acquire_deadline=DEADLINE, clock=lambda: next(ticks)):
                pytest.fail("busy and expired")


def test_open_failure_is_deferred_without_creating_a_root(tmp_path):
    missing = tmp_path / "unmounted"
    with pytest.raises(SpawnDeferred, match="fence unavailable"):
        with _guard(missing):
            pytest.fail("missing mount")
    assert not missing.exists()


def test_expired_budget_does_not_even_create_lock_file(tmp_path):
    with pytest.raises(SpawnFenceDeadlineExceeded):
        with _guard(tmp_path, now=DEADLINE):
            pytest.fail("expired before acquisition")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("deadline,now", [(None, NOW), (DEADLINE.replace(tzinfo=None), NOW),
                                         (DEADLINE, NOW.replace(tzinfo=None)), (DEADLINE, None)])
def test_invalid_absolute_deadline_or_clock_fails_closed(tmp_path, deadline, now):
    with pytest.raises(ValueError, match="requires an aware"):
        with _guard(tmp_path, deadline=deadline, now=now):
            pytest.fail("invalid deadline/clock")
    assert not list(tmp_path.iterdir())


def test_expiry_does_not_revoke_ownership_during_critical_section(tmp_path):
    now = NOW
    with exclusive_process_spawn_guard(tmp_path, acquire_deadline=DEADLINE, clock=lambda: now):
        now = DEADLINE + timedelta(seconds=1)
        with pytest.raises(SpawnDeferred):
            with process_spawn_guard(tmp_path, lambda: False):
                pytest.fail("acquisition budget is not an authority lease")
    _assert_exclusive_available(tmp_path)
