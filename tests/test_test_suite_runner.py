"""Hermetic checks for suite coverage, external gating, and lock safety."""

from contextlib import nullcontext
from multiprocessing import get_context
import os
import time

import pytest

from scripts import test_suite as suite


def test_offline_partition_requires_every_node_once(monkeypatch):
    nodes = {
        suite.OFFLINE: {"tests/a.py::unit", "tests/b.py::pg", "tests/c.py::local"},
        suite.UNIT: {"tests/a.py::unit"},
        suite.PG: {"tests/b.py::pg"},
        suite.LOCAL: {"tests/c.py::local"},
    }
    monkeypatch.setattr(suite, "collect", lambda marker, *selectors: nodes[marker])
    suite.verify_offline_partition()
    nodes[suite.PG] = set()
    with pytest.raises(RuntimeError, match="missing=.*tests/b.py::pg"):
        suite.verify_offline_partition()
    nodes[suite.PG] = {"tests/a.py::unit"}
    with pytest.raises(RuntimeError, match="duplicate=.*tests/a.py::unit"):
        suite.verify_offline_partition()


def test_external_suites_refuse_even_when_target_is_supplied():
    for name in ("service", "paid"):
        with pytest.raises(SystemExit, match="2"):
            suite.main([name])


def test_target_cannot_hide_deselected_external_cases(monkeypatch):
    monkeypatch.setattr(suite, "runner_lock", nullcontext)
    monkeypatch.setattr(suite, "collect", lambda marker, *nodes, **kwargs: (
        {"tests/example.py::safe", "tests/example.py::live"} if not marker
        else {"tests/example.py::safe"}
    ))
    monkeypatch.setattr(suite, "run", lambda *_: pytest.fail("unsafe test launched"))
    assert suite.main(["target", "tests/example.py"]) == 1


def _hold_lock(ready, release):
    with suite.runner_lock():
        ready.send(True)
        release.recv()


def test_global_runner_lock_rejects_concurrent_processes(monkeypatch, tmp_path):
    monkeypatch.setattr(suite.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.delenv(suite.LOCK_WAIT_ENV, raising=False)
    context = get_context("fork")  # The runner itself requires Unix flock.
    ready_parent, ready_child = context.Pipe()
    release_parent, release_child = context.Pipe()
    worker = context.Process(target=_hold_lock, args=(ready_child, release_child))
    worker.start()
    try:
        assert ready_parent.poll(3) and ready_parent.recv()
        with pytest.raises(RuntimeError, match="another llm-bawt test suite"):
            with suite.runner_lock():
                pass
    finally:
        release_parent.send(True)
        worker.join(timeout=3)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=3)
    assert worker.exitcode == 0
    with suite.runner_lock():
        pass


def _hold_lock_for(ready, seconds):
    with suite.runner_lock():
        ready.send(True)
        time.sleep(seconds)


def test_a_bounded_wait_acquires_after_release_and_times_out_while_held(monkeypatch, tmp_path):
    monkeypatch.setattr(suite.tempfile, "gettempdir", lambda: str(tmp_path))
    context = get_context("fork")
    ready_parent, ready_child = context.Pipe()
    worker = context.Process(target=_hold_lock_for, args=(ready_child, 0.3))
    worker.start()
    try:
        assert ready_parent.poll(3) and ready_parent.recv()
        with pytest.raises(RuntimeError, match=r"another llm-bawt test suite .*waited \d+\.\ds"):
            with suite.runner_lock(wait_seconds=0.1, poll_seconds=0.02):
                pass
        # The holder releases at ~0.3s; a 3s bounded wait must then acquire.
        with suite.runner_lock(wait_seconds=3, poll_seconds=0.02):
            pass
    finally:
        worker.join(timeout=3)
    assert worker.exitcode == 0


def test_the_runner_consumes_the_lock_wait_so_launched_tests_never_inherit_it(monkeypatch):
    seen = {}

    def lock(wait):
        seen["wait"] = wait
        return nullcontext()

    def run(*_args):
        seen["env"] = os.environ.get(suite.LOCK_WAIT_ENV)

    monkeypatch.setenv(suite.LOCK_WAIT_ENV, "90")
    monkeypatch.setattr(suite, "runner_lock", lock)
    monkeypatch.setattr(suite, "run", run)
    assert suite.main(["unit"]) == 0
    assert seen == {"wait": 90.0, "env": None}


@pytest.mark.parametrize(("value", "expected"), [("", 0.0), ("0", 0.0), ("90", 90.0)])
def test_lock_wait_comes_from_the_environment(monkeypatch, value, expected):
    monkeypatch.setenv(suite.LOCK_WAIT_ENV, value)
    assert suite.lock_wait_seconds() == expected


@pytest.mark.parametrize("value", ["soon", "-1", "100000"])
def test_a_bad_lock_wait_fails_closed(monkeypatch, value):
    monkeypatch.setenv(suite.LOCK_WAIT_ENV, value)
    with pytest.raises(RuntimeError, match="must be 0-86400"):
        suite.lock_wait_seconds()


@pytest.mark.parametrize(("value", "expected"), [("", 1.0), ("3", 3.0), ("2.5", 2.5)])
def test_unit_time_budget_comes_from_the_environment(monkeypatch, value, expected):
    monkeypatch.setenv(suite.UNIT_TIME_BUDGET_ENV, value)
    assert suite.unit_time_budget_seconds() == expected


@pytest.mark.parametrize("value", ["fast", "0", "-1", "61", "nan", "inf"])
def test_a_bad_unit_time_budget_fails_closed(monkeypatch, value):
    monkeypatch.setenv(suite.UNIT_TIME_BUDGET_ENV, value)
    with pytest.raises(RuntimeError, match="at most 60"):
        suite.unit_time_budget_seconds()


@pytest.mark.parametrize(("configured", "flag"), [(None, "--unit-time-budget=1.0"), ("3", "--unit-time-budget=3.0")])
def test_the_unit_gate_passes_the_configured_budget_to_pytest(monkeypatch, configured, flag):
    seen = []
    if configured is None:
        monkeypatch.delenv(suite.UNIT_TIME_BUDGET_ENV, raising=False)
    else:
        monkeypatch.setenv(suite.UNIT_TIME_BUDGET_ENV, configured)
    monkeypatch.setattr(suite, "runner_lock", lambda wait: nullcontext())
    monkeypatch.setattr(suite, "run", lambda marker, *extra: seen.append((marker, extra)))
    assert suite.main(["unit"]) == 0
    assert seen == [(suite.UNIT, (flag, "--require-no-skips", *suite.parallel("load")))]
