"""Hermetic checks for suite coverage, external gating, and lock safety."""

from contextlib import nullcontext
from multiprocessing import get_context

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
