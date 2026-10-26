"""Gate contracts: an absent DB or zero executed private tests cannot turn green."""

from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests import conftest


def _controller_config() -> Mock:
    """A non-worker config: pytest-xdist marks workers with ``config.workerinput``."""
    config = Mock()
    del config.workerinput
    return config


@pytest.mark.parametrize("passed,skipped,expected", [
    ([], [], pytest.ExitCode.TESTS_FAILED),
    ([object()], [object()], pytest.ExitCode.TESTS_FAILED),
    ([object()], [], None),
])
def test_acceptance_exit_status_requires_passes_without_skips(monkeypatch, passed, skipped, expected):
    monkeypatch.setattr(conftest.pg_connection_guard, "uninstall", Mock())
    config = _controller_config()
    config.getoption.return_value = True
    config.pluginmanager.get_plugin.return_value.stats = {"passed": passed, "skipped": skipped}
    session = SimpleNamespace(config=config, exitstatus=None)
    conftest.pytest_sessionfinish(session, 0)
    assert session.exitstatus == expected


def test_xdist_worker_leaves_whole_run_gates_to_the_controller(monkeypatch):
    # A worker sees only its slice (and no terminal stats); judging there would fail
    # every parallel run. The controller receives all reports and judges once.
    uninstall = Mock()
    monkeypatch.setattr(conftest.pg_connection_guard, "uninstall", uninstall)
    config = Mock(workerinput={"workerid": "gw0"})
    config.getoption.return_value = True
    config.pluginmanager.get_plugin.return_value.stats = {}
    session = SimpleNamespace(config=config, exitstatus=None)
    conftest.pytest_sessionfinish(session, 0)
    assert session.exitstatus is None
    uninstall.assert_called_once()


def _judge(monkeypatch, cases, unit_budget, real_process_budget=None):
    """Feed setup/call/teardown reports through the real hooks and finish the session.

    ``cases`` maps a nodeid to (keywords, [(when, duration), ...]).
    """
    monkeypatch.setattr(conftest.pg_connection_guard, "uninstall", Mock())
    monkeypatch.setattr(conftest, "_DURATIONS", type(conftest._DURATIONS)(float))
    monkeypatch.setattr(conftest, "_REAL_PROCESS", set())
    config = _controller_config()
    config.getoption.side_effect = lambda key: {
        "--require-pg-acceptance": False,
        "--require-no-skips": False,
        "--unit-time-budget": unit_budget,
        "--real-process-time-budget": real_process_budget,
    }[key]
    config.pluginmanager.get_plugin.return_value.stats = {"passed": [object()]}
    config._task1189_durations = conftest._DURATIONS
    for nodeid, (keywords, phases) in cases.items():
        for when, duration in phases:
            conftest.pytest_runtest_logreport(
                SimpleNamespace(nodeid=nodeid, when=when, duration=duration, keywords=keywords))
    session = SimpleNamespace(config=config, exitstatus=None)
    conftest.pytest_sessionfinish(session, 0)
    return session.exitstatus


def test_unit_budget_includes_setup_call_and_teardown(monkeypatch):
    phases = [("setup", 0.4), ("call", 0.35), ("teardown", 0.26)]
    assert _judge(monkeypatch, {"tests/virtual.py::test_summed": ({}, phases)}, 1.0) \
        == pytest.ExitCode.TESTS_FAILED


REAL_PROCESS = {"test_launcher": 1, "real_process": 1}


def test_a_real_process_case_is_judged_on_its_own_budget(monkeypatch, capsys):
    phases = [("setup", 0.1), ("call", 3.2), ("teardown", 0.1)]  # 3.4s: over 3, under 7
    assert _judge(monkeypatch, {"tests/a.py::test_launcher": (REAL_PROCESS, phases)}, 3.0, 7.0) is None
    assert "Slowest real_process case: 3.400s tests/a.py::test_launcher (budget 7.0s)" in capsys.readouterr().out


def test_an_unmarked_case_at_the_same_duration_fails_the_unit_budget(monkeypatch, capsys):
    phases = [("setup", 0.1), ("call", 3.2), ("teardown", 0.1)]
    assert _judge(monkeypatch, {"tests/a.py::test_launcher": ({"test_launcher": 1}, phases)}, 3.0, 7.0) \
        == pytest.ExitCode.TESTS_FAILED
    out = capsys.readouterr().out
    assert "Unit time budget exceeded (3.0s" in out and "3.400s tests/a.py::test_launcher" in out
    assert "Real-process" not in out


def test_a_real_process_case_over_its_budget_fails_and_names_that_budget(monkeypatch, capsys):
    cases = {"tests/a.py::test_launcher": (REAL_PROCESS, [("call", 7.5)]),
             "tests/a.py::test_unit": ({}, [("call", 0.1)])}
    assert _judge(monkeypatch, cases, 3.0, 7.0) == pytest.ExitCode.TESTS_FAILED
    out = capsys.readouterr().out
    assert "Real-process time budget exceeded (7.0s" in out and "7.500s tests/a.py::test_launcher" in out
    assert "Unit time budget exceeded" not in out


def test_without_a_real_process_budget_marked_cases_keep_the_unit_budget(monkeypatch):
    assert _judge(monkeypatch, {"tests/a.py::test_launcher": (REAL_PROCESS, [("call", 3.4)])}, 3.0) \
        == pytest.ExitCode.TESTS_FAILED


def test_acceptance_gate_rejects_deselected_private_suite():
    config = Mock()
    config.getoption.return_value = True
    config._task780_surviving_by_file = defaultdict(int)
    session = SimpleNamespace(config=config, items=[
        SimpleNamespace(nodeid="tests/test_session_search_pg.py::test_search", path="search"),
        SimpleNamespace(nodeid="tests/test_session_continuity_pg.py::test_continuity", path="continuity"),
    ])
    with pytest.raises(BaseException, match="missing selected suites"):
        conftest.pytest_collection_finish(session)
