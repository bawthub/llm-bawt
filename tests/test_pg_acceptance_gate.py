"""Gate contracts: an absent DB or zero executed private tests cannot turn green."""

from collections import defaultdict
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests import conftest, time_budget


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
    monkeypatch.setattr(conftest, "_RETIMED", {})
    config = _controller_config()
    config.getoption.side_effect = lambda key: {
        "--require-pg-acceptance": False,
        "--require-no-skips": False,
        "--unit-time-budget": unit_budget,
        "--real-process-time-budget": real_process_budget,
    }[key]
    config.pluginmanager.get_plugin.return_value.stats = {"passed": [object()]}
    config._task1189_durations = conftest._DURATIONS
    for nodeid, case in cases.items():
        keywords, phases, *retime = case
        for when, duration in phases:
            properties = [(time_budget.RETIME, retime[0])] if retime and when == phases[-1][0] else []
            conftest.pytest_runtest_logreport(SimpleNamespace(
                nodeid=nodeid, when=when, duration=duration, keywords=keywords, user_properties=properties))
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


def test_a_one_off_stall_is_judged_on_the_retry(monkeypatch, capsys):
    # TASK-1650: CI 40727877549, a cold node start took 8.06s against 7s.
    cases = {"tests/a.py::test_launcher": (REAL_PROCESS, [("call", 8.059)], 0.4)}
    assert _judge(monkeypatch, cases, 3.0, 7.0) is None
    out = capsys.readouterr().out
    assert "Re-timed after a slow first run: 8.059s then 0.400s tests/a.py::test_launcher" in out
    assert "Slowest real_process case: 0.400s" in out


@pytest.mark.parametrize("retry", [7.2, math.inf], ids=["slow-again", "retry-failed"])
def test_a_breach_that_repeats_still_fails(monkeypatch, capsys, retry):
    cases = {"tests/a.py::test_launcher": (REAL_PROCESS, [("call", 8.0)], retry)}
    assert _judge(monkeypatch, cases, 3.0, 7.0) == pytest.ExitCode.TESTS_FAILED
    assert "Real-process time budget exceeded (7.0s" in capsys.readouterr().out


def test_a_unit_case_gets_the_same_retime(monkeypatch):
    cases = {"tests/a.py::test_unit": ({}, [("setup", 0.2), ("call", 3.1), ("teardown", 0.1)], 0.3)}
    assert _judge(monkeypatch, cases, 3.0, 7.0) is None


def _report(when, duration, passed=True):
    return SimpleNamespace(when=when, duration=duration, passed=passed, user_properties=[])


def _run_protocol(monkeypatch, runs, *, real_process=False, unit_budget=3.0):
    """Drive the protocol hook with canned runs; returns (reports logged, runs used)."""
    queue = list(runs)
    used = []

    def fake_protocol(item, nextitem, log):
        assert log is False
        used.append(queue.pop(0))
        return used[-1]

    monkeypatch.setattr(conftest, "runtestprotocol", fake_protocol)
    logged = []
    item = Mock(nodeid="tests/a.py::test_x", location=("a.py", 1, "test_x"))
    item.config.getoption.side_effect = {"--unit-time-budget": unit_budget,
                                         "--real-process-time-budget": 7.0}.__getitem__
    item.get_closest_marker.return_value = object() if real_process else None
    item.ihook.pytest_runtest_logreport.side_effect = lambda report: logged.append(report)
    result = conftest.pytest_runtest_protocol(item, None)
    return result, logged, used


def test_protocol_reruns_only_a_passing_case_over_its_budget(monkeypatch):
    first = [_report("setup", 0.1), _report("call", 8.0), _report("teardown", 0.1)]
    retry = [_report("setup", 0.1), _report("call", 0.2), _report("teardown", 0.1)]
    result, logged, used = _run_protocol(monkeypatch, [first, retry], real_process=True)
    assert result is True and len(used) == 2
    assert logged == first  # the first run's outcome is what's reported
    assert first[-1].user_properties == [(time_budget.RETIME, pytest.approx(0.4))]


@pytest.mark.parametrize("first,real_process", [
    ([_report("setup", 0.1), _report("call", 0.5), _report("teardown", 0.1)], False),
    ([_report("setup", 0.1), _report("call", 5.0), _report("teardown", 0.1)], True),
    ([_report("setup", 0.1), _report("call", 9.0, passed=False), _report("teardown", 0.1)], True),
], ids=["fast", "real-process-under-7s", "failed"])
def test_protocol_runs_once_when_no_retime_applies(monkeypatch, first, real_process):
    result, logged, used = _run_protocol(monkeypatch, [first], real_process=real_process)
    assert result is True and len(used) == 1 and logged == first
    assert all(report.user_properties == [] for report in first)


def test_protocol_steps_aside_without_a_budget(monkeypatch):
    result, logged, used = _run_protocol(monkeypatch, [], unit_budget=None)
    assert result is None and used == [] and logged == []


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
