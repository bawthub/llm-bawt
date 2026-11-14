"""Gate contracts: an absent DB or zero executed private tests cannot turn green."""

from collections import defaultdict
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import ANY, Mock

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


def _judge(monkeypatch, cases, unit_budget, real_process_budget=None, alone=None, report_path=None):
    """Feed setup/call/teardown reports through the real hooks and finish the session.

    ``cases`` maps a nodeid to (keywords, [(when, duration), ...]). ``alone`` is what
    the isolated serial rerun measures per nodeid; by default every case stays slow.
    """
    isolated = Mock(side_effect=lambda nodeids, *_: {
        nodeid: (alone or {}).get(nodeid, math.inf) for nodeid in nodeids})
    monkeypatch.setattr(time_budget, "isolated_durations", isolated)
    monkeypatch.setattr(conftest.pg_connection_guard, "uninstall", Mock())
    monkeypatch.setattr(conftest, "_DURATIONS", type(conftest._DURATIONS)(float))
    monkeypatch.setattr(conftest, "_REAL_PROCESS", set())
    monkeypatch.setattr(conftest, "_RETIMED", {})
    monkeypatch.setattr(conftest, "_FAILED", set())
    config = _controller_config()
    config.getoption.side_effect = lambda key: {
        "--require-pg-acceptance": False,
        "--require-no-skips": False,
        "--unit-time-budget": unit_budget,
        "--real-process-time-budget": real_process_budget,
        time_budget.REPORT_OPTION: report_path,
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


def test_a_breach_under_load_is_judged_on_an_isolated_serial_rerun(monkeypatch, capsys):
    # TASK-2303: gate 20261114T171807Z-gavel-ccb4ea, both breached twice under load.
    unit = "tests/test_no_person_literals.py::test_src_has_no_person_or_deployment_infrastructure"
    process = "tests/test_infrastructure_error_text.py::test_sanitizer_imports_without_the_web_stack"
    cases = {unit: ({}, [("call", 1.328)], 1.169), process: (REAL_PROCESS, [("call", 1.623)], 1.558)}
    assert _judge(monkeypatch, cases, 1.0, 1.5, alone={unit: 0.4, process: 0.9}) is None
    out = capsys.readouterr().out
    assert f"Isolated serial rerun: 1.169s under load, 0.400s alone {unit}" in out
    assert f"Isolated serial rerun: 1.558s under load, 0.900s alone {process}" in out
    assert "time budget exceeded" not in out
    time_budget.isolated_durations.assert_called_once_with([process, unit], 1.0, 1.5, ANY)


@pytest.mark.parametrize("alone", [1.2, math.inf], ids=["slow-alone", "failed-alone"])
def test_a_case_slow_when_alone_still_fails(monkeypatch, capsys, alone):
    nodeid = "tests/a.py::test_slow"
    assert _judge(monkeypatch, {nodeid: ({}, [("call", 1.3)], 1.25)}, 1.0, alone={nodeid: alone}) \
        == pytest.ExitCode.TESTS_FAILED
    out = capsys.readouterr().out
    assert "Unit time budget exceeded (1.0s" in out and f"s {nodeid}" in out


def test_the_isolated_run_reports_instead_of_rerunning_again(monkeypatch, tmp_path):
    report = tmp_path / "durations.json"
    cases = {"tests/a.py::test_slow": ({}, [("call", 1.3)]), "tests/a.py::test_ok": ({}, [("call", 0.1)])}
    assert _judge(monkeypatch, cases, 1.0, report_path=str(report)) == pytest.ExitCode.TESTS_FAILED
    time_budget.isolated_durations.assert_not_called()
    assert json.loads(report.read_text()) == {"tests/a.py::test_slow": 1.3, "tests/a.py::test_ok": 0.1}


def test_isolated_durations_runs_only_those_cases_serially(tmp_path):
    def fake_run(cmd, **kwargs):
        path = cmd[cmd.index(time_budget.REPORT_OPTION) + 1]
        Path(path).write_text(json.dumps({"tests/a.py::test_fast": 0.2}))
        calls.append((cmd, kwargs))
    calls = []
    got = time_budget.isolated_durations(["tests/a.py::test_fast", "tests/a.py::test_failed"], 1.0, 1.5,
                                         str(tmp_path), run=fake_run)
    assert got == {"tests/a.py::test_fast": 0.2, "tests/a.py::test_failed": math.inf}
    cmd, kwargs = calls[0]
    assert cmd[-2:] == ["tests/a.py::test_fast", "tests/a.py::test_failed"]
    assert "no:xdist" in cmd and "--real-process-time-budget" in cmd and kwargs["cwd"] == str(tmp_path)


def test_isolated_durations_treats_a_crashed_run_as_still_slow(tmp_path):
    def crash(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 1)
    assert time_budget.isolated_durations(["t::x"], 1.0, None, str(tmp_path), run=crash) == {"t::x": math.inf}


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
