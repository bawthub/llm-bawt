"""Hermetic checks for suite coverage, external gating, and lock safety."""

from contextlib import nullcontext
from multiprocessing import get_context
import os
from pathlib import Path
import time
from types import SimpleNamespace

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


# -- stress step (TASK-1457) -------------------------------------------------


def test_stress_reads_ci_conditions_from_the_real_workflow():
    env = suite.ci_env()
    assert suite.worker_count(env[suite.WORKERS_ENV]) == 2
    assert suite.unit_time_budget_seconds(env[suite.UNIT_TIME_BUDGET_ENV]) == 3.0


@pytest.mark.parametrize(("steps", "error"), [
    ("- run: make test-all\n  env: {LLM_BAWT_TEST_WORKERS: '2'}\n", "must set"),
    ("- run: make test-all\n- run: make test-all\n", "exactly one step, found 2"),
    ("- run: make lint\n", "exactly one step, found 0"),
])
def test_a_workflow_without_one_complete_gate_step_fails_closed(tmp_path, steps, error):
    workflow = tmp_path / "tests.yml"
    workflow.write_text("jobs:\n  t:\n    steps:\n" + "".join(f"      {line}\n" for line in steps.splitlines()))
    with pytest.raises(RuntimeError, match=error):
        suite.ci_env(workflow)


def test_pinning_narrows_this_process_and_postgres_then_restores_both(monkeypatch):
    monkeypatch.setenv(suite.PG_CPUSET_ENV, "previous")
    before = os.sched_getaffinity(0)
    with suite.pinned_cpus(2) as cpuset:
        pinned = os.sched_getaffinity(0)
        assert len(pinned) == 2 and pinned <= before
        assert os.environ[suite.PG_CPUSET_ENV] == cpuset == ",".join(map(str, sorted(pinned)))
    assert os.sched_getaffinity(0) == before
    assert os.environ[suite.PG_CPUSET_ENV] == "previous"


def test_pinning_refuses_a_machine_with_fewer_cpus(monkeypatch):
    monkeypatch.setattr(suite.os, "sched_getaffinity", lambda pid: {0})
    with pytest.raises(RuntimeError, match="needs 2 CPUs, this process has 1"), suite.pinned_cpus(2):
        pass


def test_changed_tests_are_added_or_modified_test_modules_since_the_fork(monkeypatch):
    calls = []

    def git(*args):
        calls.append(args)
        return "abc123" if args[0] == "merge-base" else (
            "tests/test_b.py\ntests/helpers.py\ntests/test_a.py\ntests/data/test_x.json\n")

    monkeypatch.setattr(suite, "git", git)
    assert suite.changed_test_files("origin/master") == ("abc123", ["tests/test_a.py", "tests/test_b.py"])
    assert calls == [("merge-base", "origin/master", "HEAD"),
                     ("diff", "--name-only", "--no-renames", "--diff-filter=AM", "abc123", "--", "tests")]


def _collect_from(nodes):
    def collect(marker, *selectors, allow_empty=False):
        wanted = {selector.split("::", 1)[0] for selector in selectors}
        return {node for node in nodes[marker] if node.split("::", 1)[0] in wanted}
    return collect


def test_a_stress_plan_runs_each_changed_test_in_its_gate_partition(monkeypatch):
    nodes = {
        suite.OFFLINE: {"tests/test_u.py::a", "tests/test_p.py::b", "tests/test_l.py::c"},
        suite.UNIT: {"tests/test_u.py::a"},
        suite.PG: {"tests/test_p.py::b"},
        suite.LOCAL: {"tests/test_l.py::c"},
    }
    monkeypatch.setattr(suite, "PG_SELECTORS", ("tests/test_p.py", "tests/test_other_pg.py"))
    monkeypatch.setattr(suite, "collect", _collect_from(nodes))
    plan = suite.stress_plan(["tests/test_l.py", "tests/test_p.py", "tests/test_u.py"], 3.0)
    assert [(part.name, part.selectors, part.gate) for part in plan] == [
        ("unit", ("tests/test_u.py",), ("--unit-time-budget=3.0", "--require-no-skips")),
        ("private-pg", ("tests/test_p.py",), ("--require-no-skips",)),
        ("local-integration", ("tests/test_l.py",), ("--require-no-skips",)),
    ]
    assert [part.name for part in suite.stress_plan(["tests/test_u.py"], 3.0)] == ["unit"]


def test_a_changed_integration_test_outside_every_gate_suite_fails_the_plan(monkeypatch):
    nodes = {suite.OFFLINE: {"tests/test_p.py::b"}, suite.UNIT: set(), suite.PG: set(), suite.LOCAL: set()}
    monkeypatch.setattr(suite, "PG_SELECTORS", ())
    monkeypatch.setattr(suite, "collect", _collect_from(nodes))
    with pytest.raises(RuntimeError, match=r"register them in tests/suite_manifest.py.*test_p.py::b"):
        suite.stress_plan(["tests/test_p.py"], 3.0)


def _stress_harness(monkeypatch, fail_at=None):
    ran = []
    plan = [suite.StressPart("unit", suite.UNIT, ("tests/test_u.py",), ("--g",), "load"),
            suite.StressPart("private-pg", suite.PG, ("tests/test_p.py",), (), "loadscope")]

    def run(marker, *extra):
        ran.append((marker, extra))
        if len(ran) == fail_at:
            raise RuntimeError("test suite failed (exit 1)")

    monkeypatch.setattr(suite, "ci_env", lambda: {suite.WORKERS_ENV: "2", suite.UNIT_TIME_BUDGET_ENV: "3"})
    monkeypatch.setattr(suite, "pinned_cpus", lambda: nullcontext("14,15"))
    monkeypatch.setattr(suite, "stress_plan", lambda files, budget: plan)
    monkeypatch.setattr(suite, "run", run)
    return ran


def test_stress_repeats_every_part_with_ci_workers(monkeypatch, capsys):
    ran = _stress_harness(monkeypatch)
    suite.stress("origin/master", 2, ["tests/test_u.py", "tests/test_p.py"])
    assert ran == [(suite.UNIT, ("--g", *suite.parallel("load", 2), "tests/test_u.py")),
                   (suite.PG, (*suite.parallel("loadscope", 2), "tests/test_p.py"))] * 2
    out = capsys.readouterr().out
    assert "2 round(s) on CPUs 14,15 with CI's 2 worker(s) and 3.0s unit budget" in out
    assert "Stress: PASSED (2 round(s))" in out


def test_stress_stops_at_the_first_failing_round_and_names_it(monkeypatch):
    ran = _stress_harness(monkeypatch, fail_at=4)
    with pytest.raises(RuntimeError, match=r"stress round 2/3 failed in private-pg: test suite failed"):
        suite.stress("origin/master", 3, ["tests/test_u.py", "tests/test_p.py"])
    assert len(ran) == 4


def test_stress_with_no_changed_tests_passes_without_pinning(monkeypatch, capsys):
    monkeypatch.setattr(suite, "ci_env", lambda: {suite.WORKERS_ENV: "2", suite.UNIT_TIME_BUDGET_ENV: "3"})
    monkeypatch.setattr(suite, "changed_test_files", lambda base: ("abc123def456", []))
    monkeypatch.setattr(suite, "pinned_cpus", lambda: pytest.fail("pinned with nothing to run"))
    suite.stress("origin/master", 10)
    assert "Stress: PASSED (no changed test files to stress)" in capsys.readouterr().out


@pytest.mark.parametrize("repeat", [0, 101])
def test_stress_repeat_is_bounded(repeat):
    with pytest.raises(RuntimeError, match="repeat must be 1-100"):
        suite.stress("origin/master", repeat)


@pytest.mark.parametrize("argv", [
    ["stress", "scripts/test_suite.py"],
    ["stress", "tests/../tests/test_a.py"],
    ["stress", "--all", "tests/test_a.py"],
    ["unit", "--all"],
    ["unit", "tests/test_a.py"],
])
def test_bad_stress_arguments_are_refused(argv):
    with pytest.raises(SystemExit, match="2"):
        suite.main(argv)


def test_the_offline_suite_never_runs_the_stress_step(monkeypatch):
    # Stress is an opt-in diagnostic (make test-stress); the SHIP gate is the offline suite.
    seen = []
    monkeypatch.setattr(suite, "runner_lock", lambda wait: nullcontext())
    monkeypatch.setattr(suite, "verify_offline_partition", lambda: seen.append("partition"))
    monkeypatch.setattr(suite, "run", lambda marker, *extra: seen.append(marker))
    monkeypatch.setattr(suite, "stress", lambda *args, **kwargs: seen.append("stress"))
    assert suite.main(["offline"]) == 0
    assert seen == ["partition", suite.UNIT, suite.PG, suite.LOCAL]
    with pytest.raises(SystemExit, match="2"):
        suite.main(["gate"])


def test_the_timing_report_classifies_by_step_cost_and_honours_markers():
    base = {
        "tests/test_a.py::t": {"failed": {"when": "call", "message": "TimeoutError"}},
        "tests/test_b.py::t": {"events": {"real_time": ["time.sleep(0.1)"], "real_process": ["git"]},
                               "allowed": {"real_process": "needs git"},
                               "follow": {"executor": [2, 0.5]}},
        "tests/test_c.py::t": {"follow": {"fd": [1, 1.25]}},
    }
    late = {"failed": {"when": "call", "message": "late"}}
    stepped = {**base, "tests/test_d.py::t": late}
    # test_e fails in one base round only: D1, intermittently, never D2.
    rounds = [base, {**base, "tests/test_e.py::t": {"failed": {"when": "call", "message": "raced"}}}]
    report = suite.summarize_timing(rounds, [stepped, {**stepped, "tests/test_e.py::t": late}],
                                    {"D4": ["tests/x.py:1: w"], "D7": []})
    assert report == {
        "D1": ["tests/test_a.py::t [2/2]: TimeoutError", "tests/test_e.py::t [1/2]: raced"],
        "D2": ["tests/test_d.py::t [2/2]: late"],
        "D3": ["tests/test_b.py::t: real_time time.sleep(0.1)"],
        "D4": ["tests/x.py:1: w"],
        "D7": [],
        "FOLLOW": ["tests/test_c.py::t: fd x1, 1.25s real", "tests/test_b.py::t: executor x2, 0.50s real"],
    }


def test_a_timing_run_keeps_test_failures_but_not_a_broken_pytest(monkeypatch, tmp_path):
    from tests import timing_report

    assert (suite.TIMING_REPORT_ENV, suite.TIMING_STEP_COST_ENV) == (
        timing_report.REPORT_ENV, timing_report.STEP_COST_ENV)
    seen = []

    def fake_run(command, cwd, env):
        seen.append((command, env[suite.TIMING_STEP_COST_ENV]))
        Path(env[suite.TIMING_REPORT_ENV]).write_text("{}")
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(suite.subprocess, "run", fake_run)
    returncode = 1
    suite.timing_run(0.02, tmp_path / "r.json")
    command, step_cost = seen[0]
    assert step_cost == "0.02" and command[command.index("-p") + 1] == "tests.timing_report"
    returncode = 3
    with pytest.raises(RuntimeError, match="exit 3"):
        suite.timing_run(0.0, tmp_path / "s.json")


def test_the_timing_report_is_its_own_suite_and_never_part_of_a_gate(monkeypatch):
    seen = []
    monkeypatch.setattr(suite, "runner_lock", lambda wait: nullcontext())
    monkeypatch.setattr(suite, "verify_offline_partition", lambda: None)
    monkeypatch.setattr(suite, "run", lambda marker, *extra: seen.append(extra))
    monkeypatch.setattr(suite, "timing_report", lambda: seen.append("timing"))
    assert suite.main(["offline"]) == 0
    assert "timing" not in seen and not any("tests.timing_report" in extra for extra in seen)
    assert suite.main(["timing-report"]) == 0
    assert seen[-1] == "timing"
