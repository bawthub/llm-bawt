"""Canonical, serialized and fail-closed llm-bawt test entrypoints.

Run from repository root: python -m scripts.test_suite offline
Make and CI invoke this module rather than maintaining separate pytest recipes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import time

from tests.suite_manifest import PG_SELECTORS

ROOT = Path(__file__).resolve().parents[1]
UNIT = "not integration and not service and not llm_call and not external"
PG = "integration and not service and not llm_call and not external and not offline_integration"
LOCAL = "integration and offline_integration and not service and not llm_call and not external"
SDK = "integration and sdk_probe and not service and not llm_call and not external"
OFFLINE = "not service and not llm_call and not external and not sdk_probe"


def pytest_args(marker: str, *extra: str) -> list[str]:
    return [sys.executable, "-m", "pytest", "-o", "addopts=", "-m", marker,
            "--strict-markers", *extra]


WORKERS_ENV = "LLM_BAWT_TEST_WORKERS"


def worker_count(configured: str | None = None) -> int:
    """Parallel pytest-xdist workers for heavy suites; LLM_BAWT_TEST_WORKERS=1 runs serially.

    ``configured`` overrides the environment (the stress step passes CI's value).
    """
    if configured is None:
        configured = os.environ.get(WORKERS_ENV)
    if configured:
        if not configured.isdigit() or int(configured) < 1:
            raise RuntimeError(f"{WORKERS_ENV} must be a positive integer, got {configured!r}")
        return int(configured)
    # Leave headroom: the unit gate's 1s budget measures wall time per case.
    return max(1, min(8, (os.cpu_count() or 2) // 2))


def parallel(dist: str, workers: int | None = None) -> tuple[str, ...]:
    """xdist arguments. ``loadscope`` keeps a module on one worker so each private-PG
    module database is created once; ``load`` balances independent unit cases."""
    workers = worker_count() if workers is None else workers
    return () if workers == 1 else ("-n", str(workers), "--dist", dist)


def collect(marker: str, *selectors: str, allow_empty: bool = False) -> set[str]:
    command = pytest_args(marker, "--collect-only", "-q", *(selectors or ("tests",)))
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode and not (allow_empty and result.returncode == 5):
        raise RuntimeError(f"collection failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    nodes = {line for line in result.stdout.splitlines()
             if line.startswith("tests/") and "::" in line}
    if not nodes and not allow_empty:
        raise RuntimeError(f"collection selected no tests: {marker} {selectors}")
    return nodes


def verify_offline_partition() -> None:
    # The four collections are independent read-only pytest processes; run them concurrently.
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(collect, OFFLINE), pool.submit(collect, UNIT),
                   pool.submit(collect, PG, *PG_SELECTORS), pool.submit(collect, LOCAL)]
        all_nodes, unit, pg, local = (future.result() for future in futures)
    selections = (unit, pg, local)
    duplicate = (unit & pg) | (unit & local) | (pg & local)
    missing = all_nodes - set.union(*selections)
    extra = set.union(*selections) - all_nodes
    if duplicate or missing or extra:
        raise RuntimeError(f"offline suite coverage mismatch: duplicate={sorted(duplicate)}, "
                           f"missing={sorted(missing)}, extra={sorted(extra)}")
    print(f"Offline coverage: {len(all_nodes)} unique tests: "
          f"unit={len(unit)}, private-pg={len(pg)}, local-integration={len(local)}", flush=True)


def run(marker: str, *extra: str) -> None:
    command = pytest_args(marker, "-q", *extra)
    print("Running:", " ".join(command[1:]), flush=True)
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode:
        raise RuntimeError(f"test suite failed (exit {result.returncode})")


LOCK_WAIT_ENV = "LLM_BAWT_TEST_LOCK_WAIT_SECONDS"


def lock_wait_seconds() -> float:
    """Bounded wait for a busy runner; 0 (the default) fails immediately.

    ``make test-snapshot`` sets it so a queued snapshot gate waits its turn.
    Waiting never bypasses the lock: the suite still runs only while it holds it.
    """
    configured = os.environ.get(LOCK_WAIT_ENV, "").strip()
    if not configured:
        return 0.0
    try:
        seconds = float(configured)
    except ValueError:
        seconds = -1.0
    if not 0 <= seconds <= 24 * 3600:
        raise RuntimeError(f"{LOCK_WAIT_ENV} must be 0-86400 seconds, got {configured!r}")
    return seconds


UNIT_TIME_BUDGET_ENV = "LLM_BAWT_UNIT_TIME_BUDGET_SECONDS"
UNIT_TIME_BUDGET_DEFAULT_S = 1.0


def unit_time_budget_seconds(configured: str | None = None) -> float:
    """Wall-clock bound per unit case (setup + call + teardown).

    The 1s default is the standard every commit meets on echo through
    ``make test-snapshot`` before it is pushed. Hosted CI reruns the same suite
    on a shared virtual machine with fewer, slower cores, cold caches and a
    virtual disk, where a case that takes 0.8s here by fixed waits exceeds 1s
    with no regression. The workflow widens the bound there so CI still fails
    on hangs and new sleeps without failing on runner noise (TASK-1312).
    ``configured`` overrides the environment (the stress step passes CI's value).
    """
    if configured is None:
        configured = os.environ.get(UNIT_TIME_BUDGET_ENV, "")
    configured = configured.strip()
    if not configured:
        return UNIT_TIME_BUDGET_DEFAULT_S
    try:
        seconds = float(configured)
    except ValueError:
        seconds = 0.0
    if not 0 < seconds <= 60:
        raise RuntimeError(f"{UNIT_TIME_BUDGET_ENV} must be >0 and at most 60 seconds, got {configured!r}")
    return seconds


def unit_gate() -> tuple[str, ...]:
    """Arguments that make the unit suite a gate: per-case budget, zero skips, parallel."""
    return (f"--unit-time-budget={unit_time_budget_seconds()}", "--require-no-skips", *parallel("load"))


# ── Stress: changed tests under CI's conditions (TASK-1457) ─────────────────
#
# Echo is fast enough that a timing-dependent test passes here and fails on
# GitHub's runner. An optional diagnostic, not part of the SHIP gate: repeated
# rounds only find timing flakes by chance. The stress step reruns tests under CI's conditions: CI's
# worker count and unit budget, read from the workflow itself so the two can't
# drift, with this process, its pytest children and the private PostgreSQL
# container all pinned to STRESS_CPUS CPUs. CI derives its 2 workers from 4
# vCPUs; echo's cores are much faster, so 2 pinned CPUs is the closer match.

WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"
CI_GATE_STEP = "make test-all"
CI_ENV_KEYS = (WORKERS_ENV, UNIT_TIME_BUDGET_ENV)
PG_CPUSET_ENV = "LLM_BAWT_TEST_PG_CPUSET"
STRESS_CPUS = 2
STRESS_REPEAT_DEFAULT = 10
STRESS_REPEAT_MAX = 100
STRESS_BASE_DEFAULT = "origin/master"


def ci_env(workflow: Path = WORKFLOW) -> dict[str, str]:
    """The environment CI runs ``make test-all`` with, read from the workflow."""
    import yaml

    data = yaml.safe_load(workflow.read_text()) or {}
    steps = [step for job in (data.get("jobs") or {}).values() for step in job.get("steps") or []
             if CI_GATE_STEP in str(step.get("run", ""))]
    if len(steps) != 1:
        raise RuntimeError(f"{workflow.name} must run `{CI_GATE_STEP}` in exactly one step, found {len(steps)}")
    env = {str(key): str(value) for key, value in (steps[0].get("env") or {}).items()}
    missing = [key for key in CI_ENV_KEYS if key not in env]
    if missing:
        raise RuntimeError(f"{workflow.name}'s `{CI_GATE_STEP}` step must set {missing}")
    return env


@contextmanager
def pinned_cpus(count: int = STRESS_CPUS):
    """Pin this process (and every child it starts) to ``count`` CPUs.

    Docker doesn't inherit affinity, so the cpuset is also exported for the
    private PostgreSQL fixture, which passes it to ``docker run --cpuset-cpus``.
    """
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("the stress step needs Linux CPU affinity")
    previous = os.sched_getaffinity(0)
    if len(previous) < count:
        raise RuntimeError(f"the stress step needs {count} CPUs, this process has {len(previous)}")
    chosen = set(sorted(previous)[-count:])
    previous_env = os.environ.get(PG_CPUSET_ENV)
    cpuset = ",".join(str(cpu) for cpu in sorted(chosen))
    os.sched_setaffinity(0, chosen)
    os.environ[PG_CPUSET_ENV] = cpuset
    try:
        yield cpuset
    finally:
        os.sched_setaffinity(0, previous)
        if previous_env is None:
            os.environ.pop(PG_CPUSET_ENV, None)
        else:
            os.environ[PG_CPUSET_ENV] = previous_env


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def is_test_file(path: str) -> bool:
    name = PurePosixPath(path).name
    return path.startswith("tests/") and name.startswith("test_") and name.endswith(".py")


def changed_test_files(base: str) -> tuple[str, list[str]]:
    """Test files added or modified since ``base`` forked: the commits in
    ``base..HEAD`` plus anything staged or changed in this tree (a snapshot's
    applied patches). Deleted files can't run; untracked files aren't counted."""
    merge_base = git("merge-base", base, "HEAD")
    names = git("diff", "--name-only", "--no-renames", "--diff-filter=AM", merge_base, "--", "tests")
    return merge_base, sorted(name for name in names.splitlines() if is_test_file(name))


@dataclass(frozen=True)
class StressPart:
    name: str
    marker: str
    selectors: tuple[str, ...]
    gate: tuple[str, ...]
    dist: str


def _files_of(nodes: set[str]) -> tuple[str, ...]:
    return tuple(sorted({node.split("::", 1)[0] for node in nodes}))


def stress_plan(files: list[str], budget: float) -> list[StressPart]:
    """The offline gate's partitions, narrowed to ``files``. Each test runs in
    the partition (and with the gate flags) it gets in ``make test-all``."""
    wanted = set(files)
    pg_selectors = tuple(s for s in PG_SELECTORS if s.split("::", 1)[0] in wanted)
    with ThreadPoolExecutor(max_workers=4) as pool:
        every = pool.submit(collect, OFFLINE, *files, allow_empty=True)
        unit = pool.submit(collect, UNIT, *files, allow_empty=True)
        pg = pool.submit(collect, PG, *pg_selectors, allow_empty=True) if pg_selectors else None
        local = pool.submit(collect, LOCAL, *files, allow_empty=True)
        every_nodes, unit_nodes, local_nodes = every.result(), unit.result(), local.result()
        pg_nodes = pg.result() if pg else set()
    missing = every_nodes - unit_nodes - pg_nodes - local_nodes
    if missing:
        raise RuntimeError("changed integration tests aren't in a gate suite (register them in "
                           f"tests/suite_manifest.py): {sorted(missing)}")
    parts = [
        StressPart("unit", UNIT, _files_of(unit_nodes),
                   (f"--unit-time-budget={budget}", "--require-no-skips"), "load"),
        StressPart("private-pg", PG, pg_selectors if pg_nodes else (), ("--require-no-skips",), "loadscope"),
        StressPart("local-integration", LOCAL, _files_of(local_nodes), ("--require-no-skips",), "load"),
    ]
    return [part for part in parts if part.selectors]


def full_plan(budget: float) -> list[StressPart]:
    """Every offline test, partitioned exactly as ``make test-all`` runs it."""
    verify_offline_partition()
    return [
        StressPart("unit", UNIT, ("tests",), (f"--unit-time-budget={budget}", "--require-no-skips"), "load"),
        StressPart("private-pg", PG, PG_SELECTORS, ("--require-pg-acceptance",), "loadscope"),
        StressPart("local-integration", LOCAL, ("tests",), ("--require-no-skips",), "load"),
    ]


def stress(base: str, repeat: int, files: list[str] | None = None, everything: bool = False) -> None:
    if not 1 <= repeat <= STRESS_REPEAT_MAX:
        raise RuntimeError(f"stress repeat must be 1-{STRESS_REPEAT_MAX}, got {repeat}")
    ci = ci_env()
    workers = worker_count(ci[WORKERS_ENV])
    budget = unit_time_budget_seconds(ci[UNIT_TIME_BUDGET_ENV])
    if everything:
        scope = "the whole offline suite"
    elif files:
        scope = f"{len(files)} named test file(s)"
    else:
        merge_base, files = changed_test_files(base)
        scope = f"{len(files)} changed test file(s) in {merge_base[:12]}..HEAD"
    print(f"Stress: {scope}{': ' + ' '.join(files) if files and not everything else ''}", flush=True)
    if not everything and not files:
        print("Stress: PASSED (no changed test files to stress)", flush=True)
        return
    with pinned_cpus() as cpuset:
        plan = full_plan(budget) if everything else stress_plan(files, budget)
        print(f"Stress: {repeat} round(s) on CPUs {cpuset} with CI's {workers} worker(s) and "
              f"{budget}s unit budget; parts: {', '.join(part.name for part in plan)}", flush=True)
        for number in range(1, repeat + 1):
            for part in plan:
                try:
                    run(part.marker, *part.gate, *parallel(part.dist, workers), *part.selectors)
                except RuntimeError as exc:
                    raise RuntimeError(f"stress round {number}/{repeat} failed in {part.name}: {exc}") from exc
            print(f"Stress: round {number}/{repeat} passed", flush=True)
    print(f"Stress: PASSED ({repeat} round(s))", flush=True)


# Report-only timing check (TASK-1457 phase a). Never part of a gate.
TIMING_REPORT_ENV = "LLM_BAWT_TIMING_REPORT"
TIMING_STEP_COST_ENV = "LLM_BAWT_TIMING_STEP_COST"
TIMING_STEP_COST = 0.02
TIMING_REPORT_OUT = Path(tempfile.gettempdir()) / "llm-bawt-timing-report.json"
# The guard and its samples name the patterns on purpose.
TIMING_STATIC_SKIP = frozenset({"tests/timing_guard.py", "tests/test_timing_guard.py"})


def timing_run(step_cost: float, path: Path) -> float:
    """One unit run with every loop on the virtual clock; test failures are the
    data, so only a pytest that couldn't run (exit 2+) raises."""
    command = pytest_args(UNIT, "-q", "-p", "tests.timing_report", *parallel("load"))
    env = {**os.environ, TIMING_REPORT_ENV: str(path), TIMING_STEP_COST_ENV: str(step_cost)}
    print(f"Timing report: step cost {step_cost:g}s:", " ".join(command[1:]), flush=True)
    started = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, env=env)
    if result.returncode not in (0, 1) or not path.exists():
        raise RuntimeError(f"timing report run failed (exit {result.returncode})")
    return time.monotonic() - started


TIMING_ROUNDS = 2


def summarize_timing(base: list[dict], stepped: list[dict], static: dict[str, list]) -> dict[str, list]:
    """``base``/``stepped``: per-test records of each round at step cost 0 and
    TIMING_STEP_COST. FOLLOW races real work, so a test can fail in one round
    and pass in the next; ``[k/n]`` says how many rounds it failed in."""
    def failures(rounds: list[dict]) -> dict[str, list[str]]:
        seen: dict[str, list[str]] = {}
        for records in rounds:
            for node, entry in records.items():
                if "failed" in entry:
                    seen.setdefault(node, []).append(entry["failed"]["message"])
        return seen

    def listed(seen: dict[str, list[str]], rounds: int) -> list[str]:
        return [f"{node} [{len(messages)}/{rounds}]: {messages[0]}" for node, messages in sorted(seen.items())]

    d1 = failures(base)
    d2 = {node: messages for node, messages in failures(stepped).items() if node not in d1}
    first = base[0]
    d3 = [f"{node}: {kind} {', '.join(sorted(set(details)))}"
          for node, entry in sorted(first.items())
          for kind, details in sorted(entry.get("events", {}).items())
          if kind not in entry.get("allowed", {})]
    follow = sorted(((spent, node, cause, count) for node, entry in first.items()
                     for cause, (count, spent) in entry.get("follow", {}).items()), reverse=True)
    return {
        "D1": listed(d1, len(base)),
        "D2": listed(d2, len(stepped)),
        "D3": d3,
        "D4": [str(finding) for finding in static["D4"]],
        "D7": [str(finding) for finding in static["D7"]],
        "FOLLOW": [f"{node}: {cause} x{count}, {spent:.2f}s real" for spent, node, cause, count in follow],
    }


def timing_report(out: Path = TIMING_REPORT_OUT) -> dict[str, list]:
    from tests.timing_guard import report_findings

    with tempfile.TemporaryDirectory(prefix="llm-bawt-timing-") as scratch:
        seconds: list[float] = []
        records: dict[float, list[dict]] = {0.0: [], TIMING_STEP_COST: []}
        for number in range(TIMING_ROUNDS):
            for step_cost in records:
                path = Path(scratch) / f"round-{number}-step-{step_cost:g}.json"
                seconds.append(timing_run(step_cost, path))
                records[step_cost].append(json.loads(path.read_text()))
    report = summarize_timing(records[0.0], records[TIMING_STEP_COST],
                              report_findings(ROOT, TIMING_STATIC_SKIP))
    out.write_text(json.dumps(report, indent=1))
    for key, lines in report.items():
        print(f"\n== {key} ({len(lines)})", *lines, sep="\n  ", flush=True)
    flagged = {key: len(lines) for key, lines in report.items() if key != "FOLLOW"}
    print(f"\nTiming report: {sum(flagged.values())} finding(s) {flagged}; "
          f"{len(report['FOLLOW'])} FOLLOW record(s); runs took "
          f"{' + '.join(f'{value:.1f}s' for value in seconds)}; JSON: {out}", flush=True)
    return report


@contextmanager
def runner_lock(wait_seconds: float | None = None, poll_seconds: float = 1.0):
    # All heavy suites on one runner serialize, even across separate worktrees.
    # The descriptor releases on process exit, including Ctrl-C and pytest failure.
    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError("test-suite serialization requires a Unix runner") from exc
    wait = lock_wait_seconds() if wait_seconds is None else wait_seconds
    lock_path = Path(tempfile.gettempdir()) / "llm-bawt-test-suite.lock"
    with lock_path.open("w") as handle:
        started = time.monotonic()
        announced = False
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                waited = time.monotonic() - started
                if waited >= wait:
                    detail = f" (waited {waited:.1f}s)" if wait else ""
                    raise RuntimeError(
                        f"another llm-bawt test suite is running on this host{detail}"
                    ) from exc
                if not announced:
                    print(f"Test runner lock: busy, waiting up to {wait:.0f}s", flush=True)
                    announced = True
                time.sleep(min(poll_seconds, wait - waited))
        if announced:
            print(f"Test runner lock: acquired after {time.monotonic() - started:.1f}s",
                  flush=True)
        yield


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("unit", "pg", "offline-integration", "sdk-probe", "integration",
                                           "offline", "target", "profile", "coverage", "stress",
                                           "timing-report", "service", "paid"))
    parser.add_argument("node", nargs="*",
                        help="target: explicit tests/... node IDs; stress: test files instead of the changed ones")
    parser.add_argument("--keyword", help="profile only: pytest -k expression")
    parser.add_argument("--stress-base", default=STRESS_BASE_DEFAULT,
                        help="stress: stress test files changed since this ref forked")
    parser.add_argument("--repeat", type=int, default=STRESS_REPEAT_DEFAULT, help="stress: rounds")
    parser.add_argument("--all", action="store_true", dest="everything",
                        help="stress only: the whole offline suite instead of changed files")
    args = parser.parse_args(argv)
    if args.suite in {"service", "paid"}:
        parser.error("external CLI/LLM tests are disabled until an independently provisioned "
                     "isolated service and credential scope can be verified; never use live :8642")
    if args.node and args.suite not in {"target", "stress"}:
        parser.error("node IDs require the target or stress suite")
    if args.everything and (args.suite != "stress" or args.node):
        parser.error("--all is only for the stress suite, without files")
    if args.suite == "stress" and any(not is_test_file(node) or ".." in node for node in args.node):
        parser.error("stress files must be tests/.../test_*.py files")
    if args.keyword and args.suite != "profile":
        parser.error("--keyword is only supported for unit profiling")
    if args.suite == "target" and (not args.node or any(
        not node.startswith("tests/") or ".." in node or node.startswith("-")
        for node in args.node
    )):
        parser.error("target requires one or more safe tests/... node IDs")
    try:
        wait = lock_wait_seconds()
        # The wait is the runner's own setting; the tests it launches never inherit it.
        os.environ.pop(LOCK_WAIT_ENV, None)
        with runner_lock(wait):
            if args.suite == "offline":
                verify_offline_partition()
                run(UNIT, *unit_gate())
                run(PG, "--require-pg-acceptance", *parallel("loadscope"), *PG_SELECTORS)
                run(LOCAL, "--require-no-skips", *parallel("load"))
            elif args.suite == "stress":
                stress(args.stress_base, args.repeat, args.node, args.everything)
            elif args.suite == "integration":
                verify_offline_partition()
                run(PG, "--require-pg-acceptance", *parallel("loadscope"), *PG_SELECTORS)
                run(LOCAL, "--require-no-skips", *parallel("load"))
            elif args.suite == "unit":
                run(UNIT, *unit_gate())
            elif args.suite == "pg":
                verify_offline_partition()
                run(PG, "--require-pg-acceptance", *parallel("loadscope"), *PG_SELECTORS)
            elif args.suite == "offline-integration":
                run(LOCAL, "--require-no-skips", *parallel("load"))
            elif args.suite == "sdk-probe":
                os.environ.update({
                    "RUN_CLAUDE_OAUTH_SDK_PROBE": "1",
                    "RUN_COMPACT_SDK_PROBE": "1",
                    "RUN_PROXY_SDK_PROBE": "1",
                    "RUN_WORKER_LIVENESS_SDK_PROBE": "1",
                    "RUN_SYSTEM_PROMPT_PIN_SDK_PROBE": "1",
                })
                run(SDK, "--require-no-skips")
            elif args.suite == "profile":
                run(UNIT, "--durations=0", *(["-k", args.keyword] if args.keyword else []))
            elif args.suite == "timing-report":
                timing_report()
            elif args.suite == "coverage":
                run(UNIT, "--cov=src", "--cov-report=term-missing")
            else:
                # Explicit selection still rejects service/paid/live targets by marker.
                requested = collect("", *args.node)
                selected = collect(OFFLINE, *args.node, allow_empty=True)
                if requested != selected:
                    raise RuntimeError(f"target includes unsafe external tests: {sorted(requested - selected)}")
                print(f"Target selection: {len(selected)} tests", flush=True)
                run(OFFLINE, "--require-no-skips", *args.node)
    except (RuntimeError, KeyboardInterrupt) as exc:
        print(f"Test runner: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
