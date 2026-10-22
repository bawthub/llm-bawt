"""Canonical, serialized and fail-closed llm-bawt test entrypoints.

Run from repository root: python -m scripts.test_suite offline
Make and CI invoke this module rather than maintaining separate pytest recipes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import tempfile

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


def worker_count() -> int:
    """Parallel pytest-xdist workers for heavy suites; LLM_BAWT_TEST_WORKERS=1 runs serially."""
    configured = os.environ.get("LLM_BAWT_TEST_WORKERS")
    if configured:
        if not configured.isdigit() or int(configured) < 1:
            raise RuntimeError(f"LLM_BAWT_TEST_WORKERS must be a positive integer, got {configured!r}")
        return int(configured)
    # Leave headroom: the unit gate's 1s budget measures wall time per case.
    return max(1, min(8, (os.cpu_count() or 2) // 2))


def parallel(dist: str) -> tuple[str, ...]:
    """xdist arguments. ``loadscope`` keeps a module on one worker so each private-PG
    module database is created once; ``load`` balances independent unit cases."""
    workers = worker_count()
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


@contextmanager
def runner_lock():
    # All heavy suites on one runner serialize, even across separate worktrees.
    # The descriptor releases on process exit, including Ctrl-C and pytest failure.
    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError("test-suite serialization requires a Unix runner") from exc
    lock_path = Path(tempfile.gettempdir()) / "llm-bawt-test-suite.lock"
    with lock_path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another llm-bawt test suite is running on this host") from exc
        yield


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("unit", "pg", "offline-integration", "sdk-probe", "integration",
                                           "offline", "target", "profile", "coverage", "service", "paid"))
    parser.add_argument("node", nargs="*", help="target only: explicit tests/... node IDs")
    parser.add_argument("--keyword", help="profile only: pytest -k expression")
    args = parser.parse_args(argv)
    if args.suite in {"service", "paid"}:
        parser.error("external CLI/LLM tests are disabled until an independently provisioned "
                     "isolated service and credential scope can be verified; never use live :8642")
    if args.node and args.suite != "target":
        parser.error("node IDs require the target suite")
    if args.keyword and args.suite != "profile":
        parser.error("--keyword is only supported for unit profiling")
    if args.suite == "target" and (not args.node or any(
        not node.startswith("tests/") or ".." in node or node.startswith("-")
        for node in args.node
    )):
        parser.error("target requires one or more safe tests/... node IDs")
    try:
        with runner_lock():
            if args.suite == "offline":
                verify_offline_partition()
                run(UNIT, "--unit-time-budget=1", "--require-no-skips", *parallel("load"))
                run(PG, "--require-pg-acceptance", *parallel("loadscope"), *PG_SELECTORS)
                run(LOCAL, "--require-no-skips", *parallel("load"))
            elif args.suite == "integration":
                verify_offline_partition()
                run(PG, "--require-pg-acceptance", *parallel("loadscope"), *PG_SELECTORS)
                run(LOCAL, "--require-no-skips", *parallel("load"))
            elif args.suite == "unit":
                run(UNIT, "--unit-time-budget=1", "--require-no-skips", *parallel("load"))
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
                })
                run(SDK, "--require-no-skips")
            elif args.suite == "profile":
                run(UNIT, "--durations=0", *(["-k", args.keyword] if args.keyword else []))
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
