"""Report-only timing check for the unit suite (TASK-1457 phase a).

Loaded only by ``make test-timing-report`` (``-p tests.timing_report``); the
normal suite and ``make gate`` never import it. For one unit run it:

- D1/D2: makes every asyncio loop a :class:`tests.virtual_clock.TimingLoop`
  (JUMP/FOLLOW virtual clock, ``LLM_BAWT_TIMING_STEP_COST`` seconds charged per
  scheduler turn) and makes ``time.monotonic/perf_counter/time`` read that
  clock on the loop's thread. A test that fails here depends on timing.
- FOLLOW: records, per test, which real work kept the loop on real time and
  how many real seconds it spent there.
- D3: records ``time.sleep`` on the test's main thread and every subprocess
  spawn. Recorded, not raised, so the D1/D2 outcome of the same test still
  shows. ``@pytest.mark.real_time(reason=...)`` / ``real_process(reason=...)``
  mark one as deliberate.
- D6: a test still running after ``WATCHDOG_S`` real seconds dumps its stack
  and kills its worker, which reports it failed.

The xdist controller collects every worker's records (they travel in
``report.user_properties``) and writes one JSON file to
``LLM_BAWT_TIMING_REPORT``. Nothing here changes a test's outcome except D1/D2
themselves, which is the point.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
import faulthandler
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

import pytest

from tests import virtual_clock

REPORT_ENV = "LLM_BAWT_TIMING_REPORT"
STEP_COST_ENV = "LLM_BAWT_TIMING_STEP_COST"
WATCHDOG_S = 90.0
_PROPERTY = "task1457_timing"

_current: dict[str, Any] = {}
_saved: list[tuple[Any, str, Any]] = []
_follow_time = virtual_clock.follow_time()


def _record(kind: str, detail: str) -> None:
    events = _current.get("events")
    if events is not None:
        events[kind].append(detail)


def _observe(cause: str, real_seconds: float) -> None:
    follow = _current.get("follow")
    if follow is not None:
        entry = follow.setdefault(cause, [0, 0.0])
        entry[0] += 1
        entry[1] += real_seconds


def _patch(owner: Any, name: str, replacement: Any) -> None:
    _saved.append((owner, name, getattr(owner, name)))
    setattr(owner, name, replacement)


def _install() -> None:
    asyncio.set_event_loop_policy(virtual_clock.TimingLoopPolicy(
        step_cost=float(os.environ.get(STEP_COST_ENV) or 0), observer=_observe))
    _follow_time.__enter__()
    real_sleep, real_popen_init = time.sleep, subprocess.Popen.__init__

    def sleep(seconds: float) -> None:
        # sleep(0) only yields the GIL; it waits for nothing.
        if seconds and threading.current_thread() is threading.main_thread():
            _record("real_time", f"time.sleep({seconds!r})")
        real_sleep(seconds)

    def popen_init(self, args, *rest, **kwargs):
        _record("real_process", str(args[0] if isinstance(args, (list, tuple)) and args else args))
        real_popen_init(self, args, *rest, **kwargs)

    _patch(time, "sleep", sleep)
    _patch(subprocess.Popen, "__init__", popen_init)


def _uninstall() -> None:
    while _saved:
        owner, name, original = _saved.pop()
        setattr(owner, name, original)
    _follow_time.__exit__(None, None, None)
    asyncio.set_event_loop_policy(None)


def pytest_configure(config: pytest.Config) -> None:
    controller = not hasattr(config, "workerinput") and bool(getattr(config.option, "numprocesses", None))
    if not controller:
        _install()  # the process that runs tests: an xdist worker, or a serial run


def pytest_unconfigure(config: pytest.Config) -> None:
    if _saved:
        _uninstall()


def _allowed(item: pytest.Item) -> dict[str, str]:
    return {name: str(mark.kwargs.get("reason", ""))
            for name in ("real_time", "real_process")
            for mark in item.iter_markers(name)}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem):
    _current.update(events=defaultdict(list), follow={})
    faulthandler.dump_traceback_later(WATCHDOG_S, exit=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
        _current.clear()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call):
    outcome = yield
    report = outcome.get_result()
    if call.when == "teardown" and (_current.get("events") or _current.get("follow")):
        report.user_properties.append((_PROPERTY, json.dumps({
            "events": dict(_current["events"]), "follow": _current["follow"],
            "allowed": _allowed(item)})))


_CONTROLLER: dict[str, dict[str, Any]] = {}


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    timing = [value for name, value in report.user_properties if name == _PROPERTY]
    if not report.failed and not timing:
        return
    entry = _CONTROLLER.setdefault(report.nodeid, {})
    if report.failed:
        crash = getattr(report.longrepr, "reprcrash", None)
        entry.setdefault("failed", {"when": report.when,
                                    "message": (crash.message if crash else str(report.longrepr))[:300]})
    for value in timing:
        entry.update(json.loads(value))


def pytest_sessionfinish(session: pytest.Session, exitstatus) -> None:
    if hasattr(session.config, "workerinput"):
        return
    path = os.environ.get(REPORT_ENV)
    if path:
        notable = {nodeid: entry for nodeid, entry in sorted(_CONTROLLER.items()) if entry}
        Path(path).write_text(json.dumps(notable, indent=1, sort_keys=True))
