"""Test-suite-wide fixtures and the deselection visibility guard (TASK-780).

The default marker filter in ``pyproject.toml`` (`-m 'not service and not
llm_call and not integration'`) silently drops whole test files — the biggest
being ``test_inter_bot_delivery.py`` (~25 tests, module-scoped
``pytestmark = pytest.mark.integration``). Under a bare ``pytest`` / ``make
test`` those files vanish from the report with no visible signal, so an agent
verifying an inter-bot / dispatcher / capability change could see green and
have executed none of the relevant tests. TASK-780.

The ``pytest_deselected`` hook below records every item pytest drops (marker
filter, ``-k``, ``--lf``, whatever) and the ``pytest_terminal_summary`` hook
prints a bright end-of-run warning naming any FILE that was fully deselected
(zero surviving items) plus the exact invocation to run them.

The default stays hermetic — this hook does not change what runs, only what
gets reported. A dev/CI run without a live DB still passes; the warning just
tells whoever reads the output "there are ~25 more tests you didn't run,
here's how."
"""

from __future__ import annotations

import pytest

from collections import defaultdict
from tests import pg_connection_guard
from tests import time_budget
from tests.suite_manifest import PG_ACCEPTANCE_SUITES

_DURATIONS = defaultdict(float)
_REAL_PROCESS: set[str] = set()  # nodeids marked real_process; judged on their own budget

# Root conftest is imported before nested conftests and test modules; a
# pytest_configure hook would be too late for their import-time side effects.
# Plugins loaded before this root conftest remain outside this boundary.
pg_connection_guard.install()

pytest_plugins = ("tests.private_postgres", "tests.session_lifecycle_pg_fixtures")


def pytest_addoption(parser):
    parser.addoption("--require-pg-acceptance", action="store_true", default=False,
                     help="fail if any private-cluster acceptance suite is absent or skipped")
    parser.addoption("--unit-time-budget", type=float, default=None,
                     help="fail when setup + call + teardown for any selected unit exceeds this many seconds")
    parser.addoption("--real-process-time-budget", type=float, default=None,
                     help="the same bound for tests marked real_process (default: the unit budget)")
    parser.addoption("--require-no-skips", action="store_true", default=False,
                     help="fail when any selected test skips or xfails")


def _require_pg_acceptance(config):
    return config.getoption("--require-pg-acceptance")


def pytest_configure(config):
    if config.getoption("--require-pg-acceptance"):
        pg_connection_guard.require_private_only()
    # Per-file counters of deselected vs surviving items. Populated across the
    # collection lifecycle (pytest_deselected + pytest_collection_finish) and
    # read in pytest_terminal_summary.
    config._task780_deselected_by_file = defaultdict(int)
    config._task780_surviving_by_file = defaultdict(int)
    _DURATIONS.clear()
    _REAL_PROCESS.clear()
    config._task1189_durations = _DURATIONS


def pytest_deselected(items):
    # pytest calls this with every item removed by the -m/-k expression.
    if not items:
        return
    counters = getattr(items[0].config, "_task780_deselected_by_file", None)
    if counters is None:
        return
    for item in items:
        counters[str(item.path)] += 1


def pytest_collection_finish(session):
    # ``session.items`` here is the POST-deselection surviving list — the
    # tests that will actually run. Anything in _task780_deselected_by_file
    # whose file has 0 survivors is fully dropped by the filter.
    # (pytest_collection_modifyitems runs BEFORE deselection under `-m`,
    # so its items list would count the deselected tests as "selected"
    # and make the diff useless. Use this hook instead.)
    for item in session.items:
        session.config._task780_surviving_by_file[str(item.path)] += 1
    if _require_pg_acceptance(session.config):
        selected = [item.nodeid for item in session.items]
        missing = [suite for suite in PG_ACCEPTANCE_SUITES
                   if not any(suite in nodeid for nodeid in selected)]
        if missing:
            pytest.exit(f"PostgreSQL acceptance gate missing selected suites: {missing}", returncode=1)


def pytest_runtest_logreport(report):
    if report.when in {"setup", "call", "teardown"}:
        _DURATIONS[report.nodeid] += report.duration
        # keywords travel with an xdist report, so the controller sees the marker too.
        if time_budget.REAL_PROCESS in getattr(report, "keywords", {}):
            _REAL_PROCESS.add(report.nodeid)


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    stats = reporter.stats if reporter is not None else {}
    # Under pytest-xdist the controller receives every worker's reports, so the
    # whole-run gates (passes, skips, unit budget) are judged there exactly once;
    # a worker only sees its own slice.
    is_xdist_worker = hasattr(config, "workerinput")
    if exitstatus == 0 and not is_xdist_worker:
        if _require_pg_acceptance(config) and not stats.get("passed"):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
            print("PostgreSQL acceptance gate requires executed passes")
        if (_require_pg_acceptance(config) or config.getoption("--require-no-skips")) and (
            stats.get("skipped") or stats.get("xfailed")
        ):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
            print("Test gate requires zero skips/xfails")
        budget = config.getoption("--unit-time-budget")
        if type(budget) in (int, float):
            process_budget = config.getoption("--real-process-time-budget")
            slowest = time_budget.slowest_real_process(config._task1189_durations, _REAL_PROCESS)
            if slowest is not None:
                print(f"Slowest real_process case: {slowest[1]:.3f}s {slowest[0]} "
                      f"(budget {budget if process_budget is None else process_budget}s)")
            found = time_budget.breaches(config._task1189_durations, _REAL_PROCESS, budget, process_budget)
            if found:
                session.exitstatus = pytest.ExitCode.TESTS_FAILED
                for breach in found:
                    print("\n".join(breach.lines()))
    pg_connection_guard.uninstall()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # The detailed missing-file warning describes the *default* filter only.
    # Explicit suite selections print their own counts; listing thousands of
    # intentionally deselected tests there obscures failures.
    if config.getoption("markexpr") != "not service and not llm_call and not integration":
        return
    deselected = getattr(config, "_task780_deselected_by_file", {}) or {}
    surviving = getattr(config, "_task780_surviving_by_file", {}) or {}
    fully_dropped = sorted(
        path for path, count in deselected.items()
        if count > 0 and surviving.get(path, 0) == 0
    )
    if not fully_dropped:
        return
    tr = terminalreporter
    tr.write_sep("=", "MARKER-DESELECTED FILES (TASK-780)", yellow=True, bold=True)
    tr.write_line(
        "The default filter (-m 'not service and not llm_call and not integration')",
        yellow=True,
    )
    tr.write_line(
        "silently dropped these whole files. Green above does NOT mean they passed:",
        yellow=True,
    )
    for path in fully_dropped:
        count = deselected.get(path, 0)
        tr.write_line(f"  - {path}  ({count} test{'s' if count != 1 else ''})", yellow=True)
    tr.write_line("", yellow=True)
    tr.write_line("To run them:", yellow=True, bold=True)
    tr.write_line("  make test-integration   # optional; CLI service tests remain blocked", yellow=True)
    tr.write_line(
        "  make test-pg-acceptance # required private PostgreSQL SQL tests",
        yellow=True,
    )


@pytest.fixture(autouse=True)
def force_fs_media_backend(monkeypatch):
    """Never let the unit suite talk to Garage (TASK-847).

    Bridge containers and echo export ``LLM_BAWT_STORAGE_BACKEND=s3`` plus real
    credentials, and ``MediaStore(root=tmp_path)`` honours them — so a plain
    ``pytest`` run would spray synthetic blobs into the production bucket.
    Pin the unit tests to the filesystem backend; S3 behaviour is covered by
    ``tests/test_object_store.py`` with explicit ``S3Config`` objects.
    """
    monkeypatch.setenv("LLM_BAWT_STORAGE_BACKEND", "fs")
    monkeypatch.setenv("LLM_BAWT_S3_FALLBACK_FS", "false")


@pytest.fixture(autouse=True)
def no_derived_bawthub_service_token(monkeypatch):
    """Never derive the BawtHub service token from a real master key (TASK-1472).

    With no override set, llm-bawt derives the token from the master Fernet
    key, which on echo is the live ``~/.config/llm-bawt/secret.key``. Tests that
    need derivation patch it back in with a fixture key; everyone else sees the
    pre-derivation behaviour (no token, no bearer) and nothing is exported.
    """
    from llm_bawt.bawthub_auth import service_token

    def refuse() -> str:
        raise RuntimeError("tests never derive from a real master key")

    monkeypatch.setattr(service_token, "derive_service_token", refuse)
    monkeypatch.setattr(service_token._SOURCE, "_derived", None)
    monkeypatch.delenv(service_token.EXPORT_FILE_ENV, raising=False)
