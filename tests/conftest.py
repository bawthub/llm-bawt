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

# Root conftest is imported before nested conftests and test modules; a
# pytest_configure hook would be too late for their import-time side effects.
# Plugins loaded before this root conftest remain outside this boundary.
pg_connection_guard.install()

pytest_plugins = ("tests.private_postgres",)


_PG_ACCEPTANCE_SUITES = (
    "test_session_search_pg.py::",
    "test_session_continuity_pg.py::",
    "test_agent_bridge.py::TestEventStore::",
    "test_agent_context.py::",
    "test_inter_bot_delivery.py::",
    "test_inter_bot_delivery_retry.py::",
    "test_turn_admission_pg.py::",
    "test_session_reset_lineage_pg.py::",
    "test_session_summaries.py::TestSummarizeThreadLive::",
    "test_session_summaries.py::TestResummarizeHeuristicSummaries::",
    "test_session_scoping.py::TestScopedPersistenceProofs::",
    "test_session_scoping.py::TestScopedContextProofs::",
    "test_session_threads.py::TestRegistryInvariantsLiveDB::",
    "test_session_unique_index_pg.py::",
    "test_thread_agent_keys.py::TestAgentKeyEndpointLiveDB::",
)


def pytest_addoption(parser):
    parser.addoption("--require-pg-acceptance", action="store_true", default=False,
                     help="fail if any private-cluster acceptance suite is absent or skipped")


def _require_pg_acceptance(config):
    return config.getoption("--require-pg-acceptance")


def pytest_configure(config):
    # Per-file counters of deselected vs surviving items. Populated across the
    # collection lifecycle (pytest_deselected + pytest_collection_finish) and
    # read in pytest_terminal_summary.
    config._task780_deselected_by_file = defaultdict(int)
    config._task780_surviving_by_file = defaultdict(int)


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
        missing = [suite for suite in _PG_ACCEPTANCE_SUITES
                   if not any(suite in nodeid for nodeid in selected)]
        if missing:
            pytest.exit(f"PostgreSQL acceptance gate missing selected suites: {missing}", returncode=1)


def pytest_sessionfinish(session, exitstatus):
    if _require_pg_acceptance(session.config) and exitstatus == 0:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        stats = reporter.stats if reporter is not None else {}
        if not stats.get("passed") or stats.get("skipped") or stats.get("xfailed"):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
            print("PostgreSQL acceptance gate requires executed passes and zero skips/xfails")
    pg_connection_guard.uninstall()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
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
