"""Gate contracts: an absent DB or zero executed private tests cannot turn green."""

from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests import conftest


@pytest.mark.parametrize("passed,skipped,expected", [
    ([], [], pytest.ExitCode.TESTS_FAILED),
    ([object()], [object()], pytest.ExitCode.TESTS_FAILED),
    ([object()], [], None),
])
def test_acceptance_exit_status_requires_passes_without_skips(monkeypatch, passed, skipped, expected):
    monkeypatch.setattr(conftest.pg_connection_guard, "uninstall", Mock())
    config = Mock()
    config.getoption.return_value = True
    config.pluginmanager.get_plugin.return_value.stats = {"passed": passed, "skipped": skipped}
    session = SimpleNamespace(config=config, exitstatus=None)
    conftest.pytest_sessionfinish(session, 0)
    assert session.exitstatus == expected


def test_unit_budget_includes_setup_call_and_teardown(monkeypatch):
    monkeypatch.setattr(conftest.pg_connection_guard, "uninstall", Mock())
    config = Mock()
    config.getoption.side_effect = lambda key: {
        "--require-pg-acceptance": False,
        "--require-no-skips": False,
        "--unit-time-budget": 1.0,
    }[key]
    config.pluginmanager.get_plugin.return_value.stats = {"passed": [object()]}
    config._task1189_durations = conftest._DURATIONS
    nodeid = "tests/virtual.py::test_summed"
    try:
        for when, duration in (("setup", 0.4), ("call", 0.35), ("teardown", 0.26)):
            conftest.pytest_runtest_logreport(
                SimpleNamespace(nodeid=nodeid, when=when, duration=duration)
            )
        session = SimpleNamespace(config=config, exitstatus=None)
        conftest.pytest_sessionfinish(session, 0)
        assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
    finally:
        conftest._DURATIONS.pop(nodeid, None)


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
