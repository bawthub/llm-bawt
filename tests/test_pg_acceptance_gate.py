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
