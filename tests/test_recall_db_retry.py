"""Recall DB reads retry a transient failure exactly once (TASK-1532).

Live: the messages source hit its 2 s statement timeout on a cold query and the
whole source reported source_unavailable; the same call seconds later succeeded.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.exc import TimeoutError as PoolTimeout

from llm_bawt.recall.db_errors import db_sqlstate, is_transient_db_error
from llm_bawt.recall.postgres import RecallDatabase


class _Orig(Exception):
    def __init__(self, pgcode):
        super().__init__(pgcode)
        self.pgcode = pgcode


def _operational(code=None, invalidated=False):
    return OperationalError("SELECT", {}, _Orig(code), connection_invalidated=invalidated)


@pytest.mark.parametrize("error", [
    _operational("57014"), _operational("55P03"), _operational("40001"), _operational("40P01"),
    _operational("53300"), _operational("57P01"), _operational("08006"), _operational(None),
    _operational("XX000", invalidated=True), PoolTimeout("pool exhausted"),
])
def test_transient_errors(error):
    assert is_transient_db_error(error)


@pytest.mark.parametrize("error", [
    ProgrammingError("SELECT", {}, _Orig("42P01")), ProgrammingError("SELECT", {}, _Orig("42601")),
    _operational("28P01"), _operational("XX000"), ValueError("bad row"),
])
def test_permanent_errors(error):
    assert not is_transient_db_error(error)


def test_sqlstate_reads_wrapped_driver_error():
    assert db_sqlstate(_operational("57014")) == "57014"
    assert db_sqlstate(None) is None


class _Scripted(RecallDatabase):
    def __init__(self, *outcomes):
        super().__init__(engine=object())
        self.outcomes = list(outcomes)
        self.calls = 0

    def _read_once(self, statement, params, timeout_ms):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_transient_failure_retries_once_and_succeeds():
    database = _Scripted(_operational("57014"), [{"id": 1}])
    assert database.read("SELECT 1", {}) == [{"id": 1}]
    assert database.calls == 2


def test_second_transient_failure_is_not_retried_again():
    database = _Scripted(_operational("57014"), _operational("57014"), [{"id": 1}])
    with pytest.raises(OperationalError):
        database.read("SELECT 1", {})
    assert database.calls == 2


def test_permanent_failure_is_not_retried():
    database = _Scripted(ProgrammingError("SELECT", {}, _Orig("42P01")), [{"id": 1}])
    with pytest.raises(ProgrammingError):
        database.read("SELECT 1", {})
    assert database.calls == 1


def test_retry_can_be_disabled_for_best_effort_reads():
    database = _Scripted(_operational("57014"), [{"id": 1}])
    with pytest.raises(OperationalError):
        database.read("SELECT 1", {}, 800, False)
    assert database.calls == 1


@pytest.mark.integration
@pytest.mark.usefixtures("private_app_pg")
def test_real_statement_timeout_is_retried_on_postgres():
    """Inject a genuine PostgreSQL statement timeout on the first attempt only."""
    from llm_bawt.utils import db
    from tests.pg_isolation import require_test_db_config

    engine = create_engine(db.build_postgres_url(require_test_db_config()))
    try:
        real = RecallDatabase(engine)
        with pytest.raises(OperationalError) as cold:
            real.read("SELECT pg_sleep(1)", {}, 50, False)
        assert db_sqlstate(cold.value) == "57014" and is_transient_db_error(cold.value)

        class FirstSlow(RecallDatabase):
            attempts = 0

            def _read_once(self, statement, params, timeout_ms):
                self.attempts += 1
                if self.attempts == 1:
                    statement = "SELECT pg_sleep(1) AS id"
                return super()._read_once(statement, params, timeout_ms)

        database = FirstSlow(engine)
        assert database.read("SELECT 7 AS id", {}, 50) == [{"id": 7}]
        assert database.attempts == 2
        with engine.connect() as connection:
            assert connection.execute(text("SELECT 1")).scalar() == 1
    finally:
        engine.dispose()
