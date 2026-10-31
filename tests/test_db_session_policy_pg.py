"""The per-session safety policy survives the pool, against real PostgreSQL (TASK-1819).

The incident: the connect hook's ``SET``s ran in the DB-API's implicit
transaction, so the pool's rollback on the first return reverted them. Every
later checkout ran with no lock_timeout, and startup DDL waited forever behind
a dead host's idle-in-transaction session.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from llm_bawt.utils import db

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

POLICY = {
    "TimeZone": "UTC",
    "lock_timeout": "5s",
    "statement_timeout": "2min",
    "idle_in_transaction_session_timeout": "30s",
}
# tcp_keepalives_* are not asserted here: the private cluster is a Unix socket,
# where PostgreSQL ignores them and SHOW reads 0. The SET statements and their
# commit are pinned in tests/test_task739_lock_safety.py.


def _engine(config, application_name: str):
    # Through the module: private_app_pg patches db.build_postgres_url for its socket.
    return db.build_engine(db.build_postgres_url(config), application_name=application_name,
                           pool_size=1, max_overflow=0)


@pytest.fixture
def engine(private_app_pg):
    built = _engine(private_app_pg, "pytest-session-policy")
    try:
        yield built
    finally:
        built.dispose()


def _settings(conn) -> dict[str, str]:
    return {name: conn.execute(text(f'SHOW "{name}"')).scalar_one() for name in POLICY}


def test_policy_holds_after_the_pool_rolls_the_connection_back(engine) -> None:
    with engine.connect() as conn:
        first_pid = conn.execute(text("SELECT pg_backend_pid()")).scalar_one()
    # pool_size=1: the same server session, after the pool's reset-on-return.
    with engine.connect() as conn:
        assert conn.execute(text("SELECT pg_backend_pid()")).scalar_one() == first_pid
        assert _settings(conn) == POLICY


def test_reused_session_ddl_fails_fast_behind_an_idle_transaction(engine, private_app_pg) -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE IF NOT EXISTS session_policy_probe (id int)"))
    with engine.connect():
        pass  # first checkout returned: the pool has rolled the session back once

    blocker = _engine(private_app_pg, "pytest-session-policy-blocker")
    try:
        with blocker.connect() as held:
            held.execute(text("SELECT 1 FROM session_policy_probe"))  # AccessShareLock, left open
            with pytest.raises(OperationalError, match="lock timeout"):
                with engine.begin() as conn:
                    conn.execute(text("ALTER TABLE session_policy_probe ADD COLUMN IF NOT EXISTS probe int"))
            held.rollback()
    finally:
        blocker.dispose()
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS session_policy_probe"))
