"""Active-session index bootstrap and explicit upgrade on disposable PostgreSQL.

Every case gets a distinct database in the networkless private container. These
regressions never connect to or modify an operational/shared test database.
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import IntegrityError

from tests.pg_isolation import require_test_db_config
from tests.pg_lock_wait import wait_blocked_by
from tests.private_postgres import PORT, private_url

pytestmark = pytest.mark.integration


@pytest.fixture
def index_db(private_pg, monkeypatch):
    from llm_bawt.memory.postgresql import PostgreSQLMemoryBackend
    from llm_bawt.utils import db

    name = f"session_index_{uuid.uuid4().hex[:12]}_test"
    socket = private_pg
    admin = create_engine(private_url(socket, "postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    url = private_url(socket, name)
    monkeypatch.setattr(db, "build_postgres_url", lambda config: url)
    monkeypatch.setenv("LLM_BAWT_POSTGRES_HOST", str(socket))
    monkeypatch.setenv("LLM_BAWT_POSTGRES_PORT", PORT)
    monkeypatch.setenv("LLM_BAWT_POSTGRES_USER", "test_user")
    monkeypatch.setenv("LLM_BAWT_POSTGRES_DATABASE", name)
    monkeypatch.setenv("LLM_BAWT_POSTGRES_PASSWORD", uuid.uuid4().hex)
    engine = create_engine(url)
    try:
        config = require_test_db_config()
        backend = PostgreSQLMemoryBackend(config, bot_id="index_proof")
        yield config, engine, backend
    finally:
        engine.dispose()
        db.reset_shared_engines()
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def _insert(conn, *, bot="index_proof", user=None, status="active"):
    identity = str(uuid.uuid4())
    conn.execute(text("""
        INSERT INTO sessions (id, bot_id, user_id, status)
        VALUES (:id, :bot, :user, :status)
    """), {"id": identity, "bot": bot, "user": user, "status": status})
    return identity


def _bounded_engine(config):
    """One pooled connection, with the app's actual on-connect timeout policy."""
    from llm_bawt.utils import db

    return db.build_engine(db.build_postgres_url(config), pool_size=1, max_overflow=0)


def _timeout_state(engine):
    with engine.connect() as conn:
        return conn.execute(text("""
            SELECT pg_backend_pid(), current_setting('lock_timeout'),
                   current_setting('statement_timeout')
        """)).one()


def _set_custom_timeouts(engine):
    with engine.begin() as conn:
        conn.execute(text("SET lock_timeout = '3s'"))
        conn.execute(text("SET statement_timeout = '47s'"))
    state = _timeout_state(engine)
    assert state[1:] == ("3s", "47s")
    return state


def test_fresh_bootstrap_enforces_nullable_user_and_allows_archives(index_db):
    from llm_bawt.memory.session_unique_index import active_session_index_present

    _, engine, _ = index_db
    with engine.connect() as conn:
        assert active_session_index_present(conn)
    with engine.begin() as conn:
        _insert(conn, user=None)
        _insert(conn, user=None, status="archived")
        _insert(conn, user=None, status="archived")
        _insert(conn, user="other")
        _insert(conn, bot="another_bot", user=None)
    for user in (None, "other"):
        with pytest.raises(IntegrityError), engine.begin() as conn:
            _insert(conn, user=user)


def test_fresh_bootstrap_concurrent_first_writes(index_db):
    from llm_bawt.memory.postgresql import PostgreSQLShortTermManager

    config, engine, _ = index_db

    def first_write(_):
        return PostgreSQLShortTermManager(config, bot_id="race_proof", user_id="user").get_or_create_active_session()

    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(first_write, range(8)))
    assert len(set(ids)) == 1
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id FROM sessions WHERE bot_id='race_proof' AND user_id='user' AND status='active'
        """)).all()
    assert rows == [(ids[0],)]


def test_explicit_upgrade_is_concurrent_repeatable_and_bootstrap_does_not_upgrade(index_db):
    from llm_bawt.memory.postgresql import PostgreSQLMemoryBackend
    from llm_bawt.memory.session_unique_index import (
        active_session_index_present, upgrade_sessions_one_active_index,
    )

    config, engine, _ = index_db
    with engine.begin() as conn:
        conn.execute(text("DROP INDEX public.idx_sessions_one_active_per_bot_user"))
        _insert(conn, user="upgrade")
        _insert(conn, user="upgrade", status="archived")
    # A fresh process/engine must re-run the shared-schema bootstrap guard.
    from llm_bawt.utils import db
    db.reset_shared_engines()
    with pytest.raises(RuntimeError, match="existing sessions lacks"):
        PostgreSQLMemoryBackend(config, bot_id="held_bootstrap")
    with engine.connect() as conn:
        assert not active_session_index_present(conn)
    assert upgrade_sessions_one_active_index(engine) is True
    assert upgrade_sessions_one_active_index(engine) is False
    with engine.connect() as conn:
        assert active_session_index_present(conn)
    PostgreSQLMemoryBackend(config, bot_id="after_upgrade")
    with pytest.raises(IntegrityError), engine.begin() as conn:
        _insert(conn, user="upgrade")


def test_upgrade_duplicate_null_pair_fails_without_repair(index_db):
    from llm_bawt.memory.session_unique_index import (
        active_session_index_present, upgrade_sessions_one_active_index,
    )

    _, engine, _ = index_db
    with engine.begin() as conn:
        conn.execute(text("DROP INDEX public.idx_sessions_one_active_per_bot_user"))
        first = _insert(conn)
        second = _insert(conn)
    with pytest.raises(IntegrityError):
        upgrade_sessions_one_active_index(engine)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM sessions WHERE id IN (:a, :b)"),
                            {"a": first, "b": second}).scalar_one() == 2
        with pytest.raises(RuntimeError, match="invalid or differs"):
            active_session_index_present(conn)
    with pytest.raises(RuntimeError, match="invalid or differs"):
        upgrade_sessions_one_active_index(engine)


@pytest.mark.parametrize("duplicate", [False, True])
def test_upgrade_preserves_prior_pooled_timeouts(index_db, duplicate):
    from llm_bawt.memory.session_unique_index import upgrade_sessions_one_active_index

    config, setup_engine, _ = index_db
    with setup_engine.begin() as conn:
        conn.execute(text("DROP INDEX public.idx_sessions_one_active_per_bot_user"))
        if duplicate:
            _insert(conn)
            _insert(conn)
    engine = _bounded_engine(config)
    try:
        original = _set_custom_timeouts(engine)
        if duplicate:
            with pytest.raises(IntegrityError):
                upgrade_sessions_one_active_index(engine)
        else:
            assert upgrade_sessions_one_active_index(engine) is True
        # The same physical pooled connection must retain the pre-upgrade
        # settings, not merely replace them with role/server defaults.
        assert _timeout_state(engine) == original
    finally:
        engine.dispose()


def test_failed_restoration_invalidates_without_masking_build_error(index_db):
    from llm_bawt.memory.session_unique_index import upgrade_sessions_one_active_index

    config, setup_engine, _ = index_db
    with setup_engine.begin() as conn:
        conn.execute(text("DROP INDEX public.idx_sessions_one_active_per_bot_user"))
        _insert(conn)
        _insert(conn)
    engine = _bounded_engine(config)
    try:
        defaults = _timeout_state(engine)[1:]
        original = _set_custom_timeouts(engine)

        def fail_restore(conn, cursor, statement, params, context, many):
            if "set_config('statement_timeout'" in statement and params.get("value") == original[2]:
                raise RuntimeError("synthetic timeout restoration failure")

        event.listen(engine, "before_cursor_execute", fail_restore)
        try:
            with pytest.raises(IntegrityError) as failure:
                upgrade_sessions_one_active_index(engine)
            assert "connection invalidated" in " ".join(failure.value.__notes__)
        finally:
            event.remove(engine, "before_cursor_execute", fail_restore)
        after = _timeout_state(engine)
        assert after[0] != original[0]  # The unsafe backend was NOT returned to the pool.
        assert after[1:] == defaults  # New checkout reapplies app deadman limits.
    finally:
        engine.dispose()


def test_cancelled_concurrent_build_restores_pool_and_requires_explicit_next_action(index_db):
    from threading import Event

    from llm_bawt.memory.session_unique_index import (
        INDEX_NAME, upgrade_sessions_one_active_index,
    )

    config, setup_engine, _ = index_db
    with setup_engine.begin() as conn:
        conn.execute(text("DROP INDEX public.idx_sessions_one_active_per_bot_user"))
    engine = _bounded_engine(config)
    started = Event()

    def mark_build(conn, cursor, statement, params, context, many):
        if statement.startswith("CREATE UNIQUE INDEX CONCURRENTLY"):
            # Give CI enough time to observe/cancel the *real* blocked CREATE;
            # the test hook widens only this statement's lock wait window.
            cursor.execute("SET lock_timeout = '5s'")
            started.set()

    event.listen(engine, "before_cursor_execute", mark_build)
    try:
        original = _set_custom_timeouts(engine)
        # This uncommitted write holds the concurrent index build at its
        # transaction wait; cancel the actual CREATE, not a simulated error.
        with setup_engine.connect() as blocker:
            _insert(blocker, bot="cancel_proof", user="writer")
            with ThreadPoolExecutor(max_workers=1) as workers:
                future = workers.submit(upgrade_sessions_one_active_index, engine)
                try:
                    assert started.wait(timeout=8), "concurrent build did not start"
                    # The build is the only session that can wait on the blocker's open
                    # transaction. Observe that wait before the test's 5s lock timeout.
                    blocker_pid = blocker.execute(text("SELECT pg_backend_pid()")).scalar_one()
                    wait_blocked_by(setup_engine, blocker_pid, future, timeout=3.5)
                    with setup_engine.connect() as observer:
                        assert observer.execute(text("SELECT pg_cancel_backend(:pid)"),
                                                {"pid": original[0]}).scalar_one()
                    with pytest.raises(Exception) as cancelled:
                        future.result(timeout=8)
                    assert getattr(getattr(cancelled.value, "orig", cancelled.value), "pgcode", None) == "57014"
                finally:
                    blocker.rollback()
        assert _timeout_state(engine) == original
        # PostgreSQL can leave an absent, invalid, or completed index after an
        # interrupted concurrent build. Inspect catalog state, never auto-drop.
        with setup_engine.connect() as conn:
            state = conn.execute(text("""
                SELECT indisvalid FROM pg_index WHERE indexrelid=to_regclass(:name)
            """), {"name": f"public.{INDEX_NAME}"}).one_or_none()
        if state is None:
            assert upgrade_sessions_one_active_index(engine) is True
        elif not state[0]:
            with pytest.raises(RuntimeError, match="invalid or differs"):
                upgrade_sessions_one_active_index(engine)
        else:
            assert upgrade_sessions_one_active_index(engine) is False
        assert _timeout_state(engine) == original
    finally:
        event.remove(engine, "before_cursor_execute", mark_build)
        engine.dispose()


@pytest.mark.parametrize("replacement", [
    "CREATE INDEX idx_sessions_one_active_per_bot_user ON public.sessions (bot_id, user_id)",
    "CREATE UNIQUE INDEX idx_sessions_one_active_per_bot_user ON public.sessions (bot_id, user_id) WHERE status='active'",
])
def test_wrong_same_name_index_rejected(index_db, replacement):
    from llm_bawt.memory.session_unique_index import upgrade_sessions_one_active_index

    _, engine, _ = index_db
    with engine.begin() as conn:
        conn.execute(text("DROP INDEX public.idx_sessions_one_active_per_bot_user"))
        conn.execute(text(replacement))
    with pytest.raises(RuntimeError, match="invalid or differs"):
        upgrade_sessions_one_active_index(engine)
