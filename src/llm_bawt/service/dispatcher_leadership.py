"""The inter-bot dispatcher's leadership lock session (TASK-1311).

Leadership is a session-level PostgreSQL advisory lock: it lives exactly as
long as the connection that took it. When PostgreSQL restarts (nightly on
prod) that session is gone, and with it the singleton fence, whether or not
the process has noticed. This module holds the three pieces that make the
dispatcher notice and recover:

- ``build_lock_engine``: a dedicated single-connection engine for the lock
  session. The shared pool's connections are sized for short checkouts; the
  lock session is held for the whole leadership term, so it gets its own
  libpq keepalives and ``tcp_user_timeout`` (a half-open socket fails in a
  bounded time instead of hanging a probe for the kernel's retransmit
  budget) and no longer pins one of the shared pool's slots.
- ``probe_lock_session`` / ``probe_within``: a liveness check that runs off
  the event loop and is bounded even when the socket is half-open. A probe
  that fails or outlives its bound counts as "session lost"; the session is
  closed (at once, or once the late probe finally returns) so the server-side
  lock can never outlive the process's view of it.
- ``release_lock_session``: an unlock that tolerates a dead session. A lock
  that died with its session is already released; the unlock's failure is a
  WARNING, never an exception out of shutdown.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy import text as sa_text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import DBAPIError, InvalidRequestError

logger = logging.getLogger(__name__)

DISPATCHER_LOCK_NAME = "inter-bot-dispatcher"
LOCK_NAME_INFO_KEY = "inter_bot_dispatcher_lock_name"

# A probe that has not answered by then counts as lost. ``tcp_user_timeout``
# below makes libpq fail a half-open socket before this fires; the asyncio
# bound is the backstop for a socket the kernel still believes in.
PROBE_TIMEOUT_SECONDS = 20.0

# libpq parameters for the lock session only (psycopg2 forwards unknown
# connect kwargs to libpq). Keepalives detect a silently dead peer in about
# idle + interval * count = 60s; tcp_user_timeout caps how long an unanswered
# write (the probe itself) may wait before the socket errors out.
LOCK_SESSION_CONNECT_ARGS: dict[str, Any] = {
    "connect_timeout": 10,
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 10,
    "keepalives_count": 3,
    "tcp_user_timeout": 15_000,
}

# Errors that mean "this connection is unusable", and nothing else.
SESSION_GONE_ERRORS: tuple[type[Exception], ...] = (DBAPIError, InvalidRequestError)


def build_lock_engine(
    config: Any, *, application_name: str = "llm-bawt:dispatcher-lock"
) -> Engine | None:
    """One-connection engine for the leadership session, or None without Postgres.

    ``NullPool``: each ``connect()`` opens a fresh libpq session and ``close()``
    ends it, so a released or dead leadership session is never handed back
    out of a pool. The process holds at most one such connection at a time.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool

    from ..utils.db import build_postgres_url, set_utc_on_connect

    url = build_postgres_url(config)
    if not url:
        return None
    engine = create_engine(
        url,
        poolclass=NullPool,
        connect_args={"application_name": application_name[:63], **LOCK_SESSION_CONNECT_ARGS},
    )
    set_utc_on_connect(engine)
    return engine


def close_quietly(conn: Connection) -> None:
    """Close a lock session whose server side may already be gone."""
    try:
        conn.close()
    except SESSION_GONE_ERRORS as exc:
        logger.debug("leadership lock session close failed after loss: %s", exc)


def probe_lock_session(conn: Connection) -> bool:
    """Whether the session that holds the lock still answers.

    Runs a round trip on the lock connection itself: an advisory lock cannot
    be checked through any other session, because the lock dies with this
    one. On failure the connection is closed (it holds nothing any more) and
    False is returned; the caller drops leadership.
    """
    try:
        conn.execute(sa_text("SELECT 1"))
        # End SQLAlchemy's autobegun transaction: the session must stay
        # idle, not idle-in-transaction, between probes.
        conn.commit()
    except SESSION_GONE_ERRORS as exc:
        logger.warning(
            "leadership lock session no longer answers (%s: %s); the lock died with it",
            type(exc).__name__, str(exc).strip().splitlines()[0] if str(exc).strip() else "",
        )
        close_quietly(conn)
        return False
    return True


async def probe_within(
    probe: Callable[[], bool], *, timeout: float, abandon: Callable[[], None]
) -> bool:
    """Run ``probe`` off the loop; a probe still unanswered after ``timeout`` is a failure.

    The thread behind a timed-out probe cannot be interrupted. If it later
    reports the session alive, ``abandon`` runs (off the loop) so the session
    is closed anyway: leadership was already given up on, and a session that
    kept its lock would block this process's own re-acquisition.
    """
    future = asyncio.ensure_future(asyncio.to_thread(probe))
    try:
        return await asyncio.wait_for(asyncio.shield(future), timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "leadership lock probe did not answer within %.0fs; treating the session as lost",
            timeout,
        )
        loop = asyncio.get_running_loop()

        def _late(done: asyncio.Future) -> None:
            if not done.cancelled() and done.exception() is None and done.result():
                loop.run_in_executor(None, abandon)

        future.add_done_callback(_late)
        return False


def release_lock_session(conn: Connection) -> None:
    """Unlock and close; a session that died already released its lock.

    The narrow catch is deliberate: only connection/session errors mean "the
    lock went with the session". Anything else is a bug and still raises.
    """
    lock_name = conn.info.get(LOCK_NAME_INFO_KEY, DISPATCHER_LOCK_NAME)
    try:
        conn.execute(
            sa_text("SELECT pg_advisory_unlock(hashtext(:lock_name))"),
            {"lock_name": lock_name},
        )
        conn.commit()
    except SESSION_GONE_ERRORS as exc:
        logger.warning(
            "leadership lock session was already gone at release; the lock %r was "
            "released with its session (%s: %s)",
            lock_name, type(exc).__name__,
            str(exc).strip().splitlines()[0] if str(exc).strip() else "",
        )
    finally:
        close_quietly(conn)
