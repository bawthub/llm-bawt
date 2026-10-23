"""File-backed SQLite engines for unit fixtures that need real connections.

A ``sqlite:///`` file gives a fixture what in-memory ``sqlite://`` cannot:
several connections (``asyncio.to_thread`` workers, a second ``Session``) that
see one database with real locking and transactions. What none of those
fixtures test is durability across a power loss, yet SQLite's default
``synchronous=FULL`` fsyncs the journal and the database file on every commit.
On echo's NVMe that costs well under a millisecond; on a hosted CI runner's
virtual disk it is tens of milliseconds, so a fixture that commits a few dozen
times spends seconds inside the 1s unit time budget (TASK-1312). Turning
``synchronous`` off keeps every SQL semantic the tests rely on and drops only
the fsyncs.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine, event


def fast_sqlite_engine(path: str | Path, **connect_args) -> Engine:
    """``create_engine`` for a file-backed SQLite fixture without per-commit fsync."""
    engine = create_engine(f"sqlite:///{path}", connect_args=connect_args)

    @event.listens_for(engine, "connect")
    def fast_fixture_writes(connection, _record):
        connection.execute("PRAGMA synchronous = OFF")

    return engine
