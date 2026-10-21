"""Pytest-only PostgreSQL boundary, installed before test module collection.

This protects in-process psycopg2/SQLAlchemy connections. It cannot protect a
subprocess or a running service; those require separate pre-launch guards.
"""

from __future__ import annotations

import os
import stat
from contextlib import contextmanager
from pathlib import Path

import psycopg2
from psycopg2.extensions import make_dsn, parse_dsn


_private_sockets: set[tuple[str, str]] = set()
_original_connect = None
_private_only = False


@contextmanager
def private_cluster_socket(socket_dir: Path, data_dir: Path | None, port: str = "55439"):
    """Admit only a running Unix-socket cluster started by a test fixture."""
    socket = socket_dir.resolve(strict=True)
    if data_dir is not None and not (data_dir / "PG_VERSION").is_file():
        raise RuntimeError("private PostgreSQL cluster has not been initialized")
    if not stat.S_ISSOCK((socket / f".s.PGSQL.{port}").stat().st_mode):
        raise RuntimeError("private PostgreSQL cluster socket is not running")
    key = (str(socket), port)
    _private_sockets.add(key)
    try:
        yield
    finally:
        _private_sockets.remove(key)


def checked_connect(*args, **kwargs):
    """Reject inherited/unknown destinations BEFORE libpq opens a connection."""
    if _original_connect is None:
        raise RuntimeError("PostgreSQL test boundary is not installed")
    if os.environ.get("PGSERVICE") or kwargs.get("service"):
        raise RuntimeError("pytest refuses libpq service-based PostgreSQL routing")
    try:
        # make_dsn applies the same keyword-over-DSN precedence as psycopg2.
        dsn_kwargs = {k: v for k, v in kwargs.items()
                      if k not in {"connection_factory", "cursor_factory", "async_"}}
        params = parse_dsn(make_dsn(*args, **dsn_kwargs))
    except (TypeError, ValueError, psycopg2.Error) as exc:
        raise RuntimeError("pytest refuses an unparseable PostgreSQL target") from exc
    if params.get("service"):
        raise RuntimeError("pytest refuses libpq service-based PostgreSQL routing")
    if params.get("hostaddr") or os.environ.get("PGHOSTADDR"):
        raise RuntimeError("pytest refuses libpq hostaddr routing around the verified host")
    database = params.get("dbname")
    host = params.get("host", os.environ.get("PGHOST", ""))
    port = str(params.get("port", os.environ.get("PGPORT", "5432")))
    private_socket = (host, port) in _private_sockets and params.get("user") == "test_user"
    private = private_socket and database == "postgres"
    explicit_test_db = (
        isinstance(database, str) and database.endswith("_test")
        and os.environ.get("LLM_BAWT_POSTGRES_DATABASE") == database
        and (not _private_only or private_socket)
    )
    if not (private or explicit_test_db):
        raise RuntimeError("pytest refuses PostgreSQL target: explicitly select a *_test database or a private cluster")
    conn = _original_connect(*args, **kwargs)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT current_database(), current_setting('port'), inet_server_addr()")
            actual_db, actual_port, server_addr = cursor.fetchone()
            if actual_db != database or (private_socket and (actual_port != port or server_addr is not None)):
                raise RuntimeError("PostgreSQL connection identity did not match the verified test target")
        conn.rollback()  # Close the read-only identity probe before handing it to SQLAlchemy.
    except BaseException:
        conn.close()
        raise
    return conn


def install() -> None:
    global _original_connect
    if _original_connect is not None:
        return
    _original_connect = psycopg2.connect
    psycopg2.connect = checked_connect


def require_private_only() -> None:
    """Reject even explicit *_test URLs unless this pytest session owns the socket."""
    global _private_only
    _private_only = True


def uninstall() -> None:
    global _original_connect, _private_only
    if _original_connect is not None:
        psycopg2.connect = _original_connect
        _original_connect = None
    _private_only = False
    _private_sockets.clear()
