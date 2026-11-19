"""Alembic revision state for runtime schema ownership (TASK-631 slice 2).

A database is in one of three states:

* **unstamped** (no ``alembic_version`` table): compatibility mode. Every runtime
  ``ensure_*`` bootstrap runs exactly as before; nothing here changes behavior.
* **stamped** at one known revision: the app upgrades it to head at startup,
  and runtime bootstraps whose DDL a revision now owns are skipped.
* **invalid** (empty table, unknown revision, several rows): fail closed.

The app's startup upgrade uses one connection for the advisory lock, the state
read and every migration statement, so all three target the same database.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from weakref import WeakKeyDictionary

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection, Engine

from .schema import _ADVISORY_LOCK_NAME

logger = logging.getLogger(__name__)

VERSION_TABLE = "alembic_version"

_cache_lock = threading.Lock()
# Filled only after a committed startup upgrade; a pre-upgrade read is never cached.
_upgraded: WeakKeyDictionary[Engine, str] = WeakKeyDictionary()


class AlembicStateError(RuntimeError):
    """The database's Alembic state is not one this source can safely run on."""


def _repo_root() -> Path:
    # src/llm_bawt/utils/alembic_state.py -> repo root (or /app in images)
    return Path(__file__).resolve().parents[3]


def alembic_config(connection: Connection | None = None):
    from alembic.config import Config as AlembicConfig

    cfg = AlembicConfig(str(_repo_root() / "alembic.ini"))
    if connection is not None:
        cfg.attributes["connection"] = connection
    return cfg


def _script():
    from alembic.script import ScriptDirectory

    return ScriptDirectory.from_config(alembic_config())


def _known(rev: str) -> bool:
    from alembic.util import CommandError

    try:
        return _script().get_revision(rev) is not None
    except CommandError:
        return False


def read_revision(conn: Connection) -> str | None:
    """Return the stamped revision, None when unstamped; raise when invalid."""
    if not inspect(conn).has_table(VERSION_TABLE):
        return None
    rows = [r[0] for r in conn.execute(text(f"SELECT version_num FROM {VERSION_TABLE}"))]
    if len(rows) != 1:
        raise AlembicStateError(
            f"{VERSION_TABLE} has {len(rows)} rows; expected exactly one. Refusing to guess."
        )
    rev = rows[0]
    if not _known(rev):
        raise AlembicStateError(f"database is at unknown revision {rev!r}; this source does not know it")
    return rev


def is_at_or_after(current: str | None, target: str) -> bool:
    """True when ``current`` is ``target`` or descends from it in the revision graph."""
    if current is None:
        return False
    if not _known(target):
        raise AlembicStateError(f"unknown target revision {target!r}")
    return any(r.revision == target for r in _script().iterate_revisions(current, "base"))


def owns(engine: Engine | None, target: str) -> bool:
    """True when a revision at or before the database's state owns ``target``'s DDL.

    False for unstamped databases, so their runtime bootstrap keeps running.
    Raises ``AlembicStateError`` on an invalid state rather than guessing.
    """
    if engine is None or engine.dialect.name != "postgresql":
        return False
    with _cache_lock:
        cached = _upgraded.get(engine)
    if cached is not None:
        return is_at_or_after(cached, target)
    with engine.connect() as conn:
        return is_at_or_after(read_revision(conn), target)


def upgrade_on_startup(engine: Engine | None) -> str | None:
    """Upgrade a stamped database to head; leave an unstamped one alone.

    Returns the resulting revision, or None in compatibility mode. Raises on an
    invalid state or a failed migration so the app does not serve on a
    half-applied schema; the transaction (and the lock) roll back together.
    """
    if engine is None or engine.dialect.name != "postgresql":
        return None
    with engine.connect() as conn, conn.begin():
        conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(:n))"), {"n": _ADVISORY_LOCK_NAME})
        current = read_revision(conn)
        if current is None:
            logger.info("schema: no %s table; runtime bootstrap stays authoritative", VERSION_TABLE)
            return None
        from alembic import command  # only stamped databases need Alembic

        heads = _script().get_heads()
        if len(heads) != 1:
            raise AlembicStateError(f"source has {len(heads)} Alembic heads; expected one")
        if current != heads[0]:
            logger.info("schema: upgrading %s -> %s", current, heads[0])
            command.upgrade(alembic_config(conn), "head")
        result = read_revision(conn)
    with _cache_lock:
        _upgraded[engine] = result
    logger.info("schema: database at Alembic revision %s", result)
    return result


def _reset_cache_for_tests() -> None:
    with _cache_lock:
        _upgraded.clear()
