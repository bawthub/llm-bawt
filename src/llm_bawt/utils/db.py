"""Database utilities shared across the application.

This module centralises SQLAlchemy engine creation so the connection
pool budget is bounded across the whole service.

The rule
--------
**There is one engine per process.** Every Store class
(``BotProfileStore``, ``RuntimeSettingsStore``, ``ProfileManager``,
``TurnLogManager``, ``ModelDefinitionStore``, ``PromptTemplateStore``,
``AnimationStore``, ``MediaGenerationStore``, ``PostgreSQLMemoryBackend``,
…) connects to the same Postgres database with the same credentials.
There is no technical reason for any of them to own a separate
``Engine`` — an engine is just a connection pool, and pools that aren't
shared are pools that compete for the same fixed budget of Postgres
slots.

So: every Store fetches its engine via :func:`get_shared_engine`. The
first call builds the engine; everyone after re-uses it. Stores still
own their own table definitions, queries, and schema — just not their
pool.

Why this matters (TASK-202)
---------------------------
Before this refactor llm-bawt had ~10 Stores in the main process (plus
one PER BOT for memory backends), each constructing its own
``create_engine`` with default ``pool_size=5, max_overflow=10``. With:

    max_postgres_conns_used
        = (pool_size + max_overflow)     # per engine
        × engines_per_process
        × processes_per_container        # app runs 2: llm-service + mcp
        × container_count                # 3 (app + 2 bridges)

…the theoretical ceiling went past 900 against Postgres
``max_connections=100`` (shared with other tenants). Observed
steady-state hit 95/100 and starved every other service.

Sharing one engine collapses the formula to roughly:

    (pool_size + max_overflow) × processes_per_container × container_count

…which keeps the budget under control regardless of how many Store
classes we add.

When NOT to share
-----------------
A separate engine is justified only when the connection settings
genuinely differ — e.g. a different database, a different role, or a
test fixture pointing at SQLite in-memory. Otherwise: share.
"""

from __future__ import annotations

import logging
import threading
from typing import Any
from urllib.parse import quote_plus

from sqlalchemy import event
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


# Defaults for the SHARED process-wide engine.
#
# Because every Store in the process shares this single engine, the pool
# needs to absorb concurrent activity from *all* of them simultaneously —
# unlike the old per-store engines, which were each sized for one
# subsystem. 5 idle + 10 burst = 15 max per process is enough to handle
# overlapping chat streams, memory writes, and admin endpoint hits while
# staying well under postgres ``max_connections=100`` shared across the
# llm-service + mcp-server processes and the two bridge containers.
DEFAULT_POOL_SIZE = 5
DEFAULT_MAX_OVERFLOW = 10
# Recycle connections every 30 minutes so dropped/stale conns are replaced
# instead of held forever (Postgres restarts / NAT timeouts otherwise leave
# zombies in the pool).
DEFAULT_POOL_RECYCLE_SECONDS = 1800
# Deadman controls for lock regressions.  A request must fail and release its
# connection instead of silently wedging the shared pool/event loop forever.
DEFAULT_LOCK_TIMEOUT = "5s"
DEFAULT_STATEMENT_TIMEOUT = "120s"
DEFAULT_IDLE_TRANSACTION_TIMEOUT = "30s"


def _configure_dbapi_connection(dbapi_connection) -> None:
    """Apply the per-session safety policy to a raw DB-API connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("SET timezone = 'UTC'")
    cursor.execute(f"SET lock_timeout = '{DEFAULT_LOCK_TIMEOUT}'")
    cursor.execute(f"SET statement_timeout = '{DEFAULT_STATEMENT_TIMEOUT}'")
    cursor.execute(
        "SET idle_in_transaction_session_timeout = "
        f"'{DEFAULT_IDLE_TRANSACTION_TIMEOUT}'"
    )
    cursor.close()


def set_utc_on_connect(engine: Engine) -> None:
    """Register safe session defaults on every new connection.

    This prevents naive-datetime misinterpretation when the container's
    ``TZ`` env var is not UTC (e.g. ``America/New_York``).  The bounded lock,
    statement, and idle-transaction timeouts are a second line of defense: a
    future transaction-boundary bug fails closed instead of freezing all MCP
    tools behind an unbounded PostgreSQL wait.
    """

    @event.listens_for(engine, "connect")
    def _set_timezone(dbapi_connection, connection_record):
        _configure_dbapi_connection(dbapi_connection)


def is_postgresql(bind: Any) -> bool:
    """Whether ``bind`` (an engine or a connection) is PostgreSQL; False for None."""
    return getattr(getattr(bind, "dialect", None), "name", None) == "postgresql"


def build_postgres_url(config: Any) -> str | None:
    """Assemble a SQLAlchemy postgres URL from a config object, or None."""
    host = getattr(config, "POSTGRES_HOST", None) or "localhost"
    port = int(getattr(config, "POSTGRES_PORT", 5432) or 5432)
    user = getattr(config, "POSTGRES_USER", None)
    password = getattr(config, "POSTGRES_PASSWORD", "") or ""
    database = getattr(config, "POSTGRES_DATABASE", None)
    if not user or not database:
        return None
    return f"postgresql+psycopg2://{user}:{quote_plus(password)}@{host}:{port}/{database}"


def build_engine(
    url: str,
    *,
    application_name: str | None = None,
    pool_size: int = DEFAULT_POOL_SIZE,
    max_overflow: int = DEFAULT_MAX_OVERFLOW,
    pool_pre_ping: bool = True,
    pool_recycle: int = DEFAULT_POOL_RECYCLE_SECONDS,
    echo: bool = False,
) -> Engine:
    """Create a SQLAlchemy engine with conservative pool settings.

    All llm-bawt code that talks to the shared askllm Postgres database
    SHOULD go through this helper instead of calling ``sqlalchemy.create_engine``
    directly. The helper:

    - Bounds ``pool_size`` / ``max_overflow`` to small values so many engines
      can coexist within Postgres ``max_connections``.
    - Enables ``pool_pre_ping`` so dead connections are replaced instead of
      raising mid-query (and, more importantly for the leak case, so the
      pool will eventually close stale conns rather than hoard them).
    - Sets ``pool_recycle`` so long-idle connections rotate.
    - Tags the connection with ``application_name`` so ``pg_stat_activity``
      reveals which subsystem opened each connection. This is the
      diagnostic hook we wished we'd had during TASK-202.
    - Forces UTC and bounded lock/statement/idle-transaction timeouts on each
      new connection.

    Args:
        url: SQLAlchemy URL.
        application_name: Short label that shows up in ``pg_stat_activity``
            (e.g. ``"llm-bawt:profiles"``, ``"llm-bawt:memory:nova"``).
            Strongly recommended.
        pool_size: Steady-state pool size. **Default 2** — do not bump
            without auditing total engine count first.
        max_overflow: Burst slots above ``pool_size``. **Default 3**.
        pool_pre_ping: Validate connections before checkout. Default True.
        pool_recycle: Seconds before a connection is recycled. Default 1800.
        echo: SQLAlchemy SQL echo (debug only).

    Returns:
        Engine with UTC timezone and the connection settings above.
    """
    from sqlalchemy import create_engine

    connect_args: dict[str, Any] = {}
    if application_name:
        # psycopg2 forwards this to Postgres so it appears in pg_stat_activity.
        connect_args["application_name"] = application_name[:63]  # PG cap

    engine = create_engine(
        url,
        echo=echo,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=pool_pre_ping,
        pool_recycle=pool_recycle,
        connect_args=connect_args,
    )
    set_utc_on_connect(engine)
    return engine


def build_engine_from_config(
    config: Any,
    *,
    application_name: str | None = None,
    pool_size: int = DEFAULT_POOL_SIZE,
    max_overflow: int = DEFAULT_MAX_OVERFLOW,
    pool_pre_ping: bool = True,
    pool_recycle: int = DEFAULT_POOL_RECYCLE_SECONDS,
    echo: bool = False,
) -> Engine | None:
    """Convenience wrapper: build an engine from a config object.

    Returns ``None`` if the config has no Postgres credentials. The caller
    is expected to handle the missing-DB case (most Stores already do).

    NOTE: most callers should use :func:`get_shared_engine` instead — this
    helper exists for cases that genuinely need an isolated engine (tests,
    one-shot migration scripts, etc.).
    """
    url = build_postgres_url(config)
    if not url:
        return None
    return build_engine(
        url,
        application_name=application_name,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=pool_pre_ping,
        pool_recycle=pool_recycle,
        echo=echo,
    )


# ---------------------------------------------------------------------------
# Process-wide shared engine (TASK-202)
# ---------------------------------------------------------------------------

_SHARED_ENGINE_LOCK = threading.Lock()
_SHARED_ENGINE_CACHE: dict[str, Engine] = {}


def get_shared_engine(
    config: Any,
    *,
    application_name: str = "llm-bawt",
) -> Engine | None:
    """Return THE process-wide engine for llm-bawt's Postgres database.

    Every Store class should use this. The engine is built lazily on the
    first call and cached for the lifetime of the process, keyed on the
    connection URL (so tests pointing at a different database get a fresh
    engine without invalidating production callers).

    Returns ``None`` if Postgres credentials aren't configured. Stores
    that handle a missing DB gracefully should propagate the ``None``
    rather than raising.

    DO NOT call ``.dispose()`` on the returned engine — it is shared.
    Disposing it tears down the pool for every Store in the process.
    """
    url = build_postgres_url(config)
    if not url:
        return None
    cached = _SHARED_ENGINE_CACHE.get(url)
    if cached is not None:
        return cached
    with _SHARED_ENGINE_LOCK:
        cached = _SHARED_ENGINE_CACHE.get(url)
        if cached is not None:
            return cached
        engine = build_engine(url, application_name=application_name)
        _SHARED_ENGINE_CACHE[url] = engine
        logger.info(
            "Built shared SQLAlchemy engine (pool_size=%d, max_overflow=%d, app=%s)",
            DEFAULT_POOL_SIZE,
            DEFAULT_MAX_OVERFLOW,
            application_name,
        )
        return engine


def reset_shared_engines() -> None:
    """Dispose all shared engines. Test/shutdown helper only.

    Production code should never call this — it severs every Store's
    pool simultaneously.
    """
    with _SHARED_ENGINE_LOCK:
        for engine in list(_SHARED_ENGINE_CACHE.values()):
            try:
                engine.dispose()
            except Exception:
                pass
        _SHARED_ENGINE_CACHE.clear()
