"""Alembic environment for llm-bawt (TASK-631).

Slice 1 adopts Alembic as a version marker only. ``target_metadata`` is None,
so autogenerate is deliberately unavailable: the SQLModel metadata covers only
part of the live schema, and diffing it would propose dropping real tables.

Online mode builds a plain engine from the application's Config through
``utils.db``. It must never construct a store or call any runtime ``ensure_*``
bootstrap; runtime DDL stays authoritative until later slices retire it.
Offline mode (``--sql``) needs no database connection at all.
"""

from __future__ import annotations

import os

from alembic import context

config = context.config
target_metadata = None


def _database_url() -> str:
    explicit = os.environ.get("LLM_BAWT_ALEMBIC_URL")
    if explicit:
        return explicit
    from llm_bawt.utils.config import Config
    from llm_bawt.utils.db import build_postgres_url

    url = build_postgres_url(Config())
    if not url:
        raise RuntimeError(
            "No PostgreSQL settings found (POSTGRES_USER/POSTGRES_DATABASE); "
            "set them or LLM_BAWT_ALEMBIC_URL."
        )
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=os.environ.get("LLM_BAWT_ALEMBIC_URL", "postgresql+psycopg2://offline/offline"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    supplied = config.attributes.get("connection")
    if supplied is not None:
        # App startup (utils.alembic_state): the caller owns the connection, its
        # transaction and the schema advisory lock. Never build a second engine or
        # honor LLM_BAWT_ALEMBIC_URL here, so lock, check and DDL hit one database.
        context.configure(connection=supplied, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
        return

    from sqlalchemy import create_engine, pool

    engine = create_engine(_database_url(), poolclass=pool.NullPool)
    try:
        with engine.connect() as connection:
            context.configure(connection=connection, target_metadata=target_metadata)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
