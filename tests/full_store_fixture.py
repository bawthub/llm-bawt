"""A disposable, independently named database for full-store schema tests.

Uses the acceptance fixture's networkless pgvector cluster and Unix socket;
NEVER operational PostgreSQL. Import ``full_store_db`` into a test module.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine, text

from tests.pg_isolation import require_test_db_config
from tests.private_postgres import PORT, private_url


@pytest.fixture
def full_store_db(private_pg, monkeypatch):
    from llm_bawt.utils import db

    name = f"schema_full_{uuid.uuid4().hex[:12]}_test"
    admin = create_engine(private_url(private_pg, "postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        assert conn.execute(text("SELECT current_database()")).scalar_one() == "postgres"
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    url = private_url(private_pg, name)
    monkeypatch.setattr(db, "build_postgres_url", lambda config: url)
    monkeypatch.setenv("LLM_BAWT_POSTGRES_HOST", str(private_pg))
    monkeypatch.setenv("LLM_BAWT_POSTGRES_PORT", PORT)
    monkeypatch.setenv("LLM_BAWT_POSTGRES_USER", "test_user")
    monkeypatch.setenv("LLM_BAWT_POSTGRES_DATABASE", name)
    monkeypatch.setenv("LLM_BAWT_POSTGRES_PASSWORD", uuid.uuid4().hex)
    engine = create_engine(url)
    try:
        config = require_test_db_config()
        assert config.POSTGRES_DATABASE == engine.url.database == name
        with engine.connect() as conn:
            assert conn.execute(text("SELECT current_database()")).scalar_one() == name
        yield config, engine
    finally:
        engine.dispose()
        db.reset_shared_engines()
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()
