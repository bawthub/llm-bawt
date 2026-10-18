"""Fail closed before constructing stores for dedicated PostgreSQL tests."""

import os

import pytest
from sqlalchemy import text

from llm_bawt.utils.config import Config
from llm_bawt.utils.db import build_engine_from_config


def require_test_db_config() -> Config:
    """Assert explicit config AND actual connection identity before any bootstrap."""
    config = Config()
    database = config.POSTGRES_DATABASE
    if (not isinstance(database, str) or not database.endswith("_test")
            or os.getenv("LLM_BAWT_POSTGRES_DATABASE") != database):
        pytest.fail(
            f"refusing to run integration DB test against operational database {database!r}; "
            "set LLM_BAWT_POSTGRES_DATABASE=<x>_test explicitly"
        )
    engine = build_engine_from_config(config, application_name="pytest-test-db-identity")
    if engine is None:
        pytest.fail("isolated PostgreSQL credentials are unavailable")
    try:
        with engine.connect() as conn:
            actual = conn.execute(text("SELECT current_database()")).scalar_one()
            if actual != database:
                pytest.fail("PostgreSQL connection did not select the explicit test database")
    finally:
        engine.dispose()
    return config


def isolated_pg_store(store_type):
    """Enforce the guard before the store constructor can acquire an engine."""
    return store_type(require_test_db_config())
