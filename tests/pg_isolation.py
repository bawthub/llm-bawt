"""Fail closed before constructing stores for disposable PostgreSQL tests."""

import pytest

from llm_bawt.utils.config import Config


def require_test_db_config() -> Config:
    """Return a validated Config; never construct a store against the live DB."""
    config = Config()
    database = config.POSTGRES_DATABASE
    if not isinstance(database, str) or not database.endswith("_test"):
        pytest.fail(
            f"refusing to run integration DB test against operational database {database!r}; "
            "set LLM_BAWT_POSTGRES_DATABASE=<x>_test"
        )
    return config


def isolated_pg_store(store_type):
    """Enforce the guard before the store constructor can acquire an engine."""
    return store_type(require_test_db_config())
