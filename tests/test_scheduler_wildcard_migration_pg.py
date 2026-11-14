"""TASK-2258 on PostgreSQL: legacy per-bot jobs with runs migrate at startup."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine

from llm_bawt.utils import db
from tests.pg_isolation import require_test_db_config
from tests.private_postgres import private_app_pg  # noqa: F401  (pytest fixture)
from tests.test_scheduler_wildcard_migration import run_migration_scenario

pytestmark = pytest.mark.integration


@pytest.fixture
def engine(private_app_pg):  # noqa: F811
    engine = create_engine(db.build_postgres_url(require_test_db_config()))
    try:
        yield engine
    finally:
        engine.dispose()


def test_legacy_jobs_with_runs_migrate_on_postgres(engine, monkeypatch):
    suffix = uuid.uuid4().hex[:8]
    run_migration_scenario(engine, monkeypatch, [f"alpha-{suffix}", f"beta-{suffix}"])
