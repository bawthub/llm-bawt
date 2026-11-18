"""TASK-2470: latest harness versions come from committed successful runs only."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text as sa_text

from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

#: Far enough ahead that these runs are the newest in the private database.
FUTURE = datetime(2999, 1, 1, tzinfo=UTC)


@pytest.fixture
def engine():
    from llm_bawt.service.scheduler import create_scheduler_tables
    from llm_bawt.utils.db import get_shared_engine

    eng = get_shared_engine(require_test_db_config())
    create_scheduler_tables(eng)
    job_id = f"test-hv-{uuid.uuid4().hex[:10]}"
    with eng.begin() as conn:
        conn.execute(sa_text(
            "INSERT INTO scheduled_jobs (id, job_type, bot_id, enabled, interval_minutes) "
            "VALUES (:id, 'HARNESS_VERSION_CHECK', '*', false, 10080)"), {"id": job_id})
    eng.test_job_id = job_id
    yield eng
    with eng.begin() as conn:
        conn.execute(sa_text("DELETE FROM job_runs WHERE job_id = :id"), {"id": job_id})
        conn.execute(sa_text("DELETE FROM scheduled_jobs WHERE id = :id"), {"id": job_id})


def _run(eng, status, minutes, packages):
    with eng.begin() as conn:
        conn.execute(sa_text(
            "INSERT INTO job_runs (id, job_id, bot_id, status, started_at, finished_at, result_json) "
            "VALUES (:id, :job, '*', :status, :at, :at, :result)"),
            {"id": uuid.uuid4().hex, "job": eng.test_job_id, "status": status,
             "at": FUTURE + timedelta(minutes=minutes),
             "result": json.dumps({"packages": packages})})


def test_newest_success_wins_failed_and_pending_runs_are_ignored(engine):
    from llm_bawt.harness_versions import read_latest

    _run(engine, "SUCCESS", 1, [{"name": "claude-agent-sdk", "latest": "0.2.183",
                                  "latest_cli": "2.1.320"}])
    _run(engine, "SUCCESS", 2, [{"name": "openai-codex", "latest": "0.169.0"}])
    _run(engine, "FAILED", 3, [{"name": "openai-codex", "latest": "9.9.9"}])
    _run(engine, "RUNNING", 4, [{"name": "claude-agent-sdk", "latest": "9.9.9"}])

    latest = read_latest(engine)

    assert latest["packages"]["openai-codex"]["version"] == "0.169.0"
    assert latest["packages"]["claude-agent-sdk"]["version"] == "0.2.183"
    assert latest["claude_cli"]["version"] == "2.1.320"
