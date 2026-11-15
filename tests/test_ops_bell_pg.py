"""TASK-1183: ops bell staging and leased claims on real PostgreSQL."""

from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy import text

from llm_bawt.ops import JOB_SUCCEEDED
from llm_bawt.ops import bell
from llm_bawt.ops.store import OpsStore

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def test_stage_claim_settle_on_postgresql(private_app_pg):
    store = OpsStore(private_app_pg)
    slug = f"test.bell-{uuid.uuid4().hex[:10]}"
    op = store.create_operation(dict(
        slug=slug, title="Bell check", description="", enabled=True, target_host="",
        command_script='{"action":"restart","container_name":"test"}',
        args_schema_json='{"type":"object","additionalProperties":false,"properties":{}}',
        args_defaults_json="{}", timeout_seconds=60, start_delay_seconds=0,
        risk_level="low", category="restart", notify_bell=True))
    job = store.create_job(operation=op, args_json="{}", display_args_json="{}",
                           idempotency_key=f"bell-{uuid.uuid4().hex}", caller_user_id="test-user")
    with store.engine.begin() as conn:
        conn.execute(text("UPDATE ops_jobs SET invocation_snapshot_json=:s WHERE id=:id"),
                     {"s": json.dumps({"operation": op.to_api()}), "id": job.id})
    store.mark_dispatching(job.id, host_unit_name="u")
    store.mark_running(job.id)
    store.mark_running(job.id)
    store.mark_terminal(job.id, state=JOB_SUCCEEDED, exit_code=0)
    with store.engine.connect() as conn:
        keys = [r[0] for r in conn.execute(text(
            "SELECT event_key FROM ops_bell_outbox WHERE job_id=:id ORDER BY id"), {"id": job.id})]
    assert keys == [f"{job.id}:started", f"{job.id}:finished"]

    mine = lambda rows: [r for r in rows if r["job_id"] == job.id]  # noqa: E731 - shared DB
    first = mine(bell.claim_due(store.engine, limit=100))
    assert [r["phase"] for r in first] == ["started"]
    assert mine(bell.claim_due(store.engine, limit=100)) == []
    bell.settle(store.engine, first[0]["id"], ok=True, attempts=first[0]["attempts"])
    second = mine(bell.claim_due(store.engine, limit=100))
    assert [r["phase"] for r in second] == ["finished"]
    assert second[0]["payload"]["severity"] == "info"
