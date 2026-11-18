"""TASK-1489: the operator freeze/thaw ops executor on real PostgreSQL.

Every freeze here is scoped to this test's own bot, so parallel PG tests are
never frozen by it.
"""

from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy import text as sa_text

from llm_bawt.ops.freeze_executor import FreezeExecutor, job_actor
from llm_bawt.ops.models import JOB_FAILED, JOB_SUCCEEDED
from llm_bawt.service import operator_freeze, turn_gates
from llm_bawt.service.turn_gates import GateScope
from tests.pg_isolation import isolated_pg_store

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def rig():
    from llm_bawt.runtime_settings import BotProfileStore
    from llm_bawt.service.turn_logs import TurnLogStore

    store = isolated_pg_store(TurnLogStore)
    if store.engine is None:
        pytest.skip("PostgreSQL unavailable")
    assert BotProfileStore(store.config).engine is not None  # creates bot_profiles
    tag = uuid.uuid4().hex[:12]

    class Rig:
        engine = store.engine
        scope = GateScope.for_bots(f"test-opsfreeze-{tag}")

        def executor(self):
            return FreezeExecutor(self.engine, scope=self.scope)

        def job(self):
            return f"{tag}-{uuid.uuid4().hex[:8]}"

        def owned(self, job_id):
            with self.engine.connect() as conn:
                return conn.execute(sa_text("SELECT count(*) FROM system_gates WHERE owner_id=:o"),
                                    {"o": "freeze:" + job_actor(job_id)}).scalar()

    rig = Rig()
    yield rig
    with rig.engine.begin() as conn:
        owned = "SELECT id FROM system_gates WHERE owner_id LIKE :o"
        conn.execute(sa_text(f"DELETE FROM gate_members WHERE gate_id IN ({owned})"), {"o": f"%{tag}%"})
        conn.execute(sa_text("DELETE FROM system_gates WHERE owner_id LIKE :o"), {"o": f"%{tag}%"})


def _freeze_snapshot(**args):
    return {"spec": {"action": "operator_freeze", "op": "freeze"},
            "resolved_args": {"reason": "maintenance window", "ttl_seconds": 600, "admit_user": False, **args}}


def _thaw_snapshot(gate_id):
    return {"spec": {"action": "operator_freeze", "op": "thaw"}, "resolved_args": {"gate_id": gate_id}}


def _run(executor, job_id, snapshot):
    executor.dispatch(job_id=job_id, snapshot=snapshot)
    return executor.reconcile(job_id=job_id, snapshot=snapshot)


def _receipt(result):
    assert result.state == JOB_SUCCEEDED, result.error
    return json.loads(result.output_tail)


def test_freeze_opens_one_gate_owned_by_the_job_and_notifies(rig, monkeypatch):
    seen = []
    original = turn_gates.notify_gate_change
    monkeypatch.setattr(turn_gates, "notify_gate_change", lambda conn, gid: (seen.append(gid), original(conn, gid)))
    job = rig.job()
    receipt = _receipt(_run(rig.executor(), job, _freeze_snapshot(admit_user=True)))
    assert receipt["actor"] == job_actor(job) and receipt["live"] is True and receipt["admit_user"] is True
    assert seen == [receipt["gate_id"]]
    assert rig.owned(job) == 1


def test_freeze_crash_after_commit_recovers_the_same_gate(rig):
    job = rig.job()
    first = rig.executor()
    first.dispatch(job_id=job, snapshot=_freeze_snapshot())  # gate committed, then the process dies
    gate_id = operator_freeze.freeze(rig.engine, actor=job_actor(job), reason="x", scope=rig.scope,
                                     ttl_seconds=600, idempotent_owner=True).gate_id
    fresh = rig.executor()
    receipt = _receipt(_run(fresh, job, _freeze_snapshot()))
    assert receipt["gate_id"] == gate_id and rig.owned(job) == 1


def test_freeze_after_its_gate_was_thawed_never_reopens(rig):
    job = rig.job()
    gate_id = _receipt(_run(rig.executor(), job, _freeze_snapshot()))["gate_id"]
    operator_freeze.thaw(rig.engine, gate_id, actor=job_actor(job))
    receipt = _receipt(rig.executor().reconcile(job_id=job, snapshot=_freeze_snapshot()))
    assert receipt["gate_id"] == gate_id and receipt["live"] is False and rig.owned(job) == 1


def test_bad_freeze_args_are_a_terminal_failure(rig):
    result = _run(rig.executor(), rig.job(), _freeze_snapshot(ttl_seconds=10))
    assert result.state == JOB_FAILED and "ttl_seconds" in result.error


def test_thaw_releases_and_reports_true(rig):
    gate_id = _receipt(_run(rig.executor(), rig.job(), _freeze_snapshot()))["gate_id"]
    receipt = _receipt(_run(rig.executor(), rig.job(), _thaw_snapshot(gate_id)))
    assert receipt["released_by_this_job"] is True and receipt["live"] is False and "released_at" in receipt


def test_thaw_crash_before_release_recovers_and_reports_true(rig):
    """(a) The executor died before thawing; a fresh reconcile releases it."""
    gate_id = _receipt(_run(rig.executor(), rig.job(), _freeze_snapshot()))["gate_id"]
    receipt = _receipt(rig.executor().reconcile(job_id=rig.job(), snapshot=_thaw_snapshot(gate_id)))
    assert receipt["released_by_this_job"] is True
    assert operator_freeze.freeze_record(rig.engine, gate_id).live is False


def test_thaw_after_someone_else_released_reports_unknown(rig):
    """(b) Released by another party (or by this job before a crash)."""
    freeze_job = rig.job()
    gate_id = _receipt(_run(rig.executor(), freeze_job, _freeze_snapshot()))["gate_id"]
    operator_freeze.thaw(rig.engine, gate_id, actor=job_actor(freeze_job))
    receipt = _receipt(rig.executor().reconcile(job_id=rig.job(), snapshot=_thaw_snapshot(gate_id)))
    assert receipt["released_by_this_job"] == "unknown" and receipt["live"] is False


def test_thaw_never_reaches_a_ui_freeze_or_a_missing_gate(rig):
    """(c) A freeze by a person is not this operation's to thaw."""
    ui = operator_freeze.freeze(rig.engine, actor=f"person-{rig.job()}", reason="x", scope=rig.scope,
                                ttl_seconds=600)
    result = _run(rig.executor(), rig.job(), _thaw_snapshot(ui.gate_id))
    assert result.state == JOB_FAILED and "not opened by the freeze operation" in result.error
    assert operator_freeze.freeze_record(rig.engine, ui.gate_id).live is True
    missing = _run(rig.executor(), rig.job(), _thaw_snapshot(2**31 - 1))
    assert missing.state == JOB_FAILED and "not a freeze" in missing.error
