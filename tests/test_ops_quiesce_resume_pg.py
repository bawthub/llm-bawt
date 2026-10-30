"""TASK-1798: a quiesce job's end resumes the bots it held, exactly once.

Uses the TASK-1233 quiesce rig: a real gate opened by the driver over a real
turn row, so the held set is exactly what production records.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pytest

from llm_bawt.ops import JOB_ACCEPTED, JOB_FAILED, JOB_SUCCEEDED
from llm_bawt.ops.executor import ReconcileResult
from llm_bawt.ops.quiesce import record_ack
from llm_bawt.ops.quiesce_prompts import DEFAULT_RESUME, DEFAULT_RESUME_BLOCKED
from llm_bawt.ops.resume_sender import QuiesceResumeSender
from tests.test_ops_quiesce_pg import _agent_bot, _receipt_deliveries, rig  # noqa: F401

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def resume(rig):  # noqa: F811
    capability = {"value": None}
    sender = QuiesceResumeSender(rig.service, rig.deliveries,
                                 coordinator_capability=lambda: capability["value"])
    sender.capability = capability
    yield sender
    rig.sql("DELETE FROM ops_resume_notices WHERE job_id IN"
            " (SELECT id FROM ops_jobs WHERE operation_slug=:s)", s=rig.slug)


def _held_and_stopping(rig, note="mid TASK-9: step 3 of 5"):  # noqa: F811
    turn_id = rig.turn()
    job_id = rig.job()
    assert rig.driver.step(job_id).outcome == "opened"
    assert record_ack(rig.engine, bot_id=rig.bot, turn_id=turn_id, status="stopping", note=note)
    return turn_id, job_id


def _dispatch(rig, job_id):  # noqa: F811
    """Advance the job through its gate until the worker has accepted it."""
    for _ in range(3):
        if rig.service.get_job_status(job_id)["state"] == JOB_ACCEPTED:
            return
    raise AssertionError(f"job {job_id} was never accepted")


def _finish(rig, job_id, state=JOB_SUCCEEDED):  # noqa: F811
    """Reconcile ``job_id`` to ``state``, finishing now (inside the recent window)."""
    now = datetime.now(timezone.utc).isoformat()
    rig.executor.next_reconcile = ReconcileResult(
        state=state, exit_code=0 if state == JOB_SUCCEEDED else 1, output_tail="done\n",
        started_at=now, finished_at=now)
    assert rig.service.get_job_status(job_id)["state"] == state


def _resumes(rig, bot=None):  # noqa: F811
    return [d for d in _receipt_deliveries(rig, bot) if d["message_kind"] == "OPS_RESUME"]


def _facts(delivery):
    return json.loads(delivery["message"].split("\n", 1)[1])


def test_a_succeeded_job_resumes_each_held_bot_once_with_its_note(rig, resume):  # noqa: F811
    turn_id, job_id = _held_and_stopping(rig)
    rig.end(turn_id)
    _dispatch(rig, job_id)  # accepted, still running
    assert resume.deliver_due() == []  # nothing before the job is terminal
    _finish(rig, job_id)

    assert resume.deliver_due() == [job_id]
    [delivery] = _resumes(rig)
    assert delivery["sender_bot_id"] == "system"
    assert delivery["message"].startswith(DEFAULT_RESUME)
    facts = _facts(delivery)
    assert (facts["job_id"], facts["state"], facts["ack_status"]) == (job_id, JOB_SUCCEEDED, "stopping")
    assert facts["ack_note"] == "mid TASK-9: step 3 of 5"
    payload = delivery["payload_json"]
    assert (payload["bot_id"], payload["user"], payload["prefer_steer"]) == (rig.bot, "nick", False)

    # No duplicate on retry: the done marker stops the scan, and a crash before
    # it re-reads the same idempotent delivery.
    assert resume.deliver_due() == []
    rig.sql("DELETE FROM ops_resume_notices WHERE job_id=:j", j=job_id)
    assert resume.deliver(job_id) == [delivery["id"]]
    assert len(_resumes(rig)) == 1


def test_bots_idle_at_quiesce_get_nothing(rig, resume):  # noqa: F811
    job_id = rig.job()  # no running turn in scope: the gate has no members
    _dispatch(rig, job_id)
    _finish(rig, job_id)
    assert resume.deliver_due() == []
    assert _receipt_deliveries(rig) == []


def test_a_job_that_never_ran_still_resumes(rig, resume):  # noqa: F811
    """A quiesce timeout touched nothing, so the service is up: resume."""
    _turn, job_id = _held_and_stopping(rig)
    rig.sql("UPDATE ops_jobs SET quiesce_deadline_at = now() - interval '1 second' WHERE id=:id",
            id=job_id)
    assert rig.driver.step(job_id).outcome == "failed"
    assert rig.row(job_id)["dispatched_at"] is None
    assert resume.deliver_due() == [job_id]
    [delivery] = _resumes(rig)
    assert _facts(delivery)["state"] == JOB_FAILED


def test_a_failed_job_routes_to_the_coordinator_not_the_held_bot(rig, resume):  # noqa: F811
    tag = uuid.uuid4().hex[:10]
    role, capability = f"test_coord_{tag}", f"test_coordinate_{tag}"
    coordinator = f"{rig.bot}-coord"
    _agent_bot(rig, coordinator)
    rig.sql("INSERT INTO bot_tag_catalog (slug, name, kind, capabilities)"
            " VALUES (:s, :s, 'role', CAST(:c AS jsonb))", s=role, c=json.dumps([capability]))
    rig.sql("INSERT INTO bot_profile_tags (bot_slug, tag_slug) VALUES (:b, :t)", b=coordinator, t=role)
    try:
        turn_id, job_id = _held_and_stopping(rig)
        rig.end(turn_id)
        _dispatch(rig, job_id)
        _finish(rig, job_id, JOB_FAILED)
        assert rig.row(job_id)["dispatched_at"] is not None

        resume.capability["value"] = capability
        assert resume.deliver_due() == [job_id]
        assert _resumes(rig) == []  # the held bot is not told to resume
        [notice] = _receipt_deliveries(rig, coordinator)
        assert notice["message_kind"] == "OPS_RESUME_BLOCKED"
        assert notice["message"].startswith(DEFAULT_RESUME_BLOCKED)
        facts = _facts(notice)
        assert facts["job_id"] == job_id and facts["state"] == JOB_FAILED
        assert [m["bot_id"] for m in facts["held_bots"]] == [rig.bot]
        assert facts["held_bots"][0]["ack_note"] == "mid TASK-9: step 3 of 5"

        assert resume.deliver_due() == []
        rig.sql("DELETE FROM ops_resume_notices WHERE job_id=:j", j=job_id)
        resume.deliver(job_id)
        assert len(_receipt_deliveries(rig, coordinator)) == 1
    finally:
        rig.sql("DELETE FROM bot_profile_tags WHERE tag_slug=:t", t=role)
        rig.sql("DELETE FROM bot_tag_catalog WHERE slug=:t", t=role)
        rig.sql("DELETE FROM bot_profiles WHERE slug=:s", s=coordinator)


def test_the_caller_that_gets_the_receipt_is_not_also_resumed(rig, resume):  # noqa: F811
    turn_id, job_id = _held_and_stopping(rig)
    rig.sql("UPDATE ops_jobs SET receipt_callback_json=:j WHERE id=:id", id=job_id,
            j=json.dumps({"bot_id": rig.bot, "user_id": "nick", "session_id": "s"}))
    rig.end(turn_id)
    _dispatch(rig, job_id)
    _finish(rig, job_id)
    assert resume.deliver_due() == [job_id]
    assert _resumes(rig) == []
