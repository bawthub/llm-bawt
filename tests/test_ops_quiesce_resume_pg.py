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
from llm_bawt.ops.quiesce import AckNoteRequired, record_ack
from llm_bawt.ops.quiesce_prompts import DEFAULT_RESUME, DEFAULT_RESUME_BLOCKED
from llm_bawt.ops.resume_sender import EPOCH, QuiesceResumeSender
from tests.test_ops_quiesce_pg import _agent_bot, _receipt_deliveries, rig  # noqa: F401

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def resume(rig):  # noqa: F811
    capability = {"value": None}
    sender = QuiesceResumeSender(rig.service, rig.deliveries,
                                 coordinator_capability=lambda: capability["value"])
    sender.capability = capability
    # The sender's first-run epoch: an hour back, so jobs these tests finish count.
    rig.sql("INSERT INTO ops_resume_notices (job_id, bot_id, kind, created_at)"
            " VALUES (:e, :e, 'done', now() - interval '1 hour')"
            " ON CONFLICT (job_id, bot_id, kind) DO UPDATE SET created_at = EXCLUDED.created_at",
            e=EPOCH)
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


def _member(rig, turn_id, **cols):  # noqa: F811
    sets = ", ".join(f"{k}=:{k}" for k in cols)
    rig.sql(f"UPDATE gate_members SET {sets} WHERE turn_id=:t", t=turn_id, **cols)


def test_a_turn_that_ended_on_its_own_is_not_resumed(rig, resume):  # noqa: F811
    """not_needed / turn_not_running: the job never stopped it (e.g. the approver)."""
    turn_id = rig.turn()
    job_id = rig.job()
    assert rig.driver.step(job_id).outcome == "opened"
    _member(rig, turn_id, control_state="not_needed")
    rig.end(turn_id)
    _dispatch(rig, job_id)
    _finish(rig, job_id)
    assert resume.deliver_due() == [job_id]
    assert _resumes(rig) == []


def test_an_unstarted_turn_the_hold_requeued_is_not_resumed(rig, resume):  # noqa: F811
    """revoked_unstarted gets its own work back; a resume would duplicate it."""
    turn_id = rig.turn()
    job_id = rig.job()
    assert rig.driver.step(job_id).outcome == "opened"
    _member(rig, turn_id, disposition="revoked_unstarted", control_state="not_needed")
    rig.end(turn_id)
    _dispatch(rig, job_id)
    _finish(rig, job_id)
    assert resume.deliver_due() == [job_id]
    assert _resumes(rig) == []


def test_a_cancelled_member_without_an_ack_is_resumed(rig, resume):  # noqa: F811
    turn_id = rig.turn()
    job_id = rig.job()
    assert rig.driver.step(job_id).outcome == "opened"
    _member(rig, turn_id, control_state="cancelled")
    rig.end(turn_id)
    _dispatch(rig, job_id)
    _finish(rig, job_id)
    assert resume.deliver_due() == [job_id]
    [delivery] = _resumes(rig)
    assert _facts(delivery)["ack_status"] is None


def test_jobs_finished_before_the_sender_first_ran_are_skipped(rig, resume):  # noqa: F811
    turn_id, job_id = _held_and_stopping(rig)
    rig.end(turn_id)
    _dispatch(rig, job_id)
    _finish(rig, job_id)
    rig.sql("UPDATE ops_resume_notices SET created_at = now() WHERE job_id=:e AND bot_id=:e", e=EPOCH)
    assert resume.deliver_due() == []
    assert _resumes(rig) == []


# TASK-2121: a "finished" ack (nothing left to resume, note names what wakes
# the bot next) gets no resume; "stopping" and no-ack bots still do.

def _held_and_finished(rig, note="none: task closed"):  # noqa: F811
    turn_id = rig.turn()
    job_id = rig.job()
    assert rig.driver.step(job_id).outcome == "opened"
    _member(rig, turn_id, control_state="cancelled")  # the steer stopped it
    assert record_ack(rig.engine, bot_id=rig.bot, turn_id=turn_id, status="finished", note=note)
    return turn_id, job_id


def test_a_finished_bot_is_not_resumed(rig, resume):  # noqa: F811
    turn_id, job_id = _held_and_finished(rig)
    rig.end(turn_id)
    _dispatch(rig, job_id)
    _finish(rig, job_id)
    assert resume.deliver_due() == [job_id]
    assert _resumes(rig) == []
    assert resume.deliver_due() == []  # done marker written: no later resume either


def test_a_finished_ack_needs_a_note(rig, resume):  # noqa: F811
    turn_id = rig.turn()
    job_id = rig.job()
    assert rig.driver.step(job_id).outcome == "opened"
    for blank in (None, "", "   "):
        with pytest.raises(AckNoteRequired):
            record_ack(rig.engine, bot_id=rig.bot, turn_id=turn_id, status="finished", note=blank)
    assert rig.sql("SELECT ack_status FROM gate_members WHERE turn_id=:t", t=turn_id).scalar() is None


def test_a_later_stopping_ack_overrides_finished(rig, resume):  # noqa: F811
    """The latest ack wins: a bot that found more work and re-acked is resumed."""
    turn_id, job_id = _held_and_finished(rig)
    assert record_ack(rig.engine, bot_id=rig.bot, turn_id=turn_id, status="stopping",
                      note="found one more step")
    rig.end(turn_id)
    _dispatch(rig, job_id)
    _finish(rig, job_id)
    assert resume.deliver_due() == [job_id]
    [delivery] = _resumes(rig)
    assert _facts(delivery)["ack_note"] == "found one more step"


def test_a_failed_job_still_names_a_finished_bot_to_the_coordinator(rig, resume):  # noqa: F811
    """The OPS_RESUME_BLOCKED path is unchanged: coordinators see every held bot."""
    tag = uuid.uuid4().hex[:10]
    role, capability = f"test_coord_{tag}", f"test_coordinate_{tag}"
    coordinator = f"{rig.bot}-coord"
    _agent_bot(rig, coordinator)
    rig.sql("INSERT INTO bot_tag_catalog (slug, name, kind, capabilities)"
            " VALUES (:s, :s, 'role', CAST(:c AS jsonb))", s=role, c=json.dumps([capability]))
    rig.sql("INSERT INTO bot_profile_tags (bot_slug, tag_slug) VALUES (:b, :t)", b=coordinator, t=role)
    try:
        turn_id, job_id = _held_and_finished(rig, note="gate receipt")
        rig.end(turn_id)
        _dispatch(rig, job_id)
        _finish(rig, job_id, JOB_FAILED)
        resume.capability["value"] = capability
        assert resume.deliver_due() == [job_id]
        assert _resumes(rig) == []
        [notice] = _receipt_deliveries(rig, coordinator)
        [held] = _facts(notice)["held_bots"]
        assert (held["bot_id"], held["ack_status"], held["ack_note"]) == (
            rig.bot, "finished", "gate receipt")
    finally:
        rig.sql("DELETE FROM bot_profile_tags WHERE tag_slug=:t", t=role)
        rig.sql("DELETE FROM bot_tag_catalog WHERE slug=:t", t=role)
        rig.sql("DELETE FROM bot_profiles WHERE slug=:s", s=coordinator)
