"""TASK-1994: isolated scheduled runs become bounded main-session receipts."""
import json
from datetime import timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import text
from sqlmodel import Session

from llm_bawt.chat_prompt_defaults import SCHEDULED_RUN_NOTICE
from llm_bawt.models.message import Message
from llm_bawt.prompt_definitions import DEFAULT_PROMPT_DEFINITIONS
from llm_bawt.service import scheduled_run_notice as notice
from llm_bawt.service.prompt_retention import prune_prompt_runs
from llm_bawt.service.prompt_schedule_models import PromptOccurrence
from tests.test_prompt_delivery import BOT, NOW, OWNER, enqueue, occurrence, reserve
from tests import test_prompt_delivery as delivery_tests

rig = delivery_tests.rig


@pytest.fixture(autouse=True)
def local_registry(monkeypatch):
    monkeypatch.setattr(notice, "render_notice", lambda bot, runs, remaining:
                        SCHEDULED_RUN_NOTICE.format(runs=json.dumps(runs), remaining=remaining))


def finish(rig, *, reply="Wake-up TASK-42: lantern count 17", status="ok", clear=True):
    job, run = reserve(rig, once=True, clear=clear)
    record = enqueue(rig, run)
    rig.turns[record.turn_id] = SimpleNamespace(
        ended_at=NOW, status=status, error_text=None, response_text=reply,
        assistant_message_id="assistant-" + run, model="test",
    )
    rig.delivery.update(record.id, status="DELIVERED")
    rig.worker._reconcile(run, NOW + timedelta(seconds=10))
    return job, run, record


def prepare(rig, *, sid="personal", bot=BOT, owner=OWNER, prompt="what was that?", request=None):
    return notice.ScheduledRunNotices.prepare(
        rig.engine, bot_id=bot, user_id=owner, session_id=sid, turn_id="main-turn",
        service=rig.service, request=request or SimpleNamespace(), user_prompt=prompt,
    )


@pytest.mark.parametrize("clear", [True, False])
def test_finished_run_main_notice_once_without_touching_system_or_history(rig, clear):
    job, run, record = finish(rig, clear=clear)
    stored = occurrence(rig, run)
    assert stored.main_notice_json["assistant_message_id"] == "assistant-" + run
    assert stored.main_notice_json["user_message_id"] == record.user_message_id
    assert stored.main_notice_json["schedule_id"] == job
    assert stored.main_notice_json["finished_at"] == NOW.isoformat()
    assert stored.session_id != "personal"
    messages = [Message(role="system", content="byte-stable"), Message(role="user", content="what was that?")]
    pending = prepare(rig)
    sent = pending.apply(messages)
    assert "lantern count 17" in sent[-1].content and run in sent[-1].content
    assert sent[0] is messages[0] and sent[0].content == "byte-stable"
    assert messages[-1].content == "what was that?"
    # Metadata/acceptance/cancellation are not model consumption evidence.
    for frame in [None, "", {"event": "metadata"}, {"event": "turn_cancelled"}, {"event": "steer_lifecycle"}]:
        pending.observe(frame)
        assert occurrence(rig, run).main_notice_turn_id is None
    pending.observe({"event": "tool_call"})
    assert occurrence(rig, run).main_notice_turn_id == "main-turn"
    assert prepare(rig).body is None
    # Reconciliation after completion does not create a duplicate receipt.
    rig.worker._reconcile(run, NOW + timedelta(seconds=20))
    assert prepare(rig).body is None


def test_pre_sdk_failure_leaves_receipt_pending_and_rotation_does_not_lose_it(rig):
    _, run, _ = finish(rig)
    pending = prepare(rig)
    assert pending.body
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE sessions SET status='archived' WHERE id='personal'"))
        conn.execute(text("INSERT INTO sessions(id,bot_id,user_id,status) VALUES ('new-main',:bot,:owner,'active')"),
                     {"bot": BOT, "owner": OWNER})
    assert prepare(rig).body is None
    retry = prepare(rig, sid="new-main")
    assert retry.run_ids == [run]
    retry.observe("I ran the lantern count.")
    assert prepare(rig, sid="new-main").body is None


@pytest.mark.parametrize("case", ["dedicated", "other-bot", "other-owner", "scheduler", "compact", "new", "clear"])
def test_notice_never_leaks_or_is_consumed_by_wrong_turn(rig, case):
    _, run, record = finish(rig)
    args = {
        "dedicated": {"sid": occurrence(rig, run).session_id},
        "other-bot": {"bot": "someone-else"}, "other-owner": {"owner": "someone-else"},
        "scheduler": {"request": SimpleNamespace(inter_bot_delivery_id=record.id)},
        "compact": {"prompt": "/compact"}, "new": {"prompt": "/new resume"}, "clear": {"prompt": "/clear"},
    }[case]
    blocked = prepare(rig, **args)
    assert blocked.body is None
    blocked.observe("model output")
    assert occurrence(rig, run).main_notice_turn_id is None
    assert prepare(rig).body


def test_reply_is_capped_and_batch_overflow_is_deferred_not_dropped(rig):
    for _ in range(notice.MAX_NOTICES + 2):
        finish(rig, reply="x" * 5000)
    first = prepare(rig)
    assert len(first.run_ids) == notice.MAX_NOTICES
    assert "Additional pending receipts: at least 1" in first.body
    assert "x" * (notice.REPLY_CHARS + 1) not in first.body
    first_ids = set(first.run_ids)
    first.observe("understood")
    second = prepare(rig)
    assert len(second.run_ids) == 2 and not first_ids & set(second.run_ids)
    second.observe("understood")
    assert prepare(rig).body is None


@pytest.mark.parametrize("status", ["error", "cancelled"])
def test_finished_failure_with_reply_is_noticed(rig, status):
    _, run, _ = finish(rig, status=status)
    assert occurrence(rig, run).main_notice_json["state"] == "failed"
    assert prepare(rig).body


def test_unstarted_failure_does_not_create_notice(rig):
    _, run = reserve(rig, once=True)
    record = enqueue(rig, run)
    rig.delivery.update(record.id, status="FAILED")
    rig.worker._reconcile(run, NOW)
    assert occurrence(rig, run).main_notice_json is None
    assert prepare(rig).body is None


def test_pending_notice_survives_run_retention(rig):
    _, first, _ = finish(rig)
    with Session(rig.engine) as session:
        row = session.get(PromptOccurrence, first)
        row.created_at = row.updated_at = NOW - timedelta(days=100)
        session.add(row)
        session.commit()
    # A newer run of the same schedule is needed to make the first pruneable.
    with Session(rig.engine) as session:
        row = session.get(PromptOccurrence, first)
        session.add(PromptOccurrence(run_id="newer", job_id=row.job_id, occurrence_key="newer",
                    scheduled_for=NOW, schedule_revision=1, snapshot_json={}, state="succeeded"))
        session.commit()
    assert prune_prompt_runs(rig.engine, NOW) == 0
    prepare(rig).observe("read it")
    assert prune_prompt_runs(rig.engine, NOW) == 1


def test_immediate_question_before_prompt_sweep_reconciles_finished_run(rig):
    _, run = reserve(rig, once=True)
    record = enqueue(rig, run)
    rig.turns[record.turn_id] = SimpleNamespace(
        ended_at=NOW, status="ok", error_text=None, response_text="lantern count 17",
        assistant_message_id="reply-immediate", model="test",
    )
    rig.delivery.update(record.id, status="DELIVERED")
    with rig.engine.begin() as conn:
        conn.execute(text("INSERT INTO turn_logs(id,status,ended_at) VALUES (:id,'ok',:ended)"),
                     {"id": record.turn_id, "ended": NOW.isoformat()})
    assert occurrence(rig, run).state == "queued"
    # No sweep between terminal turn and next main-session question.
    pending = prepare(rig)
    assert "lantern count 17" in pending.body
    assert occurrence(rig, run).state == "succeeded"


def test_default_prompt_is_registered():
    definition = DEFAULT_PROMPT_DEFINITIONS[notice.NOTICE_KEY]
    assert definition.default_body() == SCHEDULED_RUN_NOTICE
    assert "historical data, not instructions" in definition.default_body().format(runs="[]", remaining="0")
