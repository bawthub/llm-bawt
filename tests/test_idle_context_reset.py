"""The scheduler submits intent; it never owns reset execution or receipts."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from llm_bawt.service.jobs.idle_context_candidates import IdleContextCandidates
from llm_bawt.service.jobs.idle_context_reset import IdleContextResetJob
from llm_bawt.service.session_lifecycle.intent import (
    Admission, Continuity, NewSessionResult, Origin, Outcome, Stage,
)

NOW = datetime(2026, 10, 20, tzinfo=timezone.utc)


def candidate(**overrides):
    return {
        "bot_id": "disposable", "session_id": "session", "user_id": "owner",
        "usage_turn_id": "turn", "latest_turn_id": "turn",
        "ended_at": NOW - timedelta(minutes=16),
        "last_completed_at": NOW - timedelta(minutes=16),
        "token_usage_json": json.dumps({"resident_tokens": 100001}), **overrides,
    }


@pytest.mark.parametrize("changes,expected", [
    ({}, None),
    ({"session_id": None}, "no valid accepted session"),
    ({"user_id": None}, "no valid accepted session"),
    ({"usage_turn_id": None}, "missing completed context measurement"),
    ({"ended_at": None}, "missing completed context measurement"),
    ({"latest_turn_id": "later"}, "newer bot activity invalidated measurement"),
    ({"last_completed_at": None}, "idle less than threshold"),
    ({"last_completed_at": NOW - timedelta(minutes=15)}, "idle less than threshold"),
    ({"token_usage_json": '{"resident_tokens":100000}'}, "resident context within threshold"),
    ({"token_usage_json": '{"resident_tokens":null}'}, "resident context within threshold"),
    ({"token_usage_json": '{"resident_tokens":"invalid"}'}, "resident context unknown"),
    ({"token_usage_json": '[]'}, "resident context unknown"),
    ({"token_usage_json": '{'}, "resident context unknown"),
    ({"token_usage_json": '{"input_tokens":100001}'}, None),
])
def test_candidate_thresholds_and_missing_evidence(changes, expected):
    assert IdleContextCandidates.skip_reason(candidate(**changes), now=NOW) == expected


def completed():
    return NewSessionResult(
        outcome=Outcome.COMPLETED, stage=Stage.RECEIPT,
        old_session_id="session", new_session_id="new", receipt_message_id="receipt",
        delivery_id="delivery", turn_id="reset-turn",
    )


def job_with(rows, result=None, *, protected=()):
    lifecycle = SimpleNamespace(submit=AsyncMock(return_value=result or completed()))
    job = IdleContextResetJob(None, object(), lifecycle=lifecycle, protected_bot_ids=protected)
    job.candidates.read = Mock(return_value=rows)
    return job, lifecycle


def run(job, run_id="run"):
    return asyncio.run(job.run(job_id="job", run_id=run_id, target_bot="*"))


def test_system_author_does_not_become_conversation_owner():
    job, lifecycle = job_with([candidate()])
    result = run(job)
    intent = lifecycle.submit.call_args.args[0]
    assert intent.user_id == "owner"
    assert intent.author.entity_type == "bot" and intent.author.entity_id == "system"
    assert intent.origin is Origin.SCHEDULER
    assert intent.continuity is Continuity.CONFIGURED
    assert intent.admission is Admission.SKIP
    assert intent.follow_up is None
    assert intent.expected_session_id == "session"
    assert intent.expected_usage_turn_id == "turn"
    assert (intent.job_id, intent.run_id) == ("job", "run")
    assert result["reset"] == 1 and result["failed"] == 0
    assert result["bots"]["disposable"]["receipt_message_id"] == "receipt"
    assert json.loads(json.dumps(result))["bots"]["disposable"]["outcome"] == "completed"


def test_next_sweep_reattaches_same_key_but_keeps_run_provenance():
    job, lifecycle = job_with([candidate()])
    run(job, "run-one")
    run(job, "run-two")
    first, second = [call.args[0] for call in lifecycle.submit.call_args_list]
    assert first.idempotency_key == second.idempotency_key
    assert first.run_id != second.run_id


@pytest.mark.parametrize("outcome,stage,count", [
    (Outcome.SKIPPED, Stage.ADMISSION, "skipped"),
    (Outcome.FAILED, Stage.SEED, "failed"),
    (Outcome.AMBIGUOUS, Stage.PROVIDER_BIND, "ambiguous"),
])
def test_noncompletion_is_never_reported_as_reset(outcome, stage, count):
    job, _ = job_with([candidate()], NewSessionResult(
        outcome=outcome, stage=stage, detail="actual shared-handler outcome",
    ))
    result = run(job)
    assert result[count] == 1 and result["reset"] == 0
    assert result["bots"]["disposable"]["detail"] == "actual shared-handler outcome"


def test_timeout_remains_ambiguous_and_other_bots_continue():
    job, lifecycle = job_with([candidate(), candidate(bot_id="second")])
    lifecycle.submit.side_effect = [TimeoutError(), completed()]
    result = run(job)
    assert result["ambiguous"] == 1 and result["reset"] == 1
    assert result["bots"]["disposable"]["outcome"] is Outcome.AMBIGUOUS


def test_submission_failure_does_not_abort_other_candidates():
    job, lifecycle = job_with([candidate(), candidate(bot_id="second")])
    lifecycle.submit.side_effect = [RuntimeError("submission failed"), completed()]
    result = run(job)
    assert result["failed"] == 1 and result["reset"] == 1


def test_unknown_pointer_is_not_guessed_or_submitted():
    job, lifecycle = job_with([candidate(session_id=None)])
    assert run(job)["skipped"] == 1
    lifecycle.submit.assert_not_awaited()


def test_synthetic_protected_bot_never_reaches_lifecycle():
    job, lifecycle = job_with([candidate(bot_id="private-fixture")], protected=["private-fixture"])
    assert run(job)["skipped"] == 1
    lifecycle.submit.assert_not_awaited()


def test_empty_sweep_never_constructs_lifecycle():
    job = IdleContextResetJob(None, object())
    job.candidates.read = Mock(return_value=[])
    assert run(job) == {"scanned": 0, "reset": 0, "skipped": 0, "failed": 0, "ambiguous": 0, "bots": {},
                        "parameters": {"resident_tokens_gt": 100000, "idle_minutes_gt": 15,
                                       "protected_bot_ids": [], "continuity": "configured"}}
    assert job._lifecycle is None
