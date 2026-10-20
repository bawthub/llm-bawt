"""Shared lifecycle intent validation, independent of delivery transport."""

import pytest

from llm_bawt.message_authorship import AuthorReference
from llm_bawt.service.session_lifecycle.intent import (
    Admission, Continuity, NewSessionIntent, NewSessionResult, Origin, Outcome, Stage,
)


def _intent(**overrides):
    options = dict(
        bot_id="byte", user_id="nick", author=AuthorReference.user("nick"),
        continuity=Continuity.CONFIGURED, admission=Admission.WAIT,
        origin=Origin.CHAT_COMMAND, idempotency_key="delivery-one",
    )
    options.update(overrides)
    return NewSessionIntent(**options)


def test_system_is_bot_author_but_not_conversation_owner():
    intent = _intent(
        author=AuthorReference.bot("system"), origin=Origin.SCHEDULER,
        admission=Admission.SKIP, expected_session_id="s1",
        expected_usage_turn_id="turn-1", job_id="job-1", run_id="run-1",
    )
    assert intent.bot_id == "byte" and intent.user_id == "nick"
    assert intent.author.to_dict()["entity_id"] == "system"
    assert intent.continuity is Continuity.CONFIGURED


@pytest.mark.parametrize("changes", [
    {"bot_id": ""}, {"user_id": " "}, {"idempotency_key": ""},
    {"author": AuthorReference.unknown()},
    {"admission": Admission.SKIP},
    {"expected_usage_turn_id": "t1"},
    {"follow_up": "  "},
])
def test_invalid_or_unfenced_intents_fail(changes):
    with pytest.raises(ValueError):
        _intent(**changes)


@pytest.mark.parametrize("mode", list(Continuity))
def test_standalone_and_follow_up_are_distinct(mode):
    assert _intent(continuity=mode).follow_up is None
    assert _intent(continuity=mode, follow_up="do this").follow_up == "do this"


def test_durable_json_enum_strings_do_not_bypass_guards():
    with pytest.raises(ValueError, match="expected_session_id"):
        _intent(admission="skip")
    with pytest.raises(ValueError, match="usage fence"):
        _intent(admission="wait", expected_usage_turn_id="t1")
    with pytest.raises(ValueError, match="'unknown' is not a valid"):
        _intent(continuity="unknown")
    decoded = _intent(continuity="retain", admission="skip", origin="bot_delivery",
                      expected_session_id="s1")
    assert decoded.continuity is Continuity.RETAIN
    assert decoded.admission is Admission.SKIP
    assert decoded.origin is Origin.BOT_DELIVERY
    with pytest.raises(ValueError, match="new session"):
        NewSessionResult(outcome="completed", stage="receipt")


def test_success_requires_actual_receipt_and_successor():
    with pytest.raises(ValueError, match="new session"):
        NewSessionResult(outcome=Outcome.COMPLETED, stage=Stage.RECEIPT)
    with pytest.raises(ValueError, match="persisted receipt"):
        NewSessionResult(outcome=Outcome.COMPLETED, stage=Stage.ROTATION,
                         new_session_id="successor")
    assert NewSessionResult(outcome=Outcome.SKIPPED, stage=Stage.ADMISSION).new_session_id is None
