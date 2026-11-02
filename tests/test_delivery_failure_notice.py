"""Pure tests for trusted failure lineage, redaction and leader announcement."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from llm_bawt.inter_bot_failure import ORIGIN, FailureOrigin, failure_code
from llm_bawt.service.delivery_cancel_notice import PassResult
from llm_bawt.service.delivery_failure_notice import FailureNoticePump
from llm_bawt.service.delivery_failure_notice_prompt import DEFAULT_FAILURE_NOTICE, PROMPT_KEY
from llm_bawt.task_turn_context import TaskTurnContext


@pytest.fixture
def caller():
    return TaskTurnContext(
        bot_id="sender", user_id="owner", session_id="11111111-1111-4111-8111-111111111111",
        turn_id="turn-" + "a" * 32, trigger_message_id="22222222-2222-4222-8222-222222222222",
        issued_at=123,
    )


def _row(caller):
    return {"sender_bot_id": caller.bot_id, "sender_turn_id": caller.turn_id,
            "author_entity_type": "bot", "author_entity_id": caller.bot_id,
            "metadata_json": {ORIGIN: FailureOrigin.capture(caller)}}


def test_capture_round_trips_only_the_trusted_origin_fields(caller):
    captured = FailureOrigin.capture(caller)
    expected = asdict(caller)
    expected.pop("issued_at")
    assert captured == {"version": 1, **expected}
    assert asdict(FailureOrigin.from_row(_row(caller))) == expected


@pytest.mark.parametrize("raw", [None, [], "forged", {}, {"version": True},
                                     {"version": "1"}, {"version": 0}, {"version": 2}])
def test_from_row_rejects_missing_or_unversioned_origin(caller, raw):
    row = _row(caller)
    row["metadata_json"] = {} if raw is None else {ORIGIN: raw}
    assert FailureOrigin.from_row(row) is None


@pytest.mark.parametrize("field", ["bot_id", "user_id", "session_id", "turn_id", "trigger_message_id"])
@pytest.mark.parametrize("value", [None, "", 12, "x" * 129])
def test_from_row_rejects_invalid_origin_fields(caller, field, value):
    row = _row(caller)
    row["metadata_json"][ORIGIN][field] = value
    assert FailureOrigin.from_row(row) is None


@pytest.mark.parametrize("field,value", [
    ("sender_bot_id", "foreign"), ("sender_turn_id", None), ("sender_turn_id", "turn-foreign"),
    ("author_entity_type", "user"), ("author_entity_id", "foreign"),
])
def test_from_row_requires_sender_turn_and_bot_authorship_to_match(caller, field, value):
    row = _row(caller)
    row[field] = value
    assert FailureOrigin.from_row(row) is None


def test_server_produced_restart_child_keeps_original_initiator(caller):
    from llm_bawt.bot_identities import SYSTEM_BOT_ID

    row = _row(caller)
    row["author_entity_id"] = SYSTEM_BOT_ID
    row["metadata_json"]["retry_original_author"] = {
        "entity_type": "bot", "entity_id": caller.bot_id, "status": "canonical",
    }
    assert FailureOrigin.from_row(row) is None  # caller metadata is not authority
    row["metadata_json"]["_request_identity_producer"] = "retry"
    assert FailureOrigin.from_row(row).session_id == caller.session_id


@pytest.mark.parametrize("error,expected", [
    (None, "worker_error"), ("", "worker_error"),
    ("ChatGPTEventTimeout: token=private-secret", "upstream_event_timeout"),
    ("EVENT TIMEOUT", "upstream_event_timeout"),
    ("context overflow", "context_limit"), ("Context is too long", "context_limit"),
    ("context window exceeded", "context_limit"),
    ("TimeoutError: tool_args={password:private-secret}", "timeout"),
    ("request timed out", "timeout"),
    ("accepted run exceeded recovery window", "outcome_unknown"),
    ("ambiguous result", "outcome_unknown"),
    ("SecretProviderFailure: private-secret tool_args=delete_everything", "worker_error"),
])
def test_failure_code_is_a_fixed_classification_not_provider_text(error, expected):
    assert failure_code(error) == expected


def test_failure_prompt_is_registered_with_only_redacted_variables():
    from llm_bawt.prompt_definitions import DEFAULT_PROMPT_DEFINITIONS

    definition = DEFAULT_PROMPT_DEFINITIONS[PROMPT_KEY]
    assert set(definition.required_vars) == {
        "worker_bot_id", "task_id", "delivery_id", "original_delivery_id", "failed_turn_id",
        "initiating_turn_id", "delivery_status", "turn_status", "error_code",
        "existing_retry_delivery_id",
    }
    assert definition.default_body() == DEFAULT_FAILURE_NOTICE
    rendered = definition.default_body().format(**{key: key for key in definition.required_vars})
    assert "not a retry" in rendered and "Prior tool effects may persist" in rendered


@pytest.mark.parametrize("engine", [None, SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))])
def test_failure_pump_is_inert_without_postgresql(engine):
    assert FailureNoticePump(SimpleNamespace(engine=engine)).run_pass() == PassResult()


@pytest.mark.parametrize("notified", [[], ["notice-one", "notice-two"]])
def test_leader_announces_failure_notices_and_wakes_only_for_new_notices(notified):
    from llm_bawt.service.leader_side_pumps import LeaderSidePumps

    emitted, woken, passes = [], [], []

    def run_pass():
        passes.append(True)
        return PassResult(notified=list(notified))

    side = LeaderSidePumps(
        SimpleNamespace(_inter_bot_dispatcher=SimpleNamespace(wake=lambda: woken.append(True))),
        SimpleNamespace(engine=None), failure_notices=SimpleNamespace(run_pass=run_pass),
    )

    async def emit(record):
        emitted.append(record)

    asyncio.run(side.run_pass(emit))
    assert passes == [True]
    assert emitted == notified
    assert woken == ([True] if notified else [])
