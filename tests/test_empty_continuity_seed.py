"""TASK-851: a /new seed starved of prior context is reported, never silent."""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_bawt.memory.session_continuity import (
    SessionContinuity,
    carries_prior_context,
    seed_empty_reason,
)
from llm_bawt.service.session_lifecycle.receipt import reset_confirmation

# Stored order: the app commits the receipt just before its command (receipt.py).
NEW_EXCHANGE = [
    {"role": "assistant", "content": "Session reset. Ready for a new conversation.", "timestamp": 9.9999},
    {"role": "user", "content": "/new", "timestamp": 10.0},
]
CAPPED = {"history_scope": "inline", "history_max_age_hours": 6}


@pytest.mark.parametrize("messages, carries", [
    ([], False),
    (NEW_EXCHANGE, False),
    (NEW_EXCHANGE * 2, False),
    (NEW_EXCHANGE[::-1], False),
    ([{"role": "assistant", "content": "Session reset. Seeded 4 recent messages (~1k tokens)."}], False),
    ([{"role": "user", "content": "/new Keep this task"}], True),
    ([{"role": "summary", "content": "s"}], True),
    ([{"role": "user", "content": "hello"}], True),
    ([{"role": "assistant", "content": "hi"}, *NEW_EXCHANGE], True),
    ([*NEW_EXCHANGE, {"role": "assistant", "content": "a second reply"}], True),
    ([{"role": "user", "content": "/newspaper"}], True),
])
def test_only_reset_bookkeeping_is_not_prior_context(messages, carries):
    assert carries_prior_context(messages) is carries


@pytest.mark.parametrize("settings, reason", [
    (CAPPED, "no messages from the last 6h (history_max_age_hours); summaries are off"),
    ({"history_scope": "inline+summaries", "history_max_age_hours": 6},
     "no messages from the last 6h (history_max_age_hours); no summaries yet"),
    ({"history_scope": "inline", "history_max_age_hours": 0}, "no earlier messages; summaries are off"),
    ({"history_scope": "summaries"}, "history is off; no summaries yet"),
    ({"history_scope": "none"}, None),
    ({}, None),
])
def test_reason_names_the_settings_that_starved_the_seed(settings, reason):
    assert seed_empty_reason(NEW_EXCHANGE, settings) == reason


def test_a_seed_with_prior_context_has_no_reason():
    assert seed_empty_reason([{"role": "user", "content": "hello"}], CAPPED) is None


def test_manifest_records_the_reason_unless_the_reset_chose_clean():
    manifest = SessionContinuity.manifest(NEW_EXCHANGE, settings=CAPPED, model="m")
    assert manifest["empty_reason"].startswith("no messages from the last 6h")
    clean = SessionContinuity.manifest([], settings=CAPPED, model="m", report_empty=False)
    assert clean["empty_reason"] is None and clean["status"] == "empty"


@pytest.mark.parametrize("status", ["restored", "empty", "prepared", "failed"])
def test_receipt_says_nothing_carried_over_instead_of_claiming_a_restore(status):
    # The incident: a seed of only the /new exchange read "Seeded 2 recent
    # messages ... Continuity restored." with nothing actually carried.
    manifest = SessionContinuity.manifest(NEW_EXCHANGE, settings=CAPPED, model="m")
    metadata = {"continuity": {**manifest, "status": status}}
    assert reset_confirmation(metadata) == (
        "Session reset. Nothing carried over: no messages from the last 6h "
        "(history_max_age_hours); summaries are off. Fresh start."
    )


def test_incident_age_cap_leaves_only_the_new_exchange_and_is_reported():
    """The 2026-09-06 snark /new: a 12h-old thread, a 6h cap, summaries off."""
    import time

    from llm_bawt.models.message import Message
    from tests.test_history_context import _getter, _history

    now = time.time()
    history = _history(_getter({"history_tokens": 12000, "history_max_age_hours": 6}))
    history.messages = [
        Message("user", "outgoing thread question", timestamp=now - 12 * 3600),
        Message("assistant", "outgoing thread answer", timestamp=now - 12 * 3600 + 60),
        Message("assistant", "Session reset. Ready for a new conversation.", timestamp=now - 2.0001),
        Message("user", "/new", timestamp=now - 2),
    ]
    payload = history.build_context_payload(
        include_history=True, include_summaries=False, delivery="seed", max_tokens=12000)
    seed = [{"role": m.role, "content": m.content} for m in payload.seed_messages]
    assert [m["content"] for m in seed] == ["Session reset. Ready for a new conversation.", "/new"]
    assert seed_empty_reason(seed, CAPPED) == (
        "no messages from the last 6h (history_max_age_hours); summaries are off")


def test_a_retained_new_task_turn_is_prior_context_through_selection():
    """Codex review: a tight budget sheds the receipt but keeps /new <task> + answer."""
    import time

    from llm_bawt.models.message import Message
    from tests.test_history_context import _getter, _history

    now = time.time()
    history = _history(_getter({"history_tokens": 1}))
    history.messages = [
        Message("assistant", "Session reset. Ready for a new conversation.", timestamp=now - 3),
        Message("user", "/new Keep this task", timestamp=now - 2),
        Message("assistant", "Task completed: the report is filed.", timestamp=now - 1),
    ]
    payload = history.build_context_payload(
        include_history=True, include_summaries=False, delivery="seed", max_tokens=12000)
    seed = [{"role": m.role, "content": m.content} for m in payload.seed_messages]
    assert [m["content"] for m in seed] == ["/new Keep this task", "Task completed: the report is filed."]
    assert seed_empty_reason(seed, {"history_scope": "inline"}) is None


def _llm(scope="inline", max_age=6):
    values = {"history_scope": scope, "history_max_age_hours": max_age}
    return SimpleNamespace(resolved_model_alias="m", config_resolver=SimpleNamespace(
        resolve_config_setting=lambda key: SimpleNamespace(value=values.get(key, 0)),
    ))


def test_capture_logs_a_warning_and_records_the_reason(monkeypatch, caplog):
    from llm_bawt.service import session_continuity
    record = Mock(return_value=True)
    monkeypatch.setattr(session_continuity, "get_storage", lambda: SimpleNamespace(engine="e"))
    monkeypatch.setattr(SessionContinuity, "record", record)
    with caplog.at_level(logging.WARNING, logger=session_continuity.__name__):
        session_continuity.record_session_continuity(
            _llm(), {"thread_session_id": "dest"}, NEW_EXCHANGE, bot_id="b", user_id="u")
    assert record.call_args.args[3]["empty_reason"].startswith("no messages from the last 6h")
    assert "carries no prior context" in caplog.text and "b/dest" in caplog.text


@pytest.mark.parametrize("binding, report_empty", [
    ({"thread_session_id": "dest"}, False),
    ({"thread_session_id": "dest", "session_policy": "reset_without_history"}, True),
])
def test_clean_resets_record_no_reason_and_no_warning(monkeypatch, caplog, binding, report_empty):
    from llm_bawt.service import session_continuity
    record = Mock(return_value=True)
    monkeypatch.setattr(session_continuity, "get_storage", lambda: SimpleNamespace(engine="e"))
    monkeypatch.setattr(SessionContinuity, "record", record)
    with caplog.at_level(logging.WARNING, logger=session_continuity.__name__):
        session_continuity.record_session_continuity(
            _llm(), binding, [], bot_id="b", user_id="u", report_empty=report_empty)
    assert record.call_args.args[3]["empty_reason"] is None
    assert "carries no prior context" not in caplog.text
