"""TASK-1231B: only a host shutdown, never a user Stop, qualifies for restart retry."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from agent_bridge.abort_contract import HOST_SHUTDOWN
from llm_bawt import inter_bot_delivery_retry as retry
from llm_bawt.inter_bot_delivery_retry import restart_abort_cause

ENDED = datetime(2026, 10, 22, 0, 26, 22, tzinfo=timezone.utc)
CAUSE = {"abort_cause": {"cause": HOST_SHUTDOWN, "at": "2026-10-22T00:26:22Z"}}


def turn(status="aborted", ended_at=ENDED, request=CAUSE, raw=None):
    return SimpleNamespace(status=status, ended_at=ended_at,
                           request_json=raw if raw is not None else json.dumps(request))


def test_host_shutdown_abort_qualifies():
    assert restart_abort_cause(turn()) == HOST_SHUTDOWN


@pytest.mark.parametrize("candidate", [
    None,
    turn(request={**CAUSE, "abort_request": {"source": "chat_stop"}}),
    turn(request={**CAUSE, "abort_request": {"source": "ops"}}),
    turn(request={}),
    turn(request={"abort_cause": {"cause": "operator"}}),
    turn(request={"abort_cause": HOST_SHUTDOWN}),
    turn(status="error"),
    turn(status="cancelling"),
    turn(ended_at=None),
    turn(raw="{not json"),
], ids=["no-turn", "user-stop", "other-stop", "no-cause", "unknown-cause", "flat-cause",
        "error", "cancelling", "not-ended", "bad-json"])
def test_everything_else_does_not_qualify(candidate):
    assert restart_abort_cause(candidate) is None


def test_continue_touches_only_the_final_user_message(monkeypatch):
    monkeypatch.setattr(retry, "resolve_restart_retry_text", lambda _bot: "continue please")
    payload = {"messages": [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [{"type": "text", "text": "do it"}, {"type": "image_url"}]},
    ], "bot_id": "snark"}
    original = SimpleNamespace(target_bot_id="snark", message="do it")
    message, built = retry.RESTART_RETRY.build(original, payload)
    assert message == "continue please\n\ndo it"
    assert built["messages"][:2] == payload["messages"][:2]
    assert built["messages"][2]["content"][0] == {"type": "text", "text": "continue please"}
    assert payload["messages"][2]["content"][0]["text"] == "do it"  # original untouched


def test_registry_failure_falls_back_to_seed(monkeypatch):
    import llm_bawt.prompt_registry as registry

    def broken(*_a, **_k):
        raise RuntimeError("db down")

    monkeypatch.setattr(registry, "get_prompt_resolver", broken)
    assert retry.resolve_restart_retry_text("snark") == retry.DEFAULT_RESTART_RETRY_CONTINUE


# --- Vex 1231B r1 B2: the continuation always carries the original request ---

def _original(message="Review TASK-9"):
    return SimpleNamespace(target_bot_id="loopy", message=message)


@pytest.fixture
def seed(monkeypatch):
    monkeypatch.setattr(retry, "resolve_restart_retry_text", lambda _target: "NOTE")


def test_continue_prefixes_text_request(seed):
    message, payload = retry._continue(_original(), {"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Review TASK-9", "name": "x"},
    ], "stream": True})
    assert message == "NOTE\n\nReview TASK-9"
    assert payload["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "NOTE\n\nReview TASK-9", "name": "x"},
    ]
    assert payload["stream"] is True


def test_continue_keeps_multimodal_parts(seed):
    parts = [{"type": "text", "text": "what is this"},
             {"type": "image_url", "image_url": {"url": "u"}}]
    _, payload = retry._continue(_original("what is this"),
                                 {"messages": [{"role": "user", "content": parts}]})
    assert payload["messages"][0]["content"] == [{"type": "text", "text": "NOTE"}, *parts]


def test_continue_without_user_message_appends_request(seed):
    _, payload = retry._continue(_original(), {"messages": []})
    assert payload["messages"] == [{"role": "user", "content": "NOTE\n\nReview TASK-9"}]


# --- Vex 1231B r1 B1: only a server-stamped system retry is trusted ---

def _record(author, **metadata):
    return SimpleNamespace(author=author, metadata=metadata)


SYSTEM = retry.AuthorReference.bot(retry.SYSTEM_BOT_ID)


def _stamp(entity_type, entity_id):
    return {"entity_type": entity_type, "entity_id": entity_id}


@pytest.mark.parametrize("record, expected", [
    (_record(SYSTEM, _request_identity_producer="retry",
             retry_original_author=_stamp("bot", "snark")), "snark"),
    (_record(SYSTEM, _request_identity_producer="bot",
             retry_original_author=_stamp("bot", "snark")), None),
    (_record(retry.AuthorReference.bot("vex"), _request_identity_producer="retry",
             retry_original_author=_stamp("bot", "snark")), None),
    (_record(SYSTEM, _request_identity_producer="retry",
             retry_original_author=_stamp("user", "nick")), None),
    (_record(SYSTEM, _request_identity_producer="retry",
             retry_original_author=_stamp("bot", "system")), None),
    (_record(SYSTEM, _request_identity_producer="retry"), None),
], ids=["trusted", "forged-producer", "not-system", "user-original", "system-original",
        "unstamped"])
def test_retry_reply_to_is_trusted_only_on_system_retries(record, expected):
    assert retry.retry_reply_to(record) == expected


def test_retry_original_author_keeps_the_user():
    record = _record(SYSTEM, _request_identity_producer="retry",
                     retry_original_author=_stamp("user", "nick"))
    assert retry.retry_original_author(record) == retry.AuthorReference.user("nick")
