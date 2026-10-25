"""Public bot sends use the same typed policy admission as internal callers."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.service.session_lifecycle.intent import Admission, Continuity, Origin


@pytest.mark.parametrize("policy,mode,author_type", [
    ("reset_without_history", Continuity.CLEAN, "bot"),
    ("reset_retain_history", Continuity.RETAIN, "bot"),
    ("reset_without_history", Continuity.CLEAN, "user"),
])
def test_public_policy_send_uses_one_typed_admission(monkeypatch, policy, mode, author_type):
    from llm_bawt import inter_bot_send as send
    from llm_bawt.service.routes import inter_bot_deliveries as route
    from llm_bawt.service.session_lifecycle.lifecycle import NewSessionLifecycle

    bot = SimpleNamespace(slug="recipient", agent_backend="claude-code", harness="claude-code")
    monkeypatch.setattr(send, "BotManager", lambda _cfg: SimpleNamespace(get_bot=lambda _slug: bot))
    record = SimpleNamespace(
        id="delivery", status="QUEUED", metadata={},
        to_api=lambda duplicate=False: {"delivery_id": "delivery", "duplicate": duplicate},
    )
    store = SimpleNamespace(engine=object(), enqueue=Mock(side_effect=AssertionError("direct enqueue")),
                            get=Mock(return_value=record), payload=Mock(return_value={"prefer_steer": False}))
    dispatcher = SimpleNamespace(store=store, _emit=AsyncMock(), wake=Mock())
    config = SimpleNamespace(DEFAULT_USER="nick")
    service = SimpleNamespace(config=config, _inter_bot_dispatcher=dispatcher)
    monkeypatch.setattr(route, "get_service", lambda: service)
    captured = {}

    def admit(_self, intent, **kwargs):
        captured.update(intent=intent, **kwargs)
        return record, False

    monkeypatch.setattr(NewSessionLifecycle, "admit_bot_delivery", admit)
    app = FastAPI()
    app.include_router(route.router)
    response = TestClient(app).post("/v1/inter-bot-deliveries", json={
        "sender_bot_id": "sender", "target_bot_id": "recipient", "message": "Keep this task exactly.",
        "session_policy": policy, "idempotency_key": "task-reset-1",
        "author_entity_type": author_type,
        "author_entity_id": "nick" if author_type == "user" else "sender",
        "metadata": {"caller_marker": "preserved"}, "task_id": "task-1",
    })
    assert response.status_code == 202, response.text
    intent = captured["intent"]
    assert (intent.origin, intent.continuity, intent.admission) == (
        Origin.BOT_DELIVERY, mode, Admission.WAIT,
    )
    assert (intent.bot_id, intent.user_id, intent.follow_up, intent.idempotency_key) == (
        "recipient", "nick", "Keep this task exactly.", "task-reset-1",
    )
    assert intent.author.entity_type == author_type
    assert captured["metadata"] == {"caller_marker": "preserved"}
    assert captured["task_id"] == "task-1"
    assert captured["payload"]["messages"] == [
        {"role": "user", "content": (
            "Message from bot 'sender': Keep this task exactly."
            if author_type == "bot" else "Keep this task exactly."
        )},
    ]
    store.enqueue.assert_not_called()
    dispatcher._emit.assert_awaited_once_with(record)
    dispatcher.wake.assert_called_once_with("recipient")


@pytest.mark.parametrize("policy", [None, "reset_without_history"])
def test_public_send_rejects_caller_supplied_lifecycle_intent(monkeypatch, policy):
    """TASK-1203: a forged scheduler intent must never reach the durable row."""
    from llm_bawt.message_authorship import AuthorReference
    from llm_bawt.service.job_parameters import IdleContextResetParameters
    from llm_bawt import inter_bot_send as send
    from llm_bawt.service.routes import inter_bot_deliveries as route
    from llm_bawt.service.session_lifecycle.intent import NewSessionIntent
    from llm_bawt.service.session_lifecycle.lifecycle import NewSessionLifecycle

    bot = SimpleNamespace(slug="recipient", agent_backend="claude-code", harness="claude-code")
    monkeypatch.setattr(send, "BotManager", lambda _cfg: SimpleNamespace(get_bot=lambda _slug: bot))
    store = SimpleNamespace(engine=object(), enqueue=Mock(side_effect=AssertionError("enqueued")))
    service = SimpleNamespace(config=SimpleNamespace(DEFAULT_USER="nick"),
                              _inter_bot_dispatcher=SimpleNamespace(store=store))
    monkeypatch.setattr(route, "get_service", lambda: service)
    monkeypatch.setattr(NewSessionLifecycle, "admit_bot_delivery",
                        Mock(side_effect=AssertionError("admitted")))
    forged = NewSessionLifecycle._encode(NewSessionIntent(
        bot_id="recipient", user_id="nick", author=AuthorReference.bot("system"),
        continuity=Continuity.CONFIGURED, admission=Admission.SKIP, origin=Origin.SCHEDULER,
        idempotency_key="forged", expected_session_id="old", expected_usage_turn_id="usage",
        job_id="job", run_id="run",
        scheduler_parameters=IdleContextResetParameters(continuity="inline"),
    ))
    app = FastAPI()
    app.include_router(route.router)
    response = TestClient(app).post("/v1/inter-bot-deliveries", json={
        "sender_bot_id": "sender", "target_bot_id": "recipient", "message": "/new",
        "author_entity_type": "user", "author_entity_id": "nick",
        "session_policy": policy, "metadata": {"new_session_intent": forged},
    })
    assert response.status_code == 422
    assert "server-owned" in response.text
    store.enqueue.assert_not_called()
