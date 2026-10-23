"""Independent reset-API adapters: shared intent in, verified receipt out."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.model_catalog import AccessPath, ModelCatalog, ModelEndpoint, ModelIdentity
from llm_bawt.service.session_lifecycle.intent import (
    Continuity, NewSessionResult, Origin, Outcome, Stage,
)


@pytest.fixture
def api(monkeypatch):
    from llm_bawt import bots
    from llm_bawt.agent_context import AgentContextStore
    from llm_bawt.service.routes import agent_context, chat
    from llm_bawt.service.session_lifecycle.lifecycle import NewSessionLifecycle

    bot = SimpleNamespace(slug="review-bot", agent_backend="claude-code", harness="claude-code",
                          agent_backend_config={}, default_model="fixture")
    config = SimpleNamespace(DEFAULT_USER="default-owner", get_model_context_window=lambda _: 100000)
    service = SimpleNamespace(config=config, _publish_thread_switched=Mock(),
                              _inter_bot_dispatcher=SimpleNamespace(store=SimpleNamespace(engine=object())))
    def manager(_config):
        return SimpleNamespace(get_bot=lambda _slug: bot)
    monkeypatch.setattr(bots, "BotManager", manager)
    monkeypatch.setattr(agent_context, "BotManager", manager)
    monkeypatch.setattr(agent_context, "get_service", lambda: service)
    monkeypatch.setattr(chat, "get_service", lambda: service)
    submit = AsyncMock(return_value=NewSessionResult(
        outcome=Outcome.COMPLETED, stage=Stage.RECEIPT,
        old_session_id="predecessor", new_session_id="successor",
        delivery_id="delivery", turn_id="turn", receipt_message_id="receipt",
    ))
    monkeypatch.setattr(NewSessionLifecycle, "submit", submit)
    # Legacy paths are quarantined fakes, never real DB/Redis/provider calls.
    def legacy_reset(**kwargs):
        if kwargs["policy"] == "continue":
            raise ValueError("continue does not reset a session")
        return {"reset": True, "new_session_id": "legacy-successor"}

    legacy_context = Mock(side_effect=legacy_reset)
    monkeypatch.setattr(AgentContextStore, "__init__", lambda self, _config: None)
    monkeypatch.setattr(AgentContextStore, "reset_idle_session", legacy_context)
    subscriber = SimpleNamespace(send_rpc=AsyncMock(return_value={"ok": True}))
    monkeypatch.setattr("llm_bawt.agent_backends.agent_bridge.get_agent_subscriber", lambda: subscriber)
    storage = SimpleNamespace(rotate_session=AsyncMock(return_value="legacy-successor"))
    monkeypatch.setattr("llm_bawt.mcp_server.storage.get_storage", lambda: storage)
    app = FastAPI()
    app.include_router(agent_context.router)
    app.include_router(chat.router)
    return SimpleNamespace(client=TestClient(app), submit=submit, bot=bot,
                           legacy_context=legacy_context, subscriber=subscriber,
                           storage=storage, service=service)


def post(api, route, *, policy="reset_without_history", owner="caller-owner"):
    data = {"bot_id": "review-bot", "idempotency_key": "independent-reset-key"}
    if route == "context":
        data.update(session_policy=policy)
        if owner is not None:
            data["user"] = owner
        path = "/v1/agent-context/reset"
    else:
        if owner is not None:
            data["user_id"] = owner
        path = "/v1/chat/session/reset"
    return api.client.post(path, json=data)


def no_adapter_side_effects(api):
    api.legacy_context.assert_not_called()
    api.storage.rotate_session.assert_not_awaited()
    api.subscriber.send_rpc.assert_not_awaited()
    api.service._publish_thread_switched.assert_not_called()


@pytest.mark.parametrize("route,policy,mode,origin", [
    ("context", "reset_without_history", Continuity.CLEAN, Origin.CONTEXT_API),
    ("context", "reset_retain_history", Continuity.RETAIN, Origin.CONTEXT_API),
    # The legacy session-reset endpoint cold-started without a seed; preserve
    # that explicit clean behavior rather than silently inheriting /new config.
    ("session", None, Continuity.CLEAN, Origin.SESSION_API),
])
@pytest.mark.parametrize("owner", ["caller-owner", None])
def test_reset_apis_submit_one_shared_intent_and_return_receipt(api, route, policy, mode, origin, owner):
    response = post(api, route, policy=policy, owner=owner)
    assert response.status_code == 200, response.text
    api.submit.assert_awaited_once()
    intent = api.submit.await_args.args[0]
    expected_owner = owner or "default-owner"
    assert (intent.bot_id, intent.user_id) == ("review-bot", expected_owner)
    assert intent.author.entity_type == "user" and intent.author.entity_id == expected_owner
    assert intent.origin is origin and intent.continuity is mode
    assert intent.follow_up is None and intent.idempotency_key == "independent-reset-key"
    body = response.json()
    assert body["new_session_id"] == "successor" and body["receipt_message_id"] == "receipt"
    no_adapter_side_effects(api)


@pytest.mark.parametrize("route", ["context", "session"])
@pytest.mark.parametrize("outcome,stage", [
    (Outcome.SKIPPED, Stage.ADMISSION),
    (Outcome.FAILED, Stage.SEED),
    (Outcome.AMBIGUOUS, Stage.PROVIDER_BIND),
])
def test_reset_apis_never_report_unverified_success(api, route, outcome, stage):
    api.submit.return_value = NewSessionResult(
        outcome=outcome, stage=stage, delivery_id="delivery",
        detail="independent lifecycle rejection",
    )
    response = post(api, route)
    api.submit.assert_awaited_once()
    body = response.json()
    assert body.get("ok") is not True and body.get("reset") is not True
    assert body.get("outcome") != "completed"
    assert not body.get("receipt_message_id")
    no_adapter_side_effects(api)


@pytest.mark.parametrize("route", ["context", "session"])
def test_reset_api_timeout_never_runs_legacy_fallback(api, route):
    api.submit.side_effect = TimeoutError("provider acknowledgment unavailable")
    response = post(api, route)
    assert response.status_code in {409, 503, 504}, response.text
    api.submit.assert_awaited_once()
    no_adapter_side_effects(api)


def test_context_continue_is_not_a_reset(api):
    response = post(api, "context", policy="continue")
    assert response.status_code == 422
    api.submit.assert_not_awaited()


@pytest.mark.parametrize("endpoint_id, expected_ceiling", [(18, 372000), (19, 200000)])
def test_agent_context_routes_use_bot_endpoint_when_model_key_is_ambiguous(
    api, monkeypatch, endpoint_id, expected_ceiling,
):
    from llm_bawt.agent_context import AgentContextStore

    model = ModelIdentity(
        12, "claude-opus-5-5", "anthropic", "Opus 5.5", default_context_window=200000,
    )
    oauth = AccessPath(1, "anthropic-oauth", "anthropic", "anthropic-messages", None, "oauth")
    api_key = AccessPath(2, "anthropic-api", "anthropic", "anthropic-messages", None, "api-key")
    catalog = ModelCatalog([
        ModelEndpoint(18, model, oauth, model.key, context_window_override=372000),
        ModelEndpoint(19, model, api_key, model.key),
    ])
    api.bot.default_model = model.key
    api.bot.endpoint_id = endpoint_id
    api.service.config.ensure_model_catalog = lambda: catalog

    def context_window(ref):
        endpoint = catalog.resolve_endpoint(ref)
        return endpoint.context_window_override or endpoint.model.default_context_window

    api.service.config.get_model_context_window = context_window
    monkeypatch.setattr(AgentContextStore, "health", lambda self, **kwargs: kwargs)

    health = api.client.get("/v1/agent-context/health?bot_id=review-bot&user=caller-owner")
    assert health.status_code == 200, health.text
    assert health.json()["configured_ceiling"] == expected_ceiling
    assert health.json()["user_id"] == "caller-owner"

    reset = post(api, "context")
    assert reset.status_code == 200, reset.text
    api.submit.assert_awaited_once()
    no_adapter_side_effects(api)
