"""Agent saves and picker lists must agree with actual bridge dispatch (TASK-1769)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from claude_code_bridge.native_auth import EndpointRouteError, resolve_native_route
from llm_bawt.model_catalog import (
    AccessPath,
    IncompatibleModelError,
    ModelCatalog,
    ModelEndpoint,
    ModelIdentity,
    ProtocolCompatibility,
)
from llm_bawt.service.routes import bot_profile_validation as validation
from llm_bawt.service.routes import settings
from llm_bawt.service.schemas import BotProfileResponse


def _endpoint(endpoint_id, key, protocol, vendor="local", engine=None):
    return ModelEndpoint(
        endpoint_id,
        ModelIdentity(1, "fixture-model", "local", "Fixture Model"),
        AccessPath(endpoint_id, key, vendor, protocol, None, "none", engine),
        "fixture-upstream",
    )


@pytest.mark.parametrize("engine", ["ollama", "vllm", "llama-cpp"])
def test_local_chat_routes_stay_chat_only(engine):
    endpoint = _endpoint(10, f"fixture-{engine}", "chat-completions", engine=engine)
    catalog = ModelCatalog([endpoint])

    assert catalog.resolve(10, harness="chat")["model_id"] == "fixture-upstream"
    for harness in ("claude-code", "claude-proxy"):
        assert catalog.endpoints(harness=harness) == ()
        with pytest.raises(IncompatibleModelError):
            catalog.resolve(10, harness=harness)


def test_proxy_eligibility_uses_dispatch_provider_decision(monkeypatch):
    endpoint = _endpoint(10, "local-anthropic", "anthropic-messages")
    catalog = ModelCatalog([endpoint])
    assert catalog.resolve(10, harness="claude-proxy")["model_id"] == "local/fixture-upstream"
    monkeypatch.setattr(ModelCatalog, "_provider_prefix", staticmethod(lambda _path: None))
    assert not ProtocolCompatibility.is_compatible("claude-proxy", endpoint.access_path)
    with pytest.raises(IncompatibleModelError):
        catalog.resolve(10, harness="claude-proxy")


@pytest.mark.parametrize(("key", "vendor", "protocol"), [
    ("unknown", "new-provider", "responses"),
    ("anthropic-custom", "anthropic", "anthropic-messages"),
])
def test_no_adapter_or_native_route_means_no_claude_harness(key, vendor, protocol):
    endpoint = _endpoint(10, key, protocol, vendor)
    assert not ProtocolCompatibility.is_compatible("claude-code", endpoint.access_path)
    assert not ProtocolCompatibility.is_compatible("claude-proxy", endpoint.access_path)


def test_native_runtime_refusal_is_preserved():
    with pytest.raises(EndpointRouteError, match="refusing to guess a credential"):
        resolve_native_route("fixture-ollama", proxy_provider=None)


@pytest.fixture
def profile_api(monkeypatch):
    """Real HTTP handlers/validation; only catalog, storage and post-save effects are fakes."""
    catalog = ModelCatalog([
        _endpoint(10, "fixture-ollama", "chat-completions", engine="ollama"),
        _endpoint(20, "local-anthropic", "anthropic-messages"),
    ])
    existing = BotProfileResponse(
        slug="fixture-bot", name="Fixture", description="", system_prompt="Fixture instruction",
        default_model="fixture-model", endpoint_id=10, harness="chat",
    )
    store = MagicMock(engine=object())
    store.get.return_value = existing
    store.upsert.side_effect = lambda payload, **_kwargs: BotProfileResponse(**payload)
    service = SimpleNamespace(config=SimpleNamespace(ensure_model_catalog=lambda: catalog))
    monkeypatch.setattr(validation, "get_service", lambda: service)
    monkeypatch.setattr(settings, "get_service", lambda: service)
    monkeypatch.setattr(settings, "get_bot_profile_store", lambda _config: store)
    probe = MagicMock(return_value=object())
    monkeypatch.setattr(settings.ProfileRouteGate, "for_payload", probe)
    for name in ("_reload_bot_registry", "_reload_service_model_catalog",
                 "_invalidate_bot_instance_cache", "_clear_session_model_overrides"):
        monkeypatch.setattr(settings, name, MagicMock())
    monkeypatch.setattr(settings, "_rotate_threads_on_runtime_switch", lambda *_args: [])
    monkeypatch.setattr(settings, "_effective_bot_settings", lambda *_args: {})
    monkeypatch.setattr(settings, "_profile_tags", lambda *_args: {})
    monkeypatch.setattr(settings, "_to_profile_response", lambda profile, **_kwargs: profile)
    app = FastAPI()
    app.include_router(settings.router)
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, store=store, probe=probe, existing=existing)


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH"])
@pytest.mark.parametrize("harness", ["claude-code", "claude-proxy"])
def test_agent_save_rejects_undispatchable_endpoint_before_any_write(profile_api, method, harness):
    rig = profile_api
    body = {"bot_type": "agent", "agent_backend": "claude-code", "harness": harness,
            "endpoint_id": 10}
    url = "/v1/bots/fixture-bot/profile"
    if method != "PATCH":
        body.update(name="Fixture", system_prompt="Fixture instruction")
    if method == "POST":
        rig.store.get.return_value = None
        body.update(slug="fixture-bot", tags=[])
        url = "/v1/bots"
    response = rig.client.request(method, url, json=body)
    assert response.status_code == 422, response.text
    assert "fixture-ollama" in response.json()["detail"]
    rig.store.upsert.assert_not_called()
    rig.probe.assert_not_called()  # not even credential probing before compatibility


def test_runtime_only_patch_does_not_silently_revert_to_chat(profile_api):
    rig = profile_api
    response = rig.client.patch("/v1/bots/fixture-bot/profile", json={"agent_backend": "claude-code"})
    assert response.status_code == 422, response.text
    assert "fixture-ollama" in response.json()["detail"]
    rig.store.upsert.assert_not_called()


def test_endpoint_only_patch_on_existing_agent_is_also_rejected(profile_api):
    rig = profile_api
    rig.existing.agent_backend = "claude-code"
    rig.existing.harness = "claude-proxy"
    rig.existing.endpoint_id = 20
    response = rig.client.patch("/v1/bots/fixture-bot/profile", json={"endpoint_id": 10})
    assert response.status_code == 422, response.text
    assert "fixture-ollama" in response.json()["detail"]
    rig.store.upsert.assert_not_called()


def test_compatible_same_model_endpoint_patch_still_saves(profile_api):
    rig = profile_api
    response = rig.client.patch("/v1/bots/fixture-bot/profile", json={
        "agent_backend": "claude-code", "harness": "claude-code", "endpoint_id": 20,
    })
    assert response.status_code == 200, response.text
    assert response.json()["harness"] == "claude-proxy"
    assert response.json()["endpoint_id"] == 20
    assert response.json()["agent_backend"] == "claude-code"
    rig.store.upsert.assert_called_once()
