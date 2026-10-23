"""HTTP contract tests for the first-run setup routes (TASK-355, TASK-1297)."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from agent_bridge.ping import BridgeReadiness
from llm_bawt.model_catalog import AccessPath, ModelCatalog, ModelEndpoint, ModelIdentity
from llm_bawt.service.routes import bot_profile_validation
from llm_bawt.service.routes import setup as setup_routes

#: TASK-1293 routes, stated independently of the code under test:
#: access path -> (vendor, protocol, harness the admin writer must settle on).
ROUTES = {
    "anthropic-oauth": ("anthropic", "anthropic-messages", "claude-code"),
    "anthropic-api": ("anthropic", "anthropic-messages", "claude-code"),
    "openai-oauth": ("openai", "responses", "claude-proxy"),
    "openai-responses": ("openai", "responses", "claude-proxy"),
    "xai-responses": ("xai", "responses", "claude-proxy"),
    "openrouter-anthropic": ("openrouter", "anthropic-messages", "claude-proxy"),
}

_READY = BridgeReadiness(ready=True, backend="claude-code", instance="boot-1", detail="claude-code bridge answered ping v1")

_FIRST_BOT = {
    "endpoint_id": 7,
    "slug": "helper",
    "name": "Helper",
    "system_prompt": "You are helpful.",
}


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(setup_routes.router)
    return TestClient(app)


def _service() -> MagicMock:
    service = MagicMock()
    service.config = MagicMock(name="config")
    return service


def _resolver(body: str = "Registry prompt body"):
    resolver = MagicMock()
    resolver.resolve.return_value = SimpleNamespace(body=body)
    return resolver


def _catalog(access_path_key: str, *, vendor: str, protocol: str, endpoint_id: int = 7) -> ModelCatalog:
    return ModelCatalog([
        ModelEndpoint(
            id=endpoint_id,
            model=ModelIdentity(id=3, key="m", vendor=vendor, display_name="M"),
            access_path=AccessPath(id=5, key=access_path_key, vendor=vendor, protocol=protocol,
                                   base_url=None, auth_mechanism="api-key"),
            upstream_model_id="m",
        ),
    ])


# ---------------------------------------------------------------------------
# GET /v1/setup/first-bot (defaults)
# ---------------------------------------------------------------------------


def test_first_bot_defaults_prompt_comes_from_the_registry():
    resolver = _resolver("Operator-edited agent prompt")
    service = _service()
    with (
        patch.object(setup_routes, "get_service", return_value=service),
        patch.object(setup_routes, "get_prompt_resolver", return_value=resolver) as get_resolver,
    ):
        response = _client().get("/v1/setup/first-bot")

    assert response.status_code == 200
    assert response.json() == {
        "default_system_prompt": "Operator-edited agent prompt",
        "bot_type": "agent",
        "agent_backend": "claude-code",
    }
    get_resolver.assert_called_once_with(service.config)
    resolver.resolve.assert_called_once_with("setup.first_agent_prompt")


def test_first_bot_defaults_report_unregistered_prompt_as_503():
    resolver = MagicMock()
    resolver.resolve.return_value = None
    with (
        patch.object(setup_routes, "get_service", return_value=_service()),
        patch.object(setup_routes, "get_prompt_resolver", return_value=resolver),
    ):
        response = _client().get("/v1/setup/first-bot")
    assert response.status_code == 503
    assert "setup.first_agent_prompt" in response.json()["detail"]


def test_wizard_keeps_no_route_table_or_model_writer_of_its_own():
    """Nick, TASK-1297: the wizard uses the admin code paths, not a parallel copy."""
    for name in ("FIRST_BOT_PROVIDERS", "first_bot_provider", "TenantSeeder", "get_adapter",
                 "_tool_eligibility", "_require_tool_capable", "_agent_profile_fields",
                 "DEFAULT_FIRST_BOT_PROMPT"):
        assert not hasattr(setup_routes, name), name
    paths = {route.path for route in setup_routes.router.routes}
    assert "/v1/setup/providers" not in paths


def test_first_agent_prompt_is_registered_with_an_owner_neutral_agent_default():
    from llm_bawt.prompt_registry import PromptResolver

    definition = PromptResolver.definition_for("setup.first_agent_prompt")
    assert definition is not None and definition.category == "setup"
    body = definition.default_body()
    assert body.strip() and "agent" in body.lower()
    for forbidden in ("nick", "nova", "mira", "vex", "personal ai assistant"):
        assert forbidden not in body.lower()


# ---------------------------------------------------------------------------
# GET /v1/setup/agent-runtime
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("readiness", [
    _READY,
    BridgeReadiness(ready=False, backend="claude-code", instance=None, detail="no reply from the claude-code bridge within 5.0s"),
])
def test_agent_runtime_reports_probe_outcome_as_200(readiness):
    service = _service()
    probe = AsyncMock(return_value=readiness)
    with (
        patch.object(setup_routes, "get_service", return_value=service),
        patch.object(setup_routes, "check_agent_runtime", probe),
    ):
        response = _client().get("/v1/setup/agent-runtime")

    assert response.status_code == 200
    assert response.json() == readiness.to_dict()
    probe.assert_awaited_once_with(service.config)


# ---------------------------------------------------------------------------
# POST /v1/setup/seed (infrastructure only)
# ---------------------------------------------------------------------------


def test_setup_seed_uses_service_config_and_returns_report():
    service = _service()
    report = {"catalog_migration": {}, "prompts": {"created": [], "existing": []}}

    with (
        patch.object(setup_routes, "get_service", return_value=service),
        patch.object(setup_routes, "seed_tenant", return_value=report) as seed,
    ):
        response = _client().post("/v1/setup/seed")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "report": report, "preferences": {}}
    seed.assert_called_once_with(service.config)


def test_setup_seed_persists_network_and_voice_preferences():
    store = MagicMock()
    store.engine = MagicMock()

    with (
        patch.object(setup_routes, "get_service", return_value=_service()),
        patch.object(setup_routes, "get_runtime_settings_store", return_value=store),
        patch.object(setup_routes, "seed_tenant", return_value={}),
    ):
        response = _client().post(
            "/v1/setup/seed",
            json={
                "exposure": "public",
                "domains": ["App.Example.com", "app.example.com"],
                "voice_enabled": False,
            },
        )

    assert response.status_code == 200
    assert response.json()["preferences"] == {
        "tenant.exposure": "public",
        "tenant.domains": ["app.example.com"],
        "tenant.voice_enabled": False,
    }
    assert store.set_value.call_args_list == [
        (("global", "*", "tenant.exposure", "public"),),
        (("global", "*", "tenant.domains", ["app.example.com"]),),
        (("global", "*", "tenant.voice_enabled", False),),
    ]


def test_setup_seed_rejects_invalid_domains():
    response = _client().post("/v1/setup/seed", json={"domains": ["not a domain"]})
    assert response.status_code == 422


def test_setup_seed_reports_database_unavailable_as_503():
    with (
        patch.object(setup_routes, "get_service", return_value=_service()),
        patch.object(
            setup_routes,
            "seed_tenant",
            side_effect=RuntimeError("Tenant seed requires database credentials"),
        ),
    ):
        response = _client().post("/v1/setup/seed")

    assert response.status_code == 503
    assert response.json()["detail"] == "Tenant seed requires database credentials"


def test_setup_seed_hides_unexpected_internal_error():
    with (
        patch.object(setup_routes, "get_service", return_value=_service()),
        patch.object(setup_routes, "seed_tenant", side_effect=KeyError("secret detail")),
    ):
        response = _client().post("/v1/setup/seed")

    assert response.status_code == 500
    assert response.json()["detail"] == "Tenant seed failed"


# ---------------------------------------------------------------------------
# POST /v1/setup/first-bot
# ---------------------------------------------------------------------------


@contextmanager
def _first_bot_env(*, persist=None, runtime: BridgeReadiness = _READY):
    """Patch the route's collaborators; yield the mocks."""
    persist = persist or AsyncMock(return_value={"slug": "helper", "name": "Helper"})
    service = _service()
    probe = AsyncMock(return_value=runtime)
    with (
        patch.object(setup_routes, "get_service", return_value=service),
        patch.object(setup_routes, "check_agent_runtime", probe),
        patch.object(setup_routes, "_persist_bot_profile", persist),
    ):
        yield SimpleNamespace(persist=persist, probe=probe, service=service)


def test_first_bot_happy_path_creates_agent_through_the_admin_writer():
    with _first_bot_env() as env:
        response = _client().post("/v1/setup/first-bot", json=_FIRST_BOT)

    assert response.status_code == 200, response.text
    env.probe.assert_awaited_once_with(env.service.config)
    payload = env.persist.call_args.args[0]
    assert env.persist.call_args.kwargs == {"create_only": True}
    assert payload["slug"] == "helper"
    assert payload["endpoint_id"] == 7
    assert (payload["bot_type"], payload["agent_backend"]) == ("agent", "claude-code")
    assert payload["uses_tools"] is True and payload["uses_search"] is True
    body = response.json()
    assert body == {
        "ok": True,
        "bot": {"slug": "helper", "name": "Helper"},
        "runtime": _READY.to_dict(),
        "preferences": {},
    }


@pytest.mark.parametrize("access_path_key", list(ROUTES))
def test_admin_writer_settles_first_bot_harness_on_every_route(access_path_key):
    """The payload first-bot builds, run through the real profile validation."""
    vendor, protocol, harness = ROUTES[access_path_key]
    service = _service()
    service.config.ensure_model_catalog.return_value = _catalog(access_path_key, vendor=vendor, protocol=protocol)
    seen: dict = {}

    async def _persist(payload, *, create_only):
        bot_profile_validation._validate_profile_payload(payload)
        seen.update(payload)
        return {"slug": payload["slug"]}

    with (
        _first_bot_env(persist=AsyncMock(side_effect=_persist)),
        patch.object(bot_profile_validation, "get_service", return_value=service),
    ):
        response = _client().post("/v1/setup/first-bot", json=_FIRST_BOT)

    assert response.status_code == 200, response.text
    assert (seen["harness"], seen["agent_backend"], seen["endpoint_id"]) == (harness, "claude-code", 7)
    assert seen["default_model"] == "m"


@pytest.mark.parametrize(("catalog", "detail"), [
    (_catalog("openai-api", vendor="openai", protocol="chat-completions"),
     "endpoint_id=7 is incompatible with harness=claude-code"),
    (_catalog("xai-responses", vendor="xai", protocol="responses", endpoint_id=8),
     "Unknown model endpoint id: 7"),
])
def test_admin_writer_refusals_reach_the_wizard_as_422(catalog, detail):
    service = _service()
    service.config.ensure_model_catalog.return_value = catalog

    async def _persist(payload, *, create_only):
        bot_profile_validation._validate_profile_payload(payload)
        raise AssertionError("validation should have refused")

    with (
        _first_bot_env(persist=AsyncMock(side_effect=_persist)),
        patch.object(bot_profile_validation, "get_service", return_value=service),
    ):
        response = _client().post("/v1/setup/first-bot", json=_FIRST_BOT)

    assert response.status_code == 422
    assert response.json()["detail"] == detail


def test_first_bot_refuses_when_agent_runtime_not_ready():
    down = BridgeReadiness(ready=False, backend="claude-code", instance=None,
                           detail="no reply from the claude-code bridge within 5.0s")
    with _first_bot_env(runtime=down) as env:
        response = _client().post("/v1/setup/first-bot", json=_FIRST_BOT)

    assert response.status_code == 503
    assert response.json()["detail"] == "Agent runtime not ready: no reply from the claude-code bridge within 5.0s"
    env.persist.assert_not_called()


@pytest.mark.parametrize("override", [
    {"slug": "Bad Slug"},
    {"system_prompt": ""},
    {"endpoint_id": 0},
    {"endpoint_id": None},
])
def test_first_bot_validates_request_shape(override):
    with _first_bot_env() as env:
        response = _client().post("/v1/setup/first-bot", json={**_FIRST_BOT, **override})
    assert response.status_code == 422
    env.probe.assert_not_awaited()


def test_first_bot_ignores_legacy_route_fields_and_client_harness():
    with _first_bot_env() as env:
        response = _client().post("/v1/setup/first-bot", json={
            **_FIRST_BOT, "provider_id": "xai", "model_id": "grok-4.3", "harness": "chat",
        })
    assert response.status_code == 200, response.text
    assert env.persist.call_args.args[0]["harness"] == "claude-code"


def test_first_bot_passes_through_duplicate_slug_conflict():
    persist = AsyncMock(side_effect=HTTPException(status_code=409, detail="Bot 'helper' already exists"))
    with _first_bot_env(persist=persist):
        response = _client().post("/v1/setup/first-bot", json=_FIRST_BOT)

    assert response.status_code == 409
    assert response.json()["detail"] == "Bot 'helper' already exists"


def test_first_bot_persists_preferences_after_bot_create():
    order: list[str] = []
    store = MagicMock()
    store.engine = MagicMock()
    store.set_value.side_effect = lambda *a: order.append("pref")

    async def _persist(payload, *, create_only):
        order.append("bot")
        return {"slug": payload["slug"]}

    with (
        _first_bot_env(persist=AsyncMock(side_effect=_persist)),
        patch.object(setup_routes, "get_runtime_settings_store", return_value=store),
    ):
        response = _client().post(
            "/v1/setup/first-bot",
            json={**_FIRST_BOT, "exposure": "private", "voice_enabled": True},
        )

    assert response.status_code == 200
    assert response.json()["preferences"] == {"tenant.exposure": "private", "tenant.voice_enabled": True}
    assert order == ["bot", "pref", "pref"]


def test_failed_bot_create_stores_no_preferences():
    store = MagicMock()
    store.engine = MagicMock()
    persist = AsyncMock(side_effect=HTTPException(status_code=422, detail="nope"))
    with (
        _first_bot_env(persist=persist),
        patch.object(setup_routes, "get_runtime_settings_store", return_value=store),
    ):
        response = _client().post("/v1/setup/first-bot", json={**_FIRST_BOT, "exposure": "private"})
    assert response.status_code == 422
    store.set_value.assert_not_called()
