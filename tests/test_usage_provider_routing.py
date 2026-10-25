"""Provider attribution tests for the per-turn context usage popover."""

from types import SimpleNamespace

import pytest

from llm_bawt.model_catalog import AccessPath, ModelCatalog, ModelEndpoint, ModelIdentity
from llm_bawt.service.routes import usage as usage_routes
from llm_bawt.service.usage import usage_adapter_for_access_path
from llm_bawt.service.usage.routing import bot_access_path


def _catalog(*endpoints: ModelEndpoint) -> ModelCatalog:
    return ModelCatalog(endpoints)


def _service_and_bot(endpoint: ModelEndpoint, *, harness: str, backend: str):
    catalog = _catalog(endpoint)
    config = SimpleNamespace(
        ensure_model_catalog=lambda: catalog,
        resolve_model=lambda ref, harness=None, default=None: catalog.resolve(
            ref, harness=harness
        ),
    )
    bot = SimpleNamespace(
        slug="al",
        default_model=endpoint.model.key,
        endpoint_id=endpoint.id,
        harness=harness,
        agent_backend=backend,
    )
    return SimpleNamespace(config=config), bot


def _patch_bot(monkeypatch, bot) -> None:
    class FakeBotManager:
        def __init__(self, _config):
            pass

        def get_bot(self, _bot_id):
            return bot

    monkeypatch.setattr("llm_bawt.bots.BotManager", FakeBotManager)


def test_usage_adapter_registration_rejects_overlapping_paths():
    from llm_bawt.service import usage
    from llm_bawt.service.usage.base import UsageAdapter

    class FakeAdapter(UsageAdapter):
        provider = "not_the_oauth_adapter"
        access_path_keys = ("anthropic-oauth",)

        async def fetch(self):
            raise AssertionError("not fetched")

    with pytest.raises(ValueError, match="anthropic-oauth"):
        usage.register(FakeAdapter())
    assert usage.usage_adapter_for_access_path("anthropic-oauth").provider == "claude"


def test_cached_snapshot_is_copy_without_fetch(monkeypatch):
    from llm_bawt.service import usage
    from llm_bawt.service.usage.canonical import ProviderUsage
    original = ProviderUsage(provider="claude", display_name="Claude", fetched_at=123)
    monkeypatch.setitem(usage._cache, "claude", (1.0, original))
    copy = usage.get_cached("claude")
    assert copy is not original and copy.cached and copy.fetched_at == 123
    assert usage.get_cached("nonexistent") is None


@pytest.mark.asyncio
async def test_unknown_access_path_returns_unavailable_without_fetch(monkeypatch):
    async def unexpected(*_args, **_kwargs):
        raise AssertionError("an unknown path must never fetch a provider report")

    monkeypatch.setattr(usage_routes, "get_usage", unexpected)
    response = await usage_routes.get_usage_endpoint(
        provider=None, bot_id=None, access_path="unrecognized-route", force=False,
    )
    assert response.provider == "unavailable"
    assert response.status == "not_implemented"
    assert response.limits == []


def test_usage_adapter_ownership_is_exact_and_unimplemented_routes_are_unknown():
    assert usage_adapter_for_access_path("anthropic-api").provider == "anthropic_api"
    assert usage_adapter_for_access_path("anthropic-oauth").provider == "claude"
    assert usage_adapter_for_access_path("openai-oauth").provider == "openai_chatgpt"
    assert usage_adapter_for_access_path("openai-api") is None
    assert usage_adapter_for_access_path("local-vllm") is None
    assert usage_adapter_for_access_path("unknown") is None


def test_openai_oauth_claude_proxy_maps_to_chatgpt_usage(monkeypatch):
    endpoint = ModelEndpoint(
        id=20,
        model=ModelIdentity(20, "reasoning-primary", "openai", "Reasoning Primary"),
        access_path=AccessPath(
            11,
            "openai-oauth",
            "openai",
            "responses",
            "https://chatgpt.com/backend-api/codex",
            "oauth",
        ),
        upstream_model_id="opaque-upstream-name",
    )
    service, bot = _service_and_bot(
        endpoint, harness="claude-proxy", backend="claude-code"
    )
    _patch_bot(monkeypatch, bot)

    assert usage_routes._provider_for_bot(service, "al") == "openai_chatgpt"


def test_unbound_kimi_model_prefix_does_not_choose_usage(monkeypatch):
    config = SimpleNamespace(
        ensure_model_catalog=lambda: None,
        resolve_model=lambda ref, harness=None, default=None: {
            "type": "claude-code",
            "model_id": "kimi_coding/k3",
        },
    )
    bot = SimpleNamespace(
        slug="al",
        default_model="kimi-k3",
        endpoint_id=None,
        harness="claude-proxy",
        agent_backend="claude-code",
    )
    _patch_bot(monkeypatch, bot)

    assert usage_routes._provider_for_bot(
        SimpleNamespace(config=config), "al"
    ) is None


def test_anthropic_api_endpoint_maps_to_spend_not_subscription(monkeypatch):
    endpoint = ModelEndpoint(
        id=18,
        model=ModelIdentity(18, "opus-current", "anthropic", "Opus"),
        access_path=AccessPath(
            7, "anthropic-api", "anthropic", "anthropic-messages",
            "https://api.anthropic.com", "api_key",
        ),
        upstream_model_id="claude-opus-current",
    )
    service, bot = _service_and_bot(endpoint, harness="claude-code", backend="claude-code")
    _patch_bot(monkeypatch, bot)

    assert bot_access_path(service.config, "al") == "anthropic-api"
    assert usage_routes._provider_for_bot(service, "al") == "anthropic_api"


def test_native_anthropic_endpoint_maps_to_claude(monkeypatch):
    endpoint = ModelEndpoint(
        id=17,
        model=ModelIdentity(17, "opus-current", "anthropic", "Opus"),
        access_path=AccessPath(
            6,
            "anthropic-oauth",
            "anthropic",
            "anthropic-messages",
            "https://api.anthropic.com",
            "oauth",
        ),
        upstream_model_id="claude-opus-current",
    )
    service, bot = _service_and_bot(
        endpoint, harness="claude-code", backend="claude-code"
    )
    _patch_bot(monkeypatch, bot)

    assert usage_routes._provider_for_bot(service, "al") == "claude"


def test_legacy_gpt_alias_does_not_borrow_subscription_usage(monkeypatch):
    config = SimpleNamespace(
        ensure_model_catalog=lambda: None,
        resolve_model=lambda ref, harness=None, default=None: {
            "type": "agent_backend",
            "model_id": "gpt-5.6-sol",
        },
    )
    bot = SimpleNamespace(
        slug="al",
        default_model="gpt-5.6-sol",
        endpoint_id=None,
        harness="claude-proxy",
        agent_backend="claude-code",
    )
    _patch_bot(monkeypatch, bot)

    assert usage_routes._provider_for_bot(SimpleNamespace(config=config), "al") is None
