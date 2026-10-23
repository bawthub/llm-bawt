"""Endpoint-aware native credentials (TASK-1295 M4 + B1a).

Every credential here is synthetic. The parent environment and the legacy
credential sources (credentials file / env token, via
``_get_fresh_oauth_token``) are deliberately loaded with *conflicting* values,
so a route that leaks any inherited value fails the test.
"""
from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest

from agent_bridge.command_publisher import CommandPublisherMixin
from agent_bridge.events import AgentEventKind
from claude_code_bridge import native_auth
from claude_code_bridge.native_auth import (
    ANTHROPIC_API_BASE_URL,
    AUTH_ENV_KEYS,
    AnthropicApiKeyRoute,
    AnthropicOAuthRoute,
    EndpointRouteError,
    LegacyNativeRoute,
    NativeCredentialUnavailable,
    resolve_native_route,
)
from claude_code_bridge.provider_key_broker import BrokerKeyError, BrokerKeyNotInstalled
from claude_code_bridge.send_errors import CLAUDE_CREDENTIAL_ERROR_MARKER
from claude_code_bridge.send_request import SendRequest
from claude_code_bridge.send_stream import ClaudeStreamMixin
from tests.test_claude_code_send_boundaries import SESSION, _SendHarness

PROXY_BASE = "http://127.0.0.1:12345"
GATEWAY = f"{PROXY_BASE}/claude-oauth"
PARENT = {
    "ANTHROPIC_API_KEY": "parent-api-key",
    "ANTHROPIC_AUTH_TOKEN": "parent-auth-token",
    "CLAUDE_CODE_OAUTH_TOKEN": "parent-oauth-token",
    "ANTHROPIC_BASE_URL": "http://parent-base.invalid",
    "CLAUDE_CODE_USE_BEDROCK": "1",
    "CLAUDE_CODE_USE_VERTEX": "1",
}


class Sources:
    """Synthetic credential sources; records which ones a route touched."""

    def __init__(self, monkeypatch, *, api_key="broker-api-key", oauth="broker-oauth-token"):
        self.calls: list[tuple] = []
        self.api_key = api_key
        self.oauth = oauth
        monkeypatch.setattr(native_auth, "fetch_provider_api_key", self._fetch_key)
        monkeypatch.setattr(native_auth, "_fetch_broker_token", self._fetch_oauth)
        monkeypatch.setattr(native_auth, "_get_fresh_oauth_token", self._legacy)

    def _fetch_key(self, provider_id):
        self.calls.append(("api-key", provider_id))
        if isinstance(self.api_key, Exception):
            raise self.api_key
        return self.api_key

    def _fetch_oauth(self, *, force=False):
        self.calls.append(("oauth", force))
        return (self.oauth, 1.0) if self.oauth else (None, None)

    def _legacy(self, *, force_refresh=False):
        self.calls.append(("legacy", force_refresh))
        return "credentials-file-token"


@pytest.fixture
def parent_env(monkeypatch):
    for key, value in PARENT.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("LLM_BAWT_API_URL", "http://app-fixture")  # gateway enabled


def _bridge() -> ClaudeStreamMixin:
    bridge = ClaudeStreamMixin()
    bridge._proxy_base_url = PROXY_BASE
    return bridge


def _sdk_env(route, *, force_refresh=False) -> dict:
    return _bridge()._build_sdk_env(
        use_proxy=False, model="claude-opus-5-5", subagent_model=None,
        force_refresh=force_refresh, bot_id="byte", session_key="byte:test",
        thread_session_id="thread", request_id="req", native_route=route,
    )


def _effective(env: dict) -> dict:
    """What the CLI sees: the SDK overlays options.env on os.environ."""
    merged = {**os.environ, **env}
    return {key: merged.get(key, "") for key in AUTH_ENV_KEYS}


# ── route resolution ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(("access_path", "proxy_provider", "path_provider", "expected"), [
    (None, None, None, LegacyNativeRoute),
    (None, "xai", None, type(None)),
    ("anthropic-api", None, None, AnthropicApiKeyRoute),
    ("anthropic-oauth", None, None, AnthropicOAuthRoute),
    ("openrouter-anthropic", "openrouter", "openrouter", type(None)),
    ("xai-responses", "xai", "xai", type(None)),
    ("local-anthropic", "local", "local", type(None)),  # a catalog row outside the seeds
])
def test_resolve_native_route(access_path, proxy_provider, path_provider, expected):
    route = resolve_native_route(
        access_path, proxy_provider=proxy_provider, access_path_provider=path_provider,
    )
    assert type(route) is expected


@pytest.mark.parametrize(("access_path", "proxy_provider", "path_provider"), [
    ("", None, None), ("", "xai", "xai"),                    # present but blank
    ("anthropic-api", "xai", None), ("anthropic-oauth", "openrouter", None),  # native on proxy
    ("openrouter-anthropic", None, "openrouter"), ("bogus", None, None),  # non-native on native
    ("bogus", "xai", None),                                  # unknown path on a proxy model
    ("xai-responses", "openai_chatgpt", "xai"),              # another provider's path
    ("openai-oauth", "xai", "openai_chatgpt"),
    ("xai-chat", "xai", None),                               # app sent no provider
    ("anthropic-api", None, "xai"),                          # native path with a proxy provider
])
def test_unknown_blank_or_inconsistent_path_is_an_error(access_path, proxy_provider, path_provider):
    with pytest.raises(EndpointRouteError):
        resolve_native_route(
            access_path, proxy_provider=proxy_provider, access_path_provider=path_provider,
        )


# ── B1a: each route uses only its own credential ─────────────────────────────


def test_api_key_route_overrides_every_inherited_value(monkeypatch, parent_env):
    sources = Sources(monkeypatch)
    env = _sdk_env(AnthropicApiKeyRoute())
    assert _effective(env) == {
        "ANTHROPIC_API_KEY": "broker-api-key",
        "ANTHROPIC_AUTH_TOKEN": "",
        "CLAUDE_CODE_OAUTH_TOKEN": "",
        "ANTHROPIC_BASE_URL": ANTHROPIC_API_BASE_URL,  # never the OAuth gateway
        "CLAUDE_CODE_USE_BEDROCK": "",
        "CLAUDE_CODE_USE_VERTEX": "",
    }
    assert sources.calls == [("api-key", "anthropic-api")]


def test_oauth_route_overrides_every_inherited_value(monkeypatch, parent_env):
    sources = Sources(monkeypatch)
    env = _sdk_env(AnthropicOAuthRoute())
    assert _effective(env) == {
        "ANTHROPIC_API_KEY": "",
        "ANTHROPIC_AUTH_TOKEN": "",
        "CLAUDE_CODE_OAUTH_TOKEN": "broker-oauth-token",
        "ANTHROPIC_BASE_URL": GATEWAY,
        "CLAUDE_CODE_USE_BEDROCK": "",
        "CLAUDE_CODE_USE_VERTEX": "",
    }
    assert sources.calls == [("oauth", False)]


def test_oauth_route_without_gateway_targets_anthropic_directly(monkeypatch, parent_env):
    monkeypatch.delenv("LLM_BAWT_API_URL")
    Sources(monkeypatch)
    assert _effective(_sdk_env(AnthropicOAuthRoute()))["ANTHROPIC_BASE_URL"] == ANTHROPIC_API_BASE_URL


def test_api_key_route_without_connected_key_fails_closed_naming_the_fix(monkeypatch, parent_env):
    sources = Sources(monkeypatch, api_key=BrokerKeyNotInstalled("503"))
    with pytest.raises(NativeCredentialUnavailable) as raised:
        _sdk_env(AnthropicApiKeyRoute())
    assert str(raised.value) == (
        "endpoint is anthropic-api but no Anthropic API key is connected; "
        "repoint the bot to anthropic-oauth or connect a key"
    )
    assert sources.calls == [("api-key", "anthropic-api")]  # no OAuth substitute


def test_api_key_route_broker_failure_fails_closed(monkeypatch, parent_env):
    sources = Sources(monkeypatch, api_key=BrokerKeyError("broker unreachable"))
    with pytest.raises(NativeCredentialUnavailable, match="broker unreachable"):
        _sdk_env(AnthropicApiKeyRoute())
    assert sources.calls == [("api-key", "anthropic-api")]


def test_oauth_route_without_subscription_fails_closed(monkeypatch, parent_env):
    sources = Sources(monkeypatch, oauth=None)
    with pytest.raises(NativeCredentialUnavailable, match="anthropic-oauth"):
        _sdk_env(AnthropicOAuthRoute())
    # Neither the file/env token nor the API key is used as a substitute.
    assert sources.calls == [("oauth", False)]


def test_oauth_retry_forces_a_broker_refresh(monkeypatch, parent_env):
    sources = Sources(monkeypatch)
    _sdk_env(AnthropicOAuthRoute(), force_refresh=True)
    assert sources.calls == [("oauth", True)]


@pytest.mark.parametrize("gateway", [True, False])
def test_legacy_route_is_unchanged(monkeypatch, parent_env, gateway):
    if not gateway:
        monkeypatch.delenv("LLM_BAWT_API_URL")
    sources = Sources(monkeypatch)
    env = _sdk_env(None)  # no access_path → legacy
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "credentials-file-token"
    assert env.get("ANTHROPIC_BASE_URL") == (GATEWAY if gateway else None)
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        assert key not in env  # legacy never touched these
    assert sources.calls == [("legacy", False)]


# ── M4: producer → wire → SendRequest → SDK env, including the retry ─────────


class _CapturingPublisher(CommandPublisherMixin):
    def __init__(self):
        self.fields: dict[str, str] = {}

        async def evaluate(_script, _numkeys, _dedupe, _stream, _maxlen, _request_id, *pairs):
            self.fields.update(zip(pairs[::2], pairs[1::2], strict=True))
            return [1, "1-0"]

        self._pub_redis = SimpleNamespace(eval=evaluate)


def _app_publishes(monkeypatch, bot_config: dict) -> dict[str, str]:
    """Drive the real app producer (AgentBridgeBackend.stream_raw) into the
    real command publisher and return the wire fields."""
    from llm_bawt.agent_backends import agent_bridge as module

    publisher = _CapturingPublisher()

    class Subscriber:
        def __init__(self, _url):
            self._redis = SimpleNamespace(connection_pool=SimpleNamespace(
                connection_kwargs={"url": "redis://localhost:6379/0"}))

        async def connect(self):
            return None

        async def send_command(self, **kwargs):
            await publisher.send_command(**kwargs)

        async def subscribe_run(self, *_args, **_kwargs):
            if False:
                yield None

        async def close(self):
            return None

    root = SimpleNamespace(_redis=SimpleNamespace(connection_pool=SimpleNamespace(
        connection_kwargs={"url": "redis://localhost:6379/0"})))
    monkeypatch.setattr(module, "get_agent_subscriber", lambda: root)
    monkeypatch.setattr("agent_bridge.subscriber.RedisSubscriber", Subscriber)
    list(module.AgentBridgeBackend().stream_raw(
        "hello", {"bot_id": "byte", "model": "claude-opus-5-5", "timeout_seconds": 1, **bot_config},
    ))
    return publisher.fields


@pytest.mark.parametrize(("access_path", "credential"), [
    ("anthropic-api", ("ANTHROPIC_API_KEY", "broker-api-key")),
    ("anthropic-oauth", ("CLAUDE_CODE_OAUTH_TOKEN", "broker-oauth-token")),
])
def test_access_path_reaches_sdk_env_end_to_end(monkeypatch, parent_env, access_path, credential):
    fields = _app_publishes(monkeypatch, {"access_path": access_path})
    assert fields["access_path"] == access_path

    req = SendRequest.from_fields(fields)
    assert req.access_path == access_path
    route = resolve_native_route(req.access_path, proxy_provider=None)
    Sources(monkeypatch)
    for force_refresh in (False, True):  # first attempt and the auth retry
        effective = _effective(_sdk_env(route, force_refresh=force_refresh))
        key, value = credential
        assert effective[key] == value
        others = {k for k in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN") if k != key}
        assert all(effective[k] == "" for k in others)


def test_absent_access_path_stays_absent_on_the_wire(monkeypatch):
    fields = _app_publishes(monkeypatch, {})
    assert "access_path" not in fields
    assert SendRequest.from_fields(fields).access_path is None


@pytest.mark.parametrize("blank", ["", "  "])
def test_blank_access_path_is_published_and_rejected_not_legacy(blank):
    publisher = _CapturingPublisher()
    asyncio.run(publisher.send_command(
        session_key="s", message="m", request_id="r", access_path=blank))
    assert "access_path" in publisher.fields
    req = SendRequest.from_fields({**publisher.fields, "bot_id": "byte"})
    assert req.access_path == ""
    with pytest.raises(EndpointRouteError):
        resolve_native_route(req.access_path, proxy_provider=None)


def _catalog_config(access_key: str, vendor: str, protocol: str, harness: str = "claude-proxy") -> dict:
    from llm_bawt.model_catalog import AccessPath, ModelCatalog, ModelEndpoint, ModelIdentity

    endpoint = ModelEndpoint(
        id=7, model=ModelIdentity(7, "m", vendor, "m"),
        access_path=AccessPath(7, access_key, vendor, protocol, None, "test"),
        upstream_model_id="up/stream", serving_config={},
    )
    return ModelCatalog([endpoint]).resolve("m", harness=harness)


@pytest.mark.parametrize(("access_key", "vendor", "protocol"), [
    ("openai-oauth", "openai", "responses"),
    ("openai-responses", "openai", "responses"),
    ("xai-responses", "xai", "responses"),
    ("zai-anthropic", "zai", "anthropic-messages"),
    ("openrouter-anthropic", "openrouter", "anthropic-messages"),
    ("local-anthropic", "local", "anthropic-messages"),  # hand-created, not a seed
    ("kimi-coding-chat", "kimi", "chat-completions"),
])
def test_catalog_proxy_endpoint_is_accepted_end_to_end(monkeypatch, access_key, vendor, protocol):
    """Catalog → app producer → wire → SendRequest → the bridge's own prefix."""
    from claude_code_bridge.bridge import ClaudeCodeBridge

    config = _catalog_config(access_key, vendor, protocol)
    # The service copies the catalog's model_id into the bot's ``model``.
    fields = _app_publishes(monkeypatch, {**config, "model": config["model_id"]})
    req = SendRequest.from_fields(fields)
    proxy_provider = ClaudeCodeBridge._model_provider_prefix(fields["model"])
    assert proxy_provider is not None and req.access_path == access_key
    assert req.access_path_provider == proxy_provider
    assert resolve_native_route(
        req.access_path, proxy_provider=proxy_provider,
        access_path_provider=req.access_path_provider,
    ) is None


def _service_bot_config(monkeypatch, model, agent_backend_config=None):
    from llm_bawt.service.core import ServiceLLMBawt

    bot = SimpleNamespace(agent_backend="claude-code", agent_backend_config=agent_backend_config or {},
                          default_model="m", harness="claude-code")
    config = SimpleNamespace(defined_models={"models": {"m": model}})

    class Client:
        def __init__(self):
            self._bot_config = {}

    monkeypatch.setattr("llm_bawt.core.base.BaseLLMBawt._init_bot",
                        lambda self, _config: setattr(self, "bot", bot))
    monkeypatch.setattr("llm_bawt.clients.agent_backend_client.AgentBackendClient", Client)
    service = ServiceLLMBawt.__new__(ServiceLLMBawt)
    service.client, service.bot_id, service.user_id, service.config = Client(), "b", "u", config
    ServiceLLMBawt._init_bot(service, config)
    return service.client._bot_config


_PROXY_MODEL = {"type": "claude-code", "backend": "claude-code", "model_id": "xai/grok-4.5",
                "access_path": "xai-responses", "access_path_provider": "xai"}
_NATIVE_MODEL = {"type": "claude-code", "backend": "claude-code", "model_id": "claude-opus-5-5",
                 "access_path": "anthropic-oauth"}
_LEGACY_MODEL = {"type": "claude-code", "backend": "claude-code", "model_id": "xai/grok-4.5"}


@pytest.mark.parametrize(("model", "per_bot", "expected"), [
    # The catalog pair replaces a stale per-bot pair as a unit.
    (_PROXY_MODEL, {"access_path": "bogus", "access_path_provider": "openrouter"},
     ("xai-responses", "xai")),
    (_NATIVE_MODEL, {"access_path_provider": "xai"}, ("anthropic-oauth", None)),
    # Only the catalog writes a provider: a hand-set one never rides along.
    (_LEGACY_MODEL, {"access_path": "bogus", "access_path_provider": "xai"}, ("bogus", None)),
    (_LEGACY_MODEL, {}, (None, None)),
])
def test_the_access_path_and_its_provider_are_one_catalog_pair(monkeypatch, model, per_bot, expected):
    """TASK-1295 r2 trust contract (native_auth.resolve_native_route): the
    bridge has no catalog and checks only the pair's consistency, so the app
    forwards the catalog's pair as a unit and never forwards a provider of
    its own. A fabricated pair therefore can't reach the bridge; a hand-set
    path arrives without a provider and the bridge refuses it on a proxy turn."""
    forwarded = _service_bot_config(monkeypatch, model, per_bot)
    assert (forwarded.get("access_path"), forwarded.get("access_path_provider")) == expected
    if expected == ("bogus", None):
        with pytest.raises(EndpointRouteError):
            resolve_native_route("bogus", proxy_provider="xai", access_path_provider=None)


def test_service_bot_init_forwards_access_path_provider(monkeypatch):
    """The agent bot's config carries the provider with its access path."""
    from llm_bawt.service.core import ServiceLLMBawt

    bot = SimpleNamespace(agent_backend="claude-code", agent_backend_config={},
                          default_model="grok", harness="claude-code")
    model = dict(_PROXY_MODEL)
    config = SimpleNamespace(defined_models={"models": {"grok": model}})

    class Client:
        def __init__(self):
            self._bot_config = {}

    monkeypatch.setattr("llm_bawt.core.base.BaseLLMBawt._init_bot",
                        lambda self, _config: setattr(self, "bot", bot))
    monkeypatch.setattr("llm_bawt.clients.agent_backend_client.AgentBackendClient", Client)
    service = ServiceLLMBawt.__new__(ServiceLLMBawt)
    service.client, service.bot_id, service.user_id, service.config = Client(), "b", "u", config
    ServiceLLMBawt._init_bot(service, config)
    forwarded = service.client._bot_config
    assert (forwarded["model"], forwarded["access_path"], forwarded["access_path_provider"]) == (
        "xai/grok-4.5", "xai-responses", "xai")


def test_catalog_native_endpoint_sends_no_access_path_provider(monkeypatch):
    fields = _app_publishes(monkeypatch, _catalog_config(
        "anthropic-oauth", "anthropic", "anthropic-messages", harness="claude-code"))
    assert "access_path_provider" not in fields
    req = SendRequest.from_fields(fields)
    assert type(resolve_native_route(
        req.access_path, proxy_provider=None, access_path_provider=req.access_path_provider,
    )) is AnthropicOAuthRoute


# ── send handler: terminal error contract ─────────────────────────────────────


def _send(harness, **fields):
    return harness._handle_send({
        "request_id": "turn-request", "session_key": SESSION, "model": "claude-opus-5-5",
        "message": "hello", "trigger_message_id": "message-turn-request",
        "thread_resume_id": "sdk-session", **fields,
    }, "send-turn-request", harness.redis)


def _errors(harness):
    return [event for event in harness.events if event.kind == AgentEventKind.ERROR]


def test_unknown_access_path_fails_the_turn_before_the_sdk(monkeypatch):
    harness = _SendHarness()
    monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient",
                        lambda **_: pytest.fail("SDK must not start"))
    asyncio.run(_send(harness, access_path="bogus"))
    error, = _errors(harness)
    assert "access_path 'bogus'" in error.text
    assert CLAUDE_CREDENTIAL_ERROR_MARKER not in error.text
    harness._build_sdk_env.assert_not_called()


@pytest.mark.parametrize("fields", [
    {"access_path": "bogus"},                                         # unknown path
    {"access_path": "xai-responses", "access_path_provider": "xai"},  # another provider's path
])
def test_mismatched_proxy_access_path_fails_the_turn_before_the_sdk(monkeypatch, fields):
    from claude_code_bridge.bridge import ClaudeCodeBridge

    harness = _SendHarness()
    harness._proxy_base_url = PROXY_BASE
    harness._model_provider_prefix = ClaudeCodeBridge._model_provider_prefix
    monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient",
                        lambda **_: pytest.fail("SDK must not start"))
    asyncio.run(_send(harness, **{**fields, "model": "openai_chatgpt/gpt-6-sol"}))
    error, = _errors(harness)
    assert f"access_path {fields['access_path']!r}" in error.text
    harness._build_sdk_env.assert_not_called()


@pytest.mark.parametrize(("access_path", "claude_marker"), [
    ("anthropic-api", False), ("anthropic-oauth", True), (None, True),
])
def test_credential_failure_offers_claude_reconnect_only_on_subscription_routes(
    monkeypatch, access_path, claude_marker,
):
    harness = _SendHarness()
    harness._build_sdk_env.side_effect = RuntimeError("API Error: 401 invalid api key")
    monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient",
                        lambda **_: pytest.fail("SDK must not start"))
    extra = {} if access_path is None else {"access_path": access_path}
    asyncio.run(_send(harness, **extra))
    error, = _errors(harness)
    assert (CLAUDE_CREDENTIAL_ERROR_MARKER in error.text) is claude_marker
    route = harness._build_sdk_env.call_args.kwargs["native_route"]
    assert route.access_path == access_path


def test_api_key_tripwire_reaches_the_user_as_the_turn_error(monkeypatch, parent_env):
    """Strict anthropic-api with no broker key: the turn fails, naming the fix."""
    harness = _SendHarness()
    harness._proxy_base_url = PROXY_BASE
    del harness._build_sdk_env  # use the real env builder
    Sources(monkeypatch, api_key=BrokerKeyNotInstalled("503"))
    monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient",
                        lambda **_: pytest.fail("SDK must not start"))
    asyncio.run(_send(harness, access_path="anthropic-api"))
    error, = _errors(harness)
    assert "endpoint is anthropic-api but no Anthropic API key is connected" in error.text
    assert "repoint the bot to anthropic-oauth or connect a key" in error.text
    assert CLAUDE_CREDENTIAL_ERROR_MARKER not in error.text
