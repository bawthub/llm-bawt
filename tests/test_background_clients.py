"""TASK-1726: background clients per serving protocol, chosen by the catalog."""
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llm_bawt.clients.anthropic_messages_client import AnthropicMessagesClient
from llm_bawt.model_catalog import (
    AccessPath, IncompatibleModelError, ModelCatalog, ModelEndpoint, ModelIdentity,
)
from llm_bawt.models.message import Message
from llm_bawt.service import background_clients as bc
from llm_bawt.service import maintenance_health as mh

# (access path key, vendor, protocol, auth, engine_kind, base_url)
PATHS = [
    ("anthropic-api", "anthropic", "anthropic-messages", "api-key", None, "https://a.example"),
    ("anthropic-oauth", "anthropic", "anthropic-messages", "oauth", None, "https://a.example"),
    ("kimi-chat", "kimi", "chat-completions", "api-key", None, "https://k.example/v1"),
    ("local-anthropic", "local", "anthropic-messages", "none", None, "http://l.example:1"),
    ("local-llamacpp", "local", "chat-completions", "none", "llama-cpp", None),
    ("local-vllm", "local", "chat-completions", "none", "vllm", None),
    ("ollama", "local", "chat-completions", "none", "ollama", "http://l.example:1/v1"),
    ("openai-api", "openai", "chat-completions", "api-key", None, "https://o.example/v1"),
    ("openai-oauth", "openai", "responses", "oauth", None, "https://c.example"),
    ("openai-responses", "openai", "responses", "api-key", None, "https://o.example/v1"),
    ("openrouter-anthropic", "openrouter", "anthropic-messages", "api-key", None, "https://r.example"),
    ("xai-chat", "xai", "chat-completions", "api-key", None, "https://x.example/v1"),
    ("xai-responses", "xai", "responses", "api-key", None, "https://x.example/v1"),
    # A subscription chat-completions path: the chat harness admits it.
    ("zai-oauth", "zai", "chat-completions", "oauth", None, "https://z.example/v1"),
]

# access path -> factory that must serve it on the background harness (None = rejected)
EXPECTED = {
    "anthropic-api": "anthropic-messages",
    "anthropic-oauth": None,
    "kimi-chat": "chat-completions",
    "local-anthropic": "anthropic-messages",
    "local-llamacpp": "local bridge",
    "local-vllm": "local bridge",
    "ollama": "ollama",
    "openai-api": "openai chat",
    "openai-oauth": None,
    "openai-responses": "responses",
    "openrouter-anthropic": "anthropic-messages",
    "xai-chat": "xai",
    "xai-responses": "xai",
    "zai-oauth": None,
}


def _catalog(paths=PATHS, models=None) -> ModelCatalog:
    endpoints = []
    for i, (key, vendor, protocol, auth, engine, base) in enumerate(paths, start=1):
        model_key = (models or {}).get(key, f"m-{key}")
        endpoints.append(ModelEndpoint(
            id=i,
            model=ModelIdentity(id=i, key=model_key, vendor=vendor, display_name=model_key),
            access_path=AccessPath(id=i, key=key, vendor=vendor, protocol=protocol,
                                   base_url=base, auth_mechanism=auth, engine_kind=engine),
            upstream_model_id=f"up-{key}",
        ))
    return ModelCatalog(endpoints)


def _config(catalog: ModelCatalog):
    return SimpleNamespace(resolve_model=lambda ref, harness=None, default=None:
                           catalog.resolve(ref, harness=harness),
                           OLLAMA_URL="http://unused:11434", TEMPERATURE=0.5, PLAIN_OUTPUT=True)


@pytest.mark.parametrize("path", [p[0] for p in PATHS])
def test_resolver_matrix(path):
    cfg = _config(_catalog())
    expected = EXPECTED[path]
    if expected is None:
        with pytest.raises(IncompatibleModelError):
            bc.resolve_background_model(cfg, f"m-{path}")
        return
    model_def = bc.resolve_background_model(cfg, f"m-{path}")
    assert model_def["access_path"] == path
    factory = bc.factory_for(model_def)
    assert factory is not None and factory.name == expected
    # Background resolution never adds a provider prefix to the upstream id.
    assert bc.BackgroundClientFactory.model_id(model_def, "alias") == f"up-{path}"


def test_chat_endpoint_keeps_priority_over_background_only_ones():
    cfg = _config(_catalog(models={"xai-chat": "shared", "xai-responses": "shared",
                                   "anthropic-api": "shared"}))
    assert bc.resolve_background_model(cfg, "shared")["access_path"] == "xai-chat"


def test_oauth_chat_endpoint_falls_through_to_an_api_key_one():
    cfg = _config(_catalog(models={"zai-oauth": "shared", "anthropic-api": "shared"}))
    assert bc.resolve_background_model(cfg, "shared")["access_path"] == "anthropic-api"


def test_no_auth_chat_endpoint_never_gets_the_platform_openai_key():
    cfg = _config(_catalog(paths=[
        ("local-oai", "local", "chat-completions", "none", None, "http://l.example:2/v1")]))
    model_def = bc.resolve_background_model(cfg, "m-local-oai")
    with patch("llm_bawt.service.providers.api_key.resolve_api_key",
               return_value="sk-platform-openai") as platform, \
         patch("llm_bawt.clients.openai_client.OpenAI") as sdk:
        bc.build_background_client(cfg, model_def, "x")
    platform.assert_not_called()
    assert sdk.call_args.kwargs["api_key"] == bc.NO_KEY


def test_every_factory_client_supports_both_maintenance_call_shapes():
    from llm_bawt.clients.base import LLMClient
    from llm_bawt.clients.grok_client import GrokClient
    from llm_bawt.clients.openai_client import OpenAIClient
    from llm_bawt.clients.responses_client import ResponsesClient

    for cls in (AnthropicMessagesClient, OpenAIClient, ResponsesClient, GrokClient):
        assert cls.query_bounded is not LLMClient.query_bounded, cls
        assert cls.query is not LLMClient.query, cls


def _adapter(key="sk-test", pid="prov"):
    return SimpleNamespace(id=pid, api_key=lambda: key)


def test_anthropic_api_key_comes_from_the_access_path_owner():
    cfg = _config(_catalog())
    model_def = bc.resolve_background_model(cfg, "m-anthropic-api")
    with patch("llm_bawt.service.providers.registry.access_path_adapter",
               return_value=_adapter()) as owner, \
         patch("llm_bawt.clients.anthropic_messages_client.Anthropic") as sdk:
        client, kind = bc.build_background_client(cfg, model_def, "m-anthropic-api")
    owner.assert_called_with(cfg, "anthropic-api")
    assert kind == "anthropic-messages" and isinstance(client, AnthropicMessagesClient)
    sdk.assert_called_once_with(api_key="sk-test", base_url="https://a.example")


def test_api_key_path_without_a_connected_key_is_rejected_visibly():
    cfg = _config(_catalog())
    model_def = bc.resolve_background_model(cfg, "m-openrouter-anthropic")
    with patch("llm_bawt.service.providers.registry.access_path_adapter",
               return_value=_adapter(key=None)), \
         pytest.raises(bc.UnsupportedBackgroundModel, match="no connected API key"):
        bc.build_background_client(cfg, model_def, "x")


def test_no_auth_paths_need_no_key():
    cfg = _config(_catalog())
    with patch("llm_bawt.clients.anthropic_messages_client.Anthropic") as sdk:
        bc.build_background_client(cfg, bc.resolve_background_model(cfg, "m-local-anthropic"), "x")
    assert sdk.call_args.kwargs["base_url"] == "http://l.example:1"
    with patch("llm_bawt.clients.openai_client.OpenAIClient") as oai:
        bc.build_background_client(cfg, bc.resolve_background_model(cfg, "m-ollama"), "x")
    assert oai.call_args.kwargs["base_url"] == "http://l.example:1/v1"
    assert oai.call_args.kwargs["api_key"] == "ollama"


def test_unknown_shape_is_unsupported_not_substituted():
    with pytest.raises(bc.UnsupportedBackgroundModel):
        bc.build_background_client(SimpleNamespace(), {"type": "claude-code"}, "x")


def _messages_client(text="hello"):
    sdk = MagicMock()
    sdk.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=f"  {text} ")])
    sdk.with_options.return_value = sdk
    cfg = SimpleNamespace(TEMPERATURE=0.5, PLAIN_OUTPUT=True, get_model_max_tokens=lambda alias: 777)
    with patch("llm_bawt.clients.anthropic_messages_client.Anthropic", return_value=sdk):
        client = AnthropicMessagesClient("up", cfg, base_url="https://a.example", api_key="k")
    return client, sdk


def test_messages_client_query_shape():
    client, sdk = _messages_client()
    out = client.query([Message(role="system", content="S1"), Message(role="system", content="S2"),
                        Message(role="user", content="u1"), Message(role="user", content="u2")],
                       plaintext_output=True, stream=False, max_tokens=50, temperature=0.1)
    assert out == "hello"
    kwargs = sdk.messages.create.call_args.kwargs
    assert kwargs["system"] == "S1\n\nS2" and kwargs["max_tokens"] == 50
    assert kwargs["temperature"] == 0.1 and kwargs["model"] == "up"
    assert kwargs["messages"] == [{"role": "user", "content": [
        {"type": "text", "text": "u1"}, {"type": "text", "text": "u2"}]}]


def test_messages_client_defaults_and_bounded():
    client, sdk = _messages_client()
    client.query([Message(role="user", content="u")])
    kwargs = sdk.messages.create.call_args.kwargs
    assert kwargs["max_tokens"] == 777 and kwargs["temperature"] == 0.5 and "system" not in kwargs
    assert client.query_bounded([Message(role="user", content="u")], timeout_seconds=3,
                                max_tokens=9) == "hello"
    sdk.with_options.assert_called_with(timeout=3, max_retries=0)


def _bad_request(message):
    import anthropic
    import httpx
    request = httpx.Request("POST", "https://a.example/v1/messages")
    return anthropic.BadRequestError(message, response=httpx.Response(400, request=request), body=None)


def test_messages_client_drops_temperature_a_model_rejects_and_remembers():
    client, sdk = _messages_client()
    ok = sdk.messages.create.return_value
    sdk.messages.create.side_effect = [_bad_request("`temperature` is deprecated for this model."), ok, ok]
    assert client.query_bounded([Message(role="user", content="u")], timeout_seconds=3, max_tokens=9) == "hello"
    first, retry = (c.kwargs for c in sdk.messages.create.call_args_list)
    assert first["temperature"] == 0.0 and "temperature" not in retry
    client.query([Message(role="user", content="u")])
    assert "temperature" not in sdk.messages.create.call_args.kwargs
    assert sdk.messages.create.call_count == 3


def test_messages_client_effort_none_disables_thinking():
    client, sdk = _messages_client()
    assert client.job_request_options("summaries", reasoning_effort="none") == {"thinking": {"type": "disabled"}}
    assert client.job_request_options("summaries", reasoning_effort="low") == {}
    assert client.job_request_options("summaries") == {}
    client.query([Message(role="user", content="u")], **client.job_request_options("x", reasoning_effort="none"))
    assert sdk.messages.create.call_args.kwargs["thinking"] == {"type": "disabled"}


def test_messages_client_drops_thinking_an_endpoint_rejects_and_keeps_temperature_memory():
    client, sdk = _messages_client()
    ok = sdk.messages.create.return_value
    sdk.messages.create.side_effect = [
        _bad_request("`temperature` is deprecated for this model."),
        _bad_request('"thinking.type.disabled" is not supported for this model.'),
        ok, ok]
    options = client.job_request_options("x", reasoning_effort="none")
    assert client.query([Message(role="user", content="u")], **options) == "hello"
    calls = [c.kwargs for c in sdk.messages.create.call_args_list]
    assert "thinking" in calls[1] and "temperature" not in calls[1]
    assert "thinking" not in calls[2] and "temperature" not in calls[2]
    client.query([Message(role="user", content="u")], **options)
    assert sdk.messages.create.call_count == 4
    last = sdk.messages.create.call_args.kwargs
    assert "thinking" not in last and "temperature" not in last


def test_messages_client_other_bad_requests_are_not_retried():
    client, sdk = _messages_client()
    sdk.messages.create.side_effect = _bad_request("max_tokens too large")
    with pytest.raises(Exception, match="max_tokens"):
        client.query_bounded([Message(role="user", content="u")], timeout_seconds=3, max_tokens=9)
    assert sdk.messages.create.call_count == 1


def test_messages_client_images():
    client, sdk = _messages_client()
    client.query([Message(role="user", content="see", content_parts=[
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
        {"type": "image_url", "image_url": {"url": "https://i.example/a.png"}}])])
    blocks = sdk.messages.create.call_args.kwargs["messages"][0]["content"]
    assert blocks[1]["source"] == {"type": "base64", "media_type": "image/png", "data": "aGk="}
    assert blocks[2]["source"] == {"type": "url", "url": "https://i.example/a.png"}


class _SpendError(Exception):
    status_code = 403
    headers: dict = {}

    def __init__(self):
        super().__init__("personal-team-blocked:spending-limit")


@pytest.mark.parametrize("backup_ref", [
    "backup",
    # The backup model also has a second endpoint, so only the endpoint ref is
    # unambiguous; labeling must use the endpoint the client was built from.
    "backup@anthropic-api",
])
def test_spending_limit_falls_back_to_a_non_xai_backup_end_to_end(monkeypatch, backup_ref):
    """Primary on xai-chat hits its spending limit; the call is replayed on an
    Anthropic-Messages backup and reported as served by the backup."""
    from llm_bawt.service import maintenance_fallback as mf
    from llm_bawt.service.instance_manager import InstanceManagerMixin

    models = {"xai-chat": "primary", "anthropic-api": "backup"}
    if "@" in backup_ref:
        models["openrouter-anthropic"] = "backup"
    catalog = _catalog(models=models)
    health = mh.MaintenanceHealth()
    monkeypatch.setattr(mh, "_registry", health)
    monkeypatch.setattr("llm_bawt.runtime_settings.resolve_job_model",
                        lambda c, k: {"maintenance_model": "primary",
                                      mf.FALLBACK_SETTING: backup_ref}.get(k))
    monkeypatch.setattr(mf, "default_store", lambda config: mf.MemoryExhaustionStore())
    resolver = MagicMock()
    resolver.return_value.resolve.return_value = 600
    monkeypatch.setattr("llm_bawt.runtime_settings.RuntimeSettingsResolver", resolver)

    owners = {"xai-chat": _adapter("xk", "xai"), "anthropic-api": _adapter("ak", "anthropic-api")}
    grok = MagicMock()
    grok.return_value.query.side_effect = _SpendError()
    sdk = MagicMock()
    sdk.messages.create.return_value = SimpleNamespace(content=[SimpleNamespace(type="text", text="from backup")])

    svc = InstanceManagerMixin.__new__(InstanceManagerMixin)
    svc.config = _config(catalog)
    svc.config.TEMPERATURE = 0.5
    svc.config.get_model_max_tokens = lambda alias: 100
    svc._bg_client_cache, svc._bg_client_lock = {}, threading.Lock()
    with patch("llm_bawt.service.providers.registry.access_path_adapter",
               side_effect=lambda cfg, key: owners.get(key)), \
         patch("llm_bawt.service.providers.base.CredentialStore") as store, \
         patch("llm_bawt.clients.grok_client.GrokClient", grok), \
         patch("llm_bawt.clients.anthropic_messages_client.Anthropic", return_value=sdk):
        store.return_value.load.return_value = None
        client, alias = svc._get_background_client(job="memory_extraction", allow_fallback=True)
        out = client.query(messages=[Message(role="user", content="x")],
                           plaintext_output=True, stream=False)

    assert alias == "primary" and out == "from backup"
    assert mf.served_model(client, alias) == "backup"
    assert sdk.messages.create.call_args.kwargs["model"] == "up-anthropic-api"


def test_no_model_or_provider_branches_in_factory_module():
    import re
    text = Path(bc.__file__).read_text()
    assert not re.search(r"\b(gpt-|grok-|claude-|gemini-|llama)", text)
    assert not re.search(r"provider\s*==", text)
