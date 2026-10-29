"""OpenRouter on the chat harness (TASK-1764).

A chat-harness bot must accept an ``openrouter-chat`` endpoint and build an
OpenAI-compatible client whose key comes from the openrouter connection.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llm_bawt.memory.model_catalog_migration import STANDARD_ACCESS_PATHS
from llm_bawt.model_catalog import (
    AccessPath,
    ModelCatalog,
    ModelEndpoint,
    ModelIdentity,
    ProtocolCompatibility,
)
from llm_bawt.service.core import ServiceLLMBawt
from llm_bawt.service.providers import registry
from llm_bawt.service.usage import usage_adapter_for_access_path

BASE = "https://openrouter.ai/api/v1"
UPSTREAM = "deepseek/deepseek-v4.1-flash"


def _path() -> AccessPath:
    seed = next(s for s in STANDARD_ACCESS_PATHS if s.key == "openrouter-chat")
    return AccessPath(id=1, key=seed.key, vendor=seed.vendor, protocol=seed.protocol,
                      base_url=seed.base_url, auth_mechanism=seed.auth_mechanism,
                      engine_kind=seed.engine_kind)


def _catalog() -> ModelCatalog:
    return ModelCatalog([ModelEndpoint(
        id=1,
        model=ModelIdentity(id=1, key="deepseek-flash", vendor="deepseek", display_name="DS"),
        access_path=_path(),
        upstream_model_id=UPSTREAM,
    )])


def test_seed_is_openrouter_chat_completions():
    path = _path()
    assert (path.vendor, path.protocol, path.base_url, path.auth_mechanism) == (
        "openrouter", "chat-completions", BASE, "api-key")


def test_chat_harness_accepts_openrouter_chat():
    assert ProtocolCompatibility.is_compatible("chat", _path())


def test_key_and_usage_are_owned_by_openrouter():
    assert registry._ACCESS_PATH_OWNERS["openrouter-chat"].id == "openrouter"
    assert usage_adapter_for_access_path("openrouter-chat").provider == "openrouter"


def test_chat_resolution_keeps_the_bare_upstream_id():
    model_def = _catalog().resolve("deepseek-flash", harness="chat")
    assert model_def["type"] == "openai-compatible"
    assert model_def["model_id"] == UPSTREAM
    assert model_def["base_url"] == BASE


def _service(model_def: dict) -> SimpleNamespace:
    return SimpleNamespace(model_definition=model_def, resolved_model_alias="deepseek-flash",
                           config=SimpleNamespace(TEMPERATURE=0.5, PLAIN_OUTPUT=True),
                           bot_id="b", user_id="u")


def test_chat_client_uses_the_broker_key():
    model_def = _catalog().resolve("deepseek-flash", harness="chat")
    svc = _service(model_def)
    owner = SimpleNamespace(id="openrouter", api_key=lambda: "sk-or-test")
    with patch("llm_bawt.service.providers.registry.access_path_adapter",
               return_value=owner) as lookup, \
         patch("llm_bawt.clients.openai_client.OpenAI") as sdk:
        client = ServiceLLMBawt._initialize_client(svc)
    lookup.assert_called_with(svc.config, "openrouter-chat")
    assert sdk.call_args.kwargs["api_key"] == "sk-or-test"
    assert sdk.call_args.kwargs["base_url"] == BASE
    assert client.model == UPSTREAM


def test_chat_client_without_a_connected_key_fails_visibly():
    svc = _service(_catalog().resolve("deepseek-flash", harness="chat"))
    owner = SimpleNamespace(id="openrouter", api_key=lambda: None)
    with patch("llm_bawt.service.providers.registry.access_path_adapter", return_value=owner), \
         pytest.raises(ValueError, match="no connected API key"):
        ServiceLLMBawt._initialize_client(svc)


def test_inline_key_never_bypasses_a_disconnected_provider():
    model_def = dict(_catalog().resolve("deepseek-flash", harness="chat"), api_key="sk-inline")
    owner = SimpleNamespace(id="openrouter", api_key=lambda: None)
    with patch("llm_bawt.service.providers.registry.access_path_adapter", return_value=owner), \
         pytest.raises(ValueError, match="no connected API key"):
        ServiceLLMBawt._initialize_client(_service(model_def))


def test_inline_key_still_serves_an_unowned_custom_path():
    from llm_bawt.service.background_clients import endpoint_api_key

    with patch("llm_bawt.service.providers.registry.access_path_adapter", return_value=None):
        assert endpoint_api_key(None, {"access_path": "openai-compatible",
                                       "auth_mechanism": "configured",
                                       "api_key": "sk-custom"}) == "sk-custom"


def test_tool_call_round_trips_over_chat_completions():
    from llm_bawt.models.message import Message

    model_def = _catalog().resolve("deepseek-flash", harness="chat")
    owner = SimpleNamespace(id="openrouter", api_key=lambda: "sk-or-test")
    tools = [{"type": "function", "function": {
        "name": "web_search", "description": "search",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}},
                       "required": ["q"]}}}]
    call = SimpleNamespace(id="call_1", type="function",
                           function=SimpleNamespace(name="web_search", arguments='{"q": "dogs"}'))
    completion = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="", tool_calls=[call]))])
    svc = _service(model_def)
    svc.config.TOP_P = 1.0
    svc.config.SYSTEM_MESSAGE = ""
    svc.config.get_model_max_tokens = lambda _alias: 256
    with patch("llm_bawt.service.providers.registry.access_path_adapter", return_value=owner), \
         patch("llm_bawt.clients.openai_client.OpenAI") as sdk:
        sdk.return_value.chat.completions.create.return_value = completion
        client = ServiceLLMBawt._initialize_client(svc)
        result, _ = client.query_with_tools([Message(role="user", content="find dogs")],
                                            tools_schema=tools, max_tokens=256)
    payload = sdk.return_value.chat.completions.create.call_args.kwargs
    assert payload["model"] == UPSTREAM
    assert payload["tools"] == tools and payload["tool_choice"] == "auto"
    assert result["tool_calls"] == [{"id": "call_1", "type": "function",
                                     "function": {"name": "web_search",
                                                  "arguments": '{"q": "dogs"}'}}]
