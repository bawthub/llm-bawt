"""TASK-1915: catalog-driven OpenRouter provider routing, end to end (hermetic).

Mirrors the ``responses_transport`` coverage: catalog validation, app ->
command serialization, command parsing, command -> proxy header -> request
context round trip, and the adapter body.  No network: the passthrough
upstream is the loopback fake from ``test_proxy_passthrough_adapters``.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from claude_code_bridge.proxy.openrouter_routing import (
    MAX_PROVIDER_NAMES,
    normalize_openrouter_provider,
    parse_openrouter_provider,
    serialize_openrouter_provider,
)
from claude_code_bridge.proxy.request_context import (
    OPENROUTER_PROVIDER_HEADER,
    REQUEST_HEADER,
    ProxyRequestContext,
    custom_header_env,
)
from tests.test_proxy_passthrough_adapters import (  # noqa: F401 — pytest fixtures
    _no_broker_key,
    fake_upstream,
)
from tests.publish_eval_args import split_publish_eval

ROUTING = {"ignore": ["Wafer"], "allow_fallbacks": True}
COMPACT = '{"allow_fallbacks":true,"ignore":["Wafer"]}'


# --- shared normalizer ---------------------------------------------------


def test_normalizer_canonicalizes_and_drops_unknown_keys():
    assert normalize_openrouter_provider({
        "order": [" DeepInfra ", "Novita", "DeepInfra"],
        "only": [],
        "ignore": ["Wafer", "google-vertex/us", "Together_AI 2.0"],
        "allow_fallbacks": False,
        "sort": "price",
        "data_collection": "deny",
    }) == {
        "order": ["DeepInfra", "Novita"],
        "ignore": ["Wafer", "google-vertex/us", "Together_AI 2.0"],
        "allow_fallbacks": False,
    }
    assert serialize_openrouter_provider(ROUTING) == COMPACT
    assert parse_openrouter_provider(COMPACT) == ROUTING


@pytest.mark.parametrize(
    "value",
    [
        None,
        "Wafer",
        ["Wafer"],
        {},
        {"sort": "price"},  # no supported key left
        {"ignore": []},  # empty after canonicalization
        {"ignore": "Wafer"},
        {"ignore": ["Wafer", 3]},
        {"ignore": [None]},
        {"ignore": [""]},
        {"ignore": ["   "]},
        {"ignore": ["Wafer\nX-Evil: 1"]},
        {"ignore": ['Wafer"']},
        {"ignore": ["Wafer;rm"]},
        {"ignore": ["Wäfer"]},
        {"ignore": ["x" * 65]},
        {"ignore": [f"p{i}" for i in range(MAX_PROVIDER_NAMES + 1)]},
        {"order": ["A"], "allow_fallbacks": "false"},
        {"allow_fallbacks": 1},
    ],
)
def test_normalizer_rejects_invalid(value):
    assert normalize_openrouter_provider(value) is None
    assert serialize_openrouter_provider(value) is None


def test_normalizer_accepts_limits():
    names = [f"{i:02d}" + "x" * 62 for i in range(MAX_PROVIDER_NAMES)]
    value = {"order": names, "only": names, "ignore": names, "allow_fallbacks": True}
    encoded = serialize_openrouter_provider(value)
    assert encoded is not None and "\n" not in encoded
    assert parse_openrouter_provider(encoded) == value


@pytest.mark.parametrize(
    "raw",
    [None, "", "not json", "[1,2]", '{"ignore":[1]}', " " * 5000, "{" * 5000, 7],
)
def test_parser_rejects_untrusted_garbage(raw):
    assert parse_openrouter_provider(raw) is None


# --- app side: catalog validation, init_bot, command ---------------------


def test_catalog_serving_config_normalizes_openrouter_provider():
    from llm_bawt.service.routes.model_catalog import EndpointWrite

    endpoint = EndpointWrite(serving_config={
        "openrouter_provider": {"ignore": [" Wafer "], "sort": "price"},
        "other": True,
    })
    assert endpoint.serving_config == {
        "openrouter_provider": {"ignore": ["Wafer"]},
        "other": True,
    }
    # Absent field: serving_config passes through untouched.
    assert EndpointWrite(serving_config={"other": 1}).serving_config == {"other": 1}


@pytest.mark.parametrize(
    "bad",
    ["Wafer", {"ignore": "Wafer"}, {"ignore": ["Wa;fer"]}, {"sort": "price"},
     {"allow_fallbacks": "no"}],
)
def test_catalog_serving_config_rejects_invalid_openrouter_provider(bad):
    from llm_bawt.service.routes.model_catalog import EndpointWrite

    with pytest.raises(ValidationError, match="openrouter_provider"):
        EndpointWrite(serving_config={"openrouter_provider": bad})


def test_init_bot_threads_catalog_openrouter_provider_to_bridge(monkeypatch):
    from tests.test_model_consolidation import _run_init_bot

    bot = SimpleNamespace(
        agent_backend="claude-code", agent_backend_config={}, default_model="flash",
    )
    model = {"type": "claude-code", "model_id": "openrouter/deepseek/deepseek-v4.1-flash"}
    service = _run_init_bot(
        monkeypatch, bot, {"flash": {**model, "openrouter_provider": ROUTING}}
    )
    assert service.client._bot_config["openrouter_provider"] == ROUTING

    absent = _run_init_bot(monkeypatch, bot, {"flash": model})
    assert "openrouter_provider" not in absent.client._bot_config

    invalid = _run_init_bot(
        monkeypatch, bot, {"flash": {**model, "openrouter_provider": {"ignore": "Wafer"}}}
    )
    assert "openrouter_provider" not in invalid.client._bot_config


def _published_command(**kwargs) -> dict:
    from agent_bridge.subscriber import RedisSubscriber

    subscriber = RedisSubscriber("redis://localhost:6379/0")
    fake = MagicMock()
    commands: list[dict] = []

    async def eval_script(_script, numkeys, *args):
        _keys, _request, flat = split_publish_eval(numkeys, args)
        commands.append(dict(zip(flat[::2], flat[1::2])))
        return [1, "1-0"]

    fake.eval = eval_script
    subscriber._pub_redis = fake
    asyncio.run(subscriber.send_command(
        session_key="snark:nick", message="hi", request_id="req-or", **kwargs,
    ))
    return commands[0]


def test_openrouter_provider_is_serialized_into_bridge_command():
    assert _published_command(openrouter_provider=ROUTING)["openrouter_provider"] == COMPACT
    assert "openrouter_provider" not in _published_command()
    assert "openrouter_provider" not in _published_command(openrouter_provider=None)


# --- bridge side: command -> header -> proxy context ----------------------

_FIELDS = {
    "request_id": "req-1",
    "session_key": "snark:nick",
    "bot_id": "snark",
    "message": "hello",
    "model": "openrouter/deepseek/deepseek-v4.1-flash",
}


def test_send_request_parses_and_ignores_invalid_openrouter_provider(caplog):
    from claude_code_bridge.send_request import SendRequest

    assert SendRequest.from_fields(
        {**_FIELDS, "openrouter_provider": COMPACT}
    ).openrouter_provider == ROUTING
    assert SendRequest.from_fields(_FIELDS).openrouter_provider is None
    with caplog.at_level(logging.WARNING):
        parsed = SendRequest.from_fields(
            {**_FIELDS, "openrouter_provider": '{"ignore":"Wafer"}'}
        )
    assert parsed.openrouter_provider is None
    assert "Ignoring invalid openrouter_provider" in caplog.text


def _headers(env_value: str) -> dict[str, str]:
    return dict(line.split(": ", 1) for line in env_value.splitlines())


def test_custom_header_carries_compact_routing_only_when_present():
    base = dict(request_id="r", provider="openrouter", bot_id="snark",
                conversation_id="0" * 32)
    headers = _headers(custom_header_env(
        ProxyRequestContext(**base, openrouter_provider=ROUTING)
    ))
    assert headers[OPENROUTER_PROVIDER_HEADER] == COMPACT
    assert OPENROUTER_PROVIDER_HEADER not in _headers(
        custom_header_env(ProxyRequestContext(**base))
    )


def test_command_to_header_to_proxy_context_round_trip(monkeypatch):
    from claude_code_bridge.proxy.routes import messages
    from claude_code_bridge.send_request import SendRequest
    from claude_code_bridge.send_stream import ClaudeStreamMixin

    class Harness(ClaudeStreamMixin):
        _proxy_base_url = "http://127.0.0.1:12345"

    req = SendRequest.from_fields({**_FIELDS, "openrouter_provider": COMPACT})
    env = Harness()._build_sdk_env(
        use_proxy=True,
        model=req.model,
        subagent_model=None,
        force_refresh=False,
        bot_id="snark",
        session_key="snark:nick",
        thread_session_id="thread-1",
        request_id="request-1",
        openrouter_provider=req.openrouter_provider,
    )
    headers = _headers(env["ANTHROPIC_CUSTOM_HEADERS"])
    assert headers[OPENROUTER_PROVIDER_HEADER] == COMPACT

    captured: list[ProxyRequestContext] = []

    class Adapter:
        def account_hash(self):
            return "mock-account"

        async def call(self, body, upstream_model, context):
            captured.append(context)
            yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'

    def request_with(header_map):
        class Request:
            headers = header_map

            async def json(self):
                return {"model": req.model,
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True}
        return Request()

    monkeypatch.setattr("claude_code_bridge.proxy.routes.lookup", lambda _p: Adapter())

    async def run_route(request):
        response = await messages(request)
        return b"".join([chunk async for chunk in response.body_iterator])

    asyncio.run(run_route(request_with(headers)))
    # Untrusted header values are re-validated; garbage is dropped, not applied.
    asyncio.run(run_route(request_with({
        REQUEST_HEADER: "request-2",
        OPENROUTER_PROVIDER_HEADER: '{"ignore":["Wa\\"fer"],"order":"x"}',
    })))
    asyncio.run(run_route(request_with({REQUEST_HEADER: "request-3"})))
    assert [c.openrouter_provider for c in captured] == [ROUTING, None, None]


# --- adapter body ----------------------------------------------------------


def _call(adapter, upstream_model, context):
    body = {
        "model": f"{adapter.name}/{upstream_model}",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }

    async def go():
        return b"".join([c async for c in adapter.call(body, upstream_model, context)])

    asyncio.run(go())
    return body


def _ctx(provider, routing):
    return ProxyRequestContext(request_id="req-or", provider=provider,
                               openrouter_provider=routing)


def test_openrouter_adapter_sends_provider_preference(fake_upstream, monkeypatch, caplog):  # noqa: F811
    from claude_code_bridge.proxy.adapters.openrouter import OpenRouterAdapter

    base_url, captured, holder = fake_upstream
    upstream = "deepseek/deepseek-v4.1-flash"
    holder["echo_model"] = upstream
    monkeypatch.setenv("OPENROUTER_BASE_URL", base_url)
    adapter = OpenRouterAdapter()
    monkeypatch.setattr(adapter, "_fetch_broker_key", lambda: "sk-or-test")

    with caplog.at_level(logging.INFO):
        original = _call(adapter, upstream, _ctx("openrouter", ROUTING))
    assert captured["body"]["provider"] == ROUTING
    assert "provider" not in original  # caller's body is never mutated
    lines = [r.getMessage() for r in caplog.records
             if "OpenRouter provider routing" in r.getMessage()]
    assert lines == [
        f"OpenRouter provider routing request_id=req-or model={upstream} "
        f"provider={COMPACT}"
    ]

    # Absent preference (or no context at all): body is unchanged.
    for context in (_ctx("openrouter", None), None):
        captured.clear()
        _call(adapter, upstream, context)
        assert "provider" not in captured["body"]


@pytest.mark.parametrize(
    "provider,upstream,key_env,url_env",
    [
        ("moonshot", "kimi-k3", "MOONSHOT_API_KEY", "MOONSHOT_BASE_URL"),
        ("zai", "glm-5.2", "ZAI_API_KEY", "ZAI_BASE_URL"),
    ],
)
def test_other_passthrough_adapters_never_send_provider(
    fake_upstream, monkeypatch, provider, upstream, key_env, url_env  # noqa: F811
):
    from claude_code_bridge.proxy.adapters import lookup

    base_url, captured, holder = fake_upstream
    holder["echo_model"] = upstream
    monkeypatch.setenv(url_env, base_url)
    monkeypatch.setenv(key_env, "secret-key")
    _call(lookup(provider), upstream, _ctx(provider, ROUTING))
    assert "provider" not in captured["body"]
