"""TASK-1636: DB-backed prompt cache TTL (bot > global > unset), native-only env."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent_bridge.prompt_cache import PROMPT_CACHE_TTLS, normalize_prompt_cache_ttl
from claude_code_bridge.context_env import PROMPT_CACHE_TTL_ENV, prompt_cache_ttl_env
from claude_code_bridge.send_request import SendRequest
from claude_code_bridge.send_stream import ClaudeStreamMixin


class _FakeStore:
    """Stands in for RuntimeSettingsStore: scope rows keyed by (scope, id)."""

    rows: dict[tuple[str, str], dict] = {}

    def __init__(self, config):
        self.engine = object()

    def get_scope_settings(self, scope_type, scope_id):
        return dict(self.rows.get((scope_type, scope_id), {}))


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr("llm_bawt.runtime_settings.RuntimeSettingsStore", _FakeStore)
    _FakeStore.rows = {}
    return _FakeStore.rows


def test_registry_default_is_unset_and_agent_scoped() -> None:
    from llm_bawt.setting_definitions import SETTING_DEFINITIONS

    definition = SETTING_DEFINITIONS["prompt_cache_ttl"]
    assert definition.default is None
    assert definition.applies_to == ("agent",)
    assert PROMPT_CACHE_TTLS == ("5m", "1h")


@pytest.mark.parametrize(
    ("global_rows", "bot_rows", "expected"),
    [
        ({}, {}, None),
        ({"prompt_cache_ttl": "1h"}, {}, "1h"),
        ({"prompt_cache_ttl": "1h"}, {"prompt_cache_ttl": "5m"}, "5m"),
        ({}, {"prompt_cache_ttl": "5m"}, "5m"),
    ],
)
def test_resolution_order_bot_then_global_then_unset(store, global_rows, bot_rows, expected) -> None:
    from llm_bawt.runtime_setting_resolution import resolve_bot_runtime_setting

    store[("global", "*")] = global_rows
    store[("bot", "byte")] = bot_rows
    store[("bot", "other")] = {"prompt_cache_ttl": "1h"}
    assert resolve_bot_runtime_setting(SimpleNamespace(), "byte", "prompt_cache_ttl") == expected


@pytest.mark.parametrize(("raw", "expected"), [
    (None, None), ("", None), ("5m", "5m"), (" 1H ", "1h"), ("10m", None), (3600, None),
])
def test_normalize_allowlist(raw, expected) -> None:
    assert normalize_prompt_cache_ttl(raw) == expected


def _client(bot_id: str):
    from llm_bawt.clients.agent_backend_client import AgentBackendClient

    configs: list[dict] = []

    class Backend:
        async def chat_full(self, prompt, config):
            configs.append(dict(config))
            return SimpleNamespace(text="ok")

    client = AgentBackendClient.__new__(AgentBackendClient)
    client._backend_name = "claude-code"
    client._bot_config = {"bot_id": bot_id}
    client._backend = Backend()
    client.config = SimpleNamespace()
    return client, configs


def test_client_resolves_per_turn_and_drops_invalid(store) -> None:
    client, configs = _client("byte")
    store[("global", "*")] = {"prompt_cache_ttl": "5m"}
    asyncio.run(client._chat_full("hi"))
    store[("bot", "byte")] = {"prompt_cache_ttl": "1h"}
    asyncio.run(client._chat_full("hi"))
    store[("bot", "byte")] = {"prompt_cache_ttl": "forever"}
    asyncio.run(client._chat_full("hi"))
    assert [c["prompt_cache_ttl"] for c in configs] == ["5m", "1h", None]
    assert "prompt_cache_ttl" not in client._bot_config


def test_non_claude_backend_gets_no_ttl(store) -> None:
    client, configs = _client("byte")
    client._backend_name = "codex"
    store[("global", "*")] = {"prompt_cache_ttl": "5m"}
    asyncio.run(client._chat_full("hi"))
    assert "prompt_cache_ttl" not in configs[0]


def test_publisher_and_send_request_round_trip() -> None:
    from agent_bridge.command_publisher import CommandPublisherMixin

    seen: list[dict] = []

    class Redis:
        async def xadd(self, stream, fields, **kwargs):
            seen.append(dict(fields))
            return "id"

    publisher = CommandPublisherMixin()
    publisher._pub_redis = Redis()
    asyncio.run(publisher.send_command("byte:nick", "hi", "r1", prompt_cache_ttl="5m"))
    asyncio.run(publisher.send_command("byte:nick", "hi", "r2"))
    assert seen[0]["prompt_cache_ttl"] == "5m"
    assert "prompt_cache_ttl" not in seen[1]
    assert SendRequest.from_fields(seen[0]).prompt_cache_ttl == "5m"
    assert SendRequest.from_fields(seen[1]).prompt_cache_ttl is None
    assert SendRequest.from_fields({**seen[0], "prompt_cache_ttl": "2h"}).prompt_cache_ttl is None


class _Harness(ClaudeStreamMixin):
    _proxy_base_url = "http://127.0.0.1:12345"


def _env(*, use_proxy: bool, ttl: str | None, model: str = "claude-opus-5-5") -> dict:
    return _Harness()._build_sdk_env(
        use_proxy=use_proxy,
        model=model,
        subagent_model=None,
        force_refresh=False,
        bot_id="byte",
        session_key="byte:nick",
        thread_session_id="thread-1",
        request_id="request-1",
        prompt_cache_ttl=ttl,
    )


@pytest.mark.parametrize("ttl", ["5m", "1h"])
def test_native_turn_emits_resolved_ttl(monkeypatch, ttl) -> None:
    monkeypatch.setattr(
        "claude_code_bridge.native_auth._get_fresh_oauth_token", lambda **_: None
    )
    assert _env(use_proxy=False, ttl=ttl)[PROMPT_CACHE_TTL_ENV] == ttl


@pytest.mark.parametrize("ttl", [None, "", "2h"])
def test_native_turn_unset_leaves_cli_default(monkeypatch, ttl) -> None:
    monkeypatch.setattr(
        "claude_code_bridge.native_auth._get_fresh_oauth_token", lambda **_: None
    )
    assert PROMPT_CACHE_TTL_ENV not in _env(use_proxy=False, ttl=ttl)
    assert prompt_cache_ttl_env(ttl) == {}


def test_proxy_turn_never_emits_ttl() -> None:
    env = _env(use_proxy=True, ttl="5m", model="local/qwen3.8:27b-64k")
    assert PROMPT_CACHE_TTL_ENV not in env
