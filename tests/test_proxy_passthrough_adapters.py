"""Hermetic tests for the Anthropic-passthrough proxy adapters (Z.AI, Moonshot).

These pin the contract of
``claude_code_bridge.proxy.adapters.anthropic_passthrough`` — the shared base
extracted so Moonshot didn't have to copy ``ZaiAdapter`` wholesale (TASK-654).

The proxy provider adapters had **no** coverage before this file, which made
refactoring the working Z.AI path risky. The Z.AI cases here exist specifically
as regression guards: Z.AI must keep authenticating with ``x-api-key`` and must
never grow a Bearer header, and the namespaced-model rewrite must survive.

No network: a throwaway ``HTTPServer`` on 127.0.0.1 impersonates the upstream
Anthropic Messages endpoint and records exactly what the adapter sent.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from claude_code_bridge.proxy.adapters import lookup
from claude_code_bridge.proxy.adapters.moonshot import MoonshotAdapter
from claude_code_bridge.proxy.adapters.zai import ZaiAdapter
from claude_code_bridge.proxy.adapters import anthropic_passthrough
from claude_code_bridge.provider_key_broker import BrokerKeyNotInstalled


@pytest.fixture(autouse=True)
def _no_broker_key(monkeypatch):
    """The app broker has no stored key, so the legacy env keys apply (TASK-1476).

    Broker-first behaviour itself is covered in test_bridge_provider_keys.py.
    """

    def fake_fetch(provider_id):
        raise BrokerKeyNotInstalled(f"no {provider_id} key")

    monkeypatch.setattr(anthropic_passthrough, "fetch_provider_api_key", fake_fetch)

# One message_start (carrying a usage block the tap parses) + a stop event.
def _sse(model: str) -> bytes:
    return (
        b"event: message_start\n"
        b'data: {"type":"message_start","message":{"id":"m","model": "'
        + model.encode()
        + b'","usage":{"input_tokens":50,"cache_read_input_tokens":150,'
        b'"cache_creation_input_tokens":0,"output_tokens":2}}}\n\n'
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","delta":{"text":"OK"}}\n\n'
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )


@pytest.fixture
def fake_upstream():
    """Spin up a fake Anthropic endpoint; yields (base_url, captured)."""
    captured: dict = {}
    holder: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 — BaseHTTPRequestHandler API
            length = int(self.headers.get("content-length", 0))
            captured["path"] = self.path
            captured["headers"] = {k.lower(): v for k, v in self.headers.items()}
            captured["body"] = json.loads(self.rfile.read(length) or b"{}")
            status = holder.get("status", 200)
            if status >= 400:
                payload = json.dumps(
                    {"error": {"message": "Invalid Authentication"}}
                ).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                self.wfile.flush()
                return
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            self.wfile.write(_sse(holder["echo_model"]))
            self.wfile.flush()

        def log_message(self, *args):  # silence stderr noise
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    holder["echo_model"] = ""
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", captured, holder
    finally:
        srv.shutdown()
        srv.server_close()


def _run(adapter, body: dict, upstream_model: str) -> bytes:
    async def go() -> bytes:
        return b"".join([chunk async for chunk in adapter.call(body, upstream_model)])

    return asyncio.run(go())


# --- registry -----------------------------------------------------------


def test_both_providers_registered():
    assert isinstance(lookup("moonshot"), MoonshotAdapter)
    assert isinstance(lookup("zai"), ZaiAdapter)


def test_endpoints_are_the_documented_ones(monkeypatch):
    monkeypatch.delenv("MOONSHOT_BASE_URL", raising=False)
    monkeypatch.delenv("ZAI_BASE_URL", raising=False)
    # Moonshot's Claude Code guide documents this exact ANTHROPIC_BASE_URL.
    assert MoonshotAdapter._base_url() == "https://api.moonshot.ai/anthropic"
    assert ZaiAdapter._base_url() == "https://api.z.ai/api/anthropic"


# --- auth: the one thing that differs between the two ------------------


def test_zai_uses_x_api_key_only():
    """Regression: Z.AI authenticated with x-api-key before the refactor."""
    headers = ZaiAdapter()._auth_headers("K")
    assert headers == {"x-api-key": "K"}


def test_moonshot_sends_bearer():
    """Moonshot documents the ANTHROPIC_AUTH_TOKEN (Bearer) style."""
    headers = MoonshotAdapter()._auth_headers("K")
    assert headers["Authorization"] == "Bearer K"
    assert headers["x-api-key"] == "K"  # harmless compatibility mirror


def test_missing_key_error_points_at_the_providers_page(monkeypatch):
    for env in ("ZAI_API_KEY", "Z_AI_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    with pytest.raises(RuntimeError) as exc:
        ZaiAdapter()._api_key()
    assert str(exc.value) == (
        "Z.AI API key required: connect it in Settings → Providers."
    )


# --- the shared passthrough round-trip ---------------------------------


@pytest.mark.parametrize(
    "provider,model_field,upstream_model,key_env,url_env",
    [
        (
            "moonshot",
            "moonshot/kimi-k3[1m]",   # bracket suffix must survive intact
            "kimi-k3[1m]",
            "MOONSHOT_API_KEY",
            "MOONSHOT_BASE_URL",
        ),
        ("zai", "zai/glm-5.2", "glm-5.2", "ZAI_API_KEY", "ZAI_BASE_URL"),
    ],
)
def test_passthrough_round_trip(
    fake_upstream, monkeypatch, provider, model_field, upstream_model, key_env, url_env
):
    base_url, captured, holder = fake_upstream
    holder["echo_model"] = upstream_model  # upstream echoes the BARE name
    monkeypatch.setenv(url_env, base_url)
    monkeypatch.setenv(key_env, "secret-key")

    body = {
        "model": model_field,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,  # adapter must force this True
    }
    out = _run(lookup(provider), body, upstream_model)

    # ...hit the messages path with the negotiated API version
    assert captured["path"] == "/v1/messages"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"

    # ...send the BARE model upstream, and always stream
    assert captured["body"]["model"] == upstream_model
    assert captured["body"]["stream"] is True

    # ...rewrite the model back to the namespaced value, or the SDK CLI
    # rejects the response as malformed.
    assert f'"model": "{model_field}"'.encode() in out
    assert f'"model": "{upstream_model}"'.encode() not in out

    # ...and relay the rest of the stream untouched.
    assert b'"text":"OK"' in out
    assert b"message_stop" in out


def test_upstream_401_raises_labeled_runtime_error(fake_upstream, monkeypatch):
    """A revoked key must surface as ``RuntimeError`` mentioning 401.

    ``routes.py::_proxy_iter`` string-matches "401" / "auth" to classify the
    failure as an Anthropic ``authentication_error``, so both the status code
    and the provider label have to appear in the message.
    """
    base_url, _captured, holder = fake_upstream
    holder["echo_model"] = "kimi-k3"
    holder["status"] = 401
    monkeypatch.setenv("MOONSHOT_BASE_URL", base_url)
    monkeypatch.setenv("MOONSHOT_API_KEY", "revoked-key")
    with pytest.raises(RuntimeError, match=r"Moonshot upstream 401") as exc:
        _run(lookup("moonshot"), {"model": "moonshot/kimi-k3"}, "kimi-k3")
    # Upstream detail is preserved for the operator, and routes.py will map
    # this to authentication_error rather than a generic api_error.
    assert "Invalid Authentication" in str(exc.value)
    assert "401" in str(exc.value)


# --- openrouter (TASK-822): broker-backed key, nested-slash models ------


def test_openrouter_registered_with_documented_endpoint(monkeypatch):
    from claude_code_bridge.proxy.adapters.openrouter import OpenRouterAdapter

    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    assert isinstance(lookup("openrouter"), OpenRouterAdapter)
    assert OpenRouterAdapter._base_url() == "https://openrouter.ai/api"


def test_openrouter_sends_bearer():
    from claude_code_bridge.proxy.adapters.openrouter import OpenRouterAdapter

    headers = OpenRouterAdapter()._auth_headers("K")
    assert headers["Authorization"] == "Bearer K"
    assert headers["x-api-key"] == "K"  # harmless compatibility mirror


def test_openrouter_key_is_broker_only(monkeypatch):
    """The DB-driven key is the whole point (TASK-822): no env-var fallback,
    and a missing broker URL fails loudly rather than silently reading env."""
    from claude_code_bridge.proxy.adapters.openrouter import OpenRouterAdapter

    assert OpenRouterAdapter.API_KEY_ENVS == ()
    monkeypatch.delenv("LLM_BAWT_API_URL", raising=False)
    with pytest.raises(RuntimeError, match="LLM_BAWT_API_URL"):
        OpenRouterAdapter()._api_key()


def test_openrouter_caches_broker_key(monkeypatch):
    from claude_code_bridge.proxy.adapters.openrouter import OpenRouterAdapter

    adapter = OpenRouterAdapter()
    calls = {"n": 0}

    def fake_fetch():
        calls["n"] += 1
        return "sk-or-test"

    monkeypatch.setattr(adapter, "_fetch_broker_key", fake_fetch)
    assert adapter._api_key() == "sk-or-test"
    assert adapter._api_key() == "sk-or-test"
    assert calls["n"] == 1  # second call served from the TTL cache


def test_openrouter_round_trip_preserves_nested_slashes(fake_upstream, monkeypatch):
    """OpenRouter upstream ids contain slashes (``qwen/qwen3-coder``); the
    proxy splits provider on the FIRST slash only, so the bare upstream model
    must keep its inner slash both upstream and in the rewritten response."""
    from claude_code_bridge.proxy.adapters.openrouter import OpenRouterAdapter

    base_url, captured, holder = fake_upstream
    upstream_model = "qwen/qwen3-coder"
    holder["echo_model"] = upstream_model
    monkeypatch.setenv("OPENROUTER_BASE_URL", base_url)

    adapter = OpenRouterAdapter()
    monkeypatch.setattr(adapter, "_fetch_broker_key", lambda: "sk-or-test")

    body = {
        "model": "openrouter/qwen/qwen3-coder",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }
    out = _run(adapter, body, upstream_model)

    assert captured["path"] == "/v1/messages"
    assert captured["headers"]["authorization"] == "Bearer sk-or-test"
    assert captured["body"]["model"] == upstream_model
    assert captured["body"]["stream"] is True
    assert b'"model": "openrouter/qwen/qwen3-coder"' in out
    assert b"message_stop" in out


# --- usage tap ----------------------------------------------------------


def test_usage_tap_computes_cache_hit(caplog):
    """cache_read / (input + cache_read + cache_create) — 150/200 = 75%."""
    buf = _sse("glm-5.2").decode()
    with caplog.at_level("INFO"):
        _remaining, logged = ZaiAdapter._tap_usage(buf, "glm-5.2", False)
    assert logged is True
    assert "cache_hit=75.0%" in caplog.text
    assert "Z.AI usage" in caplog.text  # label preserved from pre-refactor


def test_usage_tap_is_idempotent_and_bounded():
    # Already-logged short-circuits.
    assert ZaiAdapter._tap_usage("anything", "m", True) == ("anything", True)
    # A flood with no usage line must not grow the buffer without bound.
    junk = "x" * 70000
    remaining, logged = ZaiAdapter._tap_usage(junk, "m", False)
    assert len(remaining) <= 4096 and logged is True


# --- local (Ollama / llama-server): keyless passthrough ------------------


def test_local_registered_with_keyless_default(monkeypatch):
    from claude_code_bridge.proxy.adapters.local import LocalAdapter, PLACEHOLDER_KEY

    monkeypatch.delenv("LOCAL_ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("LOCAL_ANTHROPIC_API_KEY", raising=False)
    adapter = lookup("local")
    assert isinstance(adapter, LocalAdapter)
    assert LocalAdapter._base_url() == "http://10.0.0.246:11434"
    # A missing key must NOT raise — local servers don't validate it.
    assert adapter._api_key() == PLACEHOLDER_KEY
    monkeypatch.setenv("LOCAL_ANTHROPIC_API_KEY", "real")
    assert adapter._api_key() == "real"


def test_local_round_trip_strips_tool_deferral(fake_upstream, monkeypatch):
    """Ollama/llama.cpp lack the tool-search beta: inline deferred tools."""
    base_url, captured, holder = fake_upstream
    holder["echo_model"] = "qwen3.8:27b"
    monkeypatch.setenv("LOCAL_ANTHROPIC_BASE_URL", base_url)
    monkeypatch.delenv("LOCAL_ANTHROPIC_API_KEY", raising=False)
    body = {
        "model": "local/qwen3.8:27b",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"name": "a", "input_schema": {"type": "object"}, "defer_loading": True},
            {"type": "tool_search_tool_regex_20251119", "name": "tool_search"},
        ],
    }
    out = _run(lookup("local"), body, "qwen3.8:27b")
    assert captured["path"] == "/v1/messages"
    assert captured["headers"]["x-api-key"] == "local"
    assert captured["body"]["model"] == "qwen3.8:27b"
    assert captured["body"]["tools"] == [{"name": "a", "input_schema": {"type": "object"}}]
    assert b'"model": "local/qwen3.8:27b"' in out
    assert b"message_stop" in out
