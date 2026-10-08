"""Upstream model discovery contracts used by the Add Model dialog."""

from __future__ import annotations

import httpx
import pytest

from llm_bawt.service.model_discovery import (
    CodexBridgeDiscoveryProvider,
    KimiCodingDiscoveryProvider,
    ModelDiscoveryError,
    discover_models,
)
from llm_bawt.service.routes import models as model_routes


def test_kimi_discovery_preserves_display_name_and_context(monkeypatch):
    payload = {
        "data": [
            {
                "id": "k3",
                "display_name": "K3",
                "context_length": 262_144,
                "supports_reasoning": True,
            },
            {
                "id": "kimi-for-coding",
                "display_name": "K2.7 Coding",
                "context_length": 262_144,
            },
            {
                "id": "kimi-for-coding-highspeed",
                "display_name": "K2.7 Coding Highspeed",
                "context_length": 262_144,
            },
        ]
    }

    class _Response:
        status_code = 200

        def __init__(self, body):
            self.body = body

        @staticmethod
        def raise_for_status():
            return None

        def json(self):
            return self.body

    def fake_get(url, *, headers, timeout):
        assert headers == {
            "Authorization": "Bearer coding-key",
            "Accept": "application/json",
        }
        assert timeout == 20.0
        if url == "https://api.kimi.com/coding/v1/models":
            return _Response(payload)
        assert url == "https://api.kimi.com/coding/v1/usages"
        return _Response({"user": {"membership": {"level": "LEVEL_BASIC"}}})

    monkeypatch.setenv("KIMI_CODING_API_KEY", "coding-key")
    monkeypatch.setattr("llm_bawt.service.model_discovery.httpx.get", fake_get)

    assert KimiCodingDiscoveryProvider().fetch() == [
        {"id": "k3", "description": "K3", "context_length": 262_144},
        {
            "id": "kimi-for-coding",
            "description": "K2.7 Coding",
            "context_length": 262_144,
        },
    ]


def test_kimi_discovery_keeps_highspeed_for_upgraded_members(monkeypatch):
    class _Response:
        @staticmethod
        def raise_for_status():
            return None

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fake_get(url, **_kwargs):
        if url.endswith("/models"):
            return _Response(
                {
                    "data": [
                        {
                            "id": "kimi-for-coding-highspeed",
                            "display_name": "K2.7 Coding Highspeed",
                            "context_length": 262_144,
                        }
                    ]
                }
            )
        return _Response({"user": {"membership": {"level": "LEVEL_ALLEGRETTO"}}})

    monkeypatch.setenv("KIMI_CODING_API_KEY", "coding-key")
    monkeypatch.setattr("llm_bawt.service.model_discovery.httpx.get", fake_get)

    assert KimiCodingDiscoveryProvider().fetch() == [
        {
            "id": "kimi-for-coding-highspeed",
            "description": "K2.7 Coding Highspeed",
            "context_length": 262_144,
        }
    ]


def test_kimi_discovery_requires_its_subscription_key(monkeypatch):
    monkeypatch.delenv("KIMI_CODING_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    with pytest.raises(ModelDiscoveryError) as exc_info:
        KimiCodingDiscoveryProvider().fetch()

    assert exc_info.value.status_code == 503
    assert "KIMI_CODING_API_KEY" in str(exc_info.value)


def test_kimi_provider_aliases_resolve(monkeypatch):
    expected = [{"id": "k3", "description": "K3", "context_length": 262_144}]
    monkeypatch.setattr(KimiCodingDiscoveryProvider, "fetch", lambda _self: expected)

    assert discover_models("kimi") == expected
    assert discover_models("kimi_coding") == expected
    assert discover_models("kimi-code") == expected


def test_openai_discovery_enriches_live_astra_with_documented_metadata(monkeypatch):
    monkeypatch.setattr(
        "llm_bawt.service.providers.api_key.resolve_api_key",
        lambda *_args, **_kwargs: "stored-key",
    )
    monkeypatch.setattr(
        "llm_bawt.model_manager.fetch_openai_api_models",
        lambda key: (
            True,
            [
                {"id": "gpt-6-astra"},
                {"id": "gpt-future"},
            ],
        ),
    )

    models = discover_models("openai", object())

    assert models == [
        {
            "id": "gpt-6-astra",
            "description": "Astra — OpenAI's flagship model for complex reasoning and coding.",
            "context_length": 1_050_000,
        },
        {"id": "gpt-future", "description": ""},
    ]


def test_codex_discovery_preserves_live_models_and_context(monkeypatch):
    class Response:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "models": [
                    {
                        "id": "gpt-6-astra",
                        "description": "Astra",
                        "context_window": 1_100_000,
                    }
                ]
            }

    monkeypatch.setattr("llm_bawt.service.model_discovery.httpx.get", lambda *_args, **_kwargs: Response())

    assert CodexBridgeDiscoveryProvider().fetch() == [
        {
            "id": "gpt-6-astra",
            "description": "Astra",
            "context_length": 1_100_000,
        }
    ]


def test_codex_discovery_surfaces_bridge_error(monkeypatch):
    request = httpx.Request("GET", "http://codex-bridge:8682/models")
    response = httpx.Response(
        503,
        request=request,
        json={"error": "OpenAIChatGPTAdapter.extra_headers() missing responses_body"},
    )

    def fail(*_args, **_kwargs):
        raise httpx.HTTPStatusError("failure", request=request, response=response)

    monkeypatch.setattr("llm_bawt.service.model_discovery.httpx.get", fail)

    with pytest.raises(ModelDiscoveryError) as exc_info:
        CodexBridgeDiscoveryProvider().fetch()

    assert str(exc_info.value) == (
        "ChatGPT subscription model discovery failed: "
        "OpenAIChatGPTAdapter.extra_headers() missing responses_body"
    )


def test_codex_discovery_rejects_empty_live_catalog(monkeypatch):
    class Response:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"models": []}

    monkeypatch.setattr("llm_bawt.service.model_discovery.httpx.get", lambda *_args, **_kwargs: Response())

    with pytest.raises(ModelDiscoveryError, match="returned no available models"):
        CodexBridgeDiscoveryProvider().fetch()


def test_upstream_lookup_always_fetches_current_catalog(monkeypatch):
    calls = 0

    def fake_discover(provider):
        nonlocal calls
        calls += 1
        assert provider == "kimi"
        return [{"id": f"k{calls}", "description": "Kimi"}]

    monkeypatch.setattr("llm_bawt.service.model_discovery.discover_models", fake_discover)

    assert model_routes._upstream_lookup("kimi")[0]["id"] == "k1"
    assert model_routes._upstream_lookup("kimi")[0]["id"] == "k2"
    assert calls == 2


def _stub_anthropic(monkeypatch, rows):
    monkeypatch.setattr(
        "llm_bawt.service.providers.api_key.resolve_api_key",
        lambda *_args, **_kwargs: "stored-key",
    )
    monkeypatch.setattr(
        "llm_bawt.model_manager.fetch_anthropic_api_models",
        lambda key: (True, rows),
    )


def test_anthropic_discovery_offers_claude_code_slugs_with_real_windows(monkeypatch):
    _stub_anthropic(
        monkeypatch,
        [
            {"id": "claude-opus-5-5", "description": "Claude Opus 5.5", "context_length": 1_000_000},
            {"id": "claude-opus-4-5-20251101", "description": "Claude Opus 4.5", "context_length": 200_000},
            {"id": "claude-mystery", "description": "Claude Mystery"},
        ],
    )

    models = discover_models("anthropic", object())

    assert models == [
        {"id": "claude-opus-5-5", "description": "Claude Opus 5.5 · 200K context", "context_length": 200_000},
        {"id": "claude-opus-5-5[1m]", "description": "Claude Opus 5.5 · 1M context", "context_length": 1_000_000},
        {"id": "claude-opus-4-5-20251101", "description": "Claude Opus 4.5", "context_length": 200_000},
        {"id": "claude-mystery", "description": "Claude Mystery"},
    ]


def test_claude_code_alias_uses_same_anthropic_shaping(monkeypatch):
    _stub_anthropic(
        monkeypatch,
        [{"id": "claude-sonnet-5-5", "description": "Claude Sonnet 5.5", "context_length": 1_000_000}],
    )

    ids = [row["id"] for row in discover_models("claude-code", object())]

    assert ids == ["claude-sonnet-5-5", "claude-sonnet-5-5[1m]"]


def test_anthropic_fetcher_keeps_api_max_input_tokens(monkeypatch):
    from types import SimpleNamespace

    from llm_bawt import model_manager

    page = SimpleNamespace(data=[
        SimpleNamespace(id="claude-opus-5-5", display_name="Claude Opus 5.5", created_at=None, max_input_tokens=1_000_000),
        SimpleNamespace(id="claude-old", display_name="Claude Old", created_at=None, max_input_tokens=None),
    ])

    class FakeAnthropic:
        def __init__(self, api_key):
            self.models = SimpleNamespace(list=lambda limit: page)

    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)

    ok, rows = model_manager.fetch_anthropic_api_models("key")

    assert ok
    assert rows[0]["context_length"] == 1_000_000
    assert "context_length" not in rows[1]
