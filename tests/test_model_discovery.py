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

_BRIDGE_URL = "http://codex-bridge:8682/models"


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
        {
            "id": "gpt-future",
            "description": "",
        },
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

    assert CodexBridgeDiscoveryProvider(_BRIDGE_URL).fetch() == [
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
        CodexBridgeDiscoveryProvider(_BRIDGE_URL).fetch()

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
        CodexBridgeDiscoveryProvider(_BRIDGE_URL).fetch()


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


# ---------------------------------------------------------------------------
# TASK-1296 D1: the curated marker survives normalization
# ---------------------------------------------------------------------------


def test_normalize_keeps_only_a_true_curated_marker():
    from llm_bawt.service.model_discovery import _normalize

    rows = _normalize(
        [
            {"id": "a", "curated": True},
            {"id": "b", "curated": False},
            {"id": "c", "curated": "yes"},
            {"id": "d"},
        ]
    )

    assert rows == [
        {"id": "a", "description": "", "curated": True},
        {"id": "b", "description": ""},
        {"id": "c", "description": ""},
        {"id": "d", "description": ""},
    ]


# ---------------------------------------------------------------------------
# TASK-1296 D1: Claude subscription discovery
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, status_code, body=None, *, bad_json=False):
        self.status_code = status_code
        self._body = body
        self._bad_json = bad_json

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://stub")
            raise httpx.HTTPStatusError("failure", request=request, response=httpx.Response(self.status_code, request=request))

    def json(self):
        if self._bad_json:
            raise ValueError("not json")
        return self._body


def _claude_token(monkeypatch, state="ok", token="oauth-token"):
    from llm_bawt.service.usage.claude_oauth import UsageToken

    monkeypatch.setattr(
        "llm_bawt.service.usage.claude_oauth.get_access_token",
        lambda **_kw: UsageToken(token if state == "ok" else None, state),
    )


def test_claude_subscription_discovery_lists_live_models_with_oauth_headers(monkeypatch):
    observed = {}

    def fake_get(url, *, headers, timeout):
        observed.update(url=url, headers=headers, timeout=timeout)
        return _Resp(200, {"data": [
            {"id": "claude-opus-5-5", "display_name": "Claude Opus 5.5", "max_input_tokens": 1_000_000},
            {"id": "claude-fable-5-1", "display_name": "Claude Fable 5.1", "max_input_tokens": 200_000},
        ]})

    _claude_token(monkeypatch)
    monkeypatch.setattr("llm_bawt.service.model_discovery.httpx.get", fake_get)

    rows = discover_models("claude")

    assert observed["url"] == "https://api.anthropic.com/v1/models"
    assert observed["headers"] == {
        "Authorization": "Bearer oauth-token",
        "anthropic-beta": "oauth-2025-04-20",
        "anthropic-version": "2023-06-01",
        "Accept": "application/json",
    }
    assert rows == [
        {"id": "claude-opus-5-5", "description": "Claude Opus 5.5 · 200K context", "context_length": 200_000},
        {"id": "claude-opus-5-5[1m]", "description": "Claude Opus 5.5 · 1M context", "context_length": 1_000_000},
        {"id": "claude-fable-5-1", "description": "Claude Fable 5.1", "context_length": 200_000},
    ]
    assert not any(row.get("curated") for row in rows)


@pytest.mark.parametrize("status", [401, 403, 404, 400])
def test_claude_subscription_discovery_falls_back_to_curated_on_refusal(monkeypatch, status):
    _claude_token(monkeypatch)
    monkeypatch.setattr("llm_bawt.service.model_discovery.httpx.get", lambda *_a, **_k: _Resp(status, {}))

    rows = discover_models("claude")

    ids = [row["id"] for row in rows]
    assert "claude-opus-5-5" in ids and "claude-opus-5-5[1m]" in ids
    assert "claude-fable-5-1" in ids
    assert all(row["curated"] is True for row in rows)
    assert all(row["context_length"] in (200_000, 1_000_000) for row in rows)


@pytest.mark.parametrize("status", [500, 502, 503, 429, 408])
def test_claude_subscription_discovery_raises_on_transient_failure(monkeypatch, status):
    _claude_token(monkeypatch)
    monkeypatch.setattr("llm_bawt.service.model_discovery.httpx.get", lambda *_a, **_k: _Resp(status, {}))

    with pytest.raises(ModelDiscoveryError, match=f"HTTP {status}"):
        discover_models("claude")


def test_claude_subscription_discovery_raises_on_network_error(monkeypatch):
    def fail(*_a, **_k):
        raise httpx.ConnectError("boom")

    _claude_token(monkeypatch)
    monkeypatch.setattr("llm_bawt.service.model_discovery.httpx.get", fail)

    with pytest.raises(ModelDiscoveryError, match="Network error"):
        discover_models("claude")


@pytest.mark.parametrize("state", ["missing", "stale"])
def test_claude_subscription_discovery_requires_a_usable_login(monkeypatch, state):
    calls = []
    _claude_token(monkeypatch, state=state)
    monkeypatch.setattr("llm_bawt.service.model_discovery.httpx.get", lambda *a, **k: calls.append(a))

    with pytest.raises(ModelDiscoveryError) as exc_info:
        discover_models("claude")

    assert exc_info.value.status_code == 503
    assert calls == []


# ---------------------------------------------------------------------------
# TASK-1296 D2: ChatGPT subscription discovery (app-side)
# ---------------------------------------------------------------------------


def _codex_token(monkeypatch, state="ok", account_id="acct-1"):
    from llm_bawt.service.usage.codex_oauth import CodexToken

    monkeypatch.setattr(
        "llm_bawt.service.usage.codex_oauth.get_access_token",
        lambda **_kw: CodexToken("chatgpt-token" if state == "ok" else None, account_id, state=state),
    )


def test_chatgpt_subscription_discovery_calls_backend_with_app_token(monkeypatch):
    from types import SimpleNamespace

    observed = {}

    def fake_get(url, *, params, headers, timeout):
        observed.update(url=url, params=params, headers=headers, timeout=timeout)
        return _Resp(200, {"models": [
            {"slug": "gpt-6-astra", "description": "Astra", "context_window": 1_100_000},
            {"slug": "gpt-6-hidden", "context_window": 1, "visibility": "hide"},
            {"slug": "gpt-image-2", "description": "Images", "context_window": 32_000},
        ]})

    monkeypatch.delenv("CODEX_BRIDGE_MODELS_URL", raising=False)
    _codex_token(monkeypatch)
    monkeypatch.setattr("llm_bawt.service.model_discovery.httpx.get", fake_get)

    rows = discover_models("codex", SimpleNamespace(CODEX_CLIENT_VERSION="0.150.0"))

    assert observed["url"] == "https://chatgpt.com/backend-api/codex/models"
    assert observed["params"] == {"client_version": "0.150.0"}
    assert observed["headers"] == {
        "Authorization": "Bearer chatgpt-token",
        "chatgpt-account-id": "acct-1",
    }
    assert rows == [
        {"id": "gpt-6-astra", "description": "Astra", "context_length": 1_100_000},
        {"id": "gpt-image-2", "description": "Images", "context_length": 32_000},
    ]


def test_chatgpt_subscription_discovery_pins_client_version_from_config(monkeypatch):
    from llm_bawt.utils.config import Config

    observed = {}

    def fake_get(url, *, params, headers, timeout):
        observed["params"] = params
        return _Resp(200, {"models": [{"slug": "gpt-6-astra", "context_window": 1}]})

    monkeypatch.delenv("CODEX_BRIDGE_MODELS_URL", raising=False)
    _codex_token(monkeypatch)
    monkeypatch.setattr("llm_bawt.service.model_discovery.httpx.get", fake_get)

    monkeypatch.delenv("LLM_BAWT_CODEX_CLIENT_VERSION", raising=False)
    discover_models("codex", Config(_env_file=None))
    assert observed["params"] == {"client_version": "0.144.1"}

    monkeypatch.setenv("LLM_BAWT_CODEX_CLIENT_VERSION", "0.200.0")
    discover_models("codex", Config(_env_file=None))
    assert observed["params"] == {"client_version": "0.200.0"}


@pytest.mark.parametrize("status", [401, 403])
def test_chatgpt_subscription_discovery_reports_rejected_token(monkeypatch, status):
    monkeypatch.delenv("CODEX_BRIDGE_MODELS_URL", raising=False)
    _codex_token(monkeypatch)
    monkeypatch.setattr("llm_bawt.service.model_discovery.httpx.get", lambda *_a, **_k: _Resp(status, {}))

    with pytest.raises(ModelDiscoveryError) as exc_info:
        discover_models("codex")

    assert exc_info.value.status_code == 503


def test_chatgpt_subscription_discovery_requires_token_and_account(monkeypatch):
    monkeypatch.delenv("CODEX_BRIDGE_MODELS_URL", raising=False)
    calls = []
    monkeypatch.setattr("llm_bawt.service.model_discovery.httpx.get", lambda *a, **k: calls.append(a))

    _codex_token(monkeypatch, state="missing")
    with pytest.raises(ModelDiscoveryError) as exc_info:
        discover_models("codex")
    assert exc_info.value.status_code == 503

    _codex_token(monkeypatch, account_id=None)
    with pytest.raises(ModelDiscoveryError) as exc_info:
        discover_models("codex")
    assert exc_info.value.status_code == 503
    assert calls == []


def test_chatgpt_subscription_discovery_uses_bridge_only_when_configured(monkeypatch):
    observed = {}

    def fake_get(url, **kwargs):
        observed["url"] = url
        observed["kwargs"] = kwargs
        return _Resp(200, {"models": [{"id": "gpt-6-astra", "description": "Astra", "context_window": 5}]})

    def no_token(**_kw):
        raise AssertionError("bridge path must not read the app token")

    monkeypatch.setenv("CODEX_BRIDGE_MODELS_URL", "http://codex-bridge:8682/models")
    monkeypatch.setattr("llm_bawt.service.usage.codex_oauth.get_access_token", no_token)
    monkeypatch.setattr("llm_bawt.service.model_discovery.httpx.get", fake_get)

    rows = discover_models("codex")

    assert observed["url"] == "http://codex-bridge:8682/models"
    assert "headers" not in observed["kwargs"]
    assert rows == [{"id": "gpt-6-astra", "description": "Astra", "context_length": 5}]


def test_codex_bridge_provider_is_not_alias_addressable():
    from llm_bawt.service.model_discovery import CodexBridgeDiscoveryProvider

    assert CodexBridgeDiscoveryProvider.aliases == ()
