"""TASK-2047: Ollama server connection, model discovery and key use."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from llm_bawt.service import model_discovery
from llm_bawt.service.model_discovery import ModelDiscoveryError, OllamaDiscoveryProvider
from llm_bawt.service.providers import ollama
from llm_bawt.service.providers.base import STATUS_CONNECTED, ConnectionRecord
from llm_bawt.service.providers.registry import _ACCESS_PATH_OWNERS, _ADAPTER_CLASSES


class _Store:
    def __init__(self):
        self.records: dict[str, ConnectionRecord] = {}

    def load(self, provider):
        return self.records.get(provider)

    def save(self, record):
        self.records[record.provider] = record

    def delete(self, provider):
        return self.records.pop(provider, None) is not None


class _Response:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload
        self.is_error = status >= 400

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def raise_for_status(self):
        if self.is_error:
            request = httpx.Request("GET", "http://x")
            raise httpx.HTTPStatusError("err", request=request, response=httpx.Response(self.status_code, request=request))


@pytest.fixture
def path_url(monkeypatch):
    state = {"url": None}
    monkeypatch.setattr(ollama, "_stored_path_url", lambda _config: state["url"])
    monkeypatch.setattr(ollama, "_write_path_url", lambda _config, url: state.__setitem__("url", url))
    return state


def _adapter():
    adapter = ollama.OllamaAdapter.__new__(ollama.OllamaAdapter)
    adapter.config = SimpleNamespace(OLLAMA_URL="http://localhost:11434")
    adapter.store = _Store()
    return adapter


def test_registered_without_owning_route_eligibility():
    assert _ADAPTER_CLASSES["ollama"] is ollama.OllamaAdapter
    assert "ollama" not in _ACCESS_PATH_OWNERS


@pytest.mark.parametrize(("raw", "expected"), [
    ("http://box:11434", "http://box:11434"),
    ("box:11434/", "http://box:11434"),
    ("https://proxy.example/ollama/v1", "https://proxy.example/ollama"),
    ("http://box:11434/api", "http://box:11434"),
])
def test_normalize_server_url(raw, expected):
    assert ollama.normalize_server_url(raw) == expected


@pytest.mark.parametrize("raw", ["", "ftp://box", "http://box:11434?x=1"])
def test_normalize_rejects_unusable_urls(raw):
    with pytest.raises(ollama.OllamaUrlError):
        ollama.normalize_server_url(raw)


def test_server_url_prefers_the_access_path_then_config(path_url):
    config = SimpleNamespace(OLLAMA_URL="http://fallback:11434")
    assert ollama.server_url(config) == "http://fallback:11434"
    path_url["url"] = "http://box:11434/v1"
    assert ollama.server_url(config) == "http://box:11434"


def test_connect_saves_url_and_key_only_after_a_successful_probe(monkeypatch, path_url):
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append((url, headers))
        return _Response(payload={"version": "0.40.14"})

    monkeypatch.setattr(ollama.httpx, "get", fake_get)
    adapter = _adapter()
    result = adapter.connect_endpoint("box:11434", "k-1")
    assert result.ok and result.account == "Ollama 0.40.14 · http://box:11434"
    assert calls == [("http://box:11434/api/version", {"Authorization": "Bearer k-1"})]
    assert path_url["url"] == "http://box:11434/v1"
    record = adapter.store.load("ollama")
    assert record.status == STATUS_CONNECTED
    assert record.secret == {"api_key": "k-1"}
    assert adapter.api_key() == "k-1"


def test_unreachable_server_saves_nothing(monkeypatch, path_url):
    def boom(*_a, **_k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(ollama.httpx, "get", boom)
    adapter = _adapter()
    result = adapter.connect_endpoint("http://nowhere:11434")
    assert not result.ok
    assert "Couldn't reach Ollama at http://nowhere:11434" in result.detail
    assert path_url["url"] is None and adapter.store.load("ollama") is None


def test_non_ollama_server_is_rejected(monkeypatch, path_url):
    monkeypatch.setattr(ollama.httpx, "get", lambda *_a, **_k: _Response(payload={"hello": 1}))
    result = _adapter().connect_endpoint("http://web:80")
    assert not result.ok and "not like an Ollama server" in result.detail


def test_disconnect_falls_back_to_config(monkeypatch, path_url):
    monkeypatch.setattr(ollama.httpx, "get", lambda *_a, **_k: _Response(payload={"version": "1"}))
    adapter = _adapter()
    adapter.connect_endpoint("http://box:11434")
    assert adapter.disconnect() is True
    assert path_url["url"] is None
    assert ollama.server_url(adapter.config) == "http://localhost:11434"


def test_health_reports_unreachable_configured_server(monkeypatch, path_url):
    adapter = _adapter()
    adapter.store.save(ConnectionRecord(provider="ollama", status=STATUS_CONNECTED, meta={"base_url": "http://box:1"}))

    def boom(*_a, **_k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(ollama.httpx, "get", boom)
    health = adapter.health()
    assert health["state"] == "broken" and "http://box:1" in health["detail"]


def test_discovery_lists_chat_models_with_context(monkeypatch, path_url):
    path_url["url"] = "http://box:11434/v1"
    monkeypatch.setattr(ollama, "resolve_api_key", lambda _config: None)
    tags = {"models": [
        {"name": "qwen:27b", "details": {"context_length": 262144, "parameter_size": "27B", "quantization_level": "Q4_K_M"},
         "capabilities": ["completion", "tools"]},
        {"name": "nomic-embed", "details": {}, "capabilities": ["embedding"]},
        {"name": "old:7b", "details": {"parameter_size": "7B"}},
    ]}
    seen = []
    monkeypatch.setattr(model_discovery.httpx, "get", lambda url, **_k: seen.append(url) or _Response(payload=tags))
    shows = {
        "qwen:27b": {"parameters": "temperature 1\nnum_ctx                        65536", "model_info": {"q.context_length": 262144}},
        "old:7b": {"model_info": {"llama.context_length": 8192}},
    }
    monkeypatch.setattr(
        model_discovery.httpx, "post",
        lambda url, json=None, **_k: _Response(payload=shows[json["model"]]),
    )
    rows = OllamaDiscoveryProvider(SimpleNamespace(OLLAMA_URL="x")).fetch()
    assert seen == ["http://box:11434/api/tags"]
    assert rows == [
        {"id": "qwen:27b", "description": "27B · Q4_K_M", "context_length": 65536},
        {"id": "old:7b", "description": "7B", "context_length": 8192},
    ]


def test_discovery_falls_back_to_tags_context_when_show_fails(monkeypatch, path_url):
    path_url["url"] = "http://box:11434/v1"
    monkeypatch.setattr(ollama, "resolve_api_key", lambda _config: None)
    tags = {"models": [{"name": "m", "details": {"context_length": 4096}, "capabilities": ["completion"]}]}
    monkeypatch.setattr(model_discovery.httpx, "get", lambda *_a, **_k: _Response(payload=tags))
    monkeypatch.setattr(model_discovery.httpx, "post", lambda *_a, **_k: _Response(status=500))
    rows = OllamaDiscoveryProvider(SimpleNamespace(OLLAMA_URL="x")).fetch()
    assert rows == [{"id": "m", "description": "", "context_length": 4096}]


def test_discovery_error_names_the_server(monkeypatch, path_url):
    path_url["url"] = "http://box:11434/v1"
    monkeypatch.setattr(ollama, "resolve_api_key", lambda _config: None)

    def boom(*_a, **_k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(model_discovery.httpx, "get", boom)
    with pytest.raises(ModelDiscoveryError, match="Couldn't reach Ollama at http://box:11434"):
        OllamaDiscoveryProvider(SimpleNamespace(OLLAMA_URL="x")).fetch()


def test_empty_catalog_hint_is_ollama_only():
    assert "ollama pull" in model_discovery.empty_hint("ollama")
    assert model_discovery.empty_hint("openrouter") is None


def test_ollama_factory_sends_the_stored_key(monkeypatch):
    from llm_bawt.service import background_clients

    built = {}

    class FakeClient:
        def __init__(self, model_id, **kwargs):
            built.update(kwargs, model_id=model_id)

    monkeypatch.setattr("llm_bawt.clients.openai_client.OpenAIClient", FakeClient)
    monkeypatch.setattr(ollama, "resolve_api_key", lambda _config: "k-9")
    config = SimpleNamespace(OLLAMA_URL="http://localhost:11434")
    background_clients.OllamaFactory().build(config, {"type": "ollama", "model_id": "m", "base_url": "http://box:11434/v1"}, "m")
    assert built["api_key"] == "k-9" and built["base_url"] == "http://box:11434/v1"
