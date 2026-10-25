from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.service.model_discovery import ExistingFetcherProvider
from llm_bawt.service.providers.api_key import ApiKeyAdapter, resolve_api_key
from llm_bawt.service.providers.anthropic_admin import AnthropicAdminAdapter
from llm_bawt.service.providers.anthropic_key import AnthropicKeyAdapter
from llm_bawt.service.providers.base import ConnectionRecord
from llm_bawt.service.providers.registry import all_adapters
from llm_bawt.service.routes import providers as provider_routes


class FakeStore:
    def __init__(self, record: ConnectionRecord | None = None) -> None:
        self.record = record
        self.saved: list[ConnectionRecord] = []

    def load(self, provider: str) -> ConnectionRecord | None:
        if self.record and self.record.provider == provider:
            return self.record
        return None

    def save(self, record: ConnectionRecord) -> None:
        self.record = record
        self.saved.append(record)


class DummyKeyAdapter(ApiKeyAdapter):
    id = "dummy-api"
    label = "Dummy API"

    def _probe(self, key: str) -> tuple[str | None, str | None]:
        if key == "valid-key":
            return "dummy-account", None
        return None, "rejected"


def _adapter(record: ConnectionRecord | None = None) -> tuple[DummyKeyAdapter, FakeStore]:
    adapter = object.__new__(DummyKeyAdapter)
    store = FakeStore(record)
    adapter.store = store
    return adapter, store


def test_api_key_adapter_validates_before_persisting() -> None:
    adapter, store = _adapter()

    rejected = adapter.set_api_key("bad-key")
    connected = adapter.set_api_key("  valid-key  ")

    assert rejected.ok is False
    assert store.saved[0].secret == {"api_key": "valid-key"}
    assert connected.ok is True
    assert connected.account == "dummy-account"
    assert adapter.api_key() == "valid-key"


def test_resolve_api_key_prefers_explicit_then_store_then_env(monkeypatch) -> None:
    config = SimpleNamespace(LEGACY_KEY="config-key")
    record = ConnectionRecord(
        provider="dummy-api",
        status="connected",
        secret={"api_key": "stored-key"},
    )
    adapter, _ = _adapter(record)
    monkeypatch.setattr(
        "llm_bawt.service.providers.registry.get_adapter",
        lambda _config, _provider_id: adapter,
    )
    monkeypatch.setenv("DUMMY_API_KEY", "env-key")

    assert resolve_api_key(config, "dummy-api", explicit="explicit-key") == "explicit-key"
    assert resolve_api_key(config, "dummy-api", env_vars=("DUMMY_API_KEY",)) == "stored-key"

    adapter.store.record = None
    assert resolve_api_key(config, "dummy-api", env_vars=("DUMMY_API_KEY",)) == "env-key"

    monkeypatch.delenv("DUMMY_API_KEY")
    assert resolve_api_key(config, "dummy-api", config_attr="LEGACY_KEY") == "config-key"


def test_registry_exposes_all_raw_api_key_adapters(monkeypatch) -> None:
    class NoDatabaseStore:
        engine = None

    monkeypatch.setattr(
        "llm_bawt.service.providers.base.RuntimeSettingsStore",
        lambda _config: NoDatabaseStore(),
    )
    ids = {adapter.id for adapter in all_adapters(SimpleNamespace())}

    assert {"xai", "openai-api", "anthropic-api", "anthropic-admin"} <= ids


def test_api_key_broker_returns_stored_key(monkeypatch) -> None:
    record = ConnectionRecord(
        provider="dummy-api",
        status="connected",
        secret={"api_key": "stored-key"},
    )
    adapter, _ = _adapter(record)
    monkeypatch.delenv("BRIDGE_CLAUDE_TOKEN_SECRET", raising=False)
    monkeypatch.setattr(provider_routes, "_adapter_or_404", lambda _provider_id: adapter)

    app = FastAPI()
    app.include_router(provider_routes.router)
    response = TestClient(app).get("/v1/providers/dummy-api/token")

    assert response.status_code == 200
    assert response.json() == {
        "access_token": "stored-key",
        "expires_at": None,
        "state": "ok",
    }


def test_admin_key_is_not_a_model_key_and_inference_probe_explains_slot(monkeypatch) -> None:
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("admin key must not be probed against models")))
    store = FakeStore(ConnectionRecord(
        provider="anthropic-api", status="connected", secret={"api_key": "working"}))
    inference = object.__new__(AnthropicKeyAdapter)
    inference.store = store
    result = inference.set_api_key("sk-ant-admin01-example")
    assert not result.ok and "Usage reporting key" in (result.detail or "")
    assert store.record.secret == {"api_key": "working"}
    assert not issubclass(AnthropicAdminAdapter, ApiKeyAdapter)


def test_admin_probe_validates_report_then_saves_separate_slot(monkeypatch) -> None:
    import httpx

    seen = []
    def report(url, *, params, headers, timeout):
        seen.append((url, headers))
        return httpx.Response(200, json={"data": []}, request=httpx.Request("GET", url))
    monkeypatch.setattr(httpx, "get", report)
    adapter = object.__new__(AnthropicAdminAdapter)
    store = FakeStore()
    adapter.store = store
    assert not adapter.set_api_key("sk-ant-api03-inference").ok
    assert store.saved == []
    assert adapter.set_api_key("sk-ant-admin01-report").ok
    assert store.saved[0].provider == "anthropic-admin"
    assert store.saved[0].secret == {"admin_key": "sk-ant-admin01-report"}
    assert seen[0][1]["x-api-key"] == "sk-ant-admin01-report"
    assert "cost_report" in seen[0][0]


def test_model_discovery_uses_credential_store_key(monkeypatch) -> None:
    config = SimpleNamespace()
    record = ConnectionRecord(
        provider="dummy-api",
        status="connected",
        secret={"api_key": "stored-key"},
    )
    adapter, _ = _adapter(record)
    monkeypatch.setattr(
        "llm_bawt.service.providers.registry.get_adapter",
        lambda _config, _provider_id: adapter,
    )
    monkeypatch.delenv("DUMMY_API_KEY", raising=False)
    seen: list[str] = []

    def fetcher(key: str):
        seen.append(key)
        return True, [{"id": "model-1", "description": "Model One"}]

    provider = ExistingFetcherProvider(
        aliases=("dummy",),
        label="Dummy",
        fetcher=fetcher,
        key_envs=("DUMMY_API_KEY",),
        provider_id="dummy-api",
        config=config,
        pass_key=True,
    )

    assert provider.fetch() == [{"id": "model-1", "description": "Model One"}]
    assert seen == ["stored-key"]
