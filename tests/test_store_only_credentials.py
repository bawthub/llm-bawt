"""TASK-1681: retired env credentials cannot resurrect a missing connection."""

from types import SimpleNamespace

import pytest

from llm_bawt.media.clients.base import resolve_media_api_key
from llm_bawt.service.connection_secrets import ConnectionUnavailable
from llm_bawt.service.model_discovery import ExistingFetcherProvider, ModelDiscoveryError
from llm_bawt.service.providers.api_key import resolve_api_key, resolve_stored_api_key
from llm_bawt.service.providers.base import ConnectionRecord
from llm_bawt.service.providers.openai_key import OpenAIKeyAdapter
from llm_bawt.service.providers.xai import XaiAdapter
from llm_bawt.service.usage.adapters.api_key_balance import OpenAIApiUsageAdapter
from llm_bawt.service.usage.adapters.xai import XaiUsageAdapter
from tests.stored_provider_keys import real_credential_store, stub_stored_keys

PROVIDERS = [
    (OpenAIKeyAdapter, ("OPENAI_API_KEY",)),
    (XaiAdapter, ("XAI_API_KEY", "LLM_BAWT_XAI_API_KEY", "GROK_API_KEY")),
]


@pytest.mark.parametrize("adapter_cls,env_names", PROVIDERS)
@pytest.mark.parametrize("state", ["missing", "connected", "disconnected", "unreadable", "store_unavailable"])
def test_store_only_resolution_ignores_env_and_config(monkeypatch, adapter_cls, env_names, state):
    for name in env_names:
        monkeypatch.setenv(name, "retired-env-key")
    config = SimpleNamespace(XAI_API_KEY="retired-config-key")
    with real_credential_store(monkeypatch) as (store, rotate_root):
        if state not in ("missing", "store_unavailable"):
            store.save(ConnectionRecord(provider=adapter_cls.id, status="connected", auth_method="api_key",
                                        secret={"api_key": "stored-key"}))
        if state == "disconnected":
            assert adapter_cls(config).disconnect()
            assert store.load(adapter_cls.id).status == "disconnected"
            assert not store.load(adapter_cls.id).secret
        elif state == "unreadable":
            rotate_root()
        elif state == "store_unavailable":
            stub_stored_keys(monkeypatch, {adapter_cls.id: ConnectionUnavailable("store_unavailable")})

        if state in ("disconnected", "unreadable", "store_unavailable"):
            code = "not_connected" if state == "disconnected" else state
            with pytest.raises(ConnectionUnavailable) as error:
                resolve_stored_api_key(config, adapter_cls.id, env_names)
            assert error.value.code == code
        else:
            assert resolve_stored_api_key(config, adapter_cls.id, env_names) == (
                "stored-key" if state == "connected" else None
            )
        assert resolve_api_key(config, adapter_cls.id, env_names, config_attr="XAI_API_KEY") == (
            "stored-key" if state == "connected" else None
        )


@pytest.mark.parametrize("adapter_cls,env_names", PROVIDERS)
def test_explicit_credentials_still_win_without_store_access(monkeypatch, adapter_cls, env_names):
    reads = stub_stored_keys(monkeypatch, {adapter_cls.id: ConnectionUnavailable("store_unavailable")})
    assert resolve_api_key(SimpleNamespace(), adapter_cls.id, env_names, explicit="explicit-key") == "explicit-key"
    assert reads == []


@pytest.mark.parametrize("adapter_cls,env_names", PROVIDERS)
@pytest.mark.parametrize("failure", ["config", "resolver"])
def test_media_lookup_failure_never_resurrects_env(monkeypatch, adapter_cls, env_names, failure):
    for name in env_names:
        monkeypatch.setenv(name, "retired-env-key")

    def broken(*_args, **_kwargs):
        raise RuntimeError("synthetic lookup failure")

    target = "llm_bawt.utils.config.Config" if failure == "config" else (
        "llm_bawt.service.providers.api_key.resolve_api_key"
    )
    monkeypatch.setattr(target, broken)
    config = None if failure == "config" else SimpleNamespace()
    assert resolve_media_api_key(config, adapter_cls.id, env_names) == ""


@pytest.mark.parametrize("adapter_cls,env_names", PROVIDERS)
@pytest.mark.parametrize("config", [None, SimpleNamespace()])
def test_discovery_does_not_reintroduce_env_after_store_resolution(monkeypatch, adapter_cls, env_names, config):
    for name in env_names:
        monkeypatch.setenv(name, "retired-env-key")
    stub_stored_keys(monkeypatch, {})
    provider = ExistingFetcherProvider(
        aliases=("fixture",), label=adapter_cls.label,
        fetcher=lambda *_args: pytest.fail("must not call upstream with an env key"),
        key_envs=env_names, provider_id=adapter_cls.id, config=config, pass_key=True,
    )
    with pytest.raises(ModelDiscoveryError):
        provider.fetch()


@pytest.mark.parametrize("usage_cls,env_names", [
    (OpenAIApiUsageAdapter, ("OPENAI_API_KEY",)),
    (XaiUsageAdapter, ("XAI_API_KEY", "LLM_BAWT_XAI_API_KEY", "GROK_API_KEY")),
])
@pytest.mark.parametrize("state", ["missing", "disconnected"])
def test_usage_does_not_report_env_only_credentials_as_connected(monkeypatch, usage_cls, env_names, state):
    for name in env_names:
        monkeypatch.setenv(name, "retired-env-key")
    record = None if state == "missing" else ConnectionRecord(provider=usage_cls.connection_id, status="disconnected")
    monkeypatch.setattr(usage_cls, "_connection", lambda _self: record)
    assert usage_cls()._key_configured() is False
