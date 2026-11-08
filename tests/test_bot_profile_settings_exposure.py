"""Bot profile responses carry bot settings only, never global connection rows (TASK-1415)."""

import asyncio
from types import SimpleNamespace

import pytest

import llm_bawt.bots as bots_module
from llm_bawt.service.routes import settings

# Shape only; never a real secret.
_CONNECTION = {"account": "acct", "auth_method": "api_key", "secret_enc": "CIPHERTEXT-PLACEHOLDER",
               "status": "connected", "meta": {"label": "x"}}

_GLOBAL = {
    "provider_connection:anthropic-api": _CONNECTION,
    "integration_connection:radarr": _CONNECTION,
    "Provider_Connection:look-alike ": _CONNECTION,
    "ops_gate_environment:llm-bawt:abc": {"schema": 1, "evidence_json": "{}"},
    "nav_favorites": ["/tasks"],
    "home_audio": {"targets": []},
    "memory_n_results": 7,          # RuntimeTunables field
    "max_output_tokens": 4096,      # typed per-bot registry key
    "history_scope": "inline+summaries",
}

_BOT = {
    "max_output_tokens": 2048,      # bot override wins over global
    "custom_bot_key": "kept",       # explicit per-bot row stays visible
    "provider_connection:planted": _CONNECTION,
}


class _Store:
    engine = object()

    def get_scope_settings(self, scope_type, scope_id):
        if scope_type == "global":
            return dict(_GLOBAL)
        return dict(_BOT) if scope_id == "probe" else {}


@pytest.fixture
def service(monkeypatch):
    svc = SimpleNamespace(config=object(), _redis_subscriber=None)
    monkeypatch.setattr(settings, "get_runtime_settings_store", lambda _config: _Store())
    bot = SimpleNamespace(settings={"temperature": 0.8, "integration_connection:yaml": _CONNECTION})
    monkeypatch.setattr(bots_module, "BotManager",
                        lambda _config: SimpleNamespace(get_bot=lambda _slug: bot))
    return svc


def _assert_safe(result):
    assert not any(key.strip().lower().startswith(("provider_connection:", "integration_connection:"))
                   for key in result)
    assert "secret_enc" not in repr(result)
    assert not any(key.startswith("ops_gate_environment:") for key in result)
    assert "nav_favorites" not in result and "home_audio" not in result


def test_effective_settings_drop_connection_records_and_global_app_config(service):
    result = settings._effective_bot_settings(service, "probe")

    _assert_safe(result)
    assert result["memory_n_results"] == 7
    assert result["history_scope"] == "inline+summaries"
    assert result["max_output_tokens"] == 2048
    assert result["custom_bot_key"] == "kept"
    assert result["temperature"] == 0.8


def test_get_profile_route_returns_filtered_settings(service, monkeypatch):
    captured = {}
    profile = SimpleNamespace(slug="probe")
    monkeypatch.setattr(settings, "get_service", lambda: service)
    monkeypatch.setattr(settings, "get_bot_profile_store",
                        lambda _config: SimpleNamespace(engine=object(), get=lambda _slug: profile))
    monkeypatch.setattr(settings, "_profile_tags", lambda *_args: {})
    monkeypatch.setattr(settings, "_to_profile_response",
                        lambda _profile, settings, tags=None: captured.setdefault("settings", settings))

    settings.get_bot_profile("probe")

    _assert_safe(captured["settings"])
    assert captured["settings"]["memory_n_results"] == 7


def test_profile_write_response_returns_filtered_settings(service, monkeypatch):
    captured = {}
    profile = SimpleNamespace(slug="probe", agent_backend="claude-code")
    store = SimpleNamespace(engine=object(), get=lambda slug: None,
                            upsert=lambda payload, *, guard, require_existing=False: profile)
    monkeypatch.setattr(settings, "get_service", lambda: service)
    monkeypatch.setattr(settings, "get_bot_profile_store", lambda _config: store)
    monkeypatch.setattr(settings, "_validate_profile_payload", lambda _payload: None)
    monkeypatch.setattr(settings.ProfileRouteGate, "for_payload",
                        classmethod(lambda cls, *_args: cls({})))
    monkeypatch.setattr(settings, "_reload_bot_registry", lambda: None)
    monkeypatch.setattr(settings, "_reload_service_model_catalog", lambda _service: None)
    monkeypatch.setattr(settings, "_invalidate_bot_instance_cache", lambda *_args: None)
    monkeypatch.setattr(settings, "_clear_session_model_overrides", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(settings, "_profile_tags", lambda *_args: {})
    monkeypatch.setattr(settings, "_to_profile_response",
                        lambda _profile, settings, tags=None: captured.setdefault("settings", settings))

    asyncio.run(settings._persist_bot_profile({"slug": "probe", "endpoint_id": 42}, create_only=False))

    _assert_safe(captured["settings"])
    assert captured["settings"]["custom_bot_key"] == "kept"
