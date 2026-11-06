"""Per-provider "Alert on problems" switch and health fields (TASK-2104)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.service.providers import base
from llm_bawt.service.routes import providers as routes


class _Settings:
    rows: dict[str, object] = {}

    def __init__(self, _config):
        self.engine = object()

    def get_scope_settings(self, scope_type, scope_id):
        assert (scope_type, scope_id) == ("global", "*")
        return dict(self.rows)

    def set_value(self, _t, _i, key, value):
        self.rows[key] = value

    def delete_value(self, _t, _i, key):
        return self.rows.pop(key, None) is not None


class _Adapter:
    def __init__(self, id_, methods, state):
        self.id, self.label, self.auth_methods, self._state = id_, id_.title(), methods, state

    def health(self):
        return base.health_block(self._state)


@pytest.fixture
def client(monkeypatch):
    _Settings.rows = {"provider_connection:x": {"status": "connected"}}
    monkeypatch.setattr(base, "RuntimeSettingsStore", _Settings)
    adapters = [_Adapter("ollama", ("endpoint",), "offline"), _Adapter("github", ("device_oauth",), "broken")]
    monkeypatch.setattr(routes, "all_adapters", lambda _config: adapters)
    monkeypatch.setattr(routes, "get_adapter", lambda _c, pid: next((a for a in adapters if a.id == pid), None))
    monkeypatch.setattr(routes, "get_service", lambda: SimpleNamespace(config=None))
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def _health(client):
    return {p["id"]: p for p in client.get("/v1/providers/health").json()["providers"]}


def test_health_defaults_alerts_on_and_flags_endpoint_providers(client):
    health = _health(client)
    assert health["ollama"]["alerts"] is True and health["ollama"]["endpoint"] is True
    assert health["ollama"]["state"] == "offline"
    assert health["github"]["alerts"] is True and health["github"]["endpoint"] is False


def test_switch_persists_off_and_on_without_hiding_state(client):
    assert client.put("/v1/providers/github/alerts", json={"enabled": False}).json() == {"id": "github", "alerts": False}
    assert _Settings.rows["provider_alerts:github"] is False
    health = _health(client)
    assert health["github"]["alerts"] is False and health["github"]["state"] == "broken"
    assert health["ollama"]["alerts"] is True

    client.put("/v1/providers/github/alerts", json={"enabled": True})
    assert "provider_alerts:github" not in _Settings.rows
    assert _health(client)["github"]["alerts"] is True


def test_switch_rejects_unknown_provider_and_bad_body(client):
    assert client.put("/v1/providers/nope/alerts", json={"enabled": False}).status_code == 404
    assert client.put("/v1/providers/github/alerts", json={"enabled": "no", "x": 1}).status_code == 422


def test_alert_rows_are_not_credential_settings():
    assert not base.is_credential_setting_key("provider_alerts:github")
