"""Provider health aggregate flags endpoint providers (TASK-2104)."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.service.providers import base
from llm_bawt.service.routes import providers as routes


class _Adapter:
    def __init__(self, id_, methods, state):
        self.id, self.label, self.auth_methods, self._state = id_, id_.title(), methods, state

    def health(self):
        return base.health_block(self._state)


def test_health_flags_endpoint_providers_and_reports_offline(monkeypatch):
    adapters = [_Adapter("ollama", ("endpoint",), "offline"), _Adapter("github", ("device_oauth",), "broken")]
    monkeypatch.setattr(routes, "all_adapters", lambda _config: adapters)
    monkeypatch.setattr(routes, "get_service", lambda: SimpleNamespace(config=None))
    app = FastAPI()
    app.include_router(routes.router)
    health = {p["id"]: p for p in TestClient(app).get("/v1/providers/health").json()["providers"]}
    assert health["ollama"]["endpoint"] is True and health["ollama"]["state"] == "offline"
    assert health["github"]["endpoint"] is False and health["github"]["state"] == "broken"
