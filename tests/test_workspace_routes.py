"""App -> bridge facade, with no live DB/provider/workspace."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.service.routes import workspace as routes


@pytest.fixture
def client(monkeypatch):
    config = SimpleNamespace(WORKSPACE_BRIDGE_URL="http://bridge.test:8681")
    monkeypatch.setattr(routes, "get_service", lambda: SimpleNamespace(config=config))
    monkeypatch.setattr(routes, "get_runtime_settings_store", lambda _: SimpleNamespace(
        get_scope_settings=lambda *_: {},
    ))
    monkeypatch.setenv("BRIDGE_CLAUDE_TOKEN_SECRET", "fixture-only")
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def upstream(monkeypatch, response=None, error=None):
    request = AsyncMock(return_value=response, side_effect=error)
    class Client:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_):
            pass
        async def request(self, *args, **kwargs):
            return await request(*args, **kwargs)
    monkeypatch.setattr(routes.httpx, "AsyncClient", lambda **_: Client())
    return request


def test_clone_proxy(client, monkeypatch):
    request = upstream(monkeypatch, httpx.Response(200, json={"name": "repo", "path": "repo"}))
    response = client.post("/v1/workspace/clone", json={"url": "git@github.com:team/repo.git"})
    assert response.status_code == 200
    assert response.json() == {"name": "repo", "path": "repo"}
    args, kwargs = request.call_args
    assert args == ("POST", "http://bridge.test:8681/workspace/clone")
    assert kwargs["headers"] == {"X-Bridge-Token": "fixture-only"}
    assert kwargs["json"] == {"url": "https://github.com/team/repo.git", "name": "repo"}
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("body", [
    {"url": "https://user:secret@host/repo"},
    {"url": "https://host/repo?token=secret"}, {"url": "https://host/repo", "name": "../x"},
    {"url": "file:///tmp/repo"}, [], None,
])
def test_invalid_requests_not_reflected(client, monkeypatch, body):
    request = upstream(monkeypatch)
    response = client.post("/v1/workspace/clone", json=body)
    assert response.status_code == 400
    assert "secret" not in response.text
    request.assert_not_awaited()


@pytest.mark.parametrize("status", [400, 409, 502, 503, 504, 401, 302])
def test_error_mapping_static(client, monkeypatch, status):
    upstream(monkeypatch, httpx.Response(status, json={"detail": "secret"}))
    monkeypatch.setattr(routes, "_github_disconnected", lambda: False)
    response = client.post("/v1/workspace/clone", json={"url": "https://github.com/team/repo"})
    assert response.status_code == (status if status in {400, 409, 502, 503, 504} else 503)
    assert "secret" not in response.text


def test_github_disconnected_only_after_failure(client, monkeypatch):
    upstream(monkeypatch, httpx.Response(502))
    monkeypatch.setattr(routes, "_github_disconnected", lambda: True)
    response = client.post("/v1/workspace/clone", json={"url": "https://github.com/team/repo"})
    assert response.status_code == 503
    assert response.json()["code"] == "github_not_connected"


@pytest.mark.parametrize("status,expected", [(None, True), ("disconnected", True), ("error", True), ("connected", False)])
def test_github_uses_connection_not_app_configuration(client, monkeypatch, status, expected):
    from llm_bawt.service.providers import registry
    record = SimpleNamespace(status=status) if status else None
    adapter = SimpleNamespace(store=SimpleNamespace(load=lambda _: record))
    monkeypatch.setattr(registry, "get_adapter", lambda *_: adapter)
    # No is_configured() method: App configuration is not a connection.
    assert routes._github_disconnected() is expected


def test_app_prefers_exported_derived_broker(client, monkeypatch):
    monkeypatch.setenv("BRIDGE_BROKER_TOKEN_EXPORT_FILE", "/fixture/exported")
    gates = routes.bridge_gate.BridgeGates(lambda: {}, derive=lambda _: "derived-fixture")
    monkeypatch.setattr(routes.bridge_gate, "bridge_gates", lambda: gates)
    request = upstream(monkeypatch, httpx.Response(200, json={"entries": []}))
    assert client.get("/v1/workspace").status_code == 200
    assert request.call_args.kwargs["headers"] == {"X-Bridge-Token": "derived-fixture"}


@pytest.mark.parametrize("export,derived,legacy,expected", [
    (False, "derived", "legacy", "legacy"),
    (True, "derived", "legacy", "derived"),
    (True, None, "legacy", "legacy"),
    (False, "derived", None, "derived"),
    (False, None, None, None),
])
def test_outbound_broker_migration(export, derived, legacy, expected):
    env = {}
    if export:
        env["BRIDGE_BROKER_TOKEN_EXPORT_FILE"] = "/fixture/exported"
    if legacy:
        env["BRIDGE_CLAUDE_TOKEN_SECRET"] = legacy
    gates = routes.bridge_gate.BridgeGates(lambda: {}, environ=env, derive=lambda _: derived)
    assert gates.outbound_headers(routes.bridge_gate.BROKER) == (
        {"X-Bridge-Token": expected} if expected else {}
    )


def test_timeout_and_bad_success(client, monkeypatch):
    upstream(monkeypatch, error=httpx.ReadTimeout("secret"))
    assert client.get("/v1/workspace").status_code == 504
    upstream(monkeypatch, httpx.Response(200, json={"name": "repo", "path": "../secret"}))
    response = client.post("/v1/workspace/clone", json={"url": "https://host/repo"})
    assert response.status_code == 503 and "secret" not in response.text


def test_db_configuration_wins(client, monkeypatch):
    monkeypatch.setattr(routes, "get_runtime_settings_store", lambda _: SimpleNamespace(
        get_scope_settings=lambda *_: {"workspace.bridge_url": "http://codex.test:8682"},
    ))
    request = upstream(monkeypatch, httpx.Response(200, json={"entries": []}))
    assert client.get("/v1/workspace").json() == {"entries": []}
    assert request.call_args.args == ("GET", "http://codex.test:8682/workspace")


def test_bad_config_is_unavailable(client, monkeypatch):
    monkeypatch.setattr(routes, "get_runtime_settings_store", lambda _: SimpleNamespace(
        get_scope_settings=lambda *_: {"workspace.bridge_url": "http://user:secret@host"},
    ))
    response = client.get("/v1/workspace")
    assert response.status_code == 503 and "secret" not in response.text
