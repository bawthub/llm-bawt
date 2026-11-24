"""TASK-2610: remote MCP server routes; precedence over generic integration paths, no secret echo."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.service.routes import all_routers, mcp_servers
from llm_bawt.service.routes.integrations import router as integrations_router
from llm_bawt.service.routes.service_auth import require_service_token


class FakeServers:
    def __init__(self):
        self.created = []

    def list(self):
        return [{"slug": "alpha"}]

    def create(self, **kw):
        self.created.append(kw)
        return {"slug": kw["slug"]}


@pytest.fixture
def client(monkeypatch):
    fake = FakeServers()
    monkeypatch.setattr(mcp_servers, "servers", lambda: fake)
    app = FastAPI()
    app.include_router(mcp_servers.router)
    app.include_router(integrations_router)
    app.dependency_overrides[require_service_token] = lambda: None
    client = TestClient(app)
    client.fake = fake
    return client


def test_registered_before_generic_integration_routes():
    order = [getattr(r, "prefix", None) for r in all_routers]
    assert order.index("/v1/integrations/mcp-servers") < order.index("/v1/integrations")


def test_list_and_create(client):
    assert client.get("/v1/integrations/mcp-servers").json() == {"servers": [{"slug": "alpha"}]}
    response = client.post("/v1/integrations/mcp-servers", json={
        "slug": "alpha", "url": "https://alpha.example.net/mcp", "auth_kind": "bearer", "bearer_token": "tok"})
    assert response.status_code == 201
    assert client.fake.created[0]["bearer_token"] == "tok"


@pytest.mark.parametrize("body", [{"slug": "a", "bearer_token": 5}, {"slug": "a", "extra": "secret-value"}])
def test_malformed_body_never_echoes_input(client, body):
    body = {**body, "bearer_token": body.get("bearer_token", "secret-value")}
    response = client.post("/v1/integrations/mcp-servers", json=body)
    assert response.status_code == 400 and response.json() == {"detail": "invalid_field"}
    assert "secret-value" not in response.text
