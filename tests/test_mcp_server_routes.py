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


def test_callback_requires_the_service_bearer():
    app = FastAPI()
    app.include_router(mcp_servers.router)
    response = TestClient(app).get("/v1/integrations/mcp-servers/alpha/callback?state=s&code=c")
    assert response.status_code in (401, 403, 503)


def test_each_server_has_its_own_sign_in_cookie(monkeypatch, client):
    seen = {}

    class Flow:
        def start(self, server, origin, return_to):
            return {"authorization_url": "https://auth.example.net/authorize", "browser": "b-" + server.slug}

        def finish(self, server, state, browser, code, error, iss):
            seen[server.slug] = browser
            return "/settings/mcp-servers"

    client.fake.oauth = Flow()
    client.fake.get = lambda slug: type("S", (), {"slug": slug})()
    response = client.post("/v1/integrations/mcp-servers/alpha/connect", json={},
                           headers={"origin": "https://hub.example.com"})
    cookie = response.headers["set-cookie"]
    assert cookie.startswith("__Secure-bawthub-mcp-login=b-alpha")
    assert "Path=/api/chat/proxy/v1/integrations/mcp-servers/alpha/callback" in cookie
    client.cookies.set("__Secure-bawthub-mcp-login", "b-alpha")
    done = client.get("/v1/integrations/mcp-servers/alpha/callback?state=s&code=c", follow_redirects=False)
    assert seen == {"alpha": "b-alpha"} and done.status_code == 303
    cleared = done.headers["set-cookie"]
    assert "__Secure-bawthub-mcp-login=" in cleared and "mcp-servers/alpha/callback" in cleared
    assert "bawthub-integration" not in cleared


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
