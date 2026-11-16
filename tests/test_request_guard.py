"""TASK-2371: Host allowlist and cross-site write guard."""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient

from llm_bawt.service.request_guard import GuardPolicy, RequestGuardMiddleware

ENV = {
    "LLM_BAWT_HOST_GUARD": "enforce",
    "LLM_BAWT_ALLOWED_HOSTS": "api.example.test, *.lan.example.test",
    "LLM_BAWT_ALLOWED_ORIGINS": "https://ui.example.test",
    "LLM_BAWT_PUBLIC_ORIGIN": "https://public.example.test/",
}


def _client(env: dict[str, str] = ENV, host: str = "app") -> TestClient:
    app = FastAPI()

    @app.get("/v1/thing")
    def read():
        return {"ok": True}

    @app.post("/v1/thing")
    def write():
        return {"ok": True}

    @app.post("/webhook/hook")
    def hook():
        return {"ok": True}

    @app.post("/integrations/google-home/token")
    def token():
        return {"ok": True}

    @app.websocket("/v1/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        await socket.send_text("hi")
        await socket.close()

    app.add_middleware(RequestGuardMiddleware, policy=GuardPolicy.from_env(env))
    return TestClient(app, base_url=f"http://{host}")


@pytest.mark.parametrize("host", ["app", "app:8642", "localhost:8642", "127.0.0.1", "[::1]:8642",
                                  "host.docker.internal:8642", "api.example.test", "echo.lan.example.test:8642"])
def test_allowed_hosts_pass(host):
    assert _client().get("/v1/thing", headers={"host": host}).status_code == 200


@pytest.mark.parametrize("host", ["evil.example", "evil.example:8642", "lan.example.test.evil", "10.9.9.9:8642", ""])
def test_rebinding_host_is_refused(host):
    response = _client().get("/v1/thing", headers={"host": host})
    assert response.status_code == 421
    assert response.json() == {"detail": "host_not_allowed"}


def test_host_rule_applies_to_webhooks_too():
    assert _client().post("/webhook/hook", headers={"host": "evil.example"}).status_code == 421


def test_server_callers_without_origin_can_write_any_body():
    client = _client()
    assert client.post("/v1/thing", json={"a": 1}).status_code == 200
    # curl -d sends form-encoded; tools rely on it.
    assert client.post("/v1/thing", data={"bot_id": "x"}).status_code == 200
    assert client.post("/v1/thing").status_code == 200


@pytest.mark.parametrize("origin", ["https://evil.example", "null", "http://api.example.test.evil", "file://"])
def test_cross_site_browser_write_is_refused(origin):
    response = _client().post("/v1/thing", content='{"url":"http://attacker:11434"}',
                              headers={"origin": origin, "content-type": "text/plain"})
    assert response.status_code == 403
    assert response.json() == {"detail": "cross_site_origin"}


@pytest.mark.parametrize("site", ["cross-site", "same-site"])
def test_cross_site_fetch_metadata_without_origin_is_refused(site):
    assert _client().post("/v1/thing", headers={"sec-fetch-site": site}).status_code == 403


@pytest.mark.parametrize("origin", ["https://ui.example.test", "https://public.example.test",
                                    "http://api.example.test:8642", "http://localhost:3000"])
def test_allowed_origins_can_write(origin):
    assert _client().post("/v1/thing", json={}, headers={"origin": origin}).status_code == 200


def test_same_origin_fetch_metadata_passes():
    assert _client().post("/v1/thing", headers={"sec-fetch-site": "same-origin"}).status_code == 200


def test_reads_ignore_origin():
    assert _client().get("/v1/thing", headers={"origin": "https://evil.example"}).status_code == 200


@pytest.mark.parametrize("path", ["/webhook/hook", "/integrations/google-home/token"])
def test_self_authenticating_webhooks_skip_the_origin_rule(path):
    assert _client().post(path, headers={"origin": "https://evil.example"}).status_code == 200


def test_cross_site_websocket_is_refused_and_own_one_opens():
    client = _client()
    # The test client sends Host "testserver" for WebSockets regardless of base_url.
    with client.websocket_connect("/v1/ws", headers={"host": "app"}) as socket:
        assert socket.receive_text() == "hi"
    with pytest.raises(WebSocketDisconnect) as refused:
        with client.websocket_connect("/v1/ws", headers={"host": "app", "origin": "https://evil.example"}):
            pass
    assert refused.value.code == 1008


def test_report_mode_logs_once_and_refuses_nothing(caplog):
    client = _client({**ENV, "LLM_BAWT_HOST_GUARD": "report"})
    with caplog.at_level(logging.WARNING, logger="llm_bawt.service.request_guard"):
        for _ in range(3):
            assert client.get("/v1/thing", headers={"host": "evil.example"}).status_code == 200
        assert client.post("/v1/thing", headers={"origin": "https://evil.example"}).status_code == 200
    lines = [r.getMessage() for r in caplog.records if "would refuse" in r.getMessage()]
    assert len(lines) == 2
    assert "host_not_allowed" in lines[0] and "cross_site_origin" in lines[1]


def test_off_mode_and_wildcard_host():
    assert _client({**ENV, "LLM_BAWT_HOST_GUARD": "off"}).post(
        "/v1/thing", headers={"host": "evil.example", "origin": "https://evil.example"}).status_code == 200
    any_host = _client({"LLM_BAWT_HOST_GUARD": "enforce", "LLM_BAWT_ALLOWED_HOSTS": "*"})
    assert any_host.get("/v1/thing", headers={"host": "evil.example"}).status_code == 200
    # Turning the Host rule off does not turn every origin into a trusted one.
    assert any_host.post("/v1/thing", headers={"origin": "https://evil.example"}).status_code == 403


def test_default_mode_is_report_and_bad_mode_falls_back():
    assert GuardPolicy.from_env({}).mode == "report"
    assert GuardPolicy.from_env({"LLM_BAWT_HOST_GUARD": "bogus"}).mode == "report"


def test_production_app_installs_the_guard():
    from llm_bawt.service.api import app

    assert any(m.cls is RequestGuardMiddleware for m in app.user_middleware)
