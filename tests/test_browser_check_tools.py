"""``browser_check_session`` mints for the trusted turn's bot only (TASK-1659)."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from llm_bawt.bawthub_auth.service_token import BawtHubServiceAuth
from llm_bawt.mcp_server import browser_check_tools as bc
from llm_bawt.task_turn_context import TaskTurnContext

MINTED = {
    "entityId": "browser-check",
    "sessionId": "s1",
    "expiresAt": "2030-01-01T00:00:00.000Z",
    "cookies": [
        {"name": "__Secure-better-auth.session_token", "value": "tok.sig", "path": "/",
         "httpOnly": True, "secure": True, "sameSite": "Lax", "expires": 1893456000},
    ],
}


@pytest.fixture
def rig(monkeypatch):
    state: dict = {"requests": [], "bot": "fixture-bot", "status": 201}
    monkeypatch.setenv("BAWTHUB_BROWSER_CHECK_INTERNAL_URL", "http://bawthub.test:3002/")
    monkeypatch.setenv("BAWTHUB_BROWSER_CHECK_ORIGIN", "https://ui.example.test")

    class _Token:
        def read(self):
            return "fixture-service-token"

    def handle(request: httpx.Request) -> httpx.Response:
        state["requests"].append(request)
        # Like BawtHub's internal port with sessions on: no service bearer, no mint.
        if request.headers.get("authorization") != "Bearer fixture-service-token":
            return httpx.Response(401, json={"error": "unauthorized"})
        if state["status"] != 201:
            return httpx.Response(state["status"], json={"error": "nope", "code": "x"})
        return httpx.Response(201, json=MINTED)

    def context():
        if state["bot"] is None:
            raise ValueError("no capability")
        return TaskTurnContext(session_id="s", turn_id="t", trigger_message_id="m",
                               bot_id=state["bot"], user_id="u", issued_at=0)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(bc, "current_task_turn_context", context)
    monkeypatch.setattr(bc, "service_auth", lambda: BawtHubServiceAuth(_Token()))
    monkeypatch.setattr(bc.httpx, "AsyncClient", lambda **kw: real_client(
        transport=httpx.MockTransport(handle), **kw))
    return state


def _call():
    return asyncio.run(bc.browser_check_session())


def test_mints_for_the_trusted_bot_and_scopes_cookies_to_the_origin(rig):
    res = _call()
    [req] = rig["requests"]
    assert req.method == "POST"
    assert str(req.url) == "http://bawthub.test:3002/internal/auth/browser-check-session"
    assert req.headers["x-agent-bot-id"] == "fixture-bot"
    assert req.headers["authorization"] == "Bearer fixture-service-token"
    assert res["entityId"] == "browser-check"
    assert res["origin"] == "https://ui.example.test"
    assert res["cookies"] == [{
        "name": "__Secure-better-auth.session_token", "value": "tok.sig",
        "domain": "ui.example.test", "path": "/", "expires": 1893456000,
        "httpOnly": True, "secure": True, "sameSite": "Lax",
    }]


def test_without_a_trusted_turn_nothing_is_requested(rig):
    rig["bot"] = None
    assert "trusted bot turn" in _call()["error"]
    assert rig["requests"] == []


def test_unconfigured_deployment_refuses(rig, monkeypatch):
    monkeypatch.delenv("BAWTHUB_BROWSER_CHECK_INTERNAL_URL")
    error = _call()["error"]
    assert "not configured" in error and "sessions are off" in error
    assert rig["requests"] == []


def test_sessions_off_404_explains_no_session_needed(rig):
    rig["status"] = 404
    result = _call()
    assert result["status"] == 404
    assert "sessions are off" in result["error"] and "user_required" in result["error"]


def test_bawthub_refusal_is_reported(rig):
    rig["status"] = 403
    assert _call() == {"error": "nope", "status": 403}
