"""TASK-2426: OAuth bearer sign-in for human MCP clients on :8001."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from llm_bawt.api_auth import mcp
from llm_bawt.api_auth.mcp_oauth import (
    IntrospectionUnavailable, McpOAuthLayer, OAuthCaller, TokenCache, current_oauth_caller,
    load_config)
from llm_bawt.api_auth.middleware import Principal
from llm_bawt.api_auth.mode import ModeResolver

RESOURCE = "https://mcp.example.test/mcp"
# Better Auth declares its base URL as the issuer, not the bare origin.
ISSUER = "https://hub.example.test/api/auth"
ENV = {"LLM_BAWT_MCP_OAUTH_RESOURCE": RESOURCE + "/", "LLM_BAWT_MCP_OAUTH_ISSUER": ISSUER,
       "LLM_BAWT_MCP_OAUTH_PRINCIPALS": "owner, ops",
       "BAWTHUB_TASK_ASSOCIATION_INTERNAL_URL": "http://hub-internal:3002"}
ACTIVE = {"active": True, "entityId": "owner", "clientId": "c1", "scope": "openid mcp",
          "resource": RESOURCE, "exp": 4_000_000_000}


def _layer(mode="enforce", answers=None, clock=None):
    calls = []

    async def introspect(token):
        calls.append(token)
        answer = (answers or {}).get(token, {"active": False})
        if isinstance(answer, Exception):
            raise answer
        return answer

    gate = mcp.wrap(None).gate
    gate.resolver = ModeResolver(lambda: {"mcp_auth_mode": mode}, environ={},
                                 setting_key=mcp.SETTING_KEY, ttl=0)
    cache = TokenCache(clock=clock) if clock else None
    return McpOAuthLayer(gate, load_config(ENV), introspect=introspect, cache=cache), gate, calls


def _call(layer, gate, *headers, path="/mcp", method="POST", app_status=200):
    sent, reached = [], []

    async def app(scope, receive, send):
        reached.append(scope["state"]["api_principal"])
        await send({"type": "http.response.start", "status": app_status, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def send(msg):
        sent.append(msg)

    gate.app = app
    scope = {"type": "http", "path": path, "method": method, "client": ("10.0.0.9", 1),
             "headers": [(k.lower().encode(), v.encode()) for k, v in headers]}
    asyncio.run(layer(scope, None, send))
    if reached:
        return sent[0]["status"], reached[0], {}
    hdrs = {k.decode(): v.decode() for k, v in sent[0]["headers"]}
    return sent[0]["status"], json.loads(sent[1]["body"]), hdrs


def _bearer(token):
    return ("authorization", f"Bearer {token}")


def test_config_is_dormant_unless_both_urls_are_set():
    assert load_config({}) is None
    assert load_config({"LLM_BAWT_MCP_OAUTH_RESOURCE": RESOURCE}) is None
    assert load_config({**ENV, "LLM_BAWT_MCP_OAUTH_ISSUER": "ftp://x"}) is None
    config = load_config(ENV)
    assert config.resource == RESOURCE
    assert config.principals == frozenset({"owner", "ops"})
    assert config.introspect_url == "http://hub-internal:3002/internal/auth/mcp-introspect"
    assert config.metadata_url == "https://mcp.example.test/.well-known/oauth-protected-resource/mcp"


def test_protected_resource_metadata_is_public():
    layer, gate, _ = _layer()
    for path in ("/.well-known/oauth-protected-resource",
                 "/.well-known/oauth-protected-resource/mcp"):
        status, body, _ = _call(layer, gate, path=path, method="GET")
        assert status == 200
        assert body["resource"] == RESOURCE and body["authorization_servers"] == [ISSUER]
        assert body["scopes_supported"] == ["mcp"]


def test_missing_credentials_get_a_discovery_challenge():
    layer, gate, _ = _layer()
    status, body, headers = _call(layer, gate)
    assert (status, body["detail"]) == (401, "missing_credentials")
    assert headers["www-authenticate"].startswith(
        'Bearer resource_metadata="https://mcp.example.test/.well-known/oauth-protected-resource/mcp"')


def test_active_bearer_for_an_allowed_entity_is_admitted_as_a_user():
    layer, gate, _ = _layer(answers={"t1": ACTIVE})
    status, principal, _ = _call(layer, gate, _bearer("t1"))
    assert status == 200 and principal == Principal("owner", "user")


@pytest.mark.parametrize("answer", [
    {"active": False},
    {**ACTIVE, "resource": "https://other.example.test/mcp"},
])
def test_inactive_or_foreign_bearer_is_invalid_token(answer):
    layer, gate, _ = _layer(answers={"t1": answer})
    status, body, headers = _call(layer, gate, _bearer("t1"))
    assert (status, body["detail"]) == (401, "invalid_token")
    assert 'error="invalid_token"' in headers["www-authenticate"]


def test_entity_outside_the_principal_set_is_forbidden():
    layer, gate, _ = _layer(answers={"t1": {**ACTIVE, "entityId": "guest"}})
    status, body, _ = _call(layer, gate, _bearer("t1"))
    assert (status, body["detail"]) == (403, "principal_not_allowed")


def test_bearer_with_a_turn_or_key_is_ambiguous():
    layer, gate, calls = _layer(answers={"t1": ACTIVE})
    for extra in (("x-llm-bawt-task-turn-context", "x"), ("x-llm-bawt-key", "k")):
        status, body, _ = _call(layer, gate, _bearer("t1"), extra)
        assert (status, body["detail"]) == (401, "ambiguous_credentials")
    assert calls == []


def test_introspection_outage_is_503_and_not_cached():
    layer, gate, calls = _layer(answers={"t1": IntrospectionUnavailable("ConnectError")})
    assert _call(layer, gate, _bearer("t1"))[0] == 503
    assert _call(layer, gate, _bearer("t1"))[0] == 503
    assert calls == ["t1", "t1"]


def test_answers_are_cached_briefly_and_never_past_exp():
    now = [1000.0]
    layer, gate, calls = _layer(answers={"t1": {**ACTIVE, "exp": 1030}}, clock=lambda: now[0])
    assert _call(layer, gate, _bearer("t1"))[0] == 200
    assert _call(layer, gate, _bearer("t1"))[0] == 200
    assert calls == ["t1"]
    now[0] = 1031  # past exp, inside the 60 s TTL
    _call(layer, gate, _bearer("t1"))
    assert calls == ["t1", "t1"]


def test_principal_in_state_cannot_come_from_a_client_header():
    layer, gate, _ = _layer()
    status, body, _ = _call(layer, gate, ("mcp_oauth_principal", "owner"))
    assert (status, body["detail"]) == (401, "missing_credentials")


def test_wrap_installs_the_layer_only_when_configured(monkeypatch):
    for key in ENV:
        monkeypatch.delenv(key, raising=False)
    assert not isinstance(mcp.wrap(None).outer, McpOAuthLayer)
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    assert isinstance(mcp.wrap(None).outer, McpOAuthLayer)


def test_metadata_names_the_exact_provider_issuer():
    # Strict MCP clients (RFC 8414 3.3) compare this to the provider's own
    # metadata issuer, which Better Auth sets to <origin>/api/auth.
    assert load_config(ENV).metadata()["authorization_servers"] == [ISSUER]


def test_verified_user_and_client_reach_tool_execution():
    layer, gate, _ = _layer(answers={"t": ACTIVE})
    seen = []

    async def app(scope, receive, send):
        seen.append(current_oauth_caller())

    gate.app = app
    scope = {"type": "http", "path": "/mcp", "method": "POST", "client": ("10.0.0.9", 1),
             "headers": [(b"authorization", b"Bearer t")]}

    async def send(msg):
        pass

    asyncio.run(layer(scope, None, send))
    assert seen == [OAuthCaller("owner", "c1")]
    assert current_oauth_caller() is None


def test_active_answer_without_a_client_is_invalid():
    layer, gate, _ = _layer(answers={"t": {**ACTIVE, "clientId": ""}})
    assert _call(layer, gate, _bearer("t"))[0] == 401


def test_audit_records_each_accepted_oauth_request_and_escapes_client_id(caplog):
    client = 'second\n"client'
    layer, gate, calls = _layer(answers={
        "fixture-bearer-one": ACTIVE,
        "fixture-bearer-two": {**ACTIVE, "clientId": client},
    })
    with caplog.at_level("INFO"):
        for bearer in ("fixture-bearer-one", "fixture-bearer-two", "fixture-bearer-one"):
            assert _call(layer, gate, _bearer(bearer))[0] == 200
    assert calls == ["fixture-bearer-one", "fixture-bearer-two"]
    events = [r.getMessage() for r in caplog.records
              if r.getMessage().startswith("mcp_oauth_request_accepted ")]
    assert [json.loads(event.split(" ", 1)[1]) for event in events] == [
        {"entity_id": "owner", "client_id": client_id, "status": 200}
        for client_id in ("c1", client, "c1")
    ]
    assert all("\n" not in event and "fixture-bearer" not in event for event in events)


def test_audit_excludes_denied_anonymous_and_service_requests(caplog, monkeypatch):
    layer, gate, _ = _layer(answers={"valid": ACTIVE, "denied": {**ACTIVE, "entityId": "guest"}})
    monkeypatch.setattr(gate, "_verify_key", lambda _: SimpleNamespace(ok=True, principal=mcp.APP_PRINCIPAL))
    with caplog.at_level("INFO"):
        assert _call(layer, gate)[0] == 401
        assert _call(layer, gate, _bearer("inactive"))[0] == 401
        assert _call(layer, gate, _bearer("denied"))[0] == 403
        assert _call(layer, gate, _bearer("valid"), app_status=403)[0] == 403
        assert _call(layer, gate, path="/.well-known/oauth-protected-resource", method="GET")[0] == 200
        assert _call(layer, gate, ("x-llm-bawt-key", "fixture-service"))[:2] == (
            200, Principal(mcp.APP_PRINCIPAL, "key"),
        )
    assert "mcp_oauth_request_accepted" not in caplog.text


def test_oauth_identity_does_not_supply_ops_turn_authority(monkeypatch):
    from unittest.mock import AsyncMock, Mock

    from llm_bawt.api_auth import mcp_oauth
    from llm_bawt.mcp_server import approval_interceptor as ai

    server = object.__new__(ai.ApprovalAwareFastMCP)
    server._approval_store_provider = Mock()
    server._operations_requester = Mock()
    downstream = AsyncMock(return_value={})
    monkeypatch.setattr(ai.MCPServer, "call_tool", downstream)
    token = mcp_oauth._oauth_caller.set(OAuthCaller("owner", "c1"))
    try:
        result = asyncio.run(server._intercept_call("ops_run", {"operation": "x"}, None))
        assert result["status"] == "approval_context_missing"
        assert result["is_error"] is True
        downstream.assert_not_awaited()
        server._operations_requester.assert_not_called()
        server._approval_store_provider.assert_not_called()

        assert asyncio.run(server._intercept_call("tasks_board", {}, None)) == {}
        downstream.assert_awaited_once_with("tasks_board", {}, None)
        server._approval_store_provider.assert_not_called()
    finally:
        mcp_oauth._oauth_caller.reset(token)


def test_service_principals_need_a_matching_key_export():
    exports = "llm-bawt-app=/run/k/app,bawthub-web=/run/k/web"
    assert mcp.service_principals({}) == {mcp.APP_PRINCIPAL}
    assert mcp.service_principals({
        "LLM_BAWT_MCP_SERVICE_PRINCIPALS": "bawthub-web, rogue",
        "LLM_BAWT_API_KEY_EXPORTS": exports}) == {mcp.APP_PRINCIPAL, "bawthub-web"}
    assert mcp.service_principals({
        "LLM_BAWT_MCP_SERVICE_PRINCIPALS": "bawthub-web",
        "LLM_BAWT_API_KEY_EXPORTS": "not valid"}) == {mcp.APP_PRINCIPAL}
