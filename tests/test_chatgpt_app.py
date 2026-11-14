"""Production-path SIWC contracts. Synthetic signed identity + mocked HTTPS only.

Fixtures follow official sign-in.md, models-and-inference.md and preview-
limitations.md, not the disposable helper's assertions. No credential reads.
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from llm_bawt.service.providers import chatgpt_app_auth as auth
from llm_bawt.service.providers import chatgpt_app as connection
from llm_bawt.service.providers.base import ConnectionRecord, STATUS_CONNECTED
from tests.stored_provider_keys import real_credential_store

_REAL_CLIENT = httpx.Client

from authlib.integrations.httpx_client._compat import httpx2 as _authlib_httpx  # noqa: E402


@pytest.fixture(scope="module")
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def upstream(monkeypatch, signing_key):
    state = {"calls": [], "claims": {}, "token": {}, "status": 200, "refreshes": 0}
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(signing_key.public_key()))
    jwk.update(kid="test-key", use="sig", alg="RS256")

    def make_handler(hx):
        def handler(request):
            state["calls"].append(request)
            if str(request.url) == auth.JWKS:
                return hx.Response(200, json={"keys": [jwk]})
            if str(request.url) == f"{auth.ISSUER}/.well-known/openid-configuration":
                return hx.Response(200, json={"revocation_endpoint": f"{auth.ISSUER}/api/accounts/oauth/revoke"})
            if request.url.path.endswith("/revoke"):
                return hx.Response(200)
            assert str(request.url) == auth.TOKEN
            body = parse_qs(request.content.decode())
            assert body["client_id"] == ["oaiapp_test"]
            assert body["resource"] == [auth.RESOURCE]
            if body["grant_type"] == ["authorization_code"]:
                assert body["redirect_uri"] == [auth.REDIRECT_URI]
                assert body["code_verifier"] == [state["pending"].verifier]
                assert body["code"] == ["synthetic-code"]
            else:
                assert body["grant_type"] == ["refresh_token"]
                state["refreshes"] += 1
            if state["status"] != 200:
                return hx.Response(state["status"], json=state.get("error", {"error": "server_error"}))
            claims = {"iss": auth.ISSUER, "sub": "subject-a", "aud": "oaiapp_test",
                      "iat": int(time.time()), "exp": int(time.time()) + 3600,
                      "nonce": state["pending"].nonce, "email": "user@example.test", **state["claims"]}
            id_token = jwt.encode(claims, state.get("signing_key", signing_key), algorithm="RS256", headers={"kid": "test-key"})
            return hx.Response(200, json={
                "access_token": f"synthetic-access-{state['refreshes']}",
                "refresh_token": f"synthetic-refresh-{state['refreshes']}",
                "id_token": id_token, "token_type": "Bearer", "expires_in": 3600,
                "scope": auth.SCOPE, **state["token"],
            })

        return handler

    # authlib uses httpx2 when it is installed (mcp 2 brings it in); its
    # OAuth2Client needs a transport from that module, metadata_client stays httpx.
    oauth_transport = _authlib_httpx.MockTransport(make_handler(_authlib_httpx))
    transport = httpx.MockTransport(make_handler(httpx))
    real_oauth = auth.client
    monkeypatch.setattr(auth, "client", lambda client_id, **kw: real_oauth(client_id, transport=oauth_transport, **kw))
    monkeypatch.setattr(auth, "metadata_client", lambda: _REAL_CLIENT(transport=transport))
    return state


def callback(pending, **extra):
    return auth.REDIRECT_URI + "?" + urlencode({
        "state": pending.state, "code": "synthetic-code", "client_id": "oaiapp_test", **extra,
    })


def prepare(upstream, *, registration=None):
    pending, url = auth.begin("urn:uuid:synthetic", registration=registration or {}, generation=None)
    upstream["pending"] = pending
    return pending, url


def test_library_pkce_and_own_registration(upstream):
    pending, url = prepare(upstream)
    params = parse_qs(urlsplit(url).query)
    assert params["client_id"] == [auth.DYNAMIC_CLIENT]
    assert params["agent_name_hint"] == ["BawtHub"]
    assert params["scope"] == [auth.SCOPE]
    assert params["nonce"] == [pending.nonce]
    assert params["redirect_uri"] == [auth.REDIRECT_URI]
    from authlib.oauth2.rfc7636 import create_s256_code_challenge
    assert params["code_challenge"] == [create_s256_code_challenge(pending.verifier)]
    assert params["code_challenge_method"] == ["S256"]
    bundle = auth.exchange(callback(pending), pending)
    assert bundle["client_id"] == "oaiapp_test"
    assert bundle["subject"] == "subject-a"
    assert bundle["scopes"] == auth.SCOPE.split()
    assert all(request.url.host == "auth.openai.com" for request in upstream["calls"])


@pytest.mark.parametrize("change", [
    {"nonce": "wrong"}, {"iss": "https://wrong.test"}, {"aud": "wrong-client"},
    {"sub": ""}, {"exp": 1}, {"iat": 9999999999}, {"azp": "wrong-client"},
])
def test_signed_identity_claim_failures(upstream, change):
    pending, _ = prepare(upstream)
    upstream["claims"] = change
    with pytest.raises(Exception):
        auth.exchange(callback(pending), pending)


def test_signature_not_just_decoded(upstream):
    pending, _ = prepare(upstream)
    upstream["signing_key"] = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(Exception):
        auth.exchange(callback(pending), pending)


@pytest.mark.parametrize("mutate", [
    lambda url: url.replace("state=", "state=wrong"),
    lambda url: url + "&state=duplicate", lambda url: url + "&code=duplicate",
    lambda url: url.replace("127.0.0.1", "localhost"),
    lambda url: url.replace(":1455", ":1456"),
    lambda url: url.replace("/auth/callback", "/callback"),
    lambda url: url + "#fragment", lambda url: url.replace("client_id=oaiapp_test", "client_id=dynamic_agent_client"),
])
def test_bad_callbacks_make_no_exchange(upstream, mutate):
    pending, _ = prepare(upstream)
    with pytest.raises(auth.SignInError):
        auth.exchange(mutate(callback(pending)), pending)
    assert not upstream["calls"]


def test_denied_callback_no_exchange(upstream):
    pending, _ = prepare(upstream)
    with pytest.raises(auth.SignInError):
        auth.exchange(auth.REDIRECT_URI + "?" + urlencode({"state": pending.state, "error": "access_denied"}), pending)
    assert not upstream["calls"]


def test_returning_registration_and_identity_fence(upstream):
    pending, url = prepare(upstream, registration={"client_id": "oaiapp_test", "subject": "subject-a"})
    assert "agent_name_hint" not in parse_qs(urlsplit(url).query)
    assert auth.exchange(callback(pending).replace("&client_id=oaiapp_test", ""), pending)["subject"] == "subject-a"
    with pytest.raises(auth.SignInError):
        auth.callback_parameters(callback(pending, client_id="oaiapp_other"), pending)
    upstream["claims"] = {"sub": "subject-b"}
    with pytest.raises(auth.SignInError):
        auth.exchange(callback(pending), pending)


@pytest.mark.parametrize("token", [
    {"scope": "chatgpt.tokens.use.direct"}, {"expires_in": 0}, {"expires_in": True},
    {"token_type": "DPoP"}, {"refresh_token": ""},
])
def test_grant_requirements(upstream, token):
    pending, _ = prepare(upstream)
    upstream["token"] = token
    with pytest.raises(auth.SignInError):
        auth.exchange(callback(pending), pending)


@pytest.fixture
def adapter(monkeypatch):
    with real_credential_store(monkeypatch) as (store, _):
        monkeypatch.setattr(store, "durable_key_problem", lambda: None)
        obj = object.__new__(connection.ChatGPTAppAdapter)
        obj.store = store
        obj.config = SimpleNamespace()
        connection._pending.clear()
        yield obj
        connection._pending.clear()


def connect(adapter, upstream):
    result = adapter.start_browser_login()
    pending = connection._pending[result["session_id"]]
    upstream["pending"] = pending
    adapter.complete_browser_login(result["session_id"], callback(pending))
    return result, pending


def test_real_encrypted_store_and_old_provider_unchanged(adapter, upstream):
    old = ConnectionRecord(provider="openai_chatgpt", status=STATUS_CONNECTED, secret={"sentinel": "old-untouched"})
    adapter.store.save(old)
    result, pending = connect(adapter, upstream)
    assert adapter.store.load("openai_chatgpt").secret == old.secret
    public = adapter.descriptor()
    assert public["connection"]["connected"]
    assert "synthetic-access" not in json.dumps(public)
    from sqlalchemy import text
    with adapter.store._store.engine.connect() as conn:
        raw = str(conn.execute(text("select value_json from runtime_settings")).fetchall())
    assert "synthetic-access" not in raw and "synthetic-refresh" not in raw
    assert "secret_enc" in raw
    with pytest.raises(auth.SignInError):
        adapter.complete_browser_login(result["session_id"], callback(pending))


def test_refresh_rotation_single_owner_and_rejected_hash(adapter, upstream):
    connect(adapter, upstream)
    original = adapter.access_token()
    rejected = hashlib.sha256(original["access_token"].encode()).hexdigest()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: adapter.access_token(rejected_token_sha256=rejected), range(4)))
    assert upstream["refreshes"] == 1
    assert {result["access_token"] for result in results} == {"synthetic-access-1"}
    assert adapter.store.load(adapter.id).secret["refresh_token"] == "synthetic-refresh-1"
    assert {result["account_id"] for result in results} == {original["account_id"]}


def test_terminal_refresh_clears_only_new_tokens_and_commits(adapter, upstream):
    connect(adapter, upstream)
    upstream.update(status=400, error={"error": "invalid_grant"})
    token = adapter.store.load(adapter.id).secret["access_token"]
    assert adapter.access_token(rejected_token_sha256=hashlib.sha256(token.encode()).hexdigest())["state"] == "reconnect"
    record = adapter.store.load(adapter.id)
    assert record.status == "error" and "access_token" not in record.secret
    assert record.secret["client_id"] == "oaiapp_test"
    assert adapter.health()["state"] == "broken"


def test_transient_refresh_preserves_saved_pair(adapter, upstream):
    connect(adapter, upstream)
    before = adapter.store.load(adapter.id).secret
    upstream["status"] = 503
    with pytest.raises(auth.SignInError):
        adapter.access_token(rejected_token_sha256=hashlib.sha256(before["access_token"].encode()).hexdigest())
    assert adapter.store.load(adapter.id).secret == before


def test_disconnect_fences_pending_and_retains_registration(adapter, upstream):
    connect(adapter, upstream)
    started = adapter.start_browser_login()
    pending = connection._pending[started["session_id"]]
    assert adapter.disconnect()["remote_revocation"] == "confirmed"
    assert "access_token" not in adapter.store.load(adapter.id).secret
    assert adapter.store.load(adapter.id).secret["client_id"] == "oaiapp_test"
    with pytest.raises(auth.SignInError):
        adapter.complete_browser_login(started["session_id"], callback(pending))
    assert adapter.health()["state"] == "unconfigured"
    assert parse_qs(urlsplit(adapter.start_browser_login()["verification_uri"]).query)["client_id"] == ["oaiapp_test"]


def test_initial_disconnect_tombstone_fences_another_worker_pending(adapter, upstream):
    started = adapter.start_browser_login()
    pending = connection._pending[started["session_id"]]
    adapter.disconnect()
    connection._pending[started["session_id"]] = pending  # another app worker's memory
    with pytest.raises(auth.SignInError, match="changed"):
        adapter.complete_browser_login(started["session_id"], callback(pending))
    assert adapter.start_browser_login()["session_id"]


def test_discovery_uses_models_slug_and_visibility(adapter, upstream, monkeypatch):
    connect(adapter, upstream)
    from llm_bawt.service.model_discovery import ChatGPTAppDiscoveryProvider
    monkeypatch.setattr(connection.ChatGPTAppAdapter, "__init__", lambda self, config: self.__dict__.update(store=adapter.store))
    def get(url, **kwargs):
        assert url == "https://api.openai.com/v1/models"
        assert kwargs["headers"]["Authorization"].startswith("Bearer synthetic-access-")
        return httpx.Response(200, request=httpx.Request("GET", url), json={"models": [
            {"slug": "gpt-6.1-sol", "display_name": "Sol", "visibility": "list", "context_window": 100000},
            {"slug": "hidden", "visibility": "hide"},
        ]})
    monkeypatch.setattr(httpx, "get", get)
    assert ChatGPTAppDiscoveryProvider(SimpleNamespace()).fetch() == [
        {"id": "gpt-6.1-sol", "description": "Sol", "context_length": 100000},
    ]


@pytest.mark.parametrize("choice_type", ["tool", "function"])
def test_proxy_route_namespaced_tools_and_result_replay(choice_type):
    from claude_code_bridge.proxy.adapters.chatgpt_app import ChatGPTAppAdapter
    from claude_code_bridge.proxy.translate import anthropic_to_responses
    adapter = ChatGPTAppAdapter()
    body = anthropic_to_responses({
        "model": "chatgpt_app/gpt-6.1-sol", "max_tokens": 4096,
        "tools": [{"name": "noop", "description": "No-op", "input_schema": {"type": "object", "properties": {}}}],
        "tool_choice": {"type": choice_type, "name": "noop"},
        "messages": [
            {"role": "user", "content": "synthetic"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "call_one", "name": "noop", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_one", "content": "OK"}]},
        ],
    }, "gpt-6.1-sol")
    adapter.prepare_request(body)
    assert body["tools"][0]["type"] == "namespace"
    assert body["tools"][0]["tools"][0]["name"] == "noop"
    assert body["input"][1]["namespace"] == "bawthub"
    assert body["input"][2] == {"type": "function_call_output", "call_id": "call_one", "output": "OK"}
    assert body["tool_choice"] == {"type": "function", "name": "noop"}
    assert "max_output_tokens" not in body
    assert body["stream"] and not body["store"]


@pytest.mark.asyncio
async def test_proxy_resolves_broker_each_request_and_fences_accounts(monkeypatch):
    from claude_code_bridge.proxy.adapters.chatgpt_app import ChatGPTAppAdapter, BASE_URL
    adapter = ChatGPTAppAdapter()
    counter = iter(["account-a", "account-b"])
    monkeypatch.setattr(adapter, "_fetch_token", lambda: {"access_token": "synthetic", "account_id": next(counter)})
    assert await adapter.authorize() == ("synthetic", BASE_URL)
    first = adapter.reasoning_codec("gpt-6.1-sol", BASE_URL)
    assert await adapter.authorize() == ("synthetic", BASE_URL)
    assert adapter.reasoning_codec("gpt-6.1-sol", BASE_URL) != first
    headers = adapter.extra_headers({})
    assert set(headers) == {"session_id"}  # no originator/beta/account identity


def test_session_id_header_matches_prompt_cache_key():
    """TASK-1958: session_id rides with prompt_cache_key for sticky cache routing."""
    from claude_code_bridge.proxy.adapters.chatgpt_app import ChatGPTAppAdapter
    from claude_code_bridge.proxy.request_context import ProxyRequestContext
    adapter = ChatGPTAppAdapter()
    ctx = ProxyRequestContext(request_id="r", provider="chatgpt_app", conversation_id="ab" * 16)
    body = adapter.prepare_request({"model": "m", "instructions": "x", "input": []}, ctx)
    assert adapter.extra_headers(body, ctx) == {"session_id": body["prompt_cache_key"]} == {"session_id": "ab" * 16}
    bare = {"model": "m", "instructions": "stable prefix", "input": []}
    fallback = adapter.prepare_request(dict(bare), None)
    assert adapter.extra_headers(dict(bare), None) == {"session_id": fallback["prompt_cache_key"]}


@pytest.mark.asyncio
async def test_real_responses_client_stream_and_second_tool_roundtrip(monkeypatch):
    from claude_code_bridge.proxy.adapters.chatgpt_app import ChatGPTAppAdapter
    from openai import AsyncOpenAI
    captured = []
    def handler(request):
        body = json.loads(request.content)
        captured.append(body)
        assert request.url == httpx.URL("https://api.openai.com/v1/responses")
        assert request.headers["authorization"] == "Bearer synthetic"
        if len(captured) < 3:
            number = len(captured)
            item = {"type": "function_call", "id": f"fc_{number}", "call_id": f"call_{number}",
                    "namespace": "bawthub", "name": "noop", "arguments": "{}", "status": "completed"}
            events = [
                {"type": "response.created", "response": {"id": f"resp_{number}", "model": "gpt-6.1-sol"}},
                {"type": "response.output_item.added", "output_index": 0, "item": {**item, "arguments": ""}},
                {"type": "response.function_call_arguments.delta", "item_id": item["id"], "output_index": 0, "delta": "{}"},
                {"type": "response.output_item.done", "output_index": 0, "item": item},
                {"type": "response.completed", "response": {"id": f"resp_{number}", "output": [item], "usage": {"input_tokens": 20, "output_tokens": 5}}},
            ]
        else:
            assert [i["call_id"] for i in body["input"] if i.get("type") == "function_call_output"] == ["call_1", "call_2"]
            events = [
                {"type": "response.created", "response": {"id": "resp_3", "model": "gpt-6.1-sol"}},
                {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "DONE"},
                {"type": "response.completed", "response": {"id": "resp_3", "output": [], "usage": {"input_tokens": 30, "output_tokens": 2}}},
            ]
        data = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=data)
    adapter = ChatGPTAppAdapter()
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter._http_client = http
    adapter._responses_client = AsyncOpenAI(api_key="unused", http_client=http)
    monkeypatch.setattr(adapter, "_fetch_token", lambda: {"access_token": "synthetic", "account_id": "account-a"})
    messages = [{"role": "user", "content": "Run noop twice, then say DONE"}]
    try:
        for number in range(1, 4):
            chunks = b"".join([chunk async for chunk in adapter.call({
                "model": "chatgpt_app/gpt-6.1-sol", "stream": True, "messages": messages,
                "tools": [{"name": "noop", "input_schema": {"type": "object", "properties": {}}}],
            }, "gpt-6.1-sol")])
            assert b'event: message_stop' in chunks
            assert b'event: error' not in chunks
            if number < 3:
                assert b'"name":"noop"' in chunks or b'"name": "noop"' in chunks
                messages.extend([
                    {"role": "assistant", "content": [{"type": "tool_use", "id": f"call_{number}", "name": "noop", "input": {}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"call_{number}", "content": "OK"}]},
                ])
            else:
                assert b"DONE" in chunks
    finally:
        await adapter.close()
    assert len(captured) == 3


@pytest.mark.parametrize("expires", [float("inf"), float("nan"), -1, "3600"])
def test_bundle_rejects_nonfinite_or_untyped_expiry(expires):
    with pytest.raises(auth.SignInError):
        auth.token_bundle({"access_token": "synthetic", "refresh_token": "synthetic",
                           "scope": auth.SCOPE, "token_type": "Bearer", "expires_in": expires})


def test_declined_plan_retains_verified_signin_but_never_brokers_inference(adapter, upstream):
    upstream["token"] = {"scope": "openid profile email offline_access"}
    connect(adapter, upstream)
    record = adapter.store.load(adapter.id)
    assert record.status == "plan_disabled" and record.secret["subject"] == "subject-a"
    assert not adapter.descriptor()["connection"]["connected"]
    assert "not enabled" in adapter.health()["detail"]
    with pytest.raises(auth.SignInError):
        adapter.access_token()
    assert not any(request.url.path.endswith("/revoke") for request in upstream["calls"])
    enabled = adapter.start_browser_login()
    params = parse_qs(urlsplit(enabled["verification_uri"]).query)
    assert params["client_id"] == ["oaiapp_test"]
    assert params["scope"] == [auth.SCOPE]
    assert params["prompt"] == ["consent"]
    assert adapter.disconnect()["disconnected"]


def test_failed_initial_exchange_retains_registration_and_revokes_new_session(adapter, upstream):
    started = adapter.start_browser_login()
    pending = connection._pending[started["session_id"]]
    upstream.update(pending=pending, claims={"nonce": "wrong"})
    with pytest.raises(auth.SignInError):
        adapter.complete_browser_login(started["session_id"], callback(pending))
    record = adapter.store.load(adapter.id)
    assert record.secret == {"client_id": "oaiapp_test"}
    assert record.status == "disconnected"
    assert any(request.url.path.endswith("/revoke") for request in upstream["calls"])
    assert parse_qs(urlsplit(adapter.start_browser_login()["verification_uri"]).query)["client_id"] == ["oaiapp_test"]


def test_failed_reconnect_and_cancel_leave_active_pair_untouched(adapter, upstream):
    connect(adapter, upstream)
    original = adapter.store.load(adapter.id).secret
    started = adapter.start_browser_login()
    pending = connection._pending[started["session_id"]]
    upstream.update(pending=pending, status=503)
    with pytest.raises(auth.SignInError):
        adapter.complete_browser_login(started["session_id"], callback(pending))
    assert adapter.store.load(adapter.id).secret == original
    started = adapter.start_browser_login()
    adapter.cancel_browser_login(started["session_id"])
    assert started["session_id"] not in connection._pending
    assert adapter.store.load(adapter.id).secret == original


def test_expired_attempt_and_changed_generation_never_exchange(adapter, upstream):
    started = adapter.start_browser_login()
    pending = connection._pending[started["session_id"]]
    pending.deadline = 0
    with pytest.raises(auth.SignInError, match="expired"):
        adapter.complete_browser_login(started["session_id"], callback(pending))
    assert not upstream["calls"]


def test_refresh_may_omit_scope_and_refresh_token(adapter, upstream):
    connect(adapter, upstream)
    original = adapter.store.load(adapter.id).secret
    # Test token bundling directly: the real token endpoint may omit unchanged values.
    renewed = auth.token_bundle({"access_token": "new", "token_type": "bearer", "expires_in": 3600}, previous=original)
    assert renewed["refresh_token"] == original["refresh_token"]
    assert renewed["scopes"] == original["scopes"]


def test_real_browser_routes_broker_gate_and_secret_free_failures(adapter, upstream, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from llm_bawt.service.routes import providers as routes
    from llm_bawt.service import bridge_gate
    monkeypatch.setattr(routes, "_adapter_or_404", lambda _: adapter)
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        started = client.post("/v1/providers/chatgpt-app/browser/start")
        assert started.status_code == 200
        assert started.headers["cache-control"] == "no-store"
        session = started.json()["session_id"]
        pending = connection._pending[session]
        upstream["pending"] = pending
        completed = client.post("/v1/providers/chatgpt-app/browser/complete", json={
            "session_id": session, "callback_url": callback(pending),
        })
        assert completed.status_code == 200
        assert "synthetic-access" not in completed.text and "synthetic-refresh" not in completed.text
        calls = []
        monkeypatch.setattr(bridge_gate, "check", lambda request, purpose: calls.append(purpose))
        broker = client.get("/v1/providers/chatgpt-app/token")
        assert calls == [bridge_gate.BROKER]
        assert broker.json()["access_token"].startswith("synthetic-access")
        assert broker.headers["cache-control"] == "no-store"
        invalid = client.post("/v1/providers/chatgpt-app/browser/complete", json={
            "session_id": session, "callback_url": callback(pending),
        })
        assert invalid.status_code == 400 and "synthetic-code" not in invalid.text
        def broken(*args):
            raise RuntimeError("secret-access-refresh-callback")
        monkeypatch.setattr(adapter, "complete_browser_login", broken)
        failure = client.post("/v1/providers/chatgpt-app/browser/complete", json={
            "session_id": session, "callback_url": callback(pending),
        })
        assert failure.status_code == 502 and "secret-access-refresh-callback" not in failure.text
        assert client.delete("/v1/providers/chatgpt-app").json()["disconnected"]


@pytest.mark.asyncio
async def test_app_stream_does_not_publish_codex_quota(monkeypatch):
    from claude_code_bridge.proxy.adapters.chatgpt_app import ChatGPTAppAdapter
    from claude_code_bridge.proxy import usage_capture
    from openai import AsyncOpenAI
    events = [{"type": "response.created", "response": {"id": "resp"}},
              {"type": "response.completed", "response": {"id": "resp", "output": []}}]
    data = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
    adapter = ChatGPTAppAdapter()
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, headers={
        "content-type": "text/event-stream", "x-codex-primary-used-percent": "99",
    }, text=data)))
    adapter._http_client = http
    adapter._responses_client = AsyncOpenAI(api_key="unused", http_client=http)
    monkeypatch.setattr(adapter, "_fetch_token", lambda: {"access_token": "synthetic", "account_id": "account-a"})
    def unexpected(*args):
        pytest.fail("app response must not write the Codex quota snapshot")
    monkeypatch.setattr(usage_capture, "schedule_capture", unexpected)
    try:
        chunks = b"".join([chunk async for chunk in adapter.call({"messages": []}, "test")])
        assert b"message_stop" in chunks
    finally:
        await adapter.close()
    from llm_bawt.service.usage import usage_adapter_for_access_path
    assert usage_adapter_for_access_path("openai-chatgpt-app") is None
    assert usage_adapter_for_access_path("openai-oauth").provider == "openai_chatgpt"
