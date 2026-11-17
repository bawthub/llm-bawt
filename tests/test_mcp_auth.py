"""TASK-2381: the :8001 MCP gate (per-caller auth, own mode, rollback)."""

import asyncio
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet

from llm_bawt import task_turn_context as codec
from llm_bawt.api_auth import keys, mcp
from llm_bawt.api_auth.middleware import ApiAuthMiddleware, Principal
from llm_bawt.api_auth.mode import ModeResolver

ROOT = lambda: b"r" * 32  # noqa: E731
TURN = codec.TASK_TURN_CONTEXT_HEADER


class Store:
    def __init__(self, values):
        self.values = values

    def __call__(self):
        return self.values


@pytest.fixture
def fernet(monkeypatch):
    instance = Fernet(Fernet.generate_key())
    monkeypatch.setattr(codec, "_get_fernet", lambda: instance)
    return instance


def _turn(bot="loopy"):
    return codec.mint_task_turn_context(
        session_id=str(uuid4()), turn_id="turn-" + "a" * 32,
        trigger_message_id=str(uuid4()), bot_id=bot, user_id="operator")


def _gate(values):
    store = Store(values)
    gate = mcp.wrap(None).gate
    gate.resolver = ModeResolver(store, environ={}, setting_key=mcp.SETTING_KEY, ttl=0)
    gate._root = ROOT
    return gate, store


def _call(gate, *headers):
    sent, reached = [], []

    async def app(scope, receive, send):
        reached.append(scope.get("state", {}).get("api_principal"))

    async def send(msg):
        sent.append(msg)

    gate.app = app
    scope = {"type": "http", "path": "/mcp", "method": "POST", "client": ("10.0.0.9", 1),
             "headers": [(k.lower().encode(), v.encode())
                         for k, v in (("user-agent", "probe/1"),) + headers]}
    asyncio.run(gate(scope, None, send))
    if reached:
        return 200, reached[0]
    import json
    return sent[0]["status"], json.loads(sent[1]["body"])["detail"]


def _key(principal):
    return (keys.HEADER, keys.header_value(principal, root=ROOT))


def test_wrap_gates_every_path_with_service_keys_only():
    gate = mcp.wrap(None).gate
    assert isinstance(gate, ApiAuthMiddleware)
    assert gate.allowed_paths == frozenset()
    assert gate.resolver is mcp.RESOLVER and mcp.RESOLVER._key == "mcp_auth_mode"


def test_enforce_refuses_anonymous_and_log_admits_and_records():
    gate, store = _gate({"mcp_auth_mode": "enforce"})
    assert _call(gate) == (401, "missing_credentials")
    store.values = {"mcp_auth_mode": "log"}
    assert _call(gate) == (200, None)
    rows = [r["key"] for r in gate.observations.snapshot()]
    assert ["anon", "10.0.0.9", "probe/1", "POST", "/mcp"] in rows


@pytest.mark.parametrize("mode", ["log", "enforce"])
def test_each_bot_is_attributed_to_its_own_principal(fernet, mode):
    gate, _ = _gate({"mcp_auth_mode": mode})
    for bot in ("loopy", "caid"):
        assert _call(gate, (TURN, _turn(bot))) == (200, Principal(bot, "bot"))
    rows = [r["key"][:2] for r in gate.observations.snapshot()]
    assert ["bot", "loopy"] in rows and ["bot", "caid"] in rows


def test_bad_or_expired_turn_is_anonymous(fernet, monkeypatch):
    gate, _ = _gate({"mcp_auth_mode": "enforce"})
    assert _call(gate, (TURN, "garbage"))[0] == 401
    token = _turn()
    monkeypatch.setattr(codec.time, "time", lambda: 10**12)
    assert _call(gate, (TURN, token)) == (401, "missing_credentials")


@pytest.mark.parametrize("mode", ["log", "enforce"])
def test_only_service_keys_and_never_with_a_turn(fernet, mode):
    gate, _ = _gate({"mcp_auth_mode": mode})
    assert _call(gate, _key("llm-bawt-app")) == (200, Principal("llm-bawt-app", "key"))
    assert _call(gate, _key("caid")) == (401, "key_not_permitted")
    assert _call(gate, _key("bawthub-web")) == (401, "key_not_permitted")
    assert _call(gate, _key("llm-bawt-app"), (TURN, _turn())) == (401, "ambiguous_credentials")


def test_rollback_switch_and_independence_from_v1(fernet):
    gate, store = _gate({"mcp_auth_mode": "enforce", "api_auth_mode": "off"})
    assert _call(gate)[0] == 401  # /v1 off does not open :8001
    store.values = {"mcp_auth_mode": "off", "api_auth_mode": "enforce"}
    assert _call(gate) == (200, None)  # rollback write reopens :8001 only
    assert _call(gate, _key("caid")) == (200, None)  # off checks nothing
    store.values = {}
    assert _call(gate) == (200, None)  # absent row: env default (off)


def test_app_key_header(monkeypatch):
    monkeypatch.setattr(keys, "header_value", lambda p: f"{p}.k")
    assert mcp.app_key_header() == {"X-LLM-Bawt-Key": "llm-bawt-app.k"}

    def broken(p):
        raise RuntimeError("no master key")
    monkeypatch.setattr(keys, "header_value", broken)
    assert mcp.app_key_header() == {}


def test_client_transport_sends_app_key(monkeypatch):
    from llm_bawt.mcp_server import client_transport

    assert "app_key_header()" in open(client_transport.__file__).read()


def test_self_fwd_refuses_spoofed_sender_before_reading(monkeypatch):
    from llm_bawt.mcp_server import inter_bot_sender, self_tools
    from llm_bawt.mcp_server.inter_bot_sender import _SignedCaller

    monkeypatch.setattr(inter_bot_sender, "_signed_caller", lambda: _SignedCaller("caid"))

    async def tail(*a, **k):
        raise AssertionError("tail read before sender check")
    monkeypatch.setattr(self_tools, "_tail_bubbles", tail)
    fn = getattr(self_tools.self_fwd, "fn", self_tools.self_fwd)
    result = asyncio.run(fn(sender_bot_id="vex", target_bot_id="loopy"))
    assert result["success"] is False and "does not match" in result["error"]


def test_standalone_attaches_store_and_failure_stays_closed(monkeypatch):
    from llm_bawt.api_auth import wiring

    monkeypatch.setattr(wiring.SETTINGS, "store", None)
    mcp.ensure_store(lambda: (_ for _ in ()).throw(RuntimeError("db down")))
    assert wiring.SETTINGS.store is None  # unresolved -> 503, never open

    sentinel = object()
    mcp.ensure_store(lambda: sentinel)
    assert wiring.SETTINGS.store is sentinel
    mcp.ensure_store(lambda: object())  # an attached store is kept
    assert wiring.SETTINGS.store is sentinel


def test_run_server_attaches_store_before_wrapping():
    from llm_bawt.mcp_server import server

    src = open(server.__file__).read()
    assert src.index("mcp_auth.ensure_store()") < src.index("mcp_auth.wrap(")


def _get_observations(endpoint, *headers, method="GET"):
    sent = []

    async def send(msg):
        sent.append(msg)

    scope = {"type": "http", "path": mcp.OBSERVATIONS_PATH, "method": method,
             "client": ("10.0.0.9", 1),
             "headers": [(k.lower().encode(), v.encode()) for k, v in headers]}
    asyncio.run(endpoint(scope, None, send))
    import json
    return sent[0]["status"], json.loads(sent[1]["body"])


@pytest.mark.parametrize("mode", ["off", "log", "enforce"])
def test_mcp_server_serves_its_own_counts_to_the_app_key_only(fernet, mode):
    endpoint = mcp.wrap(None)
    gate, _ = _gate({"mcp_auth_mode": "log"})
    endpoint.gate = gate
    assert _call(gate, (TURN, _turn("loopy")))[0] == 200  # recorded in this process
    gate.resolver = ModeResolver(Store({"mcp_auth_mode": mode}), environ={},
                                 setting_key=mcp.SETTING_KEY, ttl=0)
    status, body = _get_observations(endpoint, _key(mcp.APP_PRINCIPAL))
    assert status == 200 and body["mode"] == mode
    assert ["bot", "loopy", "POST", "/mcp"] in [row["key"] for row in body["rows"]]
    for headers in ((), (_key("caid"),), ((keys.HEADER, "x"),)):
        assert _get_observations(endpoint, *headers) == (401, {"detail": "service_key_required"})
    assert _get_observations(endpoint, (TURN, _turn("loopy")))[0] == 401
    assert _get_observations(endpoint, _key(mcp.APP_PRINCIPAL), method="POST")[0] == 401


def test_admin_view_fetches_from_the_mcp_server(monkeypatch):
    import httpx

    seen = {}

    def handler(request):
        seen["url"], seen["key"] = str(request.url), request.headers.get(keys.HEADER)
        return httpx.Response(200, json={"mode": "enforce", "rows": []})

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(mcp, "app_key_header", lambda: {keys.HEADER: "k"})
    view = asyncio.run(mcp.fetch_observations("http://mcp:8001/"))
    assert view == {"mode": "enforce", "rows": [], "source": "mcp_server"}
    assert seen == {"url": "http://mcp:8001" + mcp.OBSERVATIONS_PATH, "key": "k"}

    fallback = asyncio.run(mcp.fetch_observations(None))
    assert fallback["source"] == "in_process" and fallback["remote_error"] == "no_server_url"
