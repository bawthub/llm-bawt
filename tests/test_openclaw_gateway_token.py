"""The OpenClaw gateway token is record-first for the app and the bridge (TASK-1476).

The app reads ``integration_connection:openclaw`` in process. The openclaw
bridge has no root key, so it reads the same record through the bridge-gated
route ``GET /v1/internal/bridge/connections/integration/openclaw``. Absent
record (404) is the only state that lets the legacy env apply; an unusable
record fails closed on both sides. No value is ever logged.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel

from llm_bawt.integrations import openclaw_credentials
from llm_bawt.integrations.connections import IntegrationStore
from llm_bawt.runtime_settings import RuntimeSetting, RuntimeSettingsStore
from llm_bawt.service import bridge_gate, connection_secrets
from llm_bawt.service.connection_secrets import BAWTHUB_READABLE, BRIDGE_READABLE, ConnectionUnavailable
from llm_bawt.service.providers import crypto
from llm_bawt.service.providers.base import STATUS_CONNECTED, STATUS_ERROR, ConnectionRecord, CredentialStore
from llm_bawt.service.routes import internal_connections as route
from openclaw_bridge import gateway_token as bridge_reader
from openclaw_bridge.gateway_token import LEGACY_ENV, ROUTE, GatewayTokenUnavailable, gateway_token
from openclaw_bridge.ws_client import OpenClawWsClient, OpenClawWsConfig

STORED = "stored-gateway-token-do-not-leak"
LEGACY = "legacy-gateway-token-do-not-leak"
BROKER_TOKEN = "broker-fixture-token"
API = "http://app.test:8642"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(connection_secrets, "_legacy_warned", set())
    monkeypatch.setattr(bridge_reader, "_warned", set())
    monkeypatch.delenv(LEGACY_ENV, raising=False)


# --- the bridge read route -----------------------------------------------

@pytest.fixture
def stores(monkeypatch):
    fernet = Fernet(Fernet.generate_key())
    monkeypatch.setattr(crypto, "_get_fernet", lambda: fernet)
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    SQLModel.metadata.create_all(engine, tables=[RuntimeSetting.__table__])
    settings_store = object.__new__(RuntimeSettingsStore)
    settings_store.config = None
    settings_store.engine = engine
    made = {}
    for kind, cls in (("provider", CredentialStore), ("integration", IntegrationStore)):
        store = object.__new__(cls)
        store._store = settings_store
        made[kind] = store
    monkeypatch.setattr(connection_secrets, "_store", lambda _config, kind: made[kind])
    return made


@pytest.fixture
def client(stores, monkeypatch):
    gates = bridge_gate.BridgeGates(lambda: {}, environ={},
                                    derive=lambda label: BROKER_TOKEN if label == bridge_gate.BROKER.label
                                    else "other-" + label)
    monkeypatch.setattr(bridge_gate, "_GATES", gates)
    monkeypatch.setattr(route, "get_config", lambda: None)
    app = FastAPI()
    app.include_router(route.bridge_router)
    return TestClient(app)


AUTH = {"X-Bridge-Token": BROKER_TOKEN}
URL = route.BRIDGE_PREFIX + "/integration/openclaw"


def _save(stores, cid="openclaw", *, secret=None, status=STATUS_CONNECTED, meta=None):
    stores["integration"].save(ConnectionRecord(provider=cid, status=status, meta=meta or {},
                                                secret=secret if secret is not None else {"token": STORED}))


def test_route_is_the_one_bridge_reader_contract():
    assert ROUTE == URL
    assert set(BRIDGE_READABLE) == {("integration", "openclaw")}
    # BawtHub's service-token route never serves the gateway token.
    assert ("integration", "openclaw") not in BAWTHUB_READABLE


def test_a_bad_bridge_token_is_refused_before_the_store(client, stores, monkeypatch):
    monkeypatch.setattr(connection_secrets, "_store", lambda *_: pytest.fail("store read before the gate"))
    for headers in ({"X-Bridge-Token": "wrong"}, {"X-Bridge-Token": ""}):
        assert client.get(URL, headers=headers).status_code == 401


def test_enforced_gate_refuses_a_missing_header(client, monkeypatch):
    monkeypatch.setattr(bridge_gate._GATES, "enforced", lambda _gate: True)
    assert client.get(URL).status_code == 401


def test_connected_record_returns_only_the_token_no_store(client, stores):
    _save(stores, meta={"note": "x"})
    response = client.get(URL, headers=AUTH)
    assert response.status_code == 200
    assert response.json() == {"kind": "integration", "id": "openclaw", "status": "connected",
                               "meta": {}, "secret": {"token": STORED}}
    assert "no-store" in response.headers["cache-control"]


def test_absent_record_is_404_not_configured(client):
    response = client.get(URL, headers=AUTH)
    assert (response.status_code, response.json()["detail"]) == (404, "not_configured")
    assert "no-store" in response.headers["cache-control"]


@pytest.mark.parametrize(("kind", "cid"), [("integration", "home-assistant"), ("provider", "xai"),
                                           ("integration", "nope")])
def test_only_allowlisted_connections_are_served(client, stores, kind, cid):
    _save(stores, "home-assistant", secret={"token": STORED}, meta={"url": "https://ha.example"})
    response = client.get(f"{route.BRIDGE_PREFIX}/{kind}/{cid}", headers=AUTH)
    assert response.status_code == 404
    assert STORED not in response.text


def test_disconnected_record_is_409(client, stores):
    _save(stores, status=STATUS_ERROR)
    response = client.get(URL, headers=AUTH)
    assert (response.status_code, response.json()["detail"]) == (409, "not_connected")


def test_blank_token_is_503_unreadable(client, stores):
    _save(stores, secret={"token": "  "})
    response = client.get(URL, headers=AUTH)
    assert (response.status_code, response.json()["detail"]) == (503, "unreadable")


def test_store_down_is_503(client, stores, monkeypatch):
    def down(*_args, **_kwargs):
        raise ConnectionUnavailable("store_unavailable")

    monkeypatch.setattr(route, "read_connection", down)
    response = client.get(URL, headers=AUTH)
    assert (response.status_code, response.json()["detail"]) == (503, "store_unavailable")


def test_route_never_logs_the_value(client, stores, caplog):
    caplog.set_level(logging.DEBUG)
    _save(stores)
    client.get(URL, headers=AUTH)
    assert STORED not in caplog.text


# --- the app's in-process reader -----------------------------------------

def _config(**fields):
    return SimpleNamespace(**{"POSTGRES_PASSWORD": "synthetic", "OPENCLAW_GATEWAY_TOKEN": "", **fields})


def _stub(monkeypatch, outcome):
    def fake_read(_config, kind, cid, spec, **_kwargs):
        assert (kind, cid) == ("integration", "openclaw")
        if isinstance(outcome, Exception):
            raise outcome
        if outcome is None:
            return None
        return connection_secrets.ConnectionSecret(kind=kind, id=cid, meta={}, secret={"token": outcome})

    monkeypatch.setattr(connection_secrets, "read_connection", fake_read)


def test_app_record_wins_over_legacy(monkeypatch):
    _stub(monkeypatch, STORED)
    assert openclaw_credentials.gateway_token(_config(OPENCLAW_GATEWAY_TOKEN=LEGACY)) == STORED


def test_app_absent_record_uses_legacy_and_warns_once_by_name(monkeypatch, caplog):
    _stub(monkeypatch, None)
    caplog.set_level(logging.WARNING)
    config = _config(OPENCLAW_GATEWAY_TOKEN=LEGACY)
    assert openclaw_credentials.gateway_token(config) == LEGACY
    assert openclaw_credentials.gateway_token(config) == LEGACY
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and openclaw_credentials.LEGACY_SETTING in warnings[0]
    assert LEGACY not in caplog.text


def test_app_nothing_configured_is_empty(monkeypatch):
    _stub(monkeypatch, None)
    assert openclaw_credentials.gateway_token(_config()) == ""


@pytest.mark.parametrize("code", ["not_connected", "unreadable", "store_unavailable"])
def test_app_unusable_record_never_falls_back(monkeypatch, code):
    _stub(monkeypatch, ConnectionUnavailable(code))
    with pytest.raises(ConnectionUnavailable) as raised:
        openclaw_credentials.gateway_token(_config(OPENCLAW_GATEWAY_TOKEN=LEGACY))
    assert LEGACY not in str(raised.value) and str(raised.value) == code


def _streamer(**fields):
    from llm_bawt.service.chat_streaming_bridge import ChatStreamingBridgeMixin

    streamer = ChatStreamingBridgeMixin.__new__(ChatStreamingBridgeMixin)
    streamer.config = _config(OPENCLAW_GATEWAY_URL="http://gw.test", **fields)
    return streamer


def test_app_stream_rebuilds_the_gateway_client_on_rotation(monkeypatch):
    # Loopy r1 on eb46516: a saved or rotated token must reach the HTTP client
    # without a restart, and an unchanged token must keep the client.
    tokens = iter(["first", "first", "second"])
    monkeypatch.setattr(openclaw_credentials, "gateway_token", lambda _config: next(tokens))
    streamer = _streamer()

    async def turns():
        return [await streamer._openclaw_gateway() for _ in range(3)]

    first, same, rotated = asyncio.run(turns())
    assert same is first and first.token == "first"
    assert rotated is not first and rotated.token == "second"
    assert rotated._client.headers["Authorization"] == "Bearer second"
    assert streamer._gateway_http is rotated


def test_app_stream_unusable_record_fails_the_turn(monkeypatch):
    def unavailable(_config):
        raise ConnectionUnavailable("not_connected")

    monkeypatch.setattr(openclaw_credentials, "gateway_token", unavailable)
    streamer = _streamer(OPENCLAW_GATEWAY_TOKEN=LEGACY)
    with pytest.raises(ConnectionUnavailable):
        asyncio.run(streamer._openclaw_gateway())
    assert getattr(streamer, "_gateway_http", None) is None


# --- the bridge's reader -------------------------------------------------

def _http(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _env(**extra):
    return {"LLM_BAWT_API_URL": API + "/", "BRIDGE_BROKER_TOKEN_FILE": "",
            "BRIDGE_CLAUDE_TOKEN_SECRET": BROKER_TOKEN, **extra}


def test_bridge_reads_the_record_with_the_broker_header():
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"secret": {"token": STORED}})

    assert gateway_token(_env(**{LEGACY_ENV: LEGACY}), client=_http(handler)) == STORED
    assert str(seen[0].url) == API + ROUTE
    assert seen[0].headers["X-Bridge-Token"] == BROKER_TOKEN


def test_bridge_404_uses_legacy_env_warning_once(caplog):
    caplog.set_level(logging.WARNING)
    client = _http(lambda _r: httpx.Response(404, json={"detail": "not_configured"}))
    for _ in range(2):
        assert gateway_token(_env(**{LEGACY_ENV: LEGACY}), client=client) == LEGACY
    warnings = [r for r in caplog.records if LEGACY_ENV in r.getMessage()]
    assert len(warnings) == 1
    assert LEGACY not in caplog.text


def test_bridge_404_without_legacy_is_no_token():
    client = _http(lambda _r: httpx.Response(404, json={"detail": "not_configured"}))
    assert gateway_token(_env(), client=client) == ""


@pytest.mark.parametrize(("status", "detail", "code"), [
    (409, "not_connected", "not_connected"),
    (503, "unreadable", "unreadable"),
    (503, "store_unavailable", "store_unavailable"),
    (401, "bad bridge token", "http_401"),
    (500, None, "http_500"),
])
def test_bridge_unusable_record_never_falls_back(status, detail, code):
    client = _http(lambda _r: httpx.Response(status, json={"detail": detail}))
    with pytest.raises(GatewayTokenUnavailable) as raised:
        gateway_token(_env(**{LEGACY_ENV: LEGACY}), client=client)
    assert raised.value.code == code
    assert LEGACY not in str(raised.value)


def test_bridge_network_error_never_falls_back():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(GatewayTokenUnavailable) as raised:
        gateway_token(_env(**{LEGACY_ENV: LEGACY}), client=_http(handler))
    assert raised.value.code == "unreachable:ConnectError"


@pytest.mark.parametrize("body", [{}, {"secret": {}}, {"secret": {"token": " "}}, {"secret": "x"}, []])
def test_bridge_malformed_body_is_unavailable(body):
    client = _http(lambda _r: httpx.Response(200, json=body))
    with pytest.raises(GatewayTokenUnavailable) as raised:
        gateway_token(_env(**{LEGACY_ENV: LEGACY}), client=client)
    assert raised.value.code == "malformed"


def test_bridge_without_an_app_url_uses_legacy_only():
    assert gateway_token({LEGACY_ENV: LEGACY}) == LEGACY
    assert gateway_token({}) == ""


def test_bridge_reader_never_logs_the_value(caplog):
    caplog.set_level(logging.DEBUG)
    client = _http(lambda _r: httpx.Response(200, json={"secret": {"token": STORED}}))
    gateway_token(_env(), client=client)
    assert STORED not in caplog.text


# --- the WS client resolves per connect ----------------------------------

def _ws_client(provider) -> OpenClawWsClient:
    return OpenClawWsClient(OpenClawWsConfig(url="ws://gw.test/v1/ws", token="static-ignored",
                                             session_keys=["main"], token_provider=provider))


def test_ws_connect_uses_the_provider_each_time():
    tokens = iter(["first", "second"])
    client = _ws_client(lambda: next(tokens))
    assert asyncio.run(client._resolve_token()) == "first"
    assert asyncio.run(client._resolve_token()) == "second"


def test_ws_without_provider_uses_static_token():
    client = OpenClawWsClient(OpenClawWsConfig(url="ws://gw.test", token="static"))
    assert asyncio.run(client._resolve_token()) == "static"


def test_ws_provider_failure_skips_the_socket_and_reconnects(caplog):
    def unavailable():
        raise GatewayTokenUnavailable("not_connected")

    client = _ws_client(unavailable)
    reconnects: list[bool] = []
    client._schedule_reconnect = lambda: reconnects.append(True)

    async def run():
        with patch("openclaw_bridge.ws_client.websockets") as mock_ws:
            mock_ws.connect = AsyncMock()
            await client._do_connect()
            return mock_ws.connect

    connect = asyncio.run(run())
    connect.assert_not_called()
    assert reconnects == [True] and not client.connected
    assert "not_connected" in caplog.text
