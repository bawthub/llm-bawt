"""TASK-1233: the operator freeze API's auth, validation and TTL cap (no PostgreSQL).

The freeze semantics on real PostgreSQL are in test_operator_freeze_pg.py.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from llm_bawt.bawthub_auth import TOKEN_ENV, TOKEN_FILE_ENV
from llm_bawt.service import operator_freeze
from llm_bawt.service.routes import admin_gates as route

TOKEN = "fixture-service-token-0123456789"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
URL = "/v1/admin/gates"
FREEZE = {"actor": "operator-1", "reason": "deploy window", "ttl_seconds": 3600}
THAW = {"actor": "operator-1"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    monkeypatch.delenv(TOKEN_FILE_ENV, raising=False)
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    app = FastAPI()
    app.include_router(route.router)
    app.dependency_overrides[route.get_gate_engine] = lambda: engine
    app.dependency_overrides[route.get_dispatch_wake] = lambda: (lambda: None)
    return TestClient(app)


def _send(client, method, url, headers, body):
    if isinstance(body, bytes):
        headers = {**headers, "Content-Type": "application/json"}
        return client.request(method, url, headers=headers, content=body)
    return client.request(method, url, headers=headers, json=body)


ROUTES = [
    ("GET", URL, None),
    ("POST", URL, FREEZE),
    ("POST", URL, {"wrong": "shape"}),  # auth runs before body validation
    ("POST", URL, b"{not json"),  # ... and before JSON decoding
    ("POST", f"{URL}/1/thaw", THAW),
    ("POST", f"{URL}/1/thaw", {"actor": "x", "override": True}),
    ("POST", f"{URL}/not-an-id/thaw", THAW),  # ... and before path validation
]


@pytest.mark.parametrize("headers", [
    {},
    {"Authorization": "Bearer wrong-token"},
    {"Authorization": f"Basic {TOKEN}"},
    {"Authorization": TOKEN},
])
@pytest.mark.parametrize("method,url,body", ROUTES)
def test_no_or_wrong_token_is_401(client, headers, method, url, body):
    response = _send(client, method, url, headers, body)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert response.headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize("method,url,body", ROUTES)
def test_unconfigured_token_fails_closed(client, monkeypatch, method, url, body):
    # No override, and conftest makes master-key derivation fail.
    monkeypatch.delenv(TOKEN_ENV)
    for headers in ({}, AUTH):
        response = _send(client, method, url, headers, body)
        assert response.status_code == 503
        assert response.json() == {"detail": "service_token_unconfigured"}


@pytest.mark.parametrize("change", [
    {"ttl_seconds": None},  # required
    {"ttl_seconds": operator_freeze.MAX_FREEZE_TTL_SECONDS + 1},  # hard maximum
    {"ttl_seconds": operator_freeze.MIN_FREEZE_TTL_SECONDS - 1},
    {"ttl_seconds": 0},
    {"ttl_seconds": -60},
    {"reason": None},
    {"reason": "   "},
    {"reason": "x" * (operator_freeze.MAX_REASON_CHARS + 1)},
    {"actor": None},
    {"actor": ""},
    {"actor": "has space"},
    {"actor": "x" * 97},
    {"scope": {}},  # names nothing
    {"scope": {"all_agents": True, "bots": ["a"]}},  # names two
    {"scope": {"bots": ["  "]}},
    {"scope": {"everything": True}},
    {"owner_kind": "job"},  # the caller can't pick the owner kind, mode or allowances
    {"mode": "hold"},
    {"allow_actors": ["reserve"]},
])
def test_invalid_freeze_is_422_and_never_reaches_the_store(client, monkeypatch, change):
    monkeypatch.setattr(route, "freeze", lambda *a, **k: pytest.fail("store reached"))
    body = {**FREEZE, **change}
    body = {key: value for key, value in body.items() if value is not None}
    response = client.post(URL, headers=AUTH, json=body)
    assert response.status_code == 422
    assert response.headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize("ttl", [operator_freeze.MIN_FREEZE_TTL_SECONDS,
                                 operator_freeze.MAX_FREEZE_TTL_SECONDS])
def test_ttl_bounds_are_inclusive(client, monkeypatch, ttl):
    seen = {}

    def fake_freeze(engine, **kwargs):
        seen.update(kwargs)
        raise operator_freeze.FreezeUnavailable("no pg")

    monkeypatch.setattr(route, "freeze", fake_freeze)
    response = client.post(URL, headers=AUTH, json={**FREEZE, "ttl_seconds": ttl})
    assert response.status_code == 503
    assert seen["ttl_seconds"] == ttl


def test_hard_maximum_is_at_most_twelve_hours():
    assert operator_freeze.MAX_FREEZE_TTL_SECONDS <= 12 * 60 * 60


def test_freeze_domain_refuses_a_ttl_past_the_cap_even_without_the_route():
    with pytest.raises(ValueError):
        operator_freeze.freeze(None, actor="a", reason="r", scope=None,
                               ttl_seconds=operator_freeze.MAX_FREEZE_TTL_SECONDS + 1)


def test_scope_defaults_to_every_agent_bot(client, monkeypatch):
    seen = {}

    def fake_freeze(engine, **kwargs):
        seen.update(kwargs)
        raise operator_freeze.FreezeUnavailable("no pg")

    monkeypatch.setattr(route, "freeze", fake_freeze)
    client.post(URL, headers=AUTH, json=FREEZE)
    assert seen["scope"].all_agents
    client.post(URL, headers=AUTH, json={**FREEZE, "scope": {"bots": [" Bot-A "]}})
    assert seen["scope"].bots == frozenset({"bot-a"})
    assert seen["actor"] == "operator-1" and seen["reason"] == "deploy window"


def test_off_postgres_status_is_empty_and_writes_are_503(client):
    status = client.get(URL, headers=AUTH)
    assert status.status_code == 200
    assert status.json() == {"frozen": False, "freezes": [],
                             "max_ttl_seconds": operator_freeze.MAX_FREEZE_TTL_SECONDS}
    assert client.post(URL, headers=AUTH, json=FREEZE).json() == {"detail": "gates_unavailable"}
    assert client.post(f"{URL}/1/thaw", headers=AUTH, json=THAW).status_code == 503


@pytest.mark.parametrize("body", [{}, {"actor": "a b"}, {"actor": "a", "force": True}])
def test_invalid_thaw_is_422(client, body):
    assert client.post(f"{URL}/1/thaw", headers=AUTH, json=body).status_code == 422


def test_router_is_registered_with_the_app():
    from llm_bawt.service.routes import all_routers

    assert route.router in all_routers
