"""Duration stats are a static route with bounded filters and safe failures."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from llm_bawt.bot_tags.facade import TagPolicyUnavailable
from llm_bawt.service.routes import turn_logs as routes

PATH = "/v1/turn-logs/duration-stats"


@pytest.fixture
def client(monkeypatch):
    store = SimpleNamespace(engine=object())
    loader = Mock(return_value={"by_bot": [], "by_tag": []})
    monkeypatch.setattr(routes, "get_turn_log_store", lambda: store)
    monkeypatch.setattr(routes, "load_turn_duration_stats", loader)
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        yield client, store, loader


def test_default_seven_day_static_route_and_exact_filters(client):
    http, store, loader = client
    assert http.get(PATH).json() == {"by_bot": [], "by_tag": []}
    loader.assert_called_once_with(store.engine, days=7, tag=None, bot_id=None, user_id=None)
    response = http.get(PATH, params={
        "days": 3, "tag": "fixture_role", "bot_id": "fixture-bot", "user_id": "fixture-user",
    })
    assert response.status_code == 200
    loader.assert_called_with(
        store.engine, days=3, tag="fixture_role", bot_id="fixture-bot", user_id="fixture-user",
    )


@pytest.mark.parametrize("params", [
    {"days": 0}, {"days": 8}, {"days": "bad"}, {"tag": "bad tag"},
    {"tag": "x' OR true--"}, {"bot_id": ""}, {"bot_id": "b" * 65},
    {"user_id": ""}, {"user_id": "u" * 129},
])
def test_invalid_filters_never_query(client, params):
    http, _, loader = client
    assert http.get(PATH, params=params).status_code == 422
    loader.assert_not_called()


def test_missing_database_returns_503_without_query(client):
    http, store, loader = client
    store.engine = None
    assert http.get(PATH).status_code == 503
    loader.assert_not_called()


@pytest.mark.parametrize("error,status,detail", [
    (TagPolicyUnavailable("sensitive internals"), 503, "Bot tags unavailable"),
    (SQLAlchemyError("sensitive internals"), 503, "Turn duration stats unavailable"),
    (ValueError("unknown bot tag"), 422, "unknown bot tag"),
])
def test_failures_are_explicit_without_database_details(client, error, status, detail):
    http, _, loader = client
    loader.side_effect = error
    response = http.get(PATH)
    assert response.status_code == status
    assert response.json() == {"detail": detail}
