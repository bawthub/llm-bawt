"""TASK-2363: scoped response reads never hydrate turn-debug envelopes."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.service import turn_logs as store_module
from llm_bawt.service.routes import turn_logs as routes
from llm_bawt.service.turn_logs import TurnLogStore


SCOPE = {"bot_id": "fixture-bot", "user_id": "fixture-user"}
URL = "/v1/turn-logs/turn-id/response"


def make_store(monkeypatch, **overrides):
    row = SimpleNamespace(
        id="turn-id", **SCOPE, response_text="raw response", status="ok", ended_at=None,
    )
    row.__dict__.update(overrides)
    statements = []
    session = Mock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)

    def execute(statement):
        statements.append(statement)
        # Model SQL equality: NULL owners never match a nonblank requested scope.
        params = statement.compile().params
        match = (row.id, row.bot_id, row.user_id) == (
            params["id_1"], params["bot_id_1"], params["user_id_1"],
        )
        return SimpleNamespace(first=lambda: row if match else None)

    session.exec.side_effect = execute
    monkeypatch.setattr(store_module, "Session", lambda engine: session)
    store = TurnLogStore.__new__(TurnLogStore)
    store.engine = object()
    for method in ("get_turn", "list_turns", "turn_identities", "_cleanup_expired_if_due"):
        setattr(store, method, Mock(side_effect=AssertionError(f"No heavy read: {method}")))
    return store, statements, session


def client_for(monkeypatch, store, **kwargs):
    monkeypatch.setattr(routes, "get_turn_log_store", lambda: store)
    for helper in (
        "get_service", "_live_tool_calls", "_turn_responses", "_turn_workers",
        "_turn_egress", "_parse_json", "_parse_token_usage", "_question_for_turn",
    ):
        monkeypatch.setattr(routes, helper, Mock(side_effect=AssertionError("No debug expansion")))
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app, **kwargs)


def test_store_projects_six_columns_with_exact_id_and_normalized_scope(monkeypatch):
    store, statements, session = make_store(monkeypatch, id=" turn-id ")
    result = store.turn_response(
        " turn-id ", bot_id=" FIXTURE-BOT ", user_id="\tFixture-User\n",
    )
    assert result == {
        "id": " turn-id ", **SCOPE, "response": "raw response", "status": "ok", "ended_at": None,
    }
    session.exec.assert_called_once()
    statement = statements[0]
    assert [column.name for column in statement.selected_columns] == [
        "id", "bot_id", "user_id", "response_text", "status", "ended_at",
    ]
    assert statement.compile().params == {
        "id_1": " turn-id ", "bot_id_1": "fixture-bot", "user_id_1": "fixture-user",
    }
    sql = str(statement).lower()
    assert "turn_logs.id = :id_1" in sql
    assert "turn_logs.bot_id = :bot_id_1" in sql
    assert "turn_logs.user_id = :user_id_1" in sql
    for fragment in (" join ", "request_json", "tool_calls", "reasoning", "created_at", "coalesce"):
        assert fragment not in sql


@pytest.mark.parametrize("overrides", [
    {"id": "missing"}, {"bot_id": "other-bot"}, {"user_id": "other-user"},
    {"bot_id": None}, {"user_id": None}, {"bot_id": None, "user_id": None},
])
def test_missing_foreign_and_null_owned_rows_are_same_404(monkeypatch, overrides):
    store, statements, _ = make_store(monkeypatch, **overrides)
    assert store.turn_response("turn-id", **SCOPE) is None
    with client_for(monkeypatch, store) as client:
        response = client.get(URL, params=SCOPE)
    assert response.status_code == 404
    assert response.json() == {"detail": "Turn log not found"}
    assert response.headers["cache-control"] == "no-store"
    assert len(statements) == 2


@pytest.mark.parametrize("text", [None, "", " \t\n ", "x" * 100_000, '{"raw":true}'])
@pytest.mark.parametrize(("status", "ended_at"), [
    ("running", None), ("ok", datetime(2026, 11, 16, 12, 34, 56, 123456, tzinfo=timezone.utc)),
    ("error", None),
])
def test_raw_response_and_terminal_fields_preserved(monkeypatch, text, status, ended_at):
    store, statements, _ = make_store(monkeypatch, response_text=text, status=status, ended_at=ended_at)
    result = store.turn_response("turn-id", **SCOPE)
    assert result["response"] == text
    assert result["status"] == status
    assert result["ended_at"] == ended_at
    with client_for(monkeypatch, store) as client:
        response = client.get(URL, params={"bot_id": " FIXTURE-BOT ", "user_id": " Fixture-User "})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    payload = response.json()
    assert payload == {
        "id": "turn-id", **SCOPE, "response": text, "status": status,
        "ended_at": None if ended_at is None else ended_at.isoformat().replace("+00:00", "Z"),
    }
    assert len(statements) == 2


@pytest.mark.parametrize("field", ["turn_id", "bot_id", "user_id"])
@pytest.mark.parametrize("value", ["", " \t\n ", None])
def test_store_rejects_invalid_scope_without_query(monkeypatch, field, value):
    store, statements, _ = make_store(monkeypatch)
    args = {"turn_id": "turn-id", **SCOPE, field: value}
    with pytest.raises(ValueError, match="nonblank"):
        store.turn_response(**args)
    assert statements == []


@pytest.mark.parametrize("params", [
    {}, {"bot_id": "fixture-bot"}, {"user_id": "fixture-user"},
    {**SCOPE, "bot_id": ""}, {**SCOPE, "user_id": ""},
    {**SCOPE, "bot_id": " \t\n "}, {**SCOPE, "user_id": " \t\n "},
])
def test_route_requires_nonblank_scope_without_query(monkeypatch, params):
    store, statements, _ = make_store(monkeypatch)
    with client_for(monkeypatch, store) as client:
        response = client.get(URL, params=params)
    assert response.status_code == 422
    assert response.headers["cache-control"] == "no-store"
    assert statements == []


def test_openapi_declares_both_owner_query_fields_required(monkeypatch):
    store, _, _ = make_store(monkeypatch)
    with client_for(monkeypatch, store) as client:
        schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/v1/turn-logs/{turn_id}/response"]["get"]
    params = {param["name"]: param for param in operation["parameters"]}
    for name in SCOPE:
        assert params[name]["in"] == "query"
        assert params[name]["required"] is True
    assert set(schema["components"]["schemas"]["TurnResponse"]["required"]) == {
        "id", "bot_id", "user_id", "response", "status", "ended_at",
    }


def test_unavailable_engine_is_503_without_query(monkeypatch):
    store, statements, _ = make_store(monkeypatch)
    store.engine = None
    with client_for(monkeypatch, store) as client:
        response = client.get(URL, params=SCOPE)
    assert response.status_code == 503
    assert response.json() == {"detail": "Turn logs DB unavailable"}
    assert response.headers["cache-control"] == "no-store"
    with pytest.raises(RuntimeError, match="DB unavailable"):
        store.turn_response("turn-id", **SCOPE)
    assert statements == []


def test_query_failure_propagates_and_is_500_not_missing(monkeypatch):
    store, _, session = make_store(monkeypatch)
    session.exec.side_effect = RuntimeError("fixture query failure")
    with pytest.raises(RuntimeError, match="fixture query failure"):
        store.turn_response("turn-id", **SCOPE)
    with client_for(monkeypatch, store) as client:
        with pytest.raises(RuntimeError, match="fixture query failure"):
            client.get(URL, params=SCOPE)
    with client_for(monkeypatch, store, raise_server_exceptions=False) as client:
        assert client.get(URL, params=SCOPE).status_code == 500


def test_response_model_strips_debug_fields(monkeypatch):
    store, statements, _ = make_store(monkeypatch)
    payload = {
        "id": "turn-id", **SCOPE, "response": None, "status": "running", "ended_at": None,
    }
    store.turn_response = Mock(return_value={
        **payload, "request": "private", "tool_calls": ["private"], "reasoning": "private",
        "responses": ["private"], "workers": ["private"], "egress_requests": ["private"],
    })
    with client_for(monkeypatch, store) as client:
        response = client.get(URL, params=SCOPE)
    assert response.status_code == 200
    assert response.json() == payload
    assert response.headers["cache-control"] == "no-store"
    store.turn_response.assert_called_once_with("turn-id", **SCOPE)
    assert statements == []


def test_no_store_policy_does_not_change_identity_route(monkeypatch):
    store, _, _ = make_store(monkeypatch)
    store.turn_identities = Mock(return_value={})
    with client_for(monkeypatch, store) as client:
        response = client.post("/v1/turn-logs/identities", json={"turn_ids": []})
    assert response.status_code == 200
    assert "cache-control" not in response.headers
