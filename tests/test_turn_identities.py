"""TASK-2364: bounded identity reads never hydrate turn-debug envelopes."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.service import turn_logs as store_module
from llm_bawt.service.routes import turn_logs as routes
from llm_bawt.service.turn_logs import MAX_TURN_IDENTITY_BATCH, TurnLogStore


def make_store(monkeypatch, rows=()):
    statements = []
    fake_session = Mock()
    fake_session.__enter__ = Mock(return_value=fake_session)
    fake_session.__exit__ = Mock(return_value=False)
    fake_session.exec.side_effect = lambda statement: (
        statements.append(statement) or SimpleNamespace(all=lambda: list(rows))
    )
    monkeypatch.setattr(store_module, "Session", lambda engine: fake_session)
    store = TurnLogStore.__new__(TurnLogStore)
    store.engine = object()
    store.get_turn = Mock(side_effect=AssertionError("No full-debug turn read"))
    store.list_turns = Mock(side_effect=AssertionError("No full-debug turn list"))
    return store, statements


def client_for(monkeypatch, store):
    monkeypatch.setattr(routes, "get_turn_log_store", lambda: store)
    monkeypatch.setattr(routes, "get_service", Mock(side_effect=AssertionError("No service expansion")))
    for helper in ("_live_tool_calls", "_turn_responses", "_turn_workers", "_turn_egress"):
        monkeypatch.setattr(routes, helper, Mock(side_effect=AssertionError("No debug expansion")))
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_store_projects_only_three_columns_in_one_deduplicated_query(monkeypatch):
    store, statements = make_store(monkeypatch, [
        SimpleNamespace(id="historical-id", bot_id="fixture-bot", user_id="fixture-user"),
        SimpleNamespace(id="legacy-id", bot_id=None, user_id=None),
    ])
    result = store.turn_identities(["historical-id", "missing", "historical-id", "legacy-id"])
    assert result == {
        "historical-id": {"id": "historical-id", "bot_id": "fixture-bot", "user_id": "fixture-user"},
        "legacy-id": {"id": "legacy-id", "bot_id": None, "user_id": None},
    }
    assert len(statements) == 1
    statement = statements[0]
    assert [column.name for column in statement.selected_columns] == ["id", "bot_id", "user_id"]
    assert list(statement.compile().params.values()) == [["historical-id", "missing", "legacy-id"]]
    sql = str(statement).lower()
    assert " join " not in sql
    assert "created_at" not in sql
    assert "request_json" not in sql
    assert "response_text" not in sql


def test_store_zero_and_exact_limit_and_oversize_before_dedup(monkeypatch):
    store, statements = make_store(monkeypatch)
    assert store.turn_identities([]) == {}
    assert statements == []
    assert store.turn_identities([f"turn-{index}" for index in range(MAX_TURN_IDENTITY_BATCH)]) == {}
    assert len(statements) == 1
    assert len(next(iter(statements[0].compile().params.values()))) == MAX_TURN_IDENTITY_BATCH
    with pytest.raises(ValueError, match="At most 500"):
        store.turn_identities(["one-id"] * (MAX_TURN_IDENTITY_BATCH + 1))
    assert len(statements) == 1


@pytest.mark.parametrize("turn_id", ["", " \t ", "x" * 1025, None])
def test_store_rejects_bad_ids_before_query(monkeypatch, turn_id):
    store, statements = make_store(monkeypatch)
    with pytest.raises(ValueError, match="nonblank"):
        store.turn_identities([turn_id])
    assert statements == []


def test_route_returns_requested_keys_missing_and_nullable_owners_without_debug(monkeypatch):
    store, statements = make_store(monkeypatch, [
        SimpleNamespace(id="historical-id", bot_id="fixture-bot", user_id="fixture-user"),
        SimpleNamespace(id="legacy", bot_id=None, user_id=None),
    ])
    with client_for(monkeypatch, store) as client:
        response = client.post("/v1/turn-logs/identities", json={
            "turn_ids": ["historical-id", "missing", "historical-id", "legacy"],
        })
    assert response.status_code == 200
    assert response.json() == {"by_turn": {
        "historical-id": {"id": "historical-id", "bot_id": "fixture-bot", "user_id": "fixture-user"},
        "missing": None,
        "legacy": {"id": "legacy", "bot_id": None, "user_id": None},
    }}
    assert len(statements) == 1


def test_route_zero_exact_limit_and_oversize_before_dedup(monkeypatch):
    store, statements = make_store(monkeypatch)
    with client_for(monkeypatch, store) as client:
        assert client.post("/v1/turn-logs/identities", json={"turn_ids": []}).json() == {"by_turn": {}}
        assert statements == []
        ids = [f"turn-{index}" for index in range(MAX_TURN_IDENTITY_BATCH)]
        assert client.post("/v1/turn-logs/identities", json={
            "turn_ids": ids,
        }).json() == {"by_turn": dict.fromkeys(ids)}
        assert len(statements) == 1
        response = client.post("/v1/turn-logs/identities", json={
            "turn_ids": ["one-id"] * (MAX_TURN_IDENTITY_BATCH + 1),
        })
    assert response.status_code == 422
    assert len(statements) == 1


@pytest.mark.parametrize("value", [[""], [" "], ["x" * 1025], [1], None])
def test_route_rejects_invalid_ids_without_query(monkeypatch, value):
    store, statements = make_store(monkeypatch)
    with client_for(monkeypatch, store) as client:
        response = client.post("/v1/turn-logs/identities", json={"turn_ids": value})
    assert response.status_code == 422
    assert statements == []


def test_route_db_unavailable_is_503(monkeypatch):
    store, statements = make_store(monkeypatch)
    store.engine = None
    with client_for(monkeypatch, store) as client:
        response = client.post("/v1/turn-logs/identities", json={"turn_ids": ["turn-id"]})
    assert response.status_code == 503
    assert statements == []
    with pytest.raises(RuntimeError, match="DB unavailable"):
        store.turn_identities(["turn-id"])


def test_route_db_error_is_not_misreported_as_missing(monkeypatch):
    store, _ = make_store(monkeypatch)
    store.turn_identities = Mock(side_effect=RuntimeError("fixture DB failure"))
    with client_for(monkeypatch, store) as client:
        with pytest.raises(RuntimeError, match="fixture DB failure"):
            client.post("/v1/turn-logs/identities", json={"turn_ids": ["turn-id"]})


def test_response_model_strips_unconsumed_debug_fields(monkeypatch):
    store, _ = make_store(monkeypatch)
    store.turn_identities = Mock(return_value={
        "turn-id": {"id": "turn-id", "bot_id": "fixture-bot", "user_id": "fixture-user",
                    "request": "do not expose", "tool_calls": ["do not expose"]},
    })
    with client_for(monkeypatch, store) as client:
        response = client.post("/v1/turn-logs/identities", json={"turn_ids": ["turn-id"]})
    assert response.json() == {"by_turn": {
        "turn-id": {"id": "turn-id", "bot_id": "fixture-bot", "user_id": "fixture-user"},
    }}
