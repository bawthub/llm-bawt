"""Standalone purge reports transactional refusal, not partial/unknown success."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError

from llm_bawt.service.routes import settings


@pytest.mark.parametrize("error,reason", [
    (ValueError("Cannot purge source: prompt occurrences are active or unknown"), "active or unknown"),
    (IntegrityError("private SQL", {"secret": "private parameter"}, Exception("private driver detail")), "dependent records"),
])
def test_purge_conflict_is_409_with_explicit_rollback_and_no_sql_leak(monkeypatch, error, reason):
    monkeypatch.setattr(settings, "get_service", lambda: SimpleNamespace(config=object()))
    purge = Mock(side_effect=error)
    monkeypatch.setattr(settings, "purge_bot_data", purge)
    app = FastAPI()
    app.include_router(settings.router)
    with TestClient(app) as client:
        response = client.post("/v1/bots/SOURCE/purge-data")
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert reason in detail
    assert "Nothing was deleted" in detail
    assert "private" not in detail
    assert purge.call_args.args[1] == "source"


@pytest.mark.parametrize("result,status", [
    ({"slug": "source", "deleted_rows": {"sessions": 0}}, 200),
    ({"error": "No database credentials configured"}, 503),
])
def test_purge_keeps_success_and_unavailable_contract_without_profile_lookup(monkeypatch, result, status):
    monkeypatch.setattr(settings, "get_service", lambda: SimpleNamespace(config=object()))
    monkeypatch.setattr(settings, "get_bot_profile_store", Mock(side_effect=AssertionError("no profile lookup")))
    monkeypatch.setattr(settings, "purge_bot_data", lambda *args: result)
    app = FastAPI()
    app.include_router(settings.router)
    with TestClient(app) as client:
        response = client.post("/v1/bots/source/purge-data")
    assert response.status_code == status
    assert response.json() == (result if status == 200 else {"detail": result["error"]})


@pytest.fixture
def delete_env(monkeypatch):
    service = SimpleNamespace(config=object())
    store = SimpleNamespace(engine=object(), get=Mock(return_value=object()), delete=Mock(return_value=True))
    monkeypatch.setattr(settings, "get_service", lambda: service)
    monkeypatch.setattr(settings, "get_bot_profile_store", lambda config: store)
    effects = []
    for name in ("_reload_bot_registry", "_reload_service_model_catalog",
                 "_invalidate_bot_instance_cache", "_clear_session_model_overrides"):
        effect = Mock()
        monkeypatch.setattr(settings, name, effect)
        effects.append(effect)
    published = AsyncMock()
    monkeypatch.setattr(settings, "_publish_bot_profile_changed", published)
    app = FastAPI()
    app.include_router(settings.router)
    with TestClient(app) as client:
        yield client, store, effects, published


@pytest.mark.parametrize("error,result,status", [
    (ValueError("active or unknown"), None, 409),
    (IntegrityError("private", {}, Exception("private")), None, 409),
    (None, {"error": "DB unavailable"}, 503),
])
def test_combined_delete_keeps_profile_and_caches_on_purge_refusal(monkeypatch, delete_env, error, result, status):
    client, store, effects, published = delete_env
    monkeypatch.setattr(settings, "purge_bot_data", Mock(side_effect=error, return_value=result))
    response = client.delete("/v1/bots/source/profile?purge=true")
    assert response.status_code == status
    store.delete.assert_not_called()
    for effect in effects:
        effect.assert_not_called()
    published.assert_not_awaited()
    if status == 409:
        assert "Nothing was deleted" in response.json()["detail"]


@pytest.mark.parametrize("purge", [False, True])
def test_combined_delete_orders_successful_purge_before_profile_delete(monkeypatch, delete_env, purge):
    client, store, effects, published = delete_env
    order = []
    result = {"deleted_rows": {"sessions": 3}}
    purge_call = Mock(side_effect=lambda *args: (order.append("purge"), result)[1])
    monkeypatch.setattr(settings, "purge_bot_data", purge_call)
    store.delete.side_effect = lambda slug: (order.append("delete"), True)[1]
    response = client.delete(f"/v1/bots/source/profile?purge={str(purge).lower()}")
    assert response.status_code == 200
    assert order == (["purge", "delete"] if purge else ["delete"])
    expected = {"success": True, "slug": "source"}
    if purge:
        expected["purge"] = result
    assert response.json() == expected
    for effect in effects:
        effect.assert_called_once()
    published.assert_awaited_once()


def test_combined_delete_missing_profile_does_not_start_purge(monkeypatch, delete_env):
    client, store, _, _ = delete_env
    store.get.return_value = None
    purge = Mock()
    monkeypatch.setattr(settings, "purge_bot_data", purge)
    assert client.delete("/v1/bots/source/profile?purge=true").status_code == 404
    purge.assert_not_called()
    store.delete.assert_not_called()
