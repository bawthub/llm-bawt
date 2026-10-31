"""Standalone purge reports transactional refusal, not partial/unknown success."""
from types import SimpleNamespace
from unittest.mock import Mock

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
