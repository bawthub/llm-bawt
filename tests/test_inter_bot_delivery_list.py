"""Compact, filtered delivery listing (TASK-1239). Hermetic: no PostgreSQL."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.inter_bot_delivery_query import (
    IN_FLIGHT_STATUSES,
    DeliveryListFilter,
    parse_statuses,
)
from llm_bawt.inter_bot_delivery_record import DeliveryRecord
from llm_bawt.mcp_server import server  # noqa: F401  # registers tools; import first
from llm_bawt.mcp_server import inter_bot_tools  # noqa: E402
from llm_bawt.service.routes import inter_bot_deliveries as routes

NOW = datetime(2026, 10, 22, 22, 0, tzinfo=timezone.utc)


def _record(**overrides) -> DeliveryRecord:
    row = {
        "id": "dlv-1", "sender_bot_id": "snark", "target_bot_id": "loopy",
        "message": "hello " * 100, "user_message_id": "u1", "turn_id": "t1",
        "status": "QUEUED", "attempt_count": 1, "max_attempts": 5,
        "idempotency_key": "k1", "task_id": "TASK-1239", "message_kind": "READY",
        "metadata_json": {"new_session_receipt": {"seeded": "x" * 4000}},
        "created_at": NOW, "updated_at": NOW, "available_at": NOW,
        "last_error": "boom " * 200, "session_policy": "continue",
        "author_entity_type": "bot", "author_entity_id": "snark",
    }
    row.update(overrides)
    return DeliveryRecord.from_mapping(row)


# --- projection ---------------------------------------------------------------

def test_compact_row_drops_metadata_and_bulky_fields():
    full = _record().to_api()
    compact = _record().to_compact()

    assert "metadata" in full
    for dropped in ("metadata", "author", "reset_status", "old_session_id",
                    "new_session_id", "session_policy", "available_at", "user_message_id"):
        assert dropped not in compact
    assert compact["delivery_id"] == "dlv-1"
    assert compact["status"] == "QUEUED"
    assert compact["sender_bot_id"] == "snark" and compact["target_bot_id"] == "loopy"
    assert compact["task_id"] == "TASK-1239"
    assert compact["attempts"] == "1/5"
    assert len(compact["last_error"]) <= DeliveryRecord.COMPACT_ERROR_CHARS
    assert len(compact["preview"]) <= DeliveryRecord.COMPACT_PREVIEW_CHARS
    assert compact["preview"].endswith("…")
    # Nulls are omitted rather than serialized.
    assert "delivered_at" not in compact and "target_turn_id" not in compact
    assert len(json.dumps(compact)) < len(json.dumps(full)) / 5


def test_compact_row_names_a_user_author_only():
    assert "author" not in _record().to_compact()
    user = _record(author_entity_type="user", author_entity_id="nick").to_compact()
    assert user["author"] == "nick"


# --- filter -------------------------------------------------------------------

def test_parse_statuses_accepts_single_comma_and_iterable():
    assert parse_statuses(None) == ()
    assert parse_statuses("queued") == ("QUEUED",)
    assert parse_statuses("QUEUED, steering,QUEUED") == ("QUEUED", "STEERING")
    assert parse_statuses(["failed", "cancelled"]) == ("FAILED", "CANCELLED")
    with pytest.raises(ValueError, match="unknown delivery status"):
        parse_statuses("RUNNING")


def test_filter_builds_either_direction_multi_status_and_chat_exclusion():
    since = NOW - timedelta(minutes=30)
    where, params = DeliveryListFilter(
        bot_id=" Loopy ", statuses=IN_FLIGHT_STATUSES, task_id="TASK-1239",
        since=since, exclude_prompt_origin=True, exclude_user_chat=True,
    ).where("postgresql")

    assert "(sender_bot_id=:either OR target_bot_id=:either)" in where
    assert params["either"] == "loopy"
    assert "status IN (:status0, :status1, :status2)" in where
    assert [params[f"status{i}"] for i in range(3)] == list(IN_FLIGHT_STATUSES)
    assert "task_id=:task_id" in where and params["task_id"] == "TASK-1239"
    assert "created_at>=:since" in where and params["since"] == since
    for key in ("prompt_schedule", "ordinary_chat", "maintenance_user_queue"):
        assert f"? '{key}'" in where
    assert "sender_bot_id <> target_bot_id" in where


def test_filter_keeps_chat_rows_unless_asked():
    where, params = DeliveryListFilter().where("postgresql")
    assert where == "1=1" and params == {}
    where, _ = DeliveryListFilter(exclude_prompt_origin=True).where("postgresql")
    assert "ordinary_chat" not in where
    assert "sender_bot_id <> target_bot_id" not in where


# --- HTTP route ---------------------------------------------------------------

class _Store:
    engine = object()

    def __init__(self):
        self.calls = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        return [_record()]


@pytest.fixture
def route_client(monkeypatch):
    store = _Store()
    monkeypatch.setattr(routes, "_dispatcher", lambda: SimpleNamespace(store=store))
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app), store


def test_route_default_shape_is_full_and_includes_chat(route_client):
    client, store = route_client
    body = client.get("/v1/inter-bot-deliveries", params={"target_bot_id": "loopy", "status": "QUEUED"}).json()

    assert "metadata" in body["deliveries"][0]
    call = store.calls[-1]
    assert call["exclude_user_chat"] is False
    assert call["exclude_prompt_origin"] is True
    assert call["status"] == ("QUEUED",)


def test_route_compact_view_and_filters(route_client):
    client, store = route_client
    body = client.get("/v1/inter-bot-deliveries", params={
        "bot_id": "loopy", "active_only": "true", "include_chat": "false",
        "task_id": "TASK-1239", "since": NOW.isoformat(), "view": "compact", "limit": 7,
    }).json()

    assert "metadata" not in body["deliveries"][0]
    call = store.calls[-1]
    assert call["bot_id"] == "loopy"
    assert call["status"] == IN_FLIGHT_STATUSES
    assert call["exclude_user_chat"] is True
    assert call["task_id"] == "TASK-1239"
    assert call["since"] == NOW
    assert call["limit"] == 7


def test_route_rejects_unknown_or_conflicting_status(route_client):
    client, _ = route_client
    assert client.get("/v1/inter-bot-deliveries", params={"status": "RUNNING"}).status_code == 422
    conflict = client.get("/v1/inter-bot-deliveries", params={"status": "FAILED", "active_only": "true"})
    assert conflict.status_code == 422


# --- MCP tool -----------------------------------------------------------------

def _capture_client(monkeypatch, *, status_code=200, body=None):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status_code, json=body if body is not None else {"deliveries": [], "total": 0})

    real = httpx.AsyncClient
    monkeypatch.setattr(inter_bot_tools.httpx, "AsyncClient",
                        lambda *a, **k: real(transport=httpx.MockTransport(handler)))
    return seen


def test_mcp_list_defaults_to_compact_bot_traffic_small_page(monkeypatch):
    seen = _capture_client(monkeypatch)
    asyncio.run(inter_bot_tools.list_deliveries())

    params = dict(seen[0].url.params)
    assert params == {"include_chat": "false", "view": "compact", "limit": "10"}


def test_mcp_list_passes_filters_and_caps_limit(monkeypatch):
    seen = _capture_client(monkeypatch)
    asyncio.run(inter_bot_tools.list_deliveries(
        bot_id="snark", active_only=True, task_id="TASK-1239",
        since_minutes=30, include_chat=True, limit=500,
    ))

    params = dict(seen[0].url.params)
    assert params["bot_id"] == "snark"
    assert params["active_only"] == "true"
    assert params["task_id"] == "TASK-1239"
    assert params["include_chat"] == "true"
    assert params["limit"] == "50"
    since = datetime.fromisoformat(params["since"])
    assert timedelta(minutes=29) < datetime.now(timezone.utc) - since < timedelta(minutes=31)


def test_mcp_list_returns_validation_error_instead_of_raising(monkeypatch):
    _capture_client(monkeypatch, status_code=422, body={"detail": "unknown delivery status 'RUNNING'"})
    result = asyncio.run(inter_bot_tools.list_deliveries(status="RUNNING"))
    assert result == {"error": "unknown delivery status 'RUNNING'"}
