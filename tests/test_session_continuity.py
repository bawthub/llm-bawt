"""TASK-1117: reference-only capture, ownership, paging and bridge receipts."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from agent_bridge.continuity import continuity_digest
from llm_bawt.memory.session_continuity import SessionContinuity
from llm_bawt.models.message import Message
from llm_bawt.utils.history import HistoryManager
from llm_bawt.service.routes import session_continuity as routes
from llm_bawt.service.routes import sessions


def payload():
    return [
        {"source_id": "summary", "role": "summary", "content": "summary text", "timestamp": 1.0},
        {"source_id": "message", "role": "user", "content": "shortened text", "timestamp": 2.0, "shortened": True},
    ]


def test_manifest_contains_ordered_references_not_bodies():
    data = SessionContinuity.manifest(payload(), settings={"history_scope": "inline+summaries"}, model="test-model")
    assert data["source_message_ids"] == ["summary", "message"]
    assert data["shortened_indices"] == [1]
    assert data["status"] == "prepared"
    assert data["summary_count"] == data["message_count"] == 1
    assert "summary text" not in json.dumps(data)
    assert "shortened text" not in json.dumps(data)
    assert SessionContinuity.manifest([], settings={}, model=None)["status"] == "empty"


def test_digest_survives_transport_but_changes_with_payload():
    original = payload()
    assert continuity_digest(original) == continuity_digest(json.loads(json.dumps(original)))
    for field, replacement in [("source_id", "different"), ("content", "changed"), ("role", "assistant")]:
        changed = payload()
        changed[0][field] = replacement
        assert continuity_digest(original) != continuity_digest(changed)
    assert continuity_digest(original) != continuity_digest(list(reversed(original)))


def test_summary_compaction_preserves_source_identity():
    manager = object.__new__(HistoryManager)
    manager.messages = [Message("summary", "Earlier discussion", timestamp=1.0, db_id="original-id")]
    manager.config = SimpleNamespace(SYSTEM_MESSAGE="System")
    manager._setting = lambda key, default: default
    summaries = manager.build_context_payload(delivery="seed").summary_messages
    assert summaries[0].db_id == "original-id"
    assert summaries[0].content != "Earlier discussion"


@pytest.fixture
def database():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE sessions (id TEXT PRIMARY KEY, bot_id TEXT, user_id TEXT, status TEXT, session_metadata TEXT)"))
        conn.execute(text("CREATE TABLE messages (id TEXT, bot_id TEXT, session_id TEXT, role TEXT, content TEXT, timestamp REAL)"))
        conn.execute(text("INSERT INTO sessions VALUES (:id,:bot,:user,:status,'{}')"), [
            {"id": "source", "bot": "test-bot", "user": "owner", "status": "archived"},
            {"id": "dest", "bot": "test-bot", "user": "owner", "status": "active"},
            {"id": "foreign", "bot": "test-bot", "user": "someone-else", "status": "archived"},
            {"id": "other-bot", "bot": "other-bot", "user": "owner", "status": "active"},
            {"id": "deleted", "bot": "test-bot", "user": "owner", "status": "deleted"},
        ])
        conn.execute(text("INSERT INTO messages VALUES (:id,:bot,:session,:role,:content,1)"), [
            {"id": "summary", "bot": "test-bot", "session": "source", "role": "summary", "content": "original summary"},
            {"id": "message", "bot": "test-bot", "session": "source", "role": "user", "content": "CURRENT source body"},
            {"id": "private", "bot": "test-bot", "session": "foreign", "role": "user", "content": "private body"},
            {"id": "other", "bot": "other-bot", "session": "other-bot", "role": "user", "content": "other bot body"},
            {"id": "gone", "bot": "test-bot", "session": "deleted", "role": "user", "content": "deleted body"},
            {"id": "receipt", "bot": "test-bot", "session": "dest", "role": "assistant", "content": "Session reset."},
        ])
    yield engine
    engine.dispose()


def test_resolve_order_pagination_and_ownership(database):
    manifest = {"source_message_ids": ["message", "summary", "private", "other", "gone", "missing", None], "shortened_indices": [0]}
    repo = SessionContinuity(database)
    first = repo.resolve(manifest, bot_id="test-bot", user_id="owner", offset=0, limit=2)
    assert [m["id"] for m in first["messages"]] == ["message", "summary"]
    assert first["messages"][0]["content"] == "CURRENT source body"
    assert first["messages"][0]["shortened"]
    assert first["total_count"] == 7 and first["has_more"]
    remainder = repo.resolve(manifest, bot_id="test-bot", user_id="owner", offset=2, limit=10)
    assert all(not m["available"] and "content" not in m for m in remainder["messages"])
    assert not remainder["has_more"]
    assert repo.resolve({}, bot_id="test-bot", user_id="owner", offset=0, limit=50)["messages"] == []


@pytest.fixture
def client(database, monkeypatch):
    async def get_session(session_id, **_):
        with database.connect() as conn:
            row = conn.execute(text("SELECT * FROM sessions WHERE id=:id"), {"id": session_id}).mappings().first()
        return {**row, "started_at": "2026-10-01 01:00:00", "session_metadata": json.loads(row["session_metadata"])} if row else None
    storage = SimpleNamespace(engine=database, get_session=get_session)
    monkeypatch.setattr(routes, "get_storage", lambda: storage)
    monkeypatch.setattr(sessions, "get_storage", lambda: storage)
    monkeypatch.setattr(routes, "get_effective_bot_id", lambda value: value)
    monkeypatch.setattr(routes, "_resolve_user", lambda value: value or "owner")
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        yield client


def test_details_legacy_by_message_and_scoping(client):
    result = client.get("/v1/sessions/dest/details?bot_id=test-bot&user=owner")
    assert result.status_code == 200
    assert result.json()["continuity"] is None
    assert result.json()["session"]["started_at"] == "2026-10-01T01:00:00+00:00"
    assert result.json()["messages"] == []
    result = client.get("/v1/sessions/by-message/receipt/details?bot_id=test-bot&user=owner")
    assert result.json()["session"]["id"] == "dest"
    for path in ["dest", "by-message/receipt"]:
        assert client.get(f"/v1/sessions/{path}/details?bot_id=test-bot&user=someone-else").status_code == 404
        assert client.get(f"/v1/sessions/{path}/details?bot_id=other-bot&user=owner").status_code == 404
    assert client.get("/v1/sessions/deleted/details?bot_id=test-bot&user=owner").status_code == 410
    assert client.get("/v1/sessions/dest/details?bot_id=test-bot&limit=101").status_code == 422


def test_bridge_receipt_follows_successful_write_and_failure():
    from claude_code_bridge.session_ops import ClaudeSessionMixin
    bridge = object.__new__(ClaudeSessionMixin)
    bridge._write_seed_transcript = Mock()
    bridge._set_thread_session = AsyncMock()
    bridge._record_continuity_receipt = AsyncMock()
    result = asyncio.run(bridge._seed_new_session("test-bot", "model", payload(), "dest"))
    assert result["seeded"]
    bridge._record_continuity_receipt.assert_awaited_once_with("dest", "test-bot", payload(), restored=True)
    bridge._record_continuity_receipt.reset_mock()
    bridge._write_seed_transcript.side_effect = OSError("disk full")
    result = asyncio.run(bridge._seed_new_session("test-bot", "model", payload(), "dest"))
    assert not result["seeded"]
    bridge._record_continuity_receipt.assert_awaited_once_with("dest", "test-bot", payload(), restored=False)


def test_no_capture_on_warm_resume(monkeypatch):
    from llm_bawt.service import session_continuity
    storage = Mock()
    monkeypatch.setattr(session_continuity, "get_storage", storage)
    session_continuity.record_session_continuity(None, {"thread_session_id": "dest"}, None, bot_id="test-bot", user_id="owner")
    storage.assert_not_called()


def test_capture_uses_destination_owner_and_actual_payload(monkeypatch):
    from llm_bawt.service import session_continuity
    record = Mock(return_value=True)
    monkeypatch.setattr(session_continuity, "get_storage", lambda: SimpleNamespace(engine="private-engine"))
    monkeypatch.setattr(SessionContinuity, "record", record)
    llm = SimpleNamespace(resolved_model_alias="test-model", config_resolver=SimpleNamespace(
        resolve_config_setting=lambda key: SimpleNamespace(value="inline" if key == "history_scope" else 42),
    ))
    messages = payload()
    session_continuity.record_session_continuity(llm, {"thread_session_id": "new-destination"}, messages,
                                               bot_id="test-bot", user_id="actual-owner")
    destination, bot, owner, manifest = record.call_args.args
    assert (destination, bot, owner) == ("new-destination", "test-bot", "actual-owner")
    assert manifest["source_message_ids"] == ["summary", "message"]
    assert manifest["settings"]["history_scope"] == "inline"
    assert manifest["model"] == "test-model"
    assert messages == payload()


def test_clean_new_session_records_an_explicit_empty_selection():
    from llm_bawt.service.routes.history_seed import maybe_build_session_seed
    from claude_code_bridge.session_ops import ClaudeSessionMixin
    llm = SimpleNamespace(bot=SimpleNamespace(agent_backend="claude-code"), config_resolver=SimpleNamespace(
        resolve_config_setting=lambda key: SimpleNamespace(value="none"),
    ))
    assert maybe_build_session_seed(llm, "test-bot", "model", "/new", None,
                                   {"thread_session_id": "dest"}) == []
    assert maybe_build_session_seed(llm, "test-bot", "model", "hello", None,
                                   {"thread_session_id": "dest", "thread_resume_id": "sdk"}) is None
    assert ClaudeSessionMixin._format_seed_ack({"clean_start": True}) == "Session reset. Ready for a new conversation."
