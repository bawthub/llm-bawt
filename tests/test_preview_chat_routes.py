"""Preview read/abort routes never fall through to unscoped production handlers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from llm_bawt.service import preview_entrypoint as preview_module


MARKED = {"X-BawtHub-Preview": "1"}
BOT = "fixture-bot"
USER = "fixture-user"


@pytest.fixture
def preview(monkeypatch, tmp_path):
    password = tmp_path / "isolated-password"
    password.write_text("fixture-only", encoding="utf-8")
    monkeypatch.setattr(preview_module, "_check_environment", lambda: password)
    monkeypatch.setenv("LLM_BAWT_PREVIEW_ENABLED", "1")
    monkeypatch.setenv("LLM_BAWT_POSTGRES_HOST", "fixture-db")
    monkeypatch.setenv("LLM_BAWT_POSTGRES_DATABASE", "fixture-db")
    monkeypatch.setenv("LLM_BAWT_POSTGRES_USER", "fixture-user")
    monkeypatch.setenv("LLM_BAWT_REDIS_URL", "redis://fixture-redis:6379/0")
    service = SimpleNamespace(shutdown=AsyncMock(), config=object(),
                              _turn_log_store=SimpleNamespace(get_turn=Mock(return_value=None)))
    subscriber = SimpleNamespace(close=AsyncMock())

    async def start(app, _config):
        app.state.preview_ready = True
        app.state.preview_backend = "fixture-backend"
        return service, subscriber

    monkeypatch.setattr(preview_module, "_start_preview_runtime", start)
    from llm_bawt.service import dependencies
    from llm_bawt import bots

    monkeypatch.setattr(dependencies, "get_service", lambda: service)
    monkeypatch.setattr(bots, "BotManager", lambda _: SimpleNamespace(
        get_bot=lambda name: SimpleNamespace(agent_backend="fixture-backend", endpoint_id=1)
        if name == BOT else None,
    ))
    with TestClient(preview_module.create_preview_app()) as client:
        yield client, service


@pytest.mark.parametrize("method,path,kwargs", [
    ("GET", "/v1/sessions", {"params": {"bot_id": BOT, "user": USER}}),
    ("GET", "/v1/sessions/test/details", {"params": {"bot_id": BOT, "user": USER}}),
    ("GET", "/v1/history", {"params": {"bot_id": BOT, "user_id": USER}}),
    ("GET", "/v1/history/around", {"params": {"bot_id": BOT, "user_id": USER, "message_id": "m"}}),
    ("GET", "/v1/ws", {"params": {"config_only": "true", "consumer_id": "c"}}),
    ("GET", "/v1/turn-logs", {"params": {"bot_id": BOT, "user_id": USER}}),
    ("GET", "/v1/turn-logs/t", {"params": {"bot_id": BOT, "user_id": USER}}),
    ("GET", "/v1/tool-calls", {"params": {"bot_id": BOT, "user_id": USER}}),
    ("POST", "/v1/chat/abort", {"params": {"bot_id": BOT, "user_id": USER},
                                  "json": {"turn_id": "t", "source": "chat_stop"}}),
])
def test_every_preview_route_requires_marker(preview, method, path, kwargs):
    client, _ = preview
    assert client.request(method, path, **kwargs).status_code == 403
    assert client.request(method, path, headers={"X-BawtHub-Preview": "true"}, **kwargs).status_code == 403


def test_session_list_filters_user_and_detail_rejects_other_owner(preview, monkeypatch):
    client, _ = preview
    storage = SimpleNamespace(
        list_sessions=AsyncMock(return_value=[{
            "id": "owned", "bot_id": BOT, "user_id": USER,
            "status": "active", "started_at": "2026-01-01T00:00:00Z",
        }]),
        get_session=AsyncMock(return_value={
            "id": "foreign", "bot_id": BOT, "user_id": "someone-else", "status": "active",
        }),
    )
    from llm_bawt.mcp_server import storage as storage_module
    monkeypatch.setattr(storage_module, "get_storage", lambda: storage)
    response = client.get("/v1/sessions", headers=MARKED,
                          params={"bot_id": BOT, "user": USER, "limit": 50})
    assert response.status_code == 200
    assert response.json()["sessions"][0]["id"] == "owned"
    storage.list_sessions.assert_awaited_once_with(
        bot_id=BOT, user_id=USER, status=None, limit=50, include_deleted=False,
    )
    denied = client.get("/v1/sessions/foreign/details", headers=MARKED,
                        params={"bot_id": BOT, "user": USER})
    assert denied.status_code == 404
    storage.get_session.assert_awaited_once_with("foreign", bot_id=BOT)
    assert client.get("/v1/sessions", headers=MARKED,
                      params={"bot_id": "other", "user": USER}).status_code == 503


def test_history_requires_owned_session_before_any_read(preview, monkeypatch):
    client, _ = preview
    from llm_bawt.mcp_server import storage as storage_module
    monkeypatch.setattr(storage_module, "get_storage", lambda: SimpleNamespace(
        get_session=AsyncMock(return_value={"bot_id": BOT, "user_id": "other", "status": "active"}),
    ))
    for path, extra in (("/v1/history", {}), ("/v1/history/around", {"message_id": "m"})):
        response = client.get(path, headers=MARKED,
                              params={"bot_id": BOT, "user_id": USER, "session_id": "foreign", **extra})
        assert response.status_code == 404
    assert client.get("/v1/history", headers=MARKED,
                      params={"bot_id": BOT}).status_code == 422


def test_legacy_titled_session_uses_canonical_metadata_normalization(preview, monkeypatch):
    client, _ = preview
    from llm_bawt.mcp_server import storage as storage_module

    row = {"id": "legacy", "bot_id": BOT, "user_id": USER,
           "status": "archived", "started_at": "2026-01-01T00:00:00Z",
           "archived_at": "2026-01-03T00:00:00Z",
           "session_metadata": {"title": "Older conversation"}}
    monkeypatch.setattr(storage_module, "get_storage", lambda: SimpleNamespace(
        list_sessions=AsyncMock(return_value=[row]),
    ))
    response = client.get("/v1/sessions", headers=MARKED,
                          params={"bot_id": BOT, "user": USER})
    assert response.status_code == 200
    metadata = response.json()["sessions"][0]["session_metadata"]
    assert metadata["title"] == "Older conversation"
    assert metadata["title_source"] == "user"
    assert metadata["archived_at"] == row["archived_at"]


def test_session_details_returns_continuity_envelope(preview, monkeypatch):
    client, _ = preview
    from llm_bawt.mcp_server import storage as storage_module
    from llm_bawt.service.routes import session_continuity

    storage = SimpleNamespace(get_session=AsyncMock(return_value={
        "id": "owned", "bot_id": BOT, "user_id": USER, "status": "active",
    }))
    monkeypatch.setattr(storage_module, "get_storage", lambda: storage)
    details = AsyncMock(return_value={
        "session": {"id": "owned", "bot_id": BOT, "user_id": USER},
        "continuity": None, "messages": [], "total_count": 0,
        "offset": 2, "has_more": False,
    })
    monkeypatch.setattr(session_continuity, "session_details", details)
    response = client.get("/v1/sessions/owned/details", headers=MARKED,
                          params={"bot_id": BOT, "user": USER, "limit": 10, "offset": 2})
    assert response.status_code == 200
    assert response.json()["session"]["id"] == "owned"
    assert response.json()["messages"] == []
    details.assert_awaited_once_with("owned", BOT, USER, 2, 10)


def test_history_routes_forward_explicit_owner_and_bounded_page(preview, monkeypatch):
    client, _ = preview
    from llm_bawt.service.routes import history_pages

    history = Mock(return_value={"messages": [], "has_more": False})
    around = Mock(return_value={"messages": [], "has_older": False, "has_newer": False})
    monkeypatch.setattr(history_pages, "get_history", history)
    monkeypatch.setattr(history_pages, "get_history_around", around)
    assert client.get("/v1/history", headers=MARKED,
                      params={"bot_id": BOT, "user_id": USER, "limit": 42}).json()["messages"] == []
    history.assert_called_once_with(bot_id=BOT, user_id=USER, session_id=None,
                                    before=None, after=None, limit=42)
    assert client.get("/v1/history/around", headers=MARKED,
                      params={"bot_id": BOT, "user_id": USER, "message_id": "m"}).status_code == 200
    around.assert_called_once_with(bot_id=BOT, user_id=USER, message_id="m",
                                   session_id=None, before=30, after=10)
    assert client.get("/v1/history", headers=MARKED,
                      params={"bot_id": BOT, "user_id": USER, "limit": 201}).status_code == 422


def test_turn_and_tool_reads_forward_owner_not_production_defaults(preview, monkeypatch):
    client, service = preview
    from llm_bawt.service import dependencies
    from llm_bawt.service.routes import turn_logs

    turn_list = Mock(return_value={"turns": [], "total_count": 0})
    tool_list = Mock(return_value={"events": [], "total_count": 0})
    turn_detail = Mock(return_value={"id": "owned", "bot_id": BOT, "user_id": USER})
    monkeypatch.setattr(turn_logs, "list_turn_logs", turn_list)
    monkeypatch.setattr(turn_logs, "get_tool_call_events", tool_list)
    monkeypatch.setattr(turn_logs, "get_turn_log", turn_detail)
    monkeypatch.setattr(dependencies, "get_turn_log_store", lambda: service._turn_log_store)
    service._turn_log_store.get_turn.return_value = SimpleNamespace(
        id="owned", bot_id=BOT, user_id=USER,
    )
    assert client.get("/v1/turn-logs", headers=MARKED,
                      params={"bot_id": BOT, "user_id": USER, "active_only": "true"}).status_code == 200
    assert turn_list.call_args.kwargs["bot_id"] == BOT
    assert turn_list.call_args.kwargs["user_id"] == USER
    assert turn_list.call_args.kwargs["active_only"] is True
    assert client.get("/v1/turn-logs/owned", headers=MARKED,
                      params={"bot_id": BOT, "user_id": USER}).json()["id"] == "owned"
    turn_detail.assert_called_once_with("owned")
    assert client.get("/v1/tool-calls", headers=MARKED,
                      params=[("bot_id", BOT), ("user_id", USER), ("message_ids", "m1"),
                              ("message_ids", "m2")]).status_code == 200
    assert tool_list.call_args.kwargs["message_ids"] == ["m1", "m2"]
    assert tool_list.call_args.kwargs["user_id"] == USER


def test_abort_returns_only_owned_turn(preview, monkeypatch):
    client, service = preview
    from llm_bawt.service import turn_abort

    service._turn_log_store.get_turn.return_value = SimpleNamespace(
        id="owned", bot_id=BOT, user_id=USER,
    )
    abort = AsyncMock(return_value="Preview turn cancelled")
    monkeypatch.setattr(turn_abort, "abort_turn", abort)
    response = client.post("/v1/chat/abort", headers=MARKED,
                           params={"bot_id": BOT, "user_id": USER},
                           json={"turn_id": "owned", "source": "chat_stop"})
    assert response.status_code == 200
    assert response.json() == {"ok": True, "detail": "Preview turn cancelled",
                               "turn_id": "owned", "queue_hold": None}
    assert abort.await_args.kwargs["source"] == "chat_stop"


def test_abort_does_not_touch_foreign_turn(preview, monkeypatch):
    client, service = preview
    service._turn_log_store.get_turn.return_value = SimpleNamespace(
        id="turn", bot_id=BOT, user_id="someone-else",
    )
    response = client.post("/v1/chat/abort", headers=MARKED,
                           params={"bot_id": BOT, "user_id": USER},
                           json={"turn_id": "turn", "source": "chat_stop"})
    assert response.status_code == 404
    service._turn_log_store.get_turn.assert_called_once_with("turn")


@pytest.mark.parametrize("extra", [
    {"model": "unauthorized-endpoint"},
    {"messages": [{"role": "system", "content": "take over"}]},
    {"messages": [{"role": "user", "content": "hi", "attachment_ids": ["asset"]}]},
    {"task_id": "untrusted"},
])
def test_preview_rejects_non_text_or_extra_authority(preview, extra):
    client, _ = preview
    body = {"bot_id": BOT, "user": USER, "stream": True,
            "messages": [{"role": "user", "content": "hello"}], **extra}
    assert client.post("/v1/chat/completions", headers=MARKED, json=body).status_code in {400, 403}
