"""TASK-2279: history/memory mutations need the service bearer; user_id narrows them."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.bawthub_auth import TokenCheck
from llm_bawt.service import dependencies
from llm_bawt.service.routes import history, history_pages, memory, service_auth

MUTATIONS = [
    ("DELETE", "/v1/history?bot_id=b"),
    ("DELETE", "/v1/history/abcdef12?bot_id=b"),
    ("DELETE", "/v1/memory/abcdef12?bot_id=b"),
    ("POST", "/v1/memory/forget?bot_id=b"),
    ("POST", "/v1/memory/restore?bot_id=b"),
]


class _Source:
    def check(self, authorization):
        return TokenCheck.OK if authorization == "Bearer good" else TokenCheck.INVALID


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(service_auth, "service_token_source", lambda: _Source())
    enforced = {"on": True}
    monkeypatch.setattr(service_auth, "mutation_auth_required", lambda: enforced["on"])
    client = Mock()
    client.ignore_message_by_id.return_value = True
    client.delete_memory.return_value = True
    client.restore_ignored_messages.return_value = 0
    client.clear_messages.return_value = 0
    client.clear_memories.return_value = True
    svc = SimpleNamespace(_default_bot="b", _llm_bawt_cache={},
                          get_memory_client=Mock(return_value=client),
                          invalidate_bot_instances=Mock(return_value=0))
    dependencies.set_service(svc)
    monkeypatch.setattr(memory, "get_service", lambda: svc)
    api = FastAPI()
    api.include_router(history.router)
    api.include_router(memory.router)
    http = TestClient(api)
    http.enforced = enforced
    yield http, svc, client
    dependencies.set_service(None)


@pytest.mark.parametrize(("method", "path"), MUTATIONS)
def test_mutation_without_bearer_is_refused(app, method, path):
    http, svc, _ = app
    body = {"message_id": "abcdef12"} if path.startswith("/v1/memory/forget") else None
    assert http.request(method, path, json=body).status_code == 401
    assert http.request(method, path, json=body, headers={"Authorization": "Bearer bad"}).status_code == 401
    svc.get_memory_client.assert_not_called()


def test_pinned_user_scopes_every_by_id_mutation(app):
    http, svc, client = app
    ok = {"Authorization": "Bearer good"}
    assert http.delete("/v1/history/abcdef12?bot_id=b&user_id=alice", headers=ok).status_code == 200
    client.ignore_message_by_id.assert_called_with("abcdef12", user_id="alice")
    assert http.delete("/v1/memory/abcdef12?bot_id=b&user_id=alice", headers=ok).status_code == 200
    client.delete_memory.assert_called_with("abcdef12", user_id="alice")
    assert http.post("/v1/memory/forget?bot_id=b", json={"message_id": "abcdef12", "user_id": "alice"},
                     headers=ok).status_code == 200
    assert http.post("/v1/memory/restore?bot_id=b&user_id=alice", headers=ok).status_code == 200
    for call in svc.get_memory_client.call_args_list:
        assert call.args == ("b", "alice")


def test_user_clear_never_clears_all(app):
    http, svc, client = app
    resp = http.delete("/v1/history?bot_id=b&user_id=alice", headers={"Authorization": "Bearer good"})
    assert resp.status_code == 200
    svc.get_memory_client.assert_called_once_with("b", "alice")
    client.clear_messages.assert_called_once_with(user_id="alice")
    client.clear_memories.assert_not_called()


def test_blank_user_is_refused(app):
    http, _, client = app
    resp = http.delete("/v1/history?bot_id=b&user_id=%20", headers={"Authorization": "Bearer good"})
    assert resp.status_code == 400
    client.clear_messages.assert_not_called()


def test_history_pages_mutations_share_the_guard():
    deps = [d.dependency for d in history_pages.mutation_router.dependencies]
    assert service_auth.require_mutation_service_token in deps


@pytest.mark.parametrize(("method", "path"), MUTATIONS)
def test_setting_off_allows_mutation_without_bearer(app, method, path):
    """Setting off (default): prod's identity-less path keeps working."""
    http, svc, _ = app
    http.enforced["on"] = False
    body = {"message_id": "abcdef12"} if path.startswith("/v1/memory/forget") else None
    assert http.request(method, path, json=body).status_code == 200
    svc.get_memory_client.assert_called()


def test_setting_off_still_scopes_a_pinned_user(app):
    http, svc, client = app
    http.enforced["on"] = False
    assert http.delete("/v1/history/abcdef12?bot_id=b&user_id=alice").status_code == 200
    client.ignore_message_by_id.assert_called_with("abcdef12", user_id="alice")
    assert http.delete("/v1/history?bot_id=b&user_id=alice").status_code == 200
    client.clear_memories.assert_not_called()
    assert all(call.args == ("b", "alice") for call in svc.get_memory_client.call_args_list)


@pytest.mark.parametrize("body", [{"count": 3}, {"minutes": 5}])
def test_setting_off_still_requires_user_for_count_or_minutes(app, body):
    http, _, client = app
    http.enforced["on"] = False
    assert http.post("/v1/memory/forget?bot_id=b", json=body).status_code == 400
    client.forget_recent_messages.assert_not_called()
    client.forget_messages_since_minutes.assert_not_called()


@pytest.mark.parametrize(("value", "expected"), [
    (None, False), (False, False), (True, True), ("true", True), ("off", False),
])
def test_mutation_auth_setting_reads_global_value(monkeypatch, value, expected):
    import llm_bawt.runtime_setting_resolution as resolution

    monkeypatch.setattr(resolution, "resolve_global_runtime_setting",
                        lambda config, key, fallback=None: fallback if value is None else value)
    dependencies.set_service(SimpleNamespace(config=object()))
    try:
        assert service_auth.mutation_auth_required() is expected
    finally:
        dependencies.set_service(None)


def test_mutation_auth_setting_is_global_only_and_default_off():
    from llm_bawt.setting_definitions import SETTING_DEFINITIONS, is_global_only_runtime_setting

    assert SETTING_DEFINITIONS[service_auth.MUTATION_AUTH_SETTING].default is False
    assert is_global_only_runtime_setting(service_auth.MUTATION_AUTH_SETTING)
