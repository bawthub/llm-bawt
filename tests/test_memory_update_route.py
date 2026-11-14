"""HTTP and tool errors from a failed memory update (TASK-1854)."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from llm_bawt.memory.errors import MemoryUpdateConflictError, MemoryUpdateDuplicateError
from llm_bawt.service.routes.memory import update_memory
from llm_bawt.service.schemas import MemoryUpdateRequest
from llm_bawt.tools.executor import ToolExecutor
from llm_bawt.tools.parser import ToolCall


@pytest.mark.parametrize("failure, expected", [
    (MemoryUpdateDuplicateError("memory update conflicts with another duplicate"), "duplicate"),
    (MemoryUpdateConflictError("memory changed concurrently; retry the update"), "retry"),
])
def test_rest_memory_update_reports_conflict(failure, expected):
    def reject(**_kwargs):
        raise failure

    client = SimpleNamespace(update_memory=reject)
    with pytest.raises(HTTPException) as exc:
        update_memory("memory-id", MemoryUpdateRequest(content="Changed"),
                      bot_id="example", client=client)
    assert exc.value.status_code == 409
    assert expected in str(exc.value.detail)


def test_rest_memory_update_leaves_unrelated_backend_failure_as_500():
    def fail(**_kwargs):
        raise RuntimeError("database unavailable")

    with pytest.raises(HTTPException) as exc:
        update_memory("memory-id", MemoryUpdateRequest(content="Changed"),
                      bot_id="example", client=SimpleNamespace(update_memory=fail))
    assert exc.value.status_code == 500


@pytest.mark.parametrize("failure, expected", [
    (MemoryUpdateDuplicateError("memory update conflicts with another duplicate"), "duplicate"),
    (MemoryUpdateConflictError("memory changed concurrently; retry the update"), "retry"),
])
def test_tool_update_shows_specific_conflict(failure, expected):
    def reject(**_kwargs):
        raise failure

    executor = ToolExecutor.__new__(ToolExecutor)
    executor.memory_client = SimpleNamespace(update_memory=reject)
    result = executor._memory_update(ToolCall(name="memory", arguments={
        "memory_id": "memory-id", "content": "Changed",
    }, raw_text=""))
    assert 'status="error"' in result
    assert expected in result


@pytest.fixture
def http_clients(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from unittest.mock import Mock
    from llm_bawt.service.routes import memory as routes

    clients = {owner: Mock() for owner in (None, "owner-a", "owner-b")}
    for owner, client in clients.items():
        client.update_memory.return_value = SimpleNamespace(
            id="owned-id", content="own fact", importance=0.5, tags=[])
        client.regenerate_embeddings.return_value = {"updated": 1, "failed": 0}
        client.consolidate_memories.return_value = {"dry_run": True, "clusters_merged": 0}
    service = Mock()
    service.get_memory_client.side_effect = lambda _bot, user_id=None: clients[user_id]
    monkeypatch.setattr(routes, "get_service", lambda: service)
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.require_mutation_service_token] = lambda: None
    with TestClient(app) as http:
        yield http, app, service, clients


@pytest.mark.parametrize("method,path,body", [
    ("PATCH", "/v1/memory/owned-id", {"importance": 0.6}),
    ("POST", "/v1/memory/regenerate-embeddings", None),
    ("POST", "/v1/memory/consolidate", {"dry_run": True}),
])
@pytest.mark.parametrize("owner", ["owner-a", "owner-b", None])
def test_http_mutations_select_personal_or_explicit_operator_client(http_clients, method, path, body, owner):
    http, _app, service, clients = http_clients
    params = {"bot_id": "fixture"}
    if owner is not None:
        params["user_id"] = " " + owner + " "
    response = http.request(method, path, params=params, json=body)
    assert response.status_code == 200, response.text
    expected = ("fixture", owner) if owner else ("fixture",)
    service.get_memory_client.assert_called_once_with(*expected)
    for other, client in clients.items():
        if other != owner:
            assert client.mock_calls == []


@pytest.mark.parametrize("method,path,body", [
    ("PATCH", "/v1/memory/owned-id", {"importance": 0.6}),
    ("POST", "/v1/memory/regenerate-embeddings", None),
    ("POST", "/v1/memory/consolidate", {"dry_run": True}),
])
def test_http_mutations_refuse_blank_owner_before_storage(http_clients, method, path, body):
    http, _app, service, _clients = http_clients
    response = http.request(method, path, params={"bot_id": "fixture", "user_id": " "}, json=body)
    assert response.status_code == 400
    service.get_memory_client.assert_not_called()


@pytest.mark.parametrize("method,path,body", [
    ("PATCH", "/v1/memory/owned-id", {"importance": 0.6}),
    ("POST", "/v1/memory/regenerate-embeddings", None),
    ("POST", "/v1/memory/consolidate", {"dry_run": True}),
])
def test_http_mutations_share_existing_service_credential_gate(http_clients, method, path, body):
    from llm_bawt.service.routes import memory as routes
    http, app, service, _clients = http_clients

    def denied():
        raise HTTPException(status_code=401, detail="service credential required")

    app.dependency_overrides[routes.require_mutation_service_token] = denied
    response = http.request(method, path, params={"bot_id": "fixture", "user_id": "owner-a"}, json=body)
    assert response.status_code == 401
    service.get_memory_client.assert_not_called()
