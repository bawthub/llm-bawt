"""TASK-1301 route contract for POST /v1/history/search_all (hermetic)."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.bot_access import BotAccessError
from llm_bawt.mcp_server import storage as storage_module
from llm_bawt.message_authorship import AuthorReference, EffectiveAuthorRule
from llm_bawt.service.routes.history_search import router


class FakeStorage:
    def __init__(self, rows=None, error: Exception | None = None):
        self.rows = rows or []
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    async def search_messages_operator(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if self.error is not None:
            raise self.error
        return self.rows


@pytest.fixture
def client_with(monkeypatch):
    def build(fake: FakeStorage) -> TestClient:
        monkeypatch.setattr(storage_module, "get_storage", lambda: fake)
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)
    return build


def test_query_string_transport_passes_parsed_filters(client_with):
    fake = FakeStorage([{
        "id": "m1", "role": "user", "content": "hi", "timestamp": 2.0,
        "source": "peer", "rank": 0.0, "total": 7, "author_class": "delivery",
        "author": {"entity_type": "bot", "entity_id": "byte", "status": "canonical"},
    }])
    response = client_with(fake).post(
        "/v1/history/search_all",
        params={"query": "", "author": "Bot:Byte", "in_bot": " Peer ", "mode": "trgm", "limit": 5},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total_count"] == 7
    assert body["messages"][0]["author_class"] == "delivery"
    query, kwargs = fake.calls[0]
    assert query == ""
    assert kwargs["author"] == AuthorReference.bot("byte")
    assert kwargs["in_bot"] == "peer"
    assert kwargs["mode"] == "trgm" and kwargs["n_results"] == 5


@pytest.mark.parametrize("params", [
    {"query": "  "},
    {},
    {"query": "x", "author": "nick"},
    {"query": "x", "author": "team:nick"},
    {"query": "x", "author": "user:"},
    {"query": "x", "author": ":nick"},
])
def test_invalid_requests_are_400_without_searching(client_with, params):
    fake = FakeStorage()
    response = client_with(fake).post("/v1/history/search_all", params=params)
    assert response.status_code == 400
    assert fake.calls == []


@pytest.mark.parametrize("code", ["policy_unavailable", "source_unavailable"])
def test_unavailable_policy_or_source_is_503_not_empty(client_with, code):
    response = client_with(FakeStorage(error=BotAccessError(code))).post(
        "/v1/history/search_all", params={"query": "x"})
    assert response.status_code == 503
    assert response.json()["detail"] == code


def test_operator_case_is_the_same_template_without_bound_singletons():
    operator = EffectiveAuthorRule.operator_case()
    assert ":authority_bot" not in operator and ":authority_user" not in operator
    assert "m.author_entity_id = p.bot_id" in operator
    assert operator.replace("p.bot_id", ":authority_bot").replace(
        "s.user_id THEN", "s.user_id\n            AND m.author_entity_id = :authority_user THEN",
    ) == EffectiveAuthorRule.sql_case
