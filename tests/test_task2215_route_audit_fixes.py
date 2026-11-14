"""Regression tests for TASK-2215 route audit fixes."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from llm_bawt.service.routes import ha_weather, sessions
from llm_bawt.service.schemas_history_memory import MemorySearchRequest


# --- session PATCH validates every field before mutating -----------------

class _FakeStorage:
    def __init__(self):
        self.calls: list[tuple] = []

    async def set_session_status(self, session_id, status, bot_id=None):
        self.calls.append(("status", status))
        return True

    async def update_session_metadata(self, session_id, meta, bot_id=None):
        self.calls.append(("meta", meta))


def test_session_patch_with_blank_title_does_not_commit_status(monkeypatch):
    storage = _FakeStorage()

    async def owned(*args, **kwargs):
        return {"id": "s1", "status": "archived"}

    monkeypatch.setattr(sessions, "get_effective_bot_id", lambda bot_id=None: "nova")
    monkeypatch.setattr(sessions, "_resolve_user", lambda user: "nick")
    monkeypatch.setattr(sessions, "_owned_session_or_404", owned)
    monkeypatch.setattr(sessions, "get_storage", lambda: storage)

    body = sessions.UpdateSessionRequest(status="deleted", title="   ")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(sessions.update_session("s1", body, bot_id=None, user=None))
    assert exc.value.status_code == 422
    assert storage.calls == []


# --- weather cache expires per entity ------------------------------------

def test_weather_cache_expiry_is_per_entity(monkeypatch):
    clock = {"now": 0.0}
    requests: list[str] = []

    def respond(request):
        requests.append(request.url.path)
        return httpx.Response(200, json={"state": f"s{len(requests)}", "attributes": {}})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(ha_weather.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw))
    monkeypatch.setattr(ha_weather, "get_service", lambda: SimpleNamespace(config=None))
    monkeypatch.setattr(ha_weather, "_ha_credentials", lambda config: ("https://ha.example", "tok"))
    monkeypatch.setattr(ha_weather.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(ha_weather, "_cache", {})

    def get(entity):
        return asyncio.run(ha_weather.get_ha_weather(entity_id=entity))

    get("weather.a")            # t=0: fetch A
    clock["now"] = 899
    get("weather.b")            # t=899: fetch B (must not refresh A's age)
    clock["now"] = 1000
    get("weather.a")            # t=1000: A is 1000 s old -> refetch
    assert requests == [
        "/api/states/weather.a",
        "/api/states/weather.b",
        "/api/states/weather.a",
    ]
    clock["now"] = 1100
    get("weather.b")            # B is 201 s old -> still cached
    assert len(requests) == 3


# --- memory search rejects the unimplemented text mode -------------------

def test_memory_search_rejects_text_mode():
    with pytest.raises(ValidationError):
        MemorySearchRequest(query="known phrase", method="text")
    assert MemorySearchRequest(query="q").method == "all"
