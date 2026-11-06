"""notifications_list contracts (TASK-2066). No live BawtHub."""
from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from llm_bawt.bot_access import BotAccessError
from llm_bawt.mcp_server import bot_access_identity, notify_tools


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def bell(monkeypatch):
    state = SimpleNamespace(caller=SimpleNamespace(bot_id="fixture-bot", user_id="fixture-user"),
                            requests=[], status=200,
                            body={"items": [{"kind": "alert", "title": "Swap firing"}]})

    def trusted_caller(name, arguments):
        assert name == "notifications_list"
        if isinstance(state.caller, Exception):
            raise state.caller
        return state.caller

    def handler(request: httpx.Request) -> httpx.Response:
        state.requests.append(request)
        return httpx.Response(state.status, json=state.body)

    monkeypatch.setattr(bot_access_identity, "trusted_caller", trusted_caller)
    monkeypatch.setattr(notify_tools, "_bell_http", lambda: httpx.AsyncClient(
        base_url="http://bell.example.test", transport=httpx.MockTransport(handler)))
    return state


@pytest.mark.anyio
async def test_reads_the_signed_callers_bell_with_filters(bell):
    result = await notify_tools.notifications_list(bot_id="fixture-bot", kind="alert",
                                                   since="2026-11-06T00:00:00Z", limit=5)
    assert result == bell.body
    [request] = bell.requests
    assert request.method == "GET" and request.url.path == "/internal/notifications"
    assert dict(request.url.params) == {"user": "fixture-user", "limit": "5", "kind": "alert",
                                        "since": "2026-11-06T00:00:00Z"}


@pytest.mark.anyio
@pytest.mark.parametrize("caller", [None, BotAccessError("caller_unavailable"),
                                    SimpleNamespace(bot_id="fixture-bot", user_id=None)])
async def test_unverified_caller_reads_nothing(bell, caller):
    bell.caller = caller
    assert await notify_tools.notifications_list(bot_id="fixture-bot") == {"error": "caller_unverified"}
    assert bell.requests == []


@pytest.mark.anyio
async def test_bell_error_is_reported(bell):
    bell.status, bell.body = 400, {"error": "unknown kind"}
    assert await notify_tools.notifications_list(bot_id="fixture-bot") == {"error": "unknown kind"}
