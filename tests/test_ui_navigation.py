"""TASK-2626: ``ui_navigate`` publishes one event and reports what a tab did."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import fakeredis.aioredis
import pytest

from llm_bawt import ui_navigation as nav
from llm_bawt.bot_access import BotAccessError
from llm_bawt.mcp_server import ui_tools

USER = "sample-user"
STREAM = f"events:vex:{USER}"


@pytest.mark.parametrize("raw,expected", [
    ("/tasks", "/tasks"),
    ("  /tasks?view=board#TASK-1  ", "/tasks?view=board#TASK-1"),
    ("/", "/"),
    ("https://example.com/tasks", None),
    ("//example.com/tasks", None),
    ("/\\example.com", None),
    ("tasks", None),
    ("javascript:alert(1)", None),
    ("/api/chat/proxy/v1/x", None),
    ("/api?x=1", None),
    ("/tasks\nx", None),
    ("", None),
    ("/" + "a" * nav.MAX_PATH, None),
])
def test_clean_path_accepts_only_in_app_paths(raw, expected):
    assert nav.clean_path(raw) == expected


def _event():
    return nav.new_event(bot_id="vex", user_id=USER, path="/tasks", reason="board")


async def _requested(client) -> dict:
    for _ in range(200):
        rows = await client.xrange(STREAM)
        if rows:
            return json.loads(rows[-1][1]["payload"])
        await asyncio.sleep(0.01)
    raise AssertionError("no event published")


async def _wait_with(acks, monkeypatch, *, first=0.5, upgrade=0.2):
    monkeypatch.setattr(nav, "FIRST_ACK_SECONDS", first)
    monkeypatch.setattr(nav, "UPGRADE_ACK_SECONDS", upgrade)
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    event = _event()
    waiter = asyncio.create_task(nav.publish_and_wait(client, event))
    published = await _requested(client)
    for outcome in acks:
        assert await nav.record_ack(client, request_id=published["request_id"], bot_id="vex",
                                    user_id=USER, outcome=outcome)
    return await waiter, published, client


def test_publishes_one_user_scoped_event_and_returns_navigated(monkeypatch):
    outcome, published, _ = asyncio.run(_wait_with(["navigated"], monkeypatch))
    assert outcome == "navigated"
    assert published["_type"] == "ui_navigate" and published["path"] == "/tasks"
    assert published["bot_id"] == "vex" and published["user_id"] == USER
    assert nav.valid_request_id(published["request_id"])
    assert published["expires_in"] == nav.EVENT_TTL_SECONDS


def test_a_tab_that_navigated_wins_over_one_that_only_offered(monkeypatch):
    outcome, _, _ = asyncio.run(_wait_with(["offered", "navigated"], monkeypatch))
    assert outcome == "navigated"


def test_offered_only(monkeypatch):
    outcome, _, _ = asyncio.run(_wait_with(["offered"], monkeypatch))
    assert outcome == "offered"


def test_no_answer_means_no_client(monkeypatch):
    outcome, _, _ = asyncio.run(_wait_with([], monkeypatch, first=0.2))
    assert outcome == "no_client"


def test_ack_refuses_another_user_bot_or_unknown_request():
    async def run():
        client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        event = _event()
        await client.set(nav.request_key(event["request_id"]), nav.owner_value("vex", USER))
        rid = event["request_id"]
        assert not await nav.record_ack(client, request_id=rid, bot_id="vex",
                                        user_id="other-user", outcome="navigated")
        assert not await nav.record_ack(client, request_id=rid, bot_id="byte",
                                        user_id=USER, outcome="navigated")
        assert not await nav.record_ack(client, request_id="0" * 32, bot_id="vex",
                                        user_id=USER, outcome="navigated")
        assert await client.llen(nav.ack_key(rid)) == 0
    asyncio.run(run())


def _caller(monkeypatch, caller=None, error=False):
    from llm_bawt.mcp_server import bot_access_identity

    def trusted(name, arguments):
        assert name == "ui_navigate"
        if error:
            raise BotAccessError("caller_unavailable")
        return caller

    monkeypatch.setattr(bot_access_identity, "trusted_caller", trusted)


def _tool(monkeypatch, outcome="navigated"):
    sent = []

    async def publish(event):
        sent.append(event)
        return outcome

    monkeypatch.setattr(ui_tools, "_publish", publish)
    return sent


@pytest.mark.parametrize("outcome,phrase", [
    ("navigated", "now shows /tasks"),
    ("offered", "was not moved"),
    ("no_client", "was not moved"),
])
def test_tool_reports_the_outcome_for_the_signed_caller(monkeypatch, outcome, phrase):
    _caller(monkeypatch, SimpleNamespace(bot_id="vex", user_id=USER))
    sent = _tool(monkeypatch, outcome)
    result = asyncio.run(ui_tools.ui_navigate(bot_id="Vex", path=" /tasks ", reason="board"))
    assert result["outcome"] == outcome and result["path"] == "/tasks"
    assert phrase in result["message"]
    assert sent[0]["user_id"] == USER and sent[0]["bot_id"] == "vex"


@pytest.mark.parametrize("caller,error,expected", [
    (None, False, "caller_unverified"),
    (None, True, "caller_unverified"),
    (SimpleNamespace(bot_id="byte", user_id=USER), False, "caller_mismatch"),
])
def test_tool_refuses_unverified_or_mismatched_callers(monkeypatch, caller, error, expected):
    _caller(monkeypatch, caller, error)
    sent = _tool(monkeypatch)
    assert asyncio.run(ui_tools.ui_navigate(bot_id="vex", path="/tasks")) == {"error": expected}
    assert sent == []


def test_tool_refuses_external_urls_before_publishing(monkeypatch):
    _caller(monkeypatch, SimpleNamespace(bot_id="vex", user_id=USER))
    sent = _tool(monkeypatch)
    result = asyncio.run(ui_tools.ui_navigate(bot_id="vex", path="https://example.com/"))
    assert "in-app path" in result["error"] and sent == []


def test_tool_reports_publish_failure_without_claiming_a_move(monkeypatch):
    _caller(monkeypatch, SimpleNamespace(bot_id="vex", user_id=USER))

    async def broken(event):
        raise ConnectionError("down")

    monkeypatch.setattr(ui_tools, "_publish", broken)
    result = asyncio.run(ui_tools.ui_navigate(bot_id="vex", path="/tasks"))
    assert "not moved" in result["error"]
