"""``initiative_state`` refuses private and inactive bots (TASK-1329)."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from llm_bawt.bot_tags import TagSnapshot
from llm_bawt.mcp_server import initiative_tools as it
from llm_bawt.task_turn_context import TaskTurnContext


def _snapshot(**tags: list[str]) -> TagSnapshot:
    return TagSnapshot.from_rows(
        {"slug": slug.replace("_", "-"), "include_in_global_search": True, "tags": value}
        for slug, value in sorted(tags.items()))


@pytest.fixture
def rig(monkeypatch):
    state = {"tags": _snapshot(fixture_public=[], fixture_private=["private"],
                               fixture_inactive=["inactive"]),
             "bot": "fixture-public", "requests": []}

    class _Tags:
        def snapshot(self):
            return state["tags"]

    def handle(request: httpx.Request) -> httpx.Response:
        state["requests"].append(request)
        if state.get("respond"):
            return state["respond"](request)
        if request.method == "GET":
            return httpx.Response(200, json={"events": []} if "events" in request.url.path
                                  else {"generation": 1})
        return httpx.Response(200, json={"ok": True, "body": json.loads(request.content)})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(it, "BotTags", _Tags)
    monkeypatch.setattr(it, "current_task_turn_context", lambda: TaskTurnContext(
        session_id="sess-1", turn_id="turn-1", trigger_message_id="msg-1",
        bot_id=state["bot"], user_id="operator", issued_at=0))
    monkeypatch.setattr(it.httpx, "AsyncClient", lambda **kw: real_client(
        transport=httpx.MockTransport(handle), **kw))
    return state


def _call(action: str):
    return asyncio.run(it.initiative_state(task_id="TASK-1", action=action, snapshot={}))


def test_a_public_bot_acquires_with_its_trusted_run(rig):
    res = _call("acquire")

    assert res["body"]["run"]["botId"] == "fixture-public"
    assert [r.url.path for r in rig["requests"]] == ["/internal/tasks/TASK-1/initiative/acquire"]


@pytest.mark.parametrize("bot,reason", [("fixture-private", "private bot"),
                                        ("fixture-inactive", "inactive bot"),
                                        ("fixture-unknown", "private bot")])
@pytest.mark.parametrize("action", ["acquire", "checkpoint"])
def test_a_private_or_inactive_bot_never_reaches_bawthub(rig, bot, reason, action):
    rig["bot"] = bot

    with pytest.raises(ValueError, match=f"{reason} is outside initiative scope"):
        _call(action)
    assert rig["requests"] == []


def test_an_unreadable_snapshot_refuses_every_write(rig):
    rig["tags"] = TagSnapshot.unavailable()

    with pytest.raises(ValueError, match="private bot"):
        _call("acquire")
    assert rig["requests"] == []


def test_get_needs_no_turn_identity(rig):
    rig["tags"] = TagSnapshot.unavailable()

    assert _call("get") == {"generation": 1, "events": []}


def test_acquire_without_snapshot_resumes_and_forwards_resolved_ref(rig):
    res = asyncio.run(it.initiative_state(task_id="TASK-1", action="acquire",
                                          resolved_ref="approval-9"))

    assert res["body"]["snapshot"] is None
    assert res["body"]["resolvedRef"] == "approval-9"


def test_checkpoint_forwards_defer(rig):
    defer = {"reason": "quota", "until": "2026-11-18T02:00:00Z"}
    res = asyncio.run(it.initiative_state(task_id="TASK-1", action="checkpoint", snapshot={},
                                          run_id="r", lease_token="t", expected_generation=1,
                                          defer=defer))

    assert res["body"]["defer"] == defer


@pytest.mark.parametrize("action,held", [("hold", True), ("resume", False)])
def test_hold_and_resume_use_the_trusted_bot_as_actor(rig, action, held):
    res = asyncio.run(it.initiative_state(task_id="TASK-1", action=action, reason="weekend"))

    assert rig["requests"][0].url.path == "/internal/tasks/TASK-1/initiative/hold"
    assert res["body"] == {"held": held, "actor": "fixture-public", "reason": "weekend"}


def test_held_and_deferred_refusals_keep_their_reason(rig):
    payload = {"error": "deferred", "code": "deferred", "deferReason": "quota",
               "deferUntil": "2026-11-18T02:00:00Z", "deferRef": None}
    rig["respond"] = lambda request: httpx.Response(429, json=payload)

    res = _call("acquire")

    assert res["status"] == 429
    assert res["code"] == "deferred"
    assert res["deferUntil"] == "2026-11-18T02:00:00Z"
