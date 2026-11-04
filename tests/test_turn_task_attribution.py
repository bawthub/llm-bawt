"""TASK-2019 accounting never guesses from prose/session history."""
import asyncio
from types import SimpleNamespace

import pytest

from llm_bawt.service import turn_task_attribution as attribution


@pytest.mark.parametrize(("payload", "expected"), [
    ({"tasks": [], "total": 0}, (None, "unassigned")),
    ({"tasks": [], "total": 2}, (None, "ambiguous")),
    ({}, (None, "unavailable")),
    ({"tasks": [], "total": 1}, (None, "unavailable")),
    ({"tasks": [{"id": "cm-task", "agentBotId": "bot", "status": "IN_PROGRESS"}],
      "total": 1}, ("cm-task", "active_assignment")),
    ({"tasks": [{"id": "cm-task", "agentBotId": "other", "status": "IN_PROGRESS"}],
      "total": 1}, (None, "unavailable")),
    ({"tasks": [{"id": "cm-task", "agentBotId": "bot", "status": "REVIEW"}],
      "total": 1}, (None, "unavailable")),
    ({"tasks": [{"id": "cm-task", "agentBotId": "bot", "status": "IN_PROGRESS",
                 "archivedAt": "date"}], "total": 1}, (None, "unavailable")),
])
def test_complete_unique_executable_assignment_only(payload, expected):
    assert attribution.assigned_task(payload, "bot") == expected


def test_delivery_precedes_assignment_without_network(monkeypatch):
    monkeypatch.setattr(attribution, "starting_delivery", lambda *args: "TASK-42")
    writes = []
    monkeypatch.setattr(attribution, "write_attribution", lambda engine, **kw: writes.append(kw))
    monkeypatch.setattr(attribution.httpx, "AsyncClient",
                        lambda **kw: pytest.fail("starting delivery should not query task list"))
    asyncio.run(attribution.stamp_turn_task(
        SimpleNamespace(engine=object()), turn_id="turn-a", bot_id="bot", user_id="user"))
    assert writes == [dict(turn_id="turn-a", bot_id="bot", user_id="user",
                           task_ref="TASK-42", source="delivery")]


def test_unavailable_snapshot_does_not_fail_chat_and_records_reason(monkeypatch):
    def offline(*args):
        raise RuntimeError("offline")
    monkeypatch.setattr(attribution, "starting_delivery", offline)
    writes = []
    monkeypatch.setattr(attribution, "write_attribution", lambda engine, **kw: writes.append(kw))
    asyncio.run(attribution.stamp_turn_task(
        SimpleNamespace(engine=object()), turn_id="turn-a", bot_id="bot", user_id="user"))
    assert writes[0]["source"] == "unavailable" and writes[0]["task_ref"] is None


def test_assignment_lookup_is_bounded_and_read_only(monkeypatch):
    monkeypatch.setattr(attribution, "starting_delivery", lambda *args: None)
    writes, requests = [], []
    monkeypatch.setattr(attribution, "write_attribution", lambda engine, **kw: writes.append(kw))

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["timeout"] == 2.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, path, params):
            requests.append((path, params))
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
                "tasks": [{"id": "cm-task", "agentBotId": "bot", "status": "IN_PROGRESS"}],
                "total": 1,
            })

    monkeypatch.setattr(attribution.httpx, "AsyncClient", Client)
    asyncio.run(attribution.stamp_turn_task(
        SimpleNamespace(engine=object()), turn_id="turn-a", bot_id="bot", user_id="user"))
    assert requests == [("/api/tasks/tasks", {"agentBotId": "bot", "status": "IN_PROGRESS",
                                             "archived": "false", "limit": 2})]
    assert writes[0]["task_ref"] == "cm-task"
    assert writes[0]["source"] == "active_assignment"
