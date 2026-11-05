"""Stale one-time task-tagged wake-ups skip on their own (TASK-2056)."""

import pytest

from llm_bawt.service import wakeup_staleness as ws


def snap(**overrides):
    base = {"creator_bot_id": "caid", "task_id": "TASK-9", "bot_id": "caid",
            "timing": {"kind": "once", "once_at": "2026-11-05T22:00:00+00:00"}}
    return {**base, **overrides}


@pytest.mark.parametrize("snapshot,kind", [
    (snap(timing={"kind": "cron", "cron_expression": "0 * * * *"}), "scheduled"),  # recurring
    (snap(creator_bot_id=None), "scheduled"),  # user-created
    (snap(task_id=None), "scheduled"),  # untagged
    (snap(), "manual"),  # explicit run-now
])
def test_exempt_occurrences_are_never_looked_up(monkeypatch, snapshot, kind):
    monkeypatch.setattr(ws, "fetch_task", lambda ref: pytest.fail("looked up"))
    assert ws.wakeup_stale_reason(snapshot, kind) is None


@pytest.mark.parametrize("task,expected", [
    ({"shortId": "TASK-9", "status": "COMPLETED", "agentBotId": "caid"}, "TASK-9 is completed"),
    ({"shortId": "TASK-9", "status": "CANCELLED", "agentBotId": "caid"}, "TASK-9 is cancelled"),
    ({"shortId": "TASK-9", "status": "REVIEW", "agentBotId": "caid"}, "TASK-9 is in REVIEW"),
    ({"shortId": "TASK-9", "status": "IN_PROGRESS", "agentBotId": "caid",
      "archivedAt": "2026-11-05T00:00:00Z"}, "TASK-9 is archived"),
    ({"shortId": "TASK-9", "status": "IN_PROGRESS", "agentBotId": "Nova"},
     "TASK-9 is now owned by nova, not caid"),
])
def test_stale_task_records_its_reason(monkeypatch, task, expected):
    monkeypatch.setattr(ws, "fetch_task", lambda ref: task)
    assert ws.wakeup_stale_reason(snap(), "scheduled") == f"Skipped: {expected}"


@pytest.mark.parametrize("task", [
    {"shortId": "TASK-9", "status": "IN_PROGRESS", "agentBotId": "caid"},
    {"shortId": "TASK-9", "status": "QUEUED", "agentBotId": "caid"},
    {"shortId": "TASK-9", "status": "BLOCKED", "agentBotId": None},
    None,  # BawtHub unreachable: fail open
])
def test_live_or_unknown_task_fires(monkeypatch, task):
    monkeypatch.setattr(ws, "fetch_task", lambda ref: task)
    assert ws.wakeup_stale_reason(snap(), "scheduled") is None


def test_fetch_failure_fails_open(monkeypatch):
    class Broken:
        def __init__(self, *a, **k):
            raise RuntimeError("down")

    monkeypatch.setattr(ws.httpx, "Client", Broken)
    assert ws.fetch_task("TASK-9") is None
