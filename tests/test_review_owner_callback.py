"""A recorded review is delivered to the task's owner (TASK-2056)."""

import asyncio

from llm_bawt.mcp_server import task_review_tools
from llm_bawt.mcp_server.server import mcp  # noqa: F401  (registers tools in order)
from llm_bawt.mcp_server import inter_bot_tools


def run(coro):
    return asyncio.run(coro)


REVIEW = {
    "id": "rev-1", "verdict": "REQUEST_CHANGES", "stage": "IMPLEMENTATION",
    "findings": [{"severity": "BLOCKER"}, {"severity": "NIT"}, {"severity": "NIT"}],
}


def _wire(monkeypatch, task):
    sends = []

    async def fake_get(path, params=None):
        return task

    async def fake_enqueue(**kwargs):
        sends.append(kwargs)
        return {"delivery_id": "d1"}

    monkeypatch.setattr(task_review_tools, "_api_get", fake_get)
    monkeypatch.setattr(inter_bot_tools, "_enqueue_durable", fake_enqueue)
    return sends


def test_owner_gets_one_compact_keyed_delivery(monkeypatch):
    sends = _wire(monkeypatch, {"id": "c1", "shortId": "TASK-7", "agentBotId": "Caid"})
    run(task_review_tools.notify_review_owner("TASK-7", "byte", REVIEW))
    assert len(sends) == 1
    send = sends[0]
    assert send["target_bot_id"] == "caid" and send["sender_bot_id"] == "byte"
    assert send["idempotency_key"] == "review-callback:rev-1"
    assert send["task_id"] == "TASK-7" and send["prefer_steer"] is True
    message = send["message"]
    assert "TASK-7" in message and "REQUEST_CHANGES" in message and "byte" in message
    assert "1 BLOCKER, 2 NIT" in message and "rev-1" in message


def test_replayed_review_reuses_the_same_key(monkeypatch):
    sends = _wire(monkeypatch, {"shortId": "TASK-7", "agentBotId": "caid"})
    run(task_review_tools.notify_review_owner("TASK-7", "byte", REVIEW))
    run(task_review_tools.notify_review_owner("TASK-7", "byte", REVIEW))
    assert {s["idempotency_key"] for s in sends} == {"review-callback:rev-1"}


def test_self_review_and_ownerless_task_send_nothing(monkeypatch):
    sends = _wire(monkeypatch, {"shortId": "TASK-7", "agentBotId": "byte"})
    run(task_review_tools.notify_review_owner("TASK-7", "Byte", REVIEW))
    sends += _wire(monkeypatch, {"shortId": "TASK-7", "agentBotId": None})
    run(task_review_tools.notify_review_owner("TASK-7", "byte", REVIEW))
    assert sends == []


def test_no_review_id_or_lookup_failure_never_raises(monkeypatch):
    sends = _wire(monkeypatch, {"agentBotId": "caid"})
    assert run(task_review_tools.notify_review_owner("TASK-7", "byte", {"verdict": "APPROVE"})) is None

    async def broken(path, params=None):
        raise RuntimeError("bawthub down")

    monkeypatch.setattr(task_review_tools, "_api_get", broken)
    assert run(task_review_tools.notify_review_owner("TASK-7", "byte", REVIEW)) is None
    assert sends == []


def test_add_notifies_only_on_a_recorded_review(monkeypatch):
    notified = []

    async def fake_record(*args):
        return fake_record.result

    async def fake_notify(task_id, reviewer, review):
        notified.append((task_id, reviewer, review["id"]))

    monkeypatch.setattr(task_review_tools, "_record_review", fake_record)
    monkeypatch.setattr(task_review_tools, "notify_review_owner", fake_notify)
    fake_record.result = {"error": "Task not found"}
    run(task_review_tools.add_task_review("TASK-7", "IMPLEMENTATION", "APPROVE", "ok", "byte"))
    assert notified == []
    fake_record.result = {"id": "rev-2", "verdict": "APPROVE"}
    run(task_review_tools.add_task_review("TASK-7", "IMPLEMENTATION", "APPROVE", "ok", "byte"))
    assert notified == [("TASK-7", "byte", "rev-2")]
