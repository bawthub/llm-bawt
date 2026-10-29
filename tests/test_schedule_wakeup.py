"""schedule_wakeup contracts (TASK-1719). No live app or production DB."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from llm_bawt.bot_access import BotAccessError
from llm_bawt.bot_tags import BehaviorTag
from llm_bawt.mcp_server import bot_access_identity, schedule_tools

NOW = datetime(2026, 10, 29, 15, 0, tzinfo=timezone.utc)


@pytest.fixture
def anyio_backend():
    return "asyncio"


class Tags:
    def __init__(self, known=("al", "byte", "mira"), private=("mira",), available=True):
        self.known_bots, self.private, self.available = frozenset(known), set(private), available

    def has(self, bot, tag):
        assert tag == BehaviorTag.PRIVATE
        return bot in self.private


class Response:
    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload

    def json(self):
        return self._payload


@pytest.fixture
def rig(monkeypatch):
    state = SimpleNamespace(caller=SimpleNamespace(bot_id="byte", user_id="owner"), tags=Tags(),
                            calls=[], schedules={})

    def trusted_caller(name, arguments):
        assert name == "schedule_wakeup"
        if isinstance(state.caller, Exception):
            raise state.caller
        return state.caller

    async def request(method, path, user, body=None, params=None):
        state.calls.append((method, path, user, body, params))
        if method == "POST":
            row = {"id": f"s{len(state.schedules) + 1}", "bot_id": body["bot_id"], "name": body["name"],
                   "description": body["description"], "timing": body["timing"],
                   "next_run_at": "x", "lifecycle": "active"}
            state.schedules[row["id"]] = row
            return Response(201, row)
        if method == "GET" and path == "/v1/prompt-schedules":
            return Response(200, {"schedules": list(state.schedules.values())})
        identifier = path.rsplit("/", 1)[-1]
        if identifier not in state.schedules:
            return Response(404, {"detail": "Schedule not found"})
        if method == "DELETE":
            state.schedules[identifier]["lifecycle"] = "cancelled"
        return Response(200, state.schedules[identifier])

    monkeypatch.setattr(bot_access_identity, "trusted_caller", trusted_caller)
    monkeypatch.setattr(schedule_tools, "BotTags", lambda: SimpleNamespace(snapshot=lambda: state.tags))
    monkeypatch.setattr(schedule_tools, "_request", request)
    monkeypatch.setattr(schedule_tools, "_now", lambda: NOW)
    return state


def result(text):
    return json.loads(text)


@pytest.mark.anyio
@pytest.mark.parametrize("caller", [None, BotAccessError("caller_unavailable"),
                                    SimpleNamespace(bot_id="byte", user_id="")])
async def test_unsigned_caller_is_refused_and_nothing_is_written(rig, caller):
    rig.caller = caller
    out = result(await schedule_tools.schedule_wakeup(message="hi", in_minutes=5))
    assert out["error"] == "unauthorized" and rig.calls == []


@pytest.mark.anyio
async def test_create_uses_signed_owner_and_callback_defaults(rig):
    out = result(await schedule_tools.schedule_wakeup(
        message="check the deploy", in_minutes=30, task_id="task-1713", idempotency_key="k1"))
    method, path, user, body, _ = rig.calls[0]
    assert (method, path, user) == ("POST", "/v1/prompt-schedules", "owner")
    assert body["bot_id"] == "byte" and body["target_mode"] == "specific"
    assert body["timing"] == {"kind": "once", "once_at": "2026-10-29T15:30:00+00:00", "timezone": "UTC"}
    assert (body["clear_context"], body["augment_memory"], body["extract_memory"],
            body["missed_policy"]) == (True, False, False, "run_latest")
    assert body["name"] == "wakeup:byte:TASK-1713" and body["description"] == "task_id=TASK-1713"
    assert body["prompt"].startswith("[Scheduled wake-up from byte, TASK-1713]\n\n")
    assert out["id"] == "s1"


@pytest.mark.anyio
async def test_idempotency_key_is_stable_per_creator(rig):
    await schedule_tools.schedule_wakeup(message="a", in_minutes=5, idempotency_key="same")
    await schedule_tools.schedule_wakeup(message="a", in_minutes=5, idempotency_key="same")
    rig.caller = SimpleNamespace(bot_id="al", user_id="owner")
    await schedule_tools.schedule_wakeup(message="a", in_minutes=5, idempotency_key="same")
    keys = [call[3]["idempotency_key"] for call in rig.calls]
    assert keys[0] == keys[1] != keys[2]


@pytest.mark.anyio
async def test_cron_and_at_timings(rig):
    await schedule_tools.schedule_wakeup(message="a", cron="17 7 * * *", timezone="America/New_York")
    await schedule_tools.schedule_wakeup(message="a", at="2026-10-30T07:17:00-04:00")
    assert rig.calls[0][3]["timing"] == {"kind": "cron", "cron_expression": "17 7 * * *",
                                         "timezone": "America/New_York"}
    assert rig.calls[1][3]["timing"]["once_at"] == "2026-10-30T07:17:00-04:00"


@pytest.mark.anyio
@pytest.mark.parametrize("kwargs,needle", [
    ({"message": " "}, "message"),
    ({"message": "a"}, "exactly one"),
    ({"message": "a", "in_minutes": 5, "cron": "* * * * *"}, "exactly one"),
    ({"message": "a", "at": "2026-10-30T07:17:00"}, "offset"),
    ({"message": "a", "at": "tomorrow"}, "ISO"),
    ({"message": "a", "in_minutes": 5, "task_id": "1713"}, "TASK-"),
])
async def test_invalid_create_writes_nothing(rig, kwargs, needle):
    out = result(await schedule_tools.schedule_wakeup(**kwargs))
    assert out["error"] == "invalid_request" and needle in out["message"] and rig.calls == []


@pytest.mark.anyio
@pytest.mark.parametrize("target,tags,needle", [
    ("mira", Tags(), "private"),
    ("ghost", Tags(), "unknown"),
    ("al", Tags(available=False), "unavailable"),
])
async def test_other_targets_are_checked(rig, target, tags, needle):
    rig.tags = tags
    out = result(await schedule_tools.schedule_wakeup(message="a", in_minutes=5, target_bot_id=target))
    assert out["error"] == "refused" and needle in out["message"] and rig.calls == []


@pytest.mark.anyio
async def test_private_bot_may_schedule_itself(rig):
    rig.caller = SimpleNamespace(bot_id="mira", user_id="owner")
    out = result(await schedule_tools.schedule_wakeup(message="a", in_minutes=5))
    assert out["bot_id"] == "mira"


@pytest.mark.anyio
async def test_list_and_cancel_see_only_own_or_targeting_wakeups(rig):
    rig.schedules["hand"] = {"id": "hand", "bot_id": "byte", "name": "Morning digest", "lifecycle": "active"}
    rig.schedules["other"] = {"id": "other", "bot_id": "al", "name": "wakeup:al:x", "lifecycle": "active"}
    rig.schedules["for-me"] = {"id": "for-me", "bot_id": "byte", "name": "wakeup:snark:x", "lifecycle": "active"}
    await schedule_tools.schedule_wakeup(message="a", in_minutes=5, target_bot_id="al")
    listed = result(await schedule_tools.schedule_wakeup(action="list"))
    assert sorted(w["id"] for w in listed["wakeups"]) == ["for-me", "s4"]

    for blocked in ("hand", "other"):
        out = result(await schedule_tools.schedule_wakeup(action="cancel", schedule_id=blocked))
        assert out["error"] == "forbidden" and rig.schedules[blocked]["lifecycle"] == "active"
    out = result(await schedule_tools.schedule_wakeup(action="cancel", schedule_id="s4"))
    assert out["lifecycle"] == "cancelled"
    missing = result(await schedule_tools.schedule_wakeup(action="cancel", schedule_id="nope"))
    assert missing["error"] == "rejected" and missing["status_code"] == 404


@pytest.mark.anyio
async def test_app_rejection_and_outage_are_reported(rig, monkeypatch):
    async def rejected(*args, **kwargs):
        return Response(422, {"detail": "Use a valid IANA timezone"})

    monkeypatch.setattr(schedule_tools, "_request", rejected)
    out = result(await schedule_tools.schedule_wakeup(message="a", cron="* * * * *", timezone="Nope"))
    assert out == {"error": "rejected", "message": "Use a valid IANA timezone", "status_code": 422}

    async def down(*args, **kwargs):
        raise OSError("down")

    monkeypatch.setattr(schedule_tools, "_request", down)
    out = result(await schedule_tools.schedule_wakeup(message="a", in_minutes=5))
    assert out["error"] == "unavailable" and "unknown" in out["message"]
