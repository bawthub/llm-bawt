"""schedule_wakeup contracts (TASK-1719). No live app or production DB."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from llm_bawt.bot_access import BotAccessError
from llm_bawt.bot_tags import BehaviorTag
from llm_bawt.mcp_server import bot_access_identity, schedule_tools
from llm_bawt.service.prompt_schedule_management import proven_wakeup_creator, schedule_identifier

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
            # The real API derives the id from owner + create key, and proves
            # the creator from that id once (PromptManagement.create).
            identifier = schedule_identifier(user, body["idempotency_key"])
            row = {"id": identifier, "owner_user_id": user,
                   "creator_bot_id": proven_wakeup_creator(user, identifier, body["name"]),
                   "bot_id": body["bot_id"], "name": body["name"],
                   "description": body["description"], "timing": body["timing"],
                   "next_run_at": "x", "lifecycle": "active"}
            if row["id"] in state.schedules:
                return Response(409, {"detail": "Idempotency key already used for another request"})
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
    method, path, user, body, _ = [c for c in rig.calls if c[0] == "POST"][0]
    assert (method, path, user) == ("POST", "/v1/prompt-schedules", "owner")
    assert body["bot_id"] == "byte" and body["target_mode"] == "specific"
    assert body["timing"] == {"kind": "once", "once_at": "2026-10-29T15:30:00+00:00", "timezone": "UTC"}
    assert (body["clear_context"], body["augment_memory"], body["extract_memory"],
            body["missed_policy"]) == (True, False, False, "run_latest")
    # A wake-up missed by hours (outage, restart) still fires: the API max, not the 1h default.
    assert body["misfire_grace_seconds"] == 604800
    assert body["name"].startswith("wakeup:byte:") and body["name"].endswith(":TASK-1713")
    assert body["description"] == "task_id=TASK-1713"
    assert body["prompt"].startswith("[Scheduled wake-up from byte, TASK-1713]\n\n")
    assert out["id"] in rig.schedules
    # TASK-1790: the tool's create key is what proves the creator on the server.
    assert rig.schedules[out["id"]]["creator_bot_id"] == "byte"


@pytest.mark.anyio
async def test_retry_with_same_key_returns_original_even_when_time_moved(rig, monkeypatch):
    first = result(await schedule_tools.schedule_wakeup(message="a", in_minutes=5, idempotency_key="same"))
    monkeypatch.setattr(schedule_tools, "_now", lambda: NOW + timedelta(seconds=40))
    again = result(await schedule_tools.schedule_wakeup(message="a", in_minutes=5, idempotency_key="same"))
    assert again["id"] == first["id"] and again["duplicate"] is True and len(rig.schedules) == 1
    rig.caller = SimpleNamespace(bot_id="al", user_id="owner")
    other = result(await schedule_tools.schedule_wakeup(message="a", in_minutes=5, idempotency_key="same"))
    assert other["id"] != first["id"] and len(rig.schedules) == 2


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
    def hand(identifier, name, bot):
        rig.schedules[identifier] = {"id": identifier, "owner_user_id": "owner", "bot_id": bot,
                                     "name": name, "lifecycle": "active"}

    hand("hand", "Morning digest", "byte")
    # The user renamed hand-made schedules to look like wake-ups (names are
    # editable, ids are derived): a claimed name is not provenance.
    hand("forged-own", "wakeup:byte:0123456789abcdef:x", "al")
    hand("forged-target", "wakeup:snark:0123456789abcdef:x", "byte")
    rig.caller = SimpleNamespace(bot_id="al", user_id="owner")
    al_own = result(await schedule_tools.schedule_wakeup(message="a", in_minutes=5))["id"]
    rig.caller = SimpleNamespace(bot_id="snark", user_id="owner")
    for_me = result(await schedule_tools.schedule_wakeup(message="a", in_minutes=5, target_bot_id="byte"))["id"]
    rig.caller = SimpleNamespace(bot_id="byte", user_id="owner")
    mine = result(await schedule_tools.schedule_wakeup(message="a", in_minutes=5, target_bot_id="al"))["id"]
    listed = result(await schedule_tools.schedule_wakeup(action="list"))
    assert sorted(w["id"] for w in listed["wakeups"]) == sorted([for_me, mine])

    for blocked in ("hand", "forged-own", "forged-target", al_own):
        out = result(await schedule_tools.schedule_wakeup(action="cancel", schedule_id=blocked))
        assert out["error"] == "forbidden" and rig.schedules[blocked]["lifecycle"] == "active"
    # The creator is persisted at create (TASK-1790): renaming a real wake-up
    # moves it to no other creator, and its original creator keeps it.
    rig.schedules[mine]["name"] = rig.schedules[mine]["name"].replace("wakeup:byte:", "wakeup:snark:")
    rig.caller = SimpleNamespace(bot_id="snark", user_id="owner")
    assert result(await schedule_tools.schedule_wakeup(action="cancel", schedule_id=mine))["error"] == "forbidden"
    rig.caller = SimpleNamespace(bot_id="byte", user_id="owner")
    out = result(await schedule_tools.schedule_wakeup(action="cancel", schedule_id=mine))
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
