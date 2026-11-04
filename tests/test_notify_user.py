"""notify_user contracts (TASK-1531). No live Home Assistant, phones or production DB."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest

from llm_bawt.bot_access import BotAccessError
from llm_bawt.bot_tags import BehaviorTag
from llm_bawt.integrations import notify, notify_bell
from llm_bawt.integrations.ha_credentials import HACredentials, HomeAssistantUnavailable
from llm_bawt.integrations.notify import (
    Notifier, NotifyNotConfigured, NotifySettings, build_payload, resolve_link,
)
from llm_bawt.integrations.notify_store import NotifyStore
from llm_bawt.mcp_server import bot_access_identity, notify_tools
from tests.sqlite_support import fast_sqlite_engine

TOKEN = "fixture-secret-token-value"
ORIGIN = "https://app.example.test"
CONFIG = SimpleNamespace(PUBLIC_ORIGIN=ORIGIN)
NOON = datetime(2026, 1, 5, 17, 0, tzinfo=timezone.utc)  # 12:00 in New York


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def store(tmp_path):
    return NotifyStore(fast_sqlite_engine(tmp_path / "notify.db"))


class Rig:
    """Settings, credentials and a recording Home Assistant for one test."""

    def __init__(self, monkeypatch):
        self.setting = {"ha_services": ["mobile_app_fixture"]}
        self.credentials = HACredentials("http://ha.example.test:8123", TOKEN)
        self.statuses: dict[str, int] = {}
        self.requests: list[httpx.Request] = []
        self.unreachable = False
        rig = self

        class Settings:
            def __init__(self, config):
                pass

            def get_scope_settings(self, scope_type, scope_id):
                assert (scope_type, scope_id) == ("global", "*")
                return {} if rig.setting is None else {"notify": rig.setting}

        def credentials(config):
            if isinstance(rig.credentials, Exception):
                raise rig.credentials
            return rig.credentials

        monkeypatch.setattr(notify, "RuntimeSettingsStore", Settings)
        monkeypatch.setattr(notify, "resolve_ha_credentials", credentials)
        monkeypatch.setattr(notify, "sender_name", lambda bot_id: f"Name-{bot_id}")

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.unreachable:
            raise httpx.ConnectError("down", request=request)
        service = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(self.statuses.get(service, 200), text=f"echo {TOKEN}")

    def http(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


@pytest.fixture
def rig(monkeypatch):
    return Rig(monkeypatch)


async def send(rig, store, *, at=NOON, bot="fixture-bot", **kwargs):
    kwargs.setdefault("message", "Approval waiting")
    async with rig.http() as http:
        return await Notifier(CONFIG, store, http, now=lambda: at).notify(bot, **kwargs)


# --- payload -------------------------------------------------------------

def test_normal_payload_has_link_and_group_but_no_priority_keys():
    payload = build_payload("Deploy done", title="Deploy", urgency="normal", bot_id="fixture-bot",
                            url=f"{ORIGIN}/tasks/1", sender="Fixture")
    assert payload == {"message": "Deploy done", "title": "Fixture: Deploy", "data": {
        "group": "bawthub-fixture-bot", "url": f"{ORIGIN}/tasks/1",
        "clickAction": f"{ORIGIN}/tasks/1"}}


def test_high_payload_is_time_sensitive_on_ios_and_immediate_on_android():
    data = build_payload("Deploy failed", title=None, urgency="high", bot_id="fixture-bot",
                         url=None)["data"]
    assert data == {"group": "bawthub-fixture-bot",
                    "push": {"interruption-level": "time-sensitive"},
                    "priority": "high", "ttl": 0}


def test_title_always_names_the_sender():
    def title(t, sender=None):
        return build_payload("m", title=t, urgency="normal", bot_id="fixture-bot", url=None,
                             sender=sender)["title"]
    assert title(None, "Fixture") == "Fixture"
    assert title("   ", "Fixture") == "Fixture"
    assert title("Deploy", "Fixture") == "Fixture: Deploy"
    assert title(None) == "fixture-bot"  # no display name: the verified id


def test_sender_name_uses_profile_name_or_falls_back_to_id(monkeypatch):
    import llm_bawt.bots as bots

    class Bot:
        name = "Fixture"

    monkeypatch.setattr(bots, "get_bot", lambda slug: Bot() if slug == "fixture-bot" else None)
    assert notify.sender_name("fixture-bot") == "Fixture"
    assert notify.sender_name("other-bot") == "other-bot"

    def boom(slug):
        raise RuntimeError("db down")

    monkeypatch.setattr(bots, "get_bot", boom)
    assert notify.sender_name("fixture-bot") == "fixture-bot"


def test_link_is_app_relative_or_on_the_public_origin():
    assert resolve_link("/tasks/TASK-1", ORIGIN + "/") == f"{ORIGIN}/tasks/TASK-1"
    assert resolve_link(f"{ORIGIN}/chat", ORIGIN) == f"{ORIGIN}/chat"
    assert resolve_link(None, ORIGIN) is None
    for bad in ("//evil.example/x", "https://evil.example/x", "http://app.example.test/x",
                "https://user@app.example.test/x", "tasks/1"):
        with pytest.raises(ValueError):
            resolve_link(bad, ORIGIN)
    with pytest.raises(ValueError, match="public origin"):
        resolve_link("/tasks", "")


# --- settings ------------------------------------------------------------

def test_settings_defaults_and_validation():
    parsed = NotifySettings.parse({"ha_services": ["mobile_app_a", "mobile_app_a", "mobile_app_b"]})
    assert parsed.services == ("mobile_app_a", "mobile_app_b")
    assert parsed.quiet_hours is None
    assert (parsed.limits.per_bot_interval_seconds, parsed.limits.global_per_hour,
            parsed.limits.dedupe_window_seconds) == (600, 6, 3600)
    for raw in (None, [], {}, {"ha_services": []}, {"ha_services": ["notify.mobile_app_a"]},
                {"ha_services": ["Mobile"]}, {"ha_services": ["a"], "global_per_hour": 0},
                {"ha_services": ["a"], "per_bot_interval_seconds": True},
                {"ha_services": ["a"], "quiet_hours": {"start": "22:00", "end": "07:00"}},
                {"ha_services": ["a"], "quiet_hours": {"start": "25:00", "end": "07:00",
                                                        "timezone": "UTC"}},
                {"ha_services": ["a"], "quiet_hours": {"start": "07:00", "end": "07:00",
                                                        "timezone": "UTC"}},
                {"ha_services": ["a"], "quiet_hours": {"start": "22:00", "end": "07:00",
                                                        "timezone": "Not/AZone"}}):
        with pytest.raises(NotifyNotConfigured):
            NotifySettings.parse(raw)


@pytest.mark.anyio
async def test_missing_or_invalid_setting_is_not_configured_and_sends_nothing(rig, store):
    rig.setting = None
    receipt = await send(rig, store)
    assert receipt["result"] == "not_configured" and rig.requests == []
    assert store.get(receipt["id"])["result"] == "not_configured"
    rig.setting = {"ha_services": ["Bad Name"]}
    assert (await send(rig, store))["result"] == "not_configured"


@pytest.mark.anyio
async def test_disconnected_home_assistant(rig, store):
    rig.credentials = None
    assert (await send(rig, store))["result"] == "not_configured"
    rig.credentials = HomeAssistantUnavailable("not_connected")
    assert (await send(rig, store))["result"] == "ha_error:not_connected"
    assert rig.requests == []


# --- sending -------------------------------------------------------------

@pytest.mark.anyio
async def test_sent_posts_to_each_configured_notify_service(rig, store):
    rig.setting = {"ha_services": ["mobile_app_one", "mobile_app_two"]}
    receipt = await send(rig, store, title="Approval", urgency="high", link="/approvals")
    assert receipt["result"] == "sent"
    assert receipt["services"] == {"mobile_app_one": "sent", "mobile_app_two": "sent"}
    assert [r.url.path for r in rig.requests] == ["/api/services/notify/mobile_app_one",
                                                  "/api/services/notify/mobile_app_two"]
    request = rig.requests[0]
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    body = json.loads(request.content)
    assert body["title"] == "Name-fixture-bot: Approval" and body["data"]["url"] == f"{ORIGIN}/approvals"
    assert body["data"]["push"] == {"interruption-level": "time-sensitive"}
    row = store.get(receipt["id"])
    assert (row["bot_id"], row["result"], row["urgency"]) == ("fixture-bot", "sent", "high")
    assert row["message_hash"] != "Approval waiting" and len(row["message_hash"]) == 64


@pytest.mark.anyio
@pytest.mark.parametrize("status", [401, 404, 500, 503])
async def test_ha_http_error_reports_status_without_token_or_body(rig, store, status):
    rig.statuses["mobile_app_fixture"] = status
    receipt = await send(rig, store)
    assert receipt["result"] == f"ha_error:{status}"
    assert TOKEN not in json.dumps(receipt) and "echo" not in json.dumps(receipt)
    assert store.get(receipt["id"])["result"] == f"ha_error:{status}"
    # Nothing reached the phone, so a retry is not rate limited.
    rig.statuses.clear()
    assert (await send(rig, store, at=NOON + timedelta(seconds=1)))["result"] == "sent"


@pytest.mark.anyio
async def test_unreachable_home_assistant(rig, store):
    rig.unreachable = True
    receipt = await send(rig, store)
    assert receipt["result"] == "ha_error:unreachable" and TOKEN not in json.dumps(receipt)


@pytest.mark.anyio
async def test_partial_delivery_still_counts_toward_the_rate_limit(rig, store):
    rig.setting = {"ha_services": ["mobile_app_one", "mobile_app_two"]}
    rig.statuses["mobile_app_two"] = 502
    receipt = await send(rig, store)
    assert receipt["result"] == "partial:ha_error:502"
    assert receipt["services"] == {"mobile_app_one": "sent", "mobile_app_two": "ha_error:502"}
    later = await send(rig, store, at=NOON + timedelta(minutes=1))
    assert later["result"] == "suppressed:rate_limit"


# --- suppression ---------------------------------------------------------

@pytest.mark.anyio
async def test_quiet_hours_let_only_high_urgency_through(rig, store):
    rig.setting = {"ha_services": ["mobile_app_fixture"],
                   "quiet_hours": {"start": "22:00", "end": "07:00",
                                   "timezone": "America/New_York"}}
    night = datetime(2026, 1, 6, 4, 30, tzinfo=timezone.utc)  # 23:30 in New York
    quiet = await send(rig, store, at=night)
    assert quiet["result"] == "suppressed:quiet_hours" and rig.requests == []
    assert store.get(quiet["id"])["result"] == "suppressed:quiet_hours"
    assert (await send(rig, store, at=night, urgency="high"))["result"] == "sent"
    assert (await send(rig, store, at=NOON, bot="other-bot"))["result"] == "sent"


@pytest.mark.anyio
async def test_dedupe_then_per_bot_rate_then_global_cap(rig, store):
    rig.setting = {"ha_services": ["mobile_app_fixture"], "per_bot_interval_seconds": 600,
                   "global_per_hour": 3, "dedupe_window_seconds": 3600}
    assert (await send(rig, store, dedupe_key="deploy:1"))["result"] == "sent"
    # Same key after the per-bot interval: dedupe still holds it.
    assert (await send(rig, store, at=NOON + timedelta(minutes=20),
                       dedupe_key="deploy:1"))["result"] == "suppressed:dedupe"
    # New key inside the interval: per-bot rate limit.
    assert (await send(rig, store, at=NOON + timedelta(minutes=5),
                       dedupe_key="deploy:2"))["result"] == "suppressed:rate_limit"
    # Other bots share the global cap of 3 per hour.
    assert (await send(rig, store, at=NOON + timedelta(minutes=6), bot="bot-b"))["result"] == "sent"
    assert (await send(rig, store, at=NOON + timedelta(minutes=7), bot="bot-c"))["result"] == "sent"
    assert (await send(rig, store, at=NOON + timedelta(minutes=8),
                       bot="bot-d"))["result"] == "suppressed:rate_limit"
    # Same key from another bot is not a duplicate; the hour has rolled over.
    assert (await send(rig, store, at=NOON + timedelta(minutes=61), bot="bot-d",
                       dedupe_key="deploy:1"))["result"] == "sent"
    assert len(rig.requests) == 4


# --- MCP tool: caller and private bots -----------------------------------

class Tags:
    def __init__(self, known=("fixture-bot", "private-bot"), private=("private-bot",),
                 available=True):
        self.known_bots, self.private, self.available = frozenset(known), set(private), available

    def has(self, bot, tag):
        assert tag == BehaviorTag.PRIVATE
        return bot in self.private


@pytest.fixture
def tool(monkeypatch, rig, store):
    state = SimpleNamespace(caller=None, tags=Tags())

    def trusted_caller(name, arguments):
        assert name == "notify_user"
        if isinstance(state.caller, Exception):
            raise state.caller
        return state.caller

    monkeypatch.setattr(bot_access_identity, "trusted_caller", trusted_caller)
    monkeypatch.setattr(notify_tools, "BotTags", lambda: SimpleNamespace(snapshot=lambda: state.tags))
    monkeypatch.setattr(notify_tools, "Config", lambda: CONFIG)
    monkeypatch.setattr(notify_tools, "_store", lambda config: store)
    monkeypatch.setattr(notify_tools, "_http", rig.http)
    state.bell_requests, state.bell_status, state.bell_down = [], 200, False

    def bell_handler(request: httpx.Request) -> httpx.Response:
        state.bell_requests.append(request)
        if state.bell_down:
            raise httpx.ConnectError("down", request=request)
        return httpx.Response(state.bell_status, json={"ok": True})

    monkeypatch.setattr(notify_tools, "_bell_http", lambda: httpx.AsyncClient(
        base_url="http://bell.example.test", transport=httpx.MockTransport(bell_handler)))
    return state


def signed(bot, user="fixture-user"):
    return SimpleNamespace(bot_id=bot, user_id=user)


@pytest.mark.anyio
@pytest.mark.parametrize("bot,caller,tags,expected", [
    ("private-bot", signed("private-bot"), Tags(), "refused:private_bot"),
    ("ghost-bot", signed("ghost-bot"), Tags(), "refused:unknown_bot"),
    ("fixture-bot", signed("fixture-bot"), Tags(available=False), "refused:policy_unavailable"),
    ("fixture-bot", BotAccessError("caller_unavailable"), Tags(), "refused:caller_unverified"),
    # An unsigned MCP client can't push by naming a known public bot.
    ("fixture-bot", None, Tags(), "refused:caller_unverified"),
    ("fixture-bot", signed("private-bot"), Tags(), "refused:caller_mismatch"),
])
async def test_tool_refusals_are_audited_and_send_nothing(tool, rig, store, bot, caller, tags,
                                                          expected):
    tool.caller, tool.tags = caller, tags
    receipt = await notify_tools.notify_user(bot_id=bot, message="hello")
    assert receipt["result"] == expected and rig.requests == []
    assert store.get(receipt["id"])["result"] == expected


@pytest.mark.anyio
async def test_tool_sends_for_the_signed_caller(tool, rig, store):
    tool.caller = signed("fixture-bot")
    receipt = await notify_tools.notify_user(bot_id="Fixture-Bot", message="Deploy failed",
                                             urgency="high", link="/tasks")
    assert receipt["result"] == "sent" and len(rig.requests) == 1
    assert store.get(receipt["id"])["bot_id"] == "fixture-bot"


@pytest.mark.anyio
async def test_off_origin_link_is_an_audited_refusal(tool, rig, store):
    tool.caller = signed("fixture-bot")
    receipt = await notify_tools.notify_user(bot_id="fixture-bot", message="hi",
                                             link="https://evil.example/x")
    assert receipt["result"] == "refused:invalid_link" and rig.requests == []
    assert store.get(receipt["id"])["result"] == "refused:invalid_link"


# --- bell history (TASK-2002) ----------------------------------------------

def bell_bodies(tool):
    return [json.loads(request.content) for request in tool.bell_requests]


@pytest.mark.anyio
async def test_delivered_push_records_one_bell_entry(tool, rig, store):
    tool.caller = signed("fixture-bot")
    receipt = await notify_tools.notify_user(
        bot_id="fixture-bot", message="Deploy failed on echo", title="Deploy",
        urgency="high", link=f"{CONFIG.PUBLIC_ORIGIN}/tasks?x=1#top")
    assert receipt["result"] == "sent"
    [request] = tool.bell_requests
    assert request.url.path == "/internal/notifications/push"
    assert bell_bodies(tool) == [{
        "pushId": receipt["id"], "botId": "fixture-bot", "userId": "fixture-user",
        "title": "Name-fixture-bot: Deploy", "message": "Deploy failed on echo",
        "urgency": "high", "link": "/tasks?x=1#top", "result": "sent"}]


@pytest.mark.anyio
async def test_partial_push_is_recorded_with_its_result(tool, rig, store):
    rig.setting = {"ha_services": ["mobile_app_a", "mobile_app_b"]}
    rig.statuses = {"mobile_app_b": 500}
    tool.caller = signed("fixture-bot")
    receipt = await notify_tools.notify_user(bot_id="fixture-bot", message="hi")
    assert receipt["result"] == "partial:ha_error:500"
    [body] = bell_bodies(tool)
    assert body["result"] == "partial:ha_error:500" and body["link"] is None
    assert body["title"] == "Name-fixture-bot" and body["urgency"] == "normal"


@pytest.mark.anyio
@pytest.mark.parametrize("setup,expected", [
    (lambda rig, tool: setattr(tool, "caller", signed("private-bot")), "refused:caller_mismatch"),
    (lambda rig, tool: setattr(rig, "statuses", {"mobile_app_fixture": 503}), "ha_error:503"),
    (lambda rig, tool: setattr(rig, "unreachable", True), "ha_error:unreachable"),
    (lambda rig, tool: setattr(rig, "setting", None), "not_configured"),
])
async def test_undelivered_pushes_never_reach_the_bell(tool, rig, store, setup, expected):
    tool.caller = signed("fixture-bot")
    setup(rig, tool)
    receipt = await notify_tools.notify_user(bot_id="fixture-bot", message="hi")
    assert receipt["result"] == expected and tool.bell_requests == []


@pytest.mark.anyio
async def test_quiet_hours_suppression_is_not_a_bell_entry(tool, rig, store, monkeypatch):
    rig.setting = {"ha_services": ["mobile_app_fixture"],
                   "quiet_hours": {"start": "11:00", "end": "13:00",
                                   "timezone": "America/New_York"}}
    monkeypatch.setattr(notify_tools, "Notifier",
                        lambda *a, **k: Notifier(*a, now=lambda: NOON, **k))
    tool.caller = signed("fixture-bot")
    receipt = await notify_tools.notify_user(bot_id="fixture-bot", message="hi")
    assert receipt["result"] == "suppressed:quiet_hours" and tool.bell_requests == []


@pytest.mark.anyio
@pytest.mark.parametrize("status,down", [(500, False), (404, False), (200, True)])
async def test_bell_failure_keeps_the_push_result_and_logs_no_text(tool, rig, store, caplog,
                                                                   status, down):
    tool.caller = signed("fixture-bot")
    tool.bell_status, tool.bell_down = status, down
    with caplog.at_level("WARNING"):
        receipt = await notify_tools.notify_user(bot_id="fixture-bot",
                                                 message="secret-ish push text")
    assert receipt["result"] == "sent" and len(tool.bell_requests) == 1
    assert store.get(receipt["id"])["result"] == "sent"
    assert receipt["id"] in caplog.text and "secret-ish" not in caplog.text


@pytest.mark.anyio
async def test_caller_without_user_skips_the_bell(tool, rig, store):
    tool.caller = SimpleNamespace(bot_id="fixture-bot", user_id=None)
    receipt = await notify_tools.notify_user(bot_id="fixture-bot", message="hi")
    assert receipt["result"] == "sent" and tool.bell_requests == []


def test_bell_link_keeps_only_the_app_path():
    assert notify_bell.bell_link(None) is None
    assert notify_bell.bell_link("/chat/nova") == "/chat/nova"
    assert notify_bell.bell_link("https://app.example.test/a/b?c=1") == "/a/b?c=1"
    assert notify_bell.bell_link("https://app.example.test") == "/"
