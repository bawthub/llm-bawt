"""Per-room Talk health, pending-verification callbacks, mapped ingress identity and
honest disconnect, against fake Nextcloud servers. All values are synthetic."""
import hashlib
import hmac
import json
from types import SimpleNamespace

import httpx
import pytest

from llm_bawt.integrations.nextcloud import webhook
from llm_bawt.integrations.nextcloud.config import NextcloudBotConfig
from llm_bawt.integrations.nextcloud.signing import binding_ref
from llm_bawt.integrations.nextcloud.setup import INTEGRATION_ID, NextcloudSetup, SetupError
from llm_bawt.service import dependencies
from llm_bawt.service.providers.base import ConnectionRecord

from tests.test_nextcloud_reconnect import NEW as BOUND_SECRET
from tests.test_nextcloud_reconnect import INSTANCE, env, profile, raw_settings, reconnect  # noqa: F401
from tests.test_nextcloud_bootstrap import start

ROOM = "room123"
CURRENT = binding_ref(ROOM, BOUND_SECRET)


class Request:
    def __init__(self, secret, actor="users/operator", room=ROOM, text="Help"):
        self.payload = json.dumps({
            "target": {"id": room}, "type": "Create",
            "actor": {"type": "Person", "name": "Operator", "id": actor},
            "object": {"id": "7", "content": json.dumps({"message": text})},
        }).encode()
        random = "synthetic-random"
        self.headers = {
            "X-Nextcloud-Talk-Backend": INSTANCE, "X-Nextcloud-Talk-Random": random,
            "X-Nextcloud-Talk-Signature": hmac.new(secret.encode(), random.encode() + self.payload,
                                                   hashlib.sha256).hexdigest(),
        }

    async def body(self):
        return self.payload


@pytest.fixture
def hook(env, monkeypatch):
    from llm_bawt.utils.config import config as app_config
    monkeypatch.setattr(app_config, "NEXTCLOUD_URL", INSTANCE)
    monkeypatch.setattr(app_config, "NEXTCLOUD_ALLOWED_ACTORS", None)
    monkeypatch.setattr("llm_bawt.integrations.nextcloud.config.get_all_raw_bot_data",
                        lambda: {"test-bot": {"nextcloud": profile(env)}} if profile(env) else {})
    monkeypatch.setattr(webhook, "get_nextcloud_manager", lambda: NextcloudBotConfig())
    turns = []

    async def complete(request):
        turns.append(request)
        return {"object": "chat.completion.queued", "delivery_id": "delivery-1",
                "user_message_id": request.user_message_id}

    dispatcher = SimpleNamespace(talk_replies=SimpleNamespace(register=lambda s: ("PENDING", False)),
                                 wake=lambda bot: None)
    monkeypatch.setattr(dependencies, "get_service", lambda: SimpleNamespace(
        chat_completion=complete, _inter_bot_dispatcher=dispatcher))
    return turns


def row(env):
    return env.bindings.list()[0]


@pytest.mark.asyncio
async def test_pending_callback_proves_inbound_without_a_model_turn(env, hook):
    secret = start(env)["secret"]
    # Not yet bound with this secret (the legacy binding has another one).
    assert await webhook.handle_nextcloud_webhook(Request(secret)) == {
        "status": "verification_received", "bot": "test-bot"}
    assert hook == []
    assert row(env)["pending_inbound_at"] and not row(env)["inbound_verified"]
    env.fake.accepted = secret
    env.bindings.complete_bootstrap("test-bot", 34)
    listed = row(env)
    assert listed["inbound_verified"] and listed["outbound_verified"] and listed["state"] == "verified"
    assert secret not in json.dumps(raw_settings(env)) and secret not in json.dumps(listed)


@pytest.mark.asyncio
async def test_outbound_201_alone_is_not_round_trip_health(env, hook):
    reconnect(env)
    listed = row(env)
    assert listed["outbound_verified"] and not listed["inbound_verified"]
    assert listed["state"] == "awaiting_inbound" and listed["identity_mapped"]


@pytest.mark.asyncio
async def test_pending_callback_from_another_secret_is_rejected(env, hook):
    start(env)
    with pytest.raises(Exception) as exc:
        await webhook.handle_nextcloud_webhook(Request("synthetic-forged-secret"))
    assert getattr(exc.value, "status_code", None) == 401
    assert hook == [] and not row(env)["pending_inbound_at"]


@pytest.mark.asyncio
async def test_mapped_room_admits_only_its_actor_as_the_configured_user(env, hook, monkeypatch):
    from tests.test_nextcloud_reconnect import NEW
    reconnect(env)
    assert await webhook.handle_nextcloud_webhook(Request(NEW, actor="users/intruder")) == {"status": "actor_not_mapped"}
    assert await webhook.handle_nextcloud_webhook(Request(NEW, actor="guests/operator")) == {"status": "actor_not_mapped"}
    assert hook == []
    sent = []

    async def send(**kwargs):
        sent.append(kwargs)
        return True

    monkeypatch.setattr(webhook, "send_nextcloud_message", send)
    result = await webhook.handle_nextcloud_webhook(Request(NEW))
    assert result["status"] == "queued"
    assert [t.user for t in hook] == ["test-user"]
    assert row(env)["inbound_verified"] and row(env)["state"] == "verified"


@pytest.mark.asyncio
async def test_mapping_without_actor_fails_closed(env, hook):
    from tests.test_nextcloud_reconnect import NEW
    reconnect(env)
    with env.engine.begin() as conn:
        from sqlalchemy import text
        data = profile(env)
        data.pop("actor_id")
        conn.execute(text("UPDATE bot_profiles SET nextcloud_config=:d WHERE slug='test-bot'"), {"d": json.dumps(data)})
    assert await webhook.handle_nextcloud_webhook(Request(NEW)) == {"status": "actor_not_mapped"}
    assert hook == []


def test_identity_is_validated_server_side(env):
    with pytest.raises(SetupError, match="unknown_user"):
        env.bindings.reconnect("test-bot", ROOM, 12, "someone-else", "x", "operator")
    with pytest.raises(SetupError, match="invalid_actor"):
        env.bindings.reconnect("test-bot", ROOM, 12, "test-user", "x", "users/operator")
    assert env.fake.calls == []


def test_outbound_failure_after_success_is_shown(env):
    reconnect(env)
    assert env.store.note("test-bot", binding=CURRENT, outbound_failed_at="9999-01-01T00:00:00+00:00")
    assert row(env)["state"] == "outbound_failing" and not row(env)["outbound_verified"]


def test_health_note_after_disconnect_does_not_resurrect(env):
    reconnect(env)
    env.bindings.disconnect("test-bot")
    assert not env.store.note("test-bot", binding=CURRENT, inbound_at="2026-01-01T00:00:00+00:00")
    assert raw_settings(env) == {}


def test_old_binding_note_after_reconnect_changes_nothing(env):
    reconnect(env)
    old = binding_ref("old-room", "old-room-secret")  # captured by an operation before the reconnect
    before = env.store.health("test-bot")
    assert not env.store.note("test-bot", binding=old, inbound_at="2026-01-01T00:00:00+00:00")
    assert env.store.health("test-bot") == before and not row(env)["inbound_verified"]
    # Same room, superseded secret: still a different binding.
    assert not env.store.note("test-bot", binding=binding_ref(ROOM, "superseded"), inbound_at="x")
    assert not row(env)["inbound_verified"]


def test_overlapping_notes_merge_without_losing_fields(env):
    reconnect(env)
    assert env.store.note("test-bot", binding=CURRENT, outbound_failed_at="9999-01-01T00:00:00+00:00")
    assert env.store.note("test-bot", binding=CURRENT, inbound_at="2026-01-01T00:00:00+00:00")
    health = env.store.health("test-bot")
    assert health["outbound_failed_at"] == "9999-01-01T00:00:00+00:00" and health["inbound_at"]
    assert row(env)["state"] == "outbound_failing"


def test_disconnect_during_note_never_recreates_health(env):
    reconnect(env)
    captured = CURRENT  # operation captured its binding, then a disconnect committed
    env.bindings.disconnect("test-bot")
    assert not env.store.note("test-bot", binding=captured, inbound_at="x")
    assert raw_settings(env) == {}


def test_note_requires_one_identity(env):
    with pytest.raises(ValueError):
        env.store.note("test-bot", inbound_at="x")


def _setup(env, status=200, account=True):
    calls = []

    def remote(request):
        calls.append(request)
        if status is None:
            raise httpx.ConnectError("synthetic down")
        return httpx.Response(status)

    rec = ConnectionRecord(provider=INTEGRATION_ID, status="connected", account="operator",
                           meta={"instance_url": INSTANCE},
                           secret={"login_name": "operator", "app_password": "synthetic-app-password"})
    setup = NextcloudSetup(SimpleNamespace(NEXTCLOUD_URL=INSTANCE),
                           store=SimpleNamespace(load=lambda p: rec if account else None),
                           client_factory=lambda **kw: httpx.Client(transport=httpx.MockTransport(remote), **kw),
                           bindings=env.bindings)
    return setup, calls


@pytest.mark.parametrize("status,disabled,remote_status", [
    (200, True, "disabled"), (404, True, "disabled"), (403, False, "failed"), (None, False, "failed")])
def test_disconnect_always_stops_locally_and_reports_remote_honestly(env, status, disabled, remote_status):
    reconnect(env)
    setup, calls = _setup(env, status)
    result = setup.disconnect_binding("test-bot")
    assert result["disconnected"] and result["remote_disabled"] is disabled
    assert result["remote_status"] == remote_status
    assert profile(env) is None and raw_settings(env) == {}
    assert calls[0].method == "DELETE" and calls[0].url.path.endswith(f"/bot/{ROOM}/12")
    if not disabled:
        assert result["manual_command"] == f"occ talk:bot:remove 12 {ROOM}"
    assert "synthetic-app-password" not in json.dumps(result)


def test_disconnect_without_account_says_remote_was_not_attempted(env):
    setup, calls = _setup(env, account=False)
    result = setup.disconnect_binding("test-bot")
    assert result["disconnected"] and result["remote_status"] == "not_attempted" and calls == []
    assert profile(env) is None
