"""Pasted-secret Talk reconnect against a fake Nextcloud. All values are synthetic."""
import hashlib
import hmac
import json
from types import SimpleNamespace

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, create_engine

from llm_bawt.bot_tags import TagSnapshot
from llm_bawt.integrations.nextcloud import setup_bindings, signing
from llm_bawt.integrations.nextcloud.config import NextcloudBotConfig
from llm_bawt.integrations.nextcloud.setup import SetupError
from llm_bawt.integrations.nextcloud.setup_bindings import NextcloudBindings
from llm_bawt.integrations.nextcloud.signing import ACTIVE, CANDIDATE, SigningStore
from llm_bawt.runtime_settings import RuntimeSetting, RuntimeSettingsStore
from llm_bawt.service.providers import crypto
from llm_bawt.service.providers.base import CredentialStore

INSTANCE = "https://cloud.example.test"
OLD = "synthetic-old-secret"
NEW = "synthetic-new-secret"


class FakeTalk:
    """Accepts a bot message only when signed with ``accepted``."""

    def __init__(self, accepted=NEW, status=None, hook=None):
        self.accepted, self.status, self.hook, self.calls = accepted, status, hook, []

    def __call__(self, request):
        self.calls.append(request)
        assert request.url.path == "/ocs/v2.php/apps/spreed/api/v1/bot/room123/message"
        if self.hook:
            self.hook()
        if self.status:
            return httpx.Response(self.status)
        message = json.loads(request.content)["message"]
        random = request.headers["X-Nextcloud-Talk-Bot-Random"]
        good = hmac.new(self.accepted.encode(), (random + message).encode(), hashlib.sha256).hexdigest()
        return httpx.Response(201 if request.headers["X-Nextcloud-Talk-Bot-Signature"] == good else 401)


@pytest.fixture
def env(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine, tables=[RuntimeSetting.__table__])
    with engine.begin() as conn:
        # Production's unique (scope_type, scope_id, key) index.
        conn.execute(text("CREATE UNIQUE INDEX rs_scope_key ON runtime_settings (scope_type, scope_id, key)"))
        conn.execute(text("CREATE TABLE bot_profiles (slug TEXT PRIMARY KEY, nextcloud_config JSON)"))
        conn.execute(text("INSERT INTO bot_profiles VALUES ('test-bot', :d), ('fixture-private', NULL)"), {
            "d": json.dumps({"bot_id": 12, "conversation_token": "room123", "secret": OLD})})
    runtime = object.__new__(RuntimeSettingsStore)
    runtime.engine = engine
    store = object.__new__(SigningStore)
    store._store = runtime
    cipher = Fernet(Fernet.generate_key())
    monkeypatch.setattr(crypto, "encrypt", lambda v: cipher.encrypt(v.encode()).decode())
    monkeypatch.setattr(crypto, "decrypt", lambda v: cipher.decrypt(v.encode()).decode())
    monkeypatch.setattr(CredentialStore, "durable_key_problem", staticmethod(lambda: None))

    class _Tags:
        def __init__(self, engine):
            self.engine = engine

        def snapshot(self):
            return TagSnapshot.from_rows([
                {"slug": "test-bot", "include_in_global_search": True, "tags": []},
                {"slug": "fixture-private", "include_in_global_search": True, "tags": ["private"]}])

    monkeypatch.setattr(setup_bindings, "BotTags", _Tags)
    # Readers resolve through the same store.
    monkeypatch.setattr(signing, "SigningStore", lambda config: store)
    fake = FakeTalk()
    bindings = NextcloudBindings(SimpleNamespace(NEXTCLOUD_URL=INSTANCE), store=SimpleNamespace(engine=engine),
                                 reload=lambda: None, signing=store, known_user=lambda u: u == "test-user",
                                 client_factory=lambda **kw: httpx.Client(transport=httpx.MockTransport(fake), **kw))
    return SimpleNamespace(engine=engine, store=store, fake=fake, bindings=bindings)


def profile(env):
    with env.engine.connect() as conn:
        raw = conn.execute(text("SELECT nextcloud_config FROM bot_profiles WHERE slug='test-bot'")).scalar_one()
    return json.loads(raw) if isinstance(raw, str) else raw


def raw_settings(env):
    with env.engine.connect() as conn:
        return dict(conn.execute(text("SELECT key, value_json FROM runtime_settings")).all())


def reconnect(env, secret=NEW):
    return env.bindings.reconnect("test-bot", "room123", 12, "test-user", secret, "operator")


def test_verified_reconnect_encrypts_secret_and_scrubs_legacy_plaintext(env, monkeypatch):
    assert reconnect(env)["outbound_verified"] is True
    assert profile(env) == {"bot_id": 12, "conversation_token": "room123", "user_id": "test-user",
                            "actor_id": "operator", "enabled": True, "signing": "encrypted"}
    assert env.store.active_secret("test-bot") == NEW
    assert NEW not in json.dumps(raw_settings(env)) and OLD not in json.dumps(raw_settings(env))
    assert not any(k.endswith(CANDIDATE + "test-bot") for k in raw_settings(env))
    listed = env.bindings.list()[0]
    assert listed["has_signing_secret"] and listed["secret_storage"] == "encrypted"
    assert NEW not in json.dumps(listed)
    # Webhook and outbox readers resolve the encrypted secret.
    monkeypatch.setattr("llm_bawt.integrations.nextcloud.config.get_all_raw_bot_data",
                        lambda: {"test-bot": {"nextcloud": profile(env)}})
    bot = NextcloudBotConfig().get_bot_by_conversation("room123")
    assert bot.secret == NEW and bot.user_id == "test-user"


def test_rejected_secret_leaves_active_config_and_no_candidate(env):
    with pytest.raises(SetupError) as exc:
        reconnect(env, "synthetic-wrong-secret")
    assert exc.value.code == "signing_rejected"
    assert "synthetic-wrong-secret" not in exc.value.message
    assert profile(env)["secret"] == OLD
    assert raw_settings(env) == {}


@pytest.mark.parametrize("status,code", [(404, "bot_not_in_room"), (503, "nextcloud_unavailable")])
def test_upstream_failures_are_bounded(env, status, code):
    env.fake.status = status
    with pytest.raises(SetupError, match=code):
        reconnect(env)
    assert profile(env)["secret"] == OLD and raw_settings(env) == {}


def test_legacy_reader_still_works_before_reconnect(env, monkeypatch):
    monkeypatch.setattr("llm_bawt.integrations.nextcloud.config.get_all_raw_bot_data",
                        lambda: {"test-bot": {"nextcloud": profile(env)}})
    assert NextcloudBotConfig().get_bot("test-bot").secret == OLD


def test_disconnect_during_verification_fences_the_commit(env):
    env.fake.hook = lambda: env.bindings.disconnect("test-bot")
    with pytest.raises(SetupError, match="reconnect_superseded"):
        reconnect(env)
    assert profile(env) is None and raw_settings(env) == {}


def test_newer_reconnect_supersedes_older_in_flight_one(env):
    other = "synthetic-newer-secret"

    def newer():
        env.fake.hook = None
        env.fake.accepted = other
        reconnect(env, other)
        env.fake.accepted = NEW

    env.fake.hook = newer
    with pytest.raises(SetupError, match="reconnect_superseded"):
        reconnect(env)
    assert env.store.active_secret("test-bot") == other


def test_disconnect_removes_encrypted_secret(env):
    reconnect(env)
    env.bindings.disconnect("test-bot")
    assert profile(env) is None and raw_settings(env) == {}


def test_private_bot_and_ephemeral_key_are_refused(env, monkeypatch):
    with pytest.raises(SetupError, match="protected_bot"):
        env.bindings.reconnect("fixture-private", "room123", 12, "test-user", NEW)
    monkeypatch.setattr(CredentialStore, "durable_key_problem", staticmethod(lambda: "ephemeral_root"))
    with pytest.raises(SetupError, match="storage_unavailable"):
        reconnect(env)
    assert env.fake.calls == []


def test_route_validation_never_echoes_the_pasted_secret(env, monkeypatch):
    from llm_bawt.service.routes import nextcloud_setup as routes
    monkeypatch.setattr(routes, "manager", lambda: SimpleNamespace(bindings=env.bindings))
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)
    bad = client.post("/v1/integrations/nextcloud-talk/bindings/test-bot/reconnect",
                      json={"room_token": "bad token!", "nextcloud_bot_id": 12, "user_id": "u", "actor_id": "operator", "secret": NEW})
    assert bad.status_code == 422 and NEW not in bad.text
    long = "synthetic-overlong-" + "x" * 2000
    too_long = client.post("/v1/integrations/nextcloud-talk/bindings/test-bot/reconnect",
                           json={"room_token": "room123", "nextcloud_bot_id": 12, "user_id": "test-user", "actor_id": "operator", "secret": long})
    assert too_long.status_code == 400 and "synthetic-overlong" not in too_long.text
    ok = client.post("/v1/integrations/nextcloud-talk/bindings/test-bot/reconnect",
                     json={"room_token": "room123", "nextcloud_bot_id": 12, "user_id": "test-user", "actor_id": "operator", "secret": NEW})
    assert ok.status_code == 200 and NEW not in ok.text and ok.headers["cache-control"] == "no-store"
