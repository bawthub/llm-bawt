"""occ bootstrap against a fake Nextcloud. All values are synthetic."""
import json
import logging
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.integrations.nextcloud.setup import SetupError
from llm_bawt.integrations.nextcloud.signing import BOOTSTRAP

from tests.test_nextcloud_reconnect import OLD, env, profile, raw_settings  # noqa: F401 - fixture reuse

HOOK = "https://bawthub.example.test/webhook/nextcloud"


def start(env):
    return env.bindings.bootstrap("test-bot", "room123", "test-user", HOOK, "Test Bot", "operator")


def test_bootstrap_reveals_once_then_verifies_and_activates(env, caplog):
    caplog.set_level(logging.DEBUG)
    out = start(env)
    secret = out["secret"]
    assert 40 <= len(secret) <= 128
    assert secret in out["commands"][0] and HOOK in out["commands"][0] and "'Test Bot'" in out["commands"][0]
    assert out["commands"][2].endswith("room123")
    # Stored encrypted, never in listings; active config untouched.
    assert secret not in json.dumps(raw_settings(env))
    listed = env.bindings.list()[0]
    assert listed["bootstrap_pending"] is True and secret not in json.dumps(listed)
    assert profile(env)["secret"] == OLD
    env.fake.accepted = secret
    assert env.bindings.complete_bootstrap("test-bot", 34)["outbound_verified"] is True
    assert env.store.active_secret("test-bot") == secret
    assert profile(env)["bot_id"] == 34 and "secret" not in profile(env)
    assert not any(k.endswith(BOOTSTRAP + "test-bot") for k in raw_settings(env))
    assert env.bindings.list()[0]["bootstrap_pending"] is False
    assert secret not in caplog.text


def test_failed_verification_keeps_pending_secret_for_retry(env):
    out = start(env)
    with pytest.raises(SetupError) as exc:
        env.bindings.complete_bootstrap("test-bot", 34)  # fake still expects another secret
    assert exc.value.code == "signing_rejected" and out["secret"] not in exc.value.message
    assert profile(env)["secret"] == OLD and env.store.has_bootstrap("test-bot")
    env.fake.accepted = out["secret"]
    env.bindings.complete_bootstrap("test-bot", 34)
    assert env.store.active_secret("test-bot") == out["secret"]


def test_restart_supersedes_earlier_secret(env):
    first, second = start(env)["secret"], start(env)["secret"]
    assert first != second
    env.fake.accepted = second
    env.bindings.complete_bootstrap("test-bot", 34)
    assert env.store.active_secret("test-bot") == second


def test_restart_during_verification_fences_old_completion(env):
    first = start(env)["secret"]
    second = {}
    env.fake.accepted = first
    env.fake.hook = lambda: second.setdefault("secret", start(env)["secret"])
    with pytest.raises(SetupError) as exc:
        env.bindings.complete_bootstrap("test-bot", 34)
    assert exc.value.code == "bootstrap_superseded"
    assert env.store.active_secret("test-bot") is None and profile(env)["secret"] == OLD
    pending = env.store.pending_bootstrap("test-bot")
    assert pending is not None and pending[0] == second["secret"]
    env.fake.hook, env.fake.accepted = None, second["secret"]
    env.bindings.complete_bootstrap("test-bot", 34)
    assert env.store.active_secret("test-bot") == second["secret"]


def test_complete_without_bootstrap_and_disconnect_clears_it(env):
    with pytest.raises(SetupError, match="no_bootstrap"):
        env.bindings.complete_bootstrap("test-bot", 34)
    start(env)
    env.bindings.disconnect("test-bot")
    assert raw_settings(env) == {}


def test_bootstrap_refuses_private_bot_bad_hook_and_ephemeral_key(env, monkeypatch):
    with pytest.raises(SetupError, match="protected_bot"):
        env.bindings.bootstrap("fixture-private", "room123", "u", HOOK, "Bot")
    with pytest.raises(SetupError, match="invalid_webhook_url"):
        env.bindings.bootstrap("test-bot", "room123", "u", "https://user:pw@x.test/h", "Bot")
    from llm_bawt.service.providers.base import CredentialStore
    monkeypatch.setattr(CredentialStore, "durable_key_problem", staticmethod(lambda: "ephemeral_root"))
    with pytest.raises(SetupError, match="storage_unavailable"):
        start(env)
    assert raw_settings(env) == {}


def test_routes_reveal_with_no_store_and_status_never_carries_secret(env, monkeypatch):
    from llm_bawt.service.routes import nextcloud_setup as routes
    monkeypatch.setattr(routes, "manager", lambda: SimpleNamespace(bindings=env.bindings))
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)
    base = "/v1/integrations/nextcloud-talk/bindings/test-bot/bootstrap"
    res = client.post(base, json={"room_token": "room123", "user_id": "test-user", "actor_id": "operator", "webhook_url": HOOK,
                                  "bot_name": "Test Bot"})
    assert res.status_code == 200 and res.headers["cache-control"] == "no-store"
    secret = res.json()["secret"]
    assert secret not in json.dumps(env.bindings.list())
    env.fake.accepted = secret
    done = client.post(base + "/complete", json={"nextcloud_bot_id": 34})
    assert done.status_code == 200 and secret not in done.text
