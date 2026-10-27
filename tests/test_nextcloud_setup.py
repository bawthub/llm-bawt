"""Hermetic Nextcloud setup tests. All credentials and instances are synthetic."""
import json
import time
from types import SimpleNamespace

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, create_engine

from llm_bawt.integrations.nextcloud.setup import (
    INTEGRATION_ID, NextcloudSetup, NextcloudSetupStore, SetupError, canonical_instance, pinned_url,
)
from llm_bawt.bot_tags import TagSnapshot
from llm_bawt.integrations.nextcloud import setup_bindings
from llm_bawt.integrations.nextcloud.setup_bindings import NextcloudBindings
from llm_bawt.runtime_settings import RuntimeSetting, RuntimeSettingsStore
from llm_bawt.service.providers import crypto
from llm_bawt.service.routes import nextcloud_setup as routes

INSTANCE = "https://cloud.example.test"
APP_PASSWORD = "synthetic-app-password"
POLL_TOKEN = "synthetic-poll-token"


@pytest.fixture
def setup(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine, tables=[RuntimeSetting.__table__])
    runtime = object.__new__(RuntimeSettingsStore)
    runtime.engine = engine
    store = object.__new__(NextcloudSetupStore)
    store._store = runtime
    cipher = Fernet(Fernet.generate_key())
    monkeypatch.setattr(crypto, "encrypt", lambda value: cipher.encrypt(value.encode()).decode())
    monkeypatch.setattr(crypto, "decrypt", lambda value: cipher.decrypt(value.encode()).decode())
    return NextcloudSetup(SimpleNamespace(NEXTCLOUD_URL=INSTANCE), store=store)


def upstream(setup, *, pending=False, start_body=None, poll_body=None, identity=None, admin_status=200, hook=None):
    calls = []
    def handle(request):
        calls.append(request)
        if hook:
            hook(request)
        if request.url.path == "/index.php/login/v2":
            return httpx.Response(200, json=start_body if start_body is not None else {
                "login": INSTANCE + "/login/v2/flow/synthetic-browser-token",
                "poll": {"token": POLL_TOKEN, "endpoint": INSTANCE + "/login/v2/poll"},
            })
        if request.url.path == "/login/v2/poll":
            return httpx.Response(404 if pending else 200, json=poll_body if poll_body is not None else {
                "server": INSTANCE, "loginName": "test-user", "appPassword": APP_PASSWORD,
            })
        if request.url.path == "/ocs/v2.php/cloud/user":
            data = identity if identity is not None else {"id": "test-user", "display-name": "Test User"}
        elif request.url.path.endswith("/room"):
            data = [{"token": "room123", "displayName": "Existing room", "participantType": 2},
                    {"token": "room456", "name": "Read-only room", "participantType": 3}]
        elif request.url.path.endswith("/bot/admin"):
            if admin_status != 200:
                return httpx.Response(admin_status, text=APP_PASSWORD)
            data = [{"id": 12, "name": "Example bot", "state": 1, "secret": "never-return-this",
                     "url": "https://credential:password@callback.example.test/webhook?secret=hidden#fragment"}]
        else:
            raise AssertionError(request.url)
        return httpx.Response(200, json={"ocs": {"meta": {"statuscode": 200}, "data": data}})
    setup.client_factory = lambda **kwargs: httpx.Client(transport=httpx.MockTransport(handle), **kwargs)
    return calls


def connect(setup):
    upstream(setup)
    flow = setup.start(INSTANCE)
    assert setup.poll(flow["browser"]) == {"status": "connected"}
    return flow


@pytest.mark.parametrize("value", [
    "http://cloud.example.test", "https://user:password@cloud.example.test", "https://127.0.0.1",
    "https://[::1]", "https://localhost", INSTANCE + "?password=x", INSTANCE + "#fragment",
    INSTANCE + "/../other", INSTANCE + "/%2fhidden", INSTANCE + "\\evil", INSTANCE + ":8443",
])
def test_reject_invalid_instances(value):
    with pytest.raises(SetupError):
        canonical_instance(value)


def test_only_operator_configured_instance_can_receive_requests(setup):
    calls = upstream(setup)
    with pytest.raises(SetupError, match="instance_not_allowed"):
        setup.start("https://other.example.test")
    assert calls == []
    assert setup.store.load("pending:" + INTEGRATION_ID) is None


@pytest.mark.parametrize("url,poll", [
    ("https://evil.example.test/login/v2/poll", True),
    (INSTANCE + "/login/v2/poll?token=x", True),
    (INSTANCE + "/arbitrary", True),
    ("https://cloud.example.test@evil.example.test/login/v2/flow/x", False),
    (INSTANCE + "/login/v2/flow/../x", False),
    (INSTANCE + "/login/v2/flow/%2fsecret", False),
    (INSTANCE + "/untrusted", False),
])
def test_pinned_returned_urls(url, poll):
    with pytest.raises(SetupError):
        pinned_url(url, INSTANCE, poll=poll)


def test_front_controller_and_subdirectory_urls_are_supported():
    base = INSTANCE + "/nextcloud"
    for prefix in ("/login/v2/", "/index.php/login/v2/"):
        assert pinned_url(base + prefix + "poll", base, poll=True)
        assert pinned_url(base + prefix + "flow/synthetic", base)
    with pytest.raises(SetupError):
        pinned_url(INSTANCE + "/login/v2/poll", base, poll=True)


def test_success_persists_encrypted_account_and_survives_recreation(setup, caplog):
    calls = upstream(setup)
    flow = setup.start(INSTANCE)
    assert flow["expires_in"] == 600
    assert setup.poll(flow["browser"]) == {"status": "connected"}
    rec = setup.store.load(INTEGRATION_ID)
    assert rec.secret["app_password"] == APP_PASSWORD
    assert rec.account == "test-user"
    status = NextcloudSetup(setup.config, store=setup.store).status()
    assert status["account"] == {"id": "test-user", "display_name": "Test User"}
    assert status["account_connected"] is True
    assert status["delivery_verified"] is False
    assert status["provisioning"]["available"] is False
    assert calls[2].headers["ocs-apirequest"] == "true"
    for public in (status, setup.descriptor(), setup.store._store.get_scope_settings("global", "*")):
        serialized = json.dumps(public)
        assert APP_PASSWORD not in serialized and POLL_TOKEN not in serialized
        assert flow["browser"] not in serialized
    assert APP_PASSWORD not in caplog.text and POLL_TOKEN not in caplog.text


def test_poll_pending_is_repeatable_and_browser_bound(setup):
    calls = upstream(setup, pending=True)
    flow = setup.start(INSTANCE)
    for browser in ("", "wrong-browser"):
        with pytest.raises(SetupError, match="login_expired"):
            setup.poll(browser)
    assert len(calls) == 1
    assert setup.poll(flow["browser"]) == {"status": "pending"}
    assert setup.poll(flow["browser"]) == {"status": "pending"}
    assert setup.store.load("pending:" + INTEGRATION_ID).status == "pending"
    upstream(setup)
    assert setup.poll(flow["browser"]) == {"status": "connected"}


def test_success_poll_retry_is_idempotent_and_bound(setup):
    flow = connect(setup)
    calls = upstream(setup)
    assert setup.poll(flow["browser"]) == {"status": "connected"}
    assert calls == []
    with pytest.raises(SetupError):
        setup.poll("other-browser")


def test_expired_flow_never_polls_upstream(setup):
    calls = upstream(setup)
    flow = setup.start(INSTANCE)
    record = setup.store.load("pending:" + INTEGRATION_ID)
    record.secret["expires_at"] = time.time() - 1
    setup.store.save(record)
    with pytest.raises(SetupError, match="login_expired"):
        setup.poll(flow["browser"])
    assert len(calls) == 1


@pytest.mark.parametrize("which", ["login", "poll"])
def test_start_rejects_untrusted_remote_urls(setup, which):
    body = {"login": INSTANCE + "/login/v2/flow/x", "poll": {"token": POLL_TOKEN, "endpoint": INSTANCE + "/login/v2/poll"}}
    if which == "login":
        body["login"] = "https://evil.example.test/login/v2/flow/x"
    else:
        body["poll"]["endpoint"] = "https://evil.example.test/login/v2/poll"
    upstream(setup, start_body=body)
    with pytest.raises(SetupError, match="untrusted_login_url"):
        setup.start(INSTANCE)
    assert setup.store.load(INTEGRATION_ID) is None


def test_returned_server_must_match_before_credentials_sent(setup):
    calls = upstream(setup, poll_body={"server": "https://evil.example.test", "loginName": "test", "appPassword": APP_PASSWORD})
    flow = setup.start(INSTANCE)
    with pytest.raises(SetupError, match="untrusted_login_url"):
        setup.poll(flow["browser"])
    assert len(calls) == 2
    assert setup.store.load(INTEGRATION_ID) is None


def test_disconnect_during_exchange_fences_late_result(setup):
    upstream(setup)
    flow = setup.start(INSTANCE)
    def hook(request):
        if request.url.path == "/ocs/v2.php/cloud/user":
            setup.disconnect()
    upstream(setup, hook=hook)
    with pytest.raises(SetupError, match="login_superseded"):
        setup.poll(flow["browser"])
    assert setup.store.load(INTEGRATION_ID) is None


def test_newer_login_owns_pending_state_even_if_older_request_finishes(setup):
    upstream(setup)
    old = setup.start(INSTANCE)
    newer = []
    def hook(request):
        if request.url.path == "/ocs/v2.php/cloud/user":
            upstream(setup)
            newer.append(setup.start(INSTANCE))
    upstream(setup, hook=hook)
    with pytest.raises(SetupError, match="login_superseded"):
        setup.poll(old["browser"])
    assert setup.store.load(INTEGRATION_ID) is None
    assert setup.poll(newer[0]["browser"]) == {"status": "connected"}


def test_parallel_poll_cannot_claim_same_flow_twice(setup):
    upstream(setup)
    flow = setup.start(INSTANCE)
    setup._claim(flow["browser"])
    with pytest.raises(SetupError, match="login_expired"):
        setup.poll(flow["browser"])


def test_reconnect_failure_keeps_old_account(setup):
    connect(setup)
    upstream(setup, identity={})
    flow = setup.start(INSTANCE)
    with pytest.raises(SetupError):
        setup.poll(flow["browser"])
    assert setup.status()["account_connected"] is True
    assert setup.store.load(INTEGRATION_ID).secret["app_password"] == APP_PASSWORD


def test_discovery_allowlists_and_moderator_permissions(setup):
    connect(setup)
    assert setup.rooms() == {"rooms": [
        {"token": "room123", "name": "Existing room", "can_moderate": True},
        {"token": "room456", "name": "Read-only room", "can_moderate": False},
    ]}
    assert setup.bots()["bots"] == [{"id": 12, "name": "Example bot", "state": 1,
                                      "url": "https://callback.example.test/webhook"}]
    assert "never-return-this" not in json.dumps(setup.bots())


def test_admin_discovery_denial_is_actionable_and_redacted(setup):
    connect(setup)
    upstream(setup, admin_status=403)
    with pytest.raises(SetupError) as exc:
        setup.bots()
    assert exc.value.code == "admin_required"
    assert "administrator" in exc.value.message
    assert APP_PASSWORD not in exc.value.message


@pytest.fixture
def tags(monkeypatch):
    """Stands in for ``BotTags`` over this store's engine: profile rows are the known bots."""
    state = {"private": set(), "down": False, "engines": []}

    class _Tags:
        def __init__(self, engine):
            self.engine = engine
            state["engines"].append(engine)

        def snapshot(self):
            if state["down"]:
                return TagSnapshot.unavailable()
            with self.engine.connect() as conn:
                slugs = conn.execute(text("SELECT slug FROM bot_profiles ORDER BY slug")).scalars()
                return TagSnapshot.from_rows(
                    {"slug": slug, "include_in_global_search": True,
                     "tags": ["private"] if slug in state["private"] else []} for slug in slugs)

    monkeypatch.setattr(setup_bindings, "BotTags", _Tags)
    return state


def _bound_bots(engine, *slugs):
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE bot_profiles (slug TEXT PRIMARY KEY, nextcloud_config JSON)"))
        for slug in slugs:
            conn.execute(text("INSERT INTO bot_profiles VALUES (:slug, :data)"), {
                "slug": slug, "data": json.dumps({"bot_id": 12, "conversation_token": "room123", "secret": "synthetic-signing-secret"}),
            })


def test_binding_blocked_honestly_and_disconnect_only_clears_local_mapping(setup, tags):
    engine = setup.store._store.engine
    _bound_bots(engine, "test-bot")
    reloads = []
    bindings = NextcloudBindings(None, store=SimpleNamespace(engine=engine), reload=lambda: reloads.append(True))
    assert bindings.list()[0]["has_signing_secret"] is True
    assert "synthetic-signing-secret" not in json.dumps(bindings.list())
    with pytest.raises(SetupError, match="provisioning_unavailable"):
        bindings.connect("test-bot", "room123", 12, "test-user")
    assert len(bindings.list()) == 1
    assert bindings.disconnect("test-bot") == {"disconnected": True, "remote_rooms_preserved": True}
    assert bindings.list() == []
    assert reloads == [True]
    assert set(tags["engines"]) == {engine}


def test_a_private_bot_is_never_listed_connected_or_disconnected(setup, tags):
    engine = setup.store._store.engine
    _bound_bots(engine, "fixture-private", "test-bot")
    tags["private"].add("fixture-private")
    bindings = NextcloudBindings(None, store=SimpleNamespace(engine=engine), reload=lambda: None)
    assert [row["bot_id"] for row in bindings.list()] == ["test-bot"]
    for attempt in (lambda: bindings.connect("Fixture-Private ", "room123", 12, "test-user"),
                    lambda: bindings.disconnect("fixture-private")):
        with pytest.raises(SetupError, match="protected_bot"):
            attempt()
    with engine.connect() as conn:
        assert conn.execute(text("SELECT nextcloud_config FROM bot_profiles WHERE slug='fixture-private'")).scalar_one()


def test_an_unreadable_tag_snapshot_lists_and_admits_no_bot(setup, tags):
    engine = setup.store._store.engine
    _bound_bots(engine, "test-bot")
    tags["down"] = True
    bindings = NextcloudBindings(None, store=SimpleNamespace(engine=engine), reload=lambda: None)
    assert bindings.list() == []
    with pytest.raises(SetupError, match="policy_unavailable") as refused:
        bindings.disconnect("test-bot")
    assert refused.value.status == 503
    with engine.connect() as conn:
        assert conn.execute(text("SELECT nextcloud_config FROM bot_profiles")).scalar_one()


def test_no_engine_refuses_without_the_process_engine(tags):
    bindings = NextcloudBindings(None, store=SimpleNamespace(engine=None))
    assert bindings.list() == []
    with pytest.raises(SetupError, match="policy_unavailable"):
        bindings.connect("test-bot", "room123", 12, "test-user")
    assert tags["engines"] == []


def test_start_never_follows_redirects(setup):
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": "https://other.example.test/login"})
    setup.client_factory = lambda **kwargs: httpx.Client(transport=httpx.MockTransport(handle), **kwargs)
    with pytest.raises(SetupError, match="login_start_failed"):
        setup.start(INSTANCE)
    assert len(requests) == 1


def test_shared_list_uses_nextcloud_descriptor_not_google_fields(setup, monkeypatch):
    from llm_bawt.integrations.connections import IntegrationConnections
    from llm_bawt.integrations.credential_integrations import CREDENTIAL_INTEGRATIONS, CredentialIntegrations
    from llm_bawt.service.routes import integrations
    monkeypatch.setattr(routes, "manager", lambda: setup)
    monkeypatch.setattr(integrations, "get_service", lambda: SimpleNamespace(config=setup.config))
    monkeypatch.setattr(integrations, "IntegrationConnections", lambda _: IntegrationConnections(None, store=setup.store))
    monkeypatch.setattr(integrations, "CredentialIntegrations", lambda _: CredentialIntegrations(None, store=setup.store))
    app = FastAPI()
    app.include_router(integrations.router)
    descriptors = TestClient(app).get("/v1/integrations").json()["integrations"]
    assert {item["id"] for item in descriptors} == {"google-keep", "nextcloud-talk", *CREDENTIAL_INTEGRATIONS}
    assert all(item["auth_method"] == "credentials" for item in descriptors if item["id"] in CREDENTIAL_INTEGRATIONS)
    descriptor = next(item for item in descriptors if item["id"] == "nextcloud-talk")
    assert descriptor["auth_method"] == "nextcloud_login_v2"
    assert "workspace" not in descriptor and "client" not in descriptor
    assert descriptor["provisioning"]["available"] is False


def test_cookie_route_contract_and_no_google_handler_fallthrough(setup, monkeypatch):
    upstream(setup)
    setup.bindings = NextcloudBindings(None, store=SimpleNamespace(engine=None))
    monkeypatch.setattr(routes, "manager", lambda: setup)
    from llm_bawt.service.routes import integrations
    app = FastAPI()
    app.include_router(routes.router)
    app.include_router(integrations.router)
    client = TestClient(app, base_url="https://ui.example.test")
    response = client.post("/v1/integrations/nextcloud-talk/login", json={"instance_url": INSTANCE})
    assert response.status_code == 200
    assert set(response.json()) == {"login_url", "expires_in"}
    cookie = response.headers["set-cookie"]
    for flag in ("HttpOnly", "Secure", "SameSite=strict", "Max-Age=600", "Path=" + routes.COOKIE_PATH):
        assert flag in cookie
    browser = client.cookies.get(routes.COOKIE)
    response = client.post("/v1/integrations/nextcloud-talk/login/poll", headers={"cookie": f"{routes.COOKIE}={browser}"})
    assert response.json() == {"status": "connected"}
    assert response.headers["cache-control"] == "no-store"
    assert client.get("/v1/integrations/nextcloud-talk/status").json()["account_connected"]
    response = client.post("/v1/integrations/nextcloud-talk/bindings", json={
        "bot_id": "test-bot", "room_token": "room123", "nextcloud_bot_id": 12, "user_id": "test-user",
    })
    assert response.status_code == 503
    # No engine: the private-bot check fails closed before provisioning is tried.
    assert response.json()["detail"]["code"] == "policy_unavailable"
    assert client.put("/v1/integrations/nextcloud-talk/workspace", json={"service_account_json": "{}", "subject": "x@y.test"}).status_code == 404
    assert client.delete("/v1/integrations/nextcloud-talk").json()["remote_rooms_preserved"]
    assert not setup.status()["account_connected"]
