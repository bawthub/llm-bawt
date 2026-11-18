"""TASK-905 slice B: Google Tasks browser OAuth and import runs against fake Google and BawtHub."""
from __future__ import annotations

import json
import time
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from cryptography.fernet import Fernet
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, create_engine

from llm_bawt.integrations.connections import (
    TASKS_LISTS_URL, TASKS_SCOPE, IntegrationConnections, IntegrationError, IntegrationStore, callback_path,
)
from llm_bawt.integrations.google_tasks_sync import GoogleTasksSync
from llm_bawt.runtime_settings import RuntimeSetting, RuntimeSettingsStore
from llm_bawt.service.providers import crypto

ORIGIN = "https://dev.bawthub.com"
CLIENT_ID = "test.apps.googleusercontent.com"
TID = "google-tasks"


@pytest.fixture
def manager(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine, tables=[RuntimeSetting.__table__])
    runtime = object.__new__(RuntimeSettingsStore)
    runtime.engine = engine
    store = object.__new__(IntegrationStore)
    store._store = runtime
    cipher = Fernet(Fernet.generate_key())
    monkeypatch.setattr(crypto, "encrypt", lambda text: cipher.encrypt(text.encode()).decode())
    monkeypatch.setattr(crypto, "decrypt", lambda text: cipher.decrypt(text.encode()).decode())
    recorded: list = []
    manager = IntegrationConnections(None, store=store, record_connection=recorded.append)
    manager.recorded = recorded
    manager.configure(TID, CLIENT_ID, "client-secret", ORIGIN + callback_path(TID))
    return manager


def fake_google(manager, *, scope=f"openid email {TASKS_SCOPE}", probe_status=200, refresh=True, sub="g-1"):
    calls = []

    def handle(request):
        calls.append(request)
        if request.url.path == "/token":
            body = dict(parse_qs(request.content.decode()))
            if body.get("grant_type") == ["refresh_token"]:
                return httpx.Response(200, json={"access_token": "refreshed", "expires_in": 3600})
            return httpx.Response(200, json={"scope": scope, "access_token": "access-secret", "expires_in": 3600,
                                             **({"refresh_token": "refresh-secret"} if refresh else {})})
        if request.url.path == "/v1/userinfo":
            return httpx.Response(200, json={"sub": sub, "email": "operator@example.com", "email_verified": True})
        if str(request.url).startswith(TASKS_LISTS_URL):
            return httpx.Response(probe_status, json={"items": []} if probe_status == 200 else {})
        raise AssertionError(f"unexpected Google call {request.url}")

    manager.client_factory = lambda **kw: httpx.Client(transport=httpx.MockTransport(handle), **kw)
    return calls


def connect(manager):
    flow = manager.start(TID, "/tools/settings/integrations", ORIGIN)
    params = parse_qs(urlsplit(flow["authorization_url"]).query)
    return flow, params, manager.finish(TID, params["state"][0], flow["browser"], "code", "")


def test_registry_and_authorization_scopes(manager):
    flow = manager.start(TID, "/tools/settings/integrations", ORIGIN)
    params = parse_qs(urlsplit(flow["authorization_url"]).query)
    assert params["scope"][0].split() == [TASKS_SCOPE, "openid", "https://www.googleapis.com/auth/userinfo.email"]
    assert params["redirect_uri"][0] == ORIGIN + callback_path(TID)
    with pytest.raises(IntegrationError, match="invalid_redirect_uri"):
        manager.configure(TID, CLIENT_ID, None, ORIGIN + callback_path("google-keep"))


def test_connect_records_bawthub_generation_before_success(manager):
    calls = fake_google(manager)
    *_, return_to = connect(manager)
    assert return_to == "/tools/settings/integrations"
    record = manager.store.load(TID)
    assert record.status == "connected" and record.meta["read_only"] is True
    assert manager.recorded == ["g-1"]
    assert any(str(c.url).startswith(TASKS_LISTS_URL) for c in calls)
    assert manager.descriptor(TID)["reauthorization_required"] is False


def test_bawthub_failure_leaves_nothing_connected(manager):
    fake_google(manager)

    def fail(subject):
        raise IntegrationError("bawthub_unavailable")

    manager.record_connection = fail
    with pytest.raises(IntegrationError, match="bawthub_unavailable"):
        connect(manager)
    assert manager.store.load(TID) is None


def test_stale_callback_cannot_notify_after_newer_login(manager):
    """Callback A exchanges, login B starts and commits; A must fail before telling BawtHub."""
    fake_google(manager, sub="g-a")
    flow_a = manager.start(TID, "/tools/settings/integrations", ORIGIN)
    state_a = parse_qs(urlsplit(flow_a["authorization_url"]).query)["state"][0]
    pending_a = manager.store.consume_state(TID, state_a, flow_a["browser"])  # A is mid-exchange
    fake_google(manager, sub="g-b")
    connect(manager)  # B starts, finishes and commits
    with pytest.raises(IntegrationError, match="invalid_state"):
        manager.store.finalize_claim(TID, pending_a["claim"])
    assert manager.recorded == ["g-b"] and manager.store.load(TID).meta["google_subject"] == "g-b"


def test_finalizing_callback_fences_start_configure_and_disconnect(manager):
    fake_google(manager)
    flow = manager.start(TID, "/tools/settings/integrations", ORIGIN)
    state = parse_qs(urlsplit(flow["authorization_url"]).query)["state"][0]
    pending = manager.store.consume_state(TID, state, flow["browser"])
    claim = manager.store.finalize_claim(TID, pending["claim"])
    for action in (lambda: manager.start(TID, "/tools/settings/integrations", ORIGIN),
                   lambda: manager.configure(TID, CLIENT_ID, None, ORIGIN + callback_path(TID)),
                   lambda: manager.disconnect(TID)):
        with pytest.raises(IntegrationError, match="connection_in_progress"):
            action()
    manager.store.release_claim(TID, claim)
    assert manager.store.finalizing(TID) is False
    manager.start(TID, "/tools/settings/integrations", ORIGIN)


def test_ambiguous_bawthub_bump_restores_local_subject(manager):
    fake_google(manager, sub="g-1")
    connect(manager)
    fake_google(manager, sub="g-2")
    sent = []

    def flaky(subject):
        sent.append(subject)
        if subject == "g-2":
            raise IntegrationError("bawthub_unavailable")  # may have landed

    manager.record_connection = flaky
    with pytest.raises(IntegrationError, match="bawthub_unavailable"):
        connect(manager)
    assert sent == ["g-2", "g-1"]  # restored to the still-committed account
    assert manager.store.load(TID).meta["google_subject"] == "g-1"
    assert manager.store.finalizing(TID) is False


def test_tasks_forces_consent_and_keep_does_not(manager):
    flow = manager.start(TID, "/tools/settings/integrations", ORIGIN)
    assert parse_qs(urlsplit(flow["authorization_url"]).query)["prompt"] == ["consent select_account"]
    from llm_bawt.integrations.connections import OAUTH
    assert OAUTH["google-keep"].get("prompt") is None  # Keep keeps select_account only


def test_missing_scope_and_probe_errors_are_bounded(manager):
    fake_google(manager, scope="openid email")
    with pytest.raises(IntegrationError, match="missing_tasks_scope"):
        connect(manager)
    fake_google(manager, probe_status=403)
    with pytest.raises(IntegrationError, match="tasks_access_denied"):
        connect(manager)
    assert manager.store.load(TID) is None and manager.recorded == []


def test_refresh_token_reused_only_for_same_subject(manager):
    fake_google(manager)
    connect(manager)
    fake_google(manager, refresh=False)
    connect(manager)
    assert manager.store.load(TID).secret["refresh_token"] == "refresh-secret"
    fake_google(manager, refresh=False, sub="g-2")
    with pytest.raises(IntegrationError, match="offline_access_required"):
        connect(manager)


def test_disconnect_fences_bawthub_first(manager):
    fake_google(manager)
    connect(manager)
    assert manager.disconnect(TID) is True
    assert manager.recorded == ["g-1", None]


class FakeHub:
    def __init__(self, pages_needed=2, start_status=200, updated_min="2026-11-18T00:00:00Z"):
        self.ingested, self.runs, self.failures = [], 0, []
        self.start_status, self.updated_min = start_status, updated_min

    def client(self):
        def handle(request):
            path = request.url.path
            if request.method == "GET" and path.endswith("/sources"):
                return httpx.Response(200, json={"sources": [{"id": "src-1", "subject": "g-1", "tasklistId": "list/1"}]})
            if path.endswith("/runs"):
                self.runs += 1
                if self.start_status != 200:
                    return httpx.Response(self.start_status, json={"error": "busy"})
                return httpx.Response(200, json={"ok": True, "baseline": False, "updatedMin": self.updated_min,
                    "pageToken": None, "fence": {"sourceId": "src-1", "runId": "r", "subject": "g-1",
                                                 "connectionGeneration": 3, "sourceGeneration": 4}})
            if path.endswith("/failure"):
                self.failures.append(json.loads(request.content))
                return httpx.Response(200, json={"recorded": True})
            if path.endswith("/ingest"):
                body = json.loads(request.content)
                self.ingested.append(body)
                return httpx.Response(200, json={"ok": True, "final": body["nextPageToken"] is None,
                                                 "created": len(body["tasks"]), "updated": 0, "pendingLinkIds": []})
            raise AssertionError(path)
        return httpx.Client(base_url="http://hub", transport=httpx.MockTransport(handle))


def task_pages(seen):
    def handle(request):
        seen.append(request)
        if request.url.path == "/token":
            return httpx.Response(200, json={"access_token": "refreshed", "expires_in": 3600})
        assert request.url.raw_path.split(b"?")[0] == b"/tasks/v1/lists/list%2F1/tasks"
        q = parse_qs(request.url.query.decode())
        assert q["showCompleted"] == q["showHidden"] == q["showDeleted"] == ["true"]
        assert q["updatedMin"] == ["2026-11-18T00:00:00Z"]
        if "pageToken" not in q:
            return httpx.Response(200, json={"items": [{"id": "a", "title": "Buy milk", "etag": "x"}], "nextPageToken": "p2"})
        return httpx.Response(200, json={"items": [{"id": "b", "status": "completed", "deleted": True}, {"bad": 1}]})
    return lambda **kw: httpx.Client(transport=httpx.MockTransport(handle), **kw)


def test_sync_paginates_and_posts_each_page_with_fence(manager):
    fake_google(manager)
    connect(manager)
    record = manager.store.load(TID)
    record.meta["expires_at"] = time.time() - 1  # forces a refresh
    manager.store.save(record)
    hub, seen = FakeHub(), []
    summary = GoogleTasksSync(manager, google_factory=task_pages(seen), bawthub_factory=hub.client).sync_all()
    assert summary == {"sources": 1, "created": 2, "updated": 0, "skipped": 0, "errors": []}
    assert [p["nextPageToken"] for p in hub.ingested] == ["p2", None]
    assert hub.ingested[0]["tasks"] == [{"id": "a", "title": "Buy milk"}]  # unknown fields dropped
    assert hub.ingested[1]["tasks"] == [{"id": "b", "status": "completed", "deleted": True}]
    assert all(p["runId"] == "r" and p["connectionGeneration"] == 3 for p in hub.ingested)
    assert manager.store.load(TID).secret["access_token"] == "refreshed"


def test_busy_run_is_skipped_without_google_calls(manager):
    fake_google(manager)
    connect(manager)
    hub, seen = FakeHub(start_status=409), []
    summary = GoogleTasksSync(manager, google_factory=task_pages(seen), bawthub_factory=hub.client).sync_all()
    assert summary["skipped"] == 1 and seen == [] and hub.ingested == []


def test_sync_requires_connection(manager):
    with pytest.raises(IntegrationError, match="not_connected"):
        GoogleTasksSync(manager, bawthub_factory=FakeHub().client).sync_all()


def test_failed_run_reports_bounded_code_to_bawthub(manager):
    fake_google(manager)
    connect(manager)
    def denied(**kw):
        def handle(request):
            return httpx.Response(403, json={"error": {"message": "secret upstream text"}})
        return httpx.Client(transport=httpx.MockTransport(handle), **kw)
    hub = FakeHub()
    summary = GoogleTasksSync(manager, google_factory=denied, bawthub_factory=hub.client).sync_all()
    assert len(summary["errors"]) == 1 and hub.ingested == []
    assert hub.failures == [{"runId": "r", "error": summary["errors"][0]["error"]}]
    assert "secret" not in json.dumps(hub.failures)


def test_task_lists_pages_ids_and_titles(manager):
    fake_google(manager)
    connect(manager)
    def lists(**kw):
        def handle(request):
            assert str(request.url).startswith(TASKS_LISTS_URL)
            if "pageToken" not in parse_qs(request.url.query.decode()):
                return httpx.Response(200, json={"items": [{"id": "l1", "title": "My Tasks", "etag": "e"}], "nextPageToken": "n"})
            return httpx.Response(200, json={"items": [{"id": "l2"}, {"title": "no id"}]})
        return httpx.Client(transport=httpx.MockTransport(handle), **kw)
    assert GoogleTasksSync(manager, google_factory=lists).task_lists() == [{"id": "l1", "title": "My Tasks"}, {"id": "l2", "title": ""}]


def test_lifecycle_admitted_before_finalize_serializes_with_callback(manager):
    """Codex cmwmmnoh0006f: start B passes admission and pauses before writing pending;
    callback A finishes its exchange meanwhile. A must wait for B, then lose its claim."""
    import threading

    fake_google(manager, sub="g-a")
    flow_a = manager.start(TID, "/tools/settings/integrations", ORIGIN)
    state_a = parse_qs(urlsplit(flow_a["authorization_url"]).query)["state"][0]
    admitted, release = threading.Event(), threading.Event()
    original_load = manager.store.load

    def load(provider):
        if threading.current_thread().name == "start-b" and provider.startswith("client:") and not admitted.is_set():
            admitted.set()
            release.wait(5)
        return original_load(provider)

    manager.store.load = load
    results: dict = {}

    def run(name, fn):
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001
            results[name] = exc

    b = threading.Thread(name="start-b", target=run,
                         args=("b", lambda: manager.start(TID, "/tools/settings/integrations", ORIGIN)))
    b.start()
    assert admitted.wait(5)
    a = threading.Thread(target=run, args=("a", lambda: manager.finish(TID, state_a, flow_a["browser"], "code", "")))
    a.start()
    a.join(0.5)
    assert a.is_alive() and manager.recorded == []  # A waits at the lifecycle lock, BawtHub untouched
    release.set()
    b.join(5)
    a.join(5)
    assert isinstance(results["b"], dict)
    assert isinstance(results["a"], IntegrationError) and str(results["a"]) == "invalid_state"
    assert manager.recorded == [] and manager.store.load(TID) is None


def test_disconnect_waits_for_a_finalizing_callback(manager):
    """Disconnect's pre-notify phase is inside the same lock as the callback's commit."""
    import threading

    fake_google(manager, sub="g-a")
    entered, release = threading.Event(), threading.Event()
    order: list = []

    def record(subject):
        order.append(subject)
        if subject == "g-a":
            entered.set()
            release.wait(5)

    manager.record_connection = record
    t = threading.Thread(target=lambda: connect(manager))
    t.start()
    assert entered.wait(5)
    d = threading.Thread(target=lambda: manager.disconnect(TID))
    d.start()
    d.join(0.5)
    assert d.is_alive() and order == ["g-a"]
    release.set()
    t.join(5)
    d.join(5)
    assert order == ["g-a", None] and manager.store.load(TID) is None


@pytest.mark.parametrize("fails", [False, True])
def test_run_once_publishes_even_without_imports(monkeypatch, fails):
    import asyncio
    from llm_bawt.integrations import google_tasks_sync
    from llm_bawt.service import state_events

    class FakeSync:
        def __init__(self, _):
            pass

        def sync_all(self):
            if fails:
                raise IntegrationError("google_unavailable")
            return {"created": 0, "updated": 0, "errors": []}  # a baseline-only run

    published: list = []

    async def publish(service, topics):
        published.append(list(topics))
        return True

    monkeypatch.setattr(google_tasks_sync, "GoogleTasksSync", FakeSync)
    monkeypatch.setattr(google_tasks_sync, "IntegrationConnections", lambda config: None)
    monkeypatch.setattr(state_events, "publish_state_changed", publish)
    service = type("S", (), {"config": None})()
    if fails:
        with pytest.raises(IntegrationError):
            asyncio.run(google_tasks_sync.run_once(service))
    else:
        asyncio.run(google_tasks_sync.run_once(service))
    assert published == [[state_events.GOOGLE_TASKS]]
