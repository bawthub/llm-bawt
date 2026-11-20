"""Google Tasks over Workspace delegation (TASK-905); no live Google credentials."""
import json
import time
from urllib.parse import parse_qs

import httpx
import jwt
import pytest

from llm_bawt.integrations.connections import (
    KEEP_SCOPE, TASKS_LISTS_URL, TASKS_SCOPE, IntegrationError, access_token,
)
from llm_bawt.integrations.keep_workspace import TOKEN_URL
from tests.test_integration_connections import manager as integration_manager
from tests.test_keep_workspace import key  # noqa: F401 -- RSA service-account fixture

manager = pytest.fixture(name="manager")(integration_manager.__wrapped__)  # noqa: F401
SUBJECT = "operator@example.com"


@pytest.fixture
def notified(manager):
    seen = []
    manager.record_connection = seen.append
    return seen


def google(manager, key, *, list_status=200, scope=TASKS_SCOPE):
    calls = []

    def handle(request):
        calls.append(request)
        if request.url.path == "/token":
            assertion = parse_qs(request.content.decode())["assertion"][0]
            claims = jwt.decode(assertion, key[1].public_key(), algorithms=["RS256"], audience=TOKEN_URL)
            assert claims["sub"] == SUBJECT
            assert claims["scope"] == scope
            return httpx.Response(200, json={"access_token": f"tok-{len(calls)}", "expires_in": 3600})
        assert str(request.url).startswith(TASKS_LISTS_URL)
        return httpx.Response(list_status, json={"items": [{"id": "l1", "title": "PRIVATE LIST"}]})

    manager.client_factory = lambda **kw: httpx.Client(transport=httpx.MockTransport(handle), **kw)
    return calls


def test_tasks_offers_both_methods_and_its_own_scope(manager):
    d = manager.descriptor("google-tasks")
    assert d["auth_methods"] == ["oauth_web", "service_account_delegation"]
    assert d["workspace_admin_scopes"] == [TASKS_SCOPE]
    assert manager.descriptor("google-keep")["workspace_admin_scopes"] == [KEEP_SCOPE]


def test_verify_mints_tasks_scope_probes_lists_and_tells_bawthub(manager, key, notified):
    manager.configure_workspace("google-tasks", json.dumps(key[0]), "Operator@Example.com")
    calls = google(manager, key)
    result = manager.verify_workspace("google-tasks")
    assert len(calls) == 2
    conn = result["connection"]
    assert conn["connected"] and conn["auth_method"] == "service_account_delegation"
    assert conn["account"] == SUBJECT
    assert conn["meta"]["scopes"] == [TASKS_SCOPE] and conn["meta"]["read_only"] is True
    assert notified == [SUBJECT]
    assert "PRIVATE LIST" not in json.dumps(result) and "BEGIN PRIVATE KEY" not in json.dumps(result)


def test_list_denial_does_not_connect_or_notify(manager, key, notified):
    manager.configure_workspace("google-tasks", json.dumps(key[0]), SUBJECT)
    google(manager, key, list_status=403)
    with pytest.raises(IntegrationError):
        manager.verify_workspace("google-tasks")
    assert not manager.descriptor("google-tasks")["connection"]["connected"]
    assert notified == []


def test_bawthub_failure_leaves_tasks_disconnected(manager, key):
    def fail(subject):
        raise IntegrationError("bawthub_unavailable")
    manager.record_connection = fail
    manager.configure_workspace("google-tasks", json.dumps(key[0]), SUBJECT)
    google(manager, key)
    with pytest.raises(IntegrationError, match="bawthub_unavailable"):
        manager.verify_workspace("google-tasks")
    assert not manager.descriptor("google-tasks")["connection"]["connected"]


def test_copy_from_keep_reuses_the_key_server_side(manager, key, notified):
    with pytest.raises(IntegrationError, match="workspace_source_unavailable"):
        manager.configure_workspace("google-tasks", None, SUBJECT, copy_from="google-keep")
    manager.configure_workspace("google-keep", json.dumps(key[0]), SUBJECT)
    assert manager.descriptor("google-tasks")["workspace_sources"] == ["google-keep"]
    manager.configure_workspace("google-tasks", None, SUBJECT, copy_from="google-keep")
    tasks = manager.store.load("workspace:google-tasks")
    assert tasks.secret == manager.store.load("workspace:google-keep").secret
    assert tasks.meta["client_email"] == key[0]["client_email"]
    # Keep never offers its own sources, and a self-copy or unknown source is refused.
    assert manager.descriptor("google-keep")["workspace_sources"] == []
    for bad in ("google-tasks", "nextcloud-talk"):
        with pytest.raises(IntegrationError, match="workspace_source_unavailable"):
            manager.configure_workspace("google-tasks", None, SUBJECT, copy_from=bad)


def test_access_token_remints_for_delegation_and_requires_matching_config(manager, key, notified):
    manager.configure_workspace("google-tasks", json.dumps(key[0]), SUBJECT)
    google(manager, key)
    manager.verify_workspace("google-tasks")
    calls = google(manager, key)
    with manager.client_factory() as http:
        token, subject = access_token(manager, "google-tasks", http)
    assert subject == SUBJECT and calls == []  # fresh verified token reused

    record = manager.store.load("google-tasks")
    record.meta = {**record.meta, "expires_at": time.time() - 1}
    manager.store.save(record)
    with manager.client_factory() as http:
        token, _ = access_token(manager, "google-tasks", http)
    assert len(calls) == 1 and token.startswith("tok-")
    assert manager.store.load("google-tasks").meta["expires_at"] > time.time()

    # A changed target user, without re-verification, is never silently used.
    workspace = manager.store.load("workspace:google-tasks")
    workspace.meta = {**workspace.meta, "subject": "someone@example.com"}
    manager.store.save(workspace)
    record = manager.store.load("google-tasks")
    record.meta = {**record.meta, "expires_at": time.time() - 1}
    manager.store.save(record)
    with manager.client_factory() as http, pytest.raises(IntegrationError, match="reauthorization_required"):
        access_token(manager, "google-tasks", http)


def test_replacing_the_key_fences_bawthub_before_dropping_the_connection(manager, key, notified):
    manager.configure_workspace("google-tasks", json.dumps(key[0]), SUBJECT)
    google(manager, key)
    manager.verify_workspace("google-tasks")
    manager.configure_workspace("google-tasks", json.dumps(key[0]), "other@example.com")
    assert notified == [SUBJECT, None]
    assert not manager.descriptor("google-tasks")["connection"]["connected"]


def test_unsupported_integration_refuses_workspace(manager, key):
    with pytest.raises((IntegrationError, KeyError)):
        manager.configure_workspace("nextcloud-talk", json.dumps(key[0]), SUBJECT)


def test_failed_local_commit_realigns_bawthub_and_releases_the_claim(manager, key, notified):
    manager.configure_workspace("google-tasks", json.dumps(key[0]), SUBJECT)
    google(manager, key)

    def boom(*a, **kw):
        raise RuntimeError("disk full")
    manager.store.commit_connection = boom
    with pytest.raises(RuntimeError):
        manager.verify_workspace("google-tasks")
    assert notified == [SUBJECT, None]  # told, then set back to the local truth
    del manager.store.commit_connection
    manager.record_connection = notified.append
    google(manager, key)
    assert manager.verify_workspace("google-tasks")["connection"]["connected"]  # claim was released


def test_refresh_refuses_a_key_other_than_the_verified_one(manager, key, notified):
    manager.configure_workspace("google-tasks", json.dumps(key[0]), SUBJECT)
    google(manager, key)
    manager.verify_workspace("google-tasks")
    workspace = manager.store.load("workspace:google-tasks")
    workspace.secret = {**workspace.secret, "private_key": workspace.secret["private_key"] + "\n"}
    manager.store.save(workspace)
    record = manager.store.load("google-tasks")
    record.meta = {**record.meta, "expires_at": time.time() - 1}
    manager.store.save(record)
    calls = google(manager, key)
    with manager.client_factory() as http, pytest.raises(IntegrationError, match="reauthorization_required"):
        access_token(manager, "google-tasks", http)
    assert calls == []


def test_copy_from_a_source_removed_after_the_allowlist_read(manager, key, monkeypatch):
    manager.configure_workspace("google-keep", json.dumps(key[0]), SUBJECT)
    load = manager.store.load
    monkeypatch.setattr(manager.store, "load", lambda k: None if k == "workspace:google-keep" else load(k))
    monkeypatch.setattr(manager, "descriptor", lambda i: {"workspace_sources": ["google-keep"]})
    with pytest.raises(IntegrationError, match="workspace_source_unavailable"):
        manager.configure_workspace("google-tasks", None, SUBJECT, copy_from="google-keep")


def _expire(manager):
    record = manager.store.load("google-tasks")
    record.meta = {**record.meta, "expires_at": time.time() - 1}
    manager.store.save(record)


def test_refresh_fails_closed_when_the_connection_is_removed_during_the_exchange(manager, key, notified):
    manager.configure_workspace("google-tasks", json.dumps(key[0]), SUBJECT)
    google(manager, key)
    manager.verify_workspace("google-tasks")
    _expire(manager)
    google(manager, key)
    inner = manager.client_factory

    def racing(**kw):
        client = inner(**kw)
        post = client.post

        def post_then_disconnect(*a, **k):
            response = post(*a, **k)
            manager.store.delete("google-tasks")  # disconnect lands mid-exchange
            return response
        client.post = post_then_disconnect
        return client
    with racing() as http, pytest.raises(IntegrationError, match="reauthorization_required"):
        access_token(manager, "google-tasks", http)


def test_refresh_that_loses_to_a_concurrent_refresh_uses_the_winner(manager, key, notified):
    manager.configure_workspace("google-tasks", json.dumps(key[0]), SUBJECT)
    google(manager, key)
    manager.verify_workspace("google-tasks")
    _expire(manager)
    google(manager, key)
    inner = manager.client_factory

    def racing(**kw):
        client = inner(**kw)
        post = client.post

        def post_then_win(*a, **k):
            response = post(*a, **k)
            winner = manager.store.load("google-tasks")
            winner.secret = {**winner.secret, "access_token": "winner"}
            winner.meta = {**winner.meta, "expires_at": time.time() + 3600}
            manager.store.save(winner)
            return response
        client.post = post_then_win
        return client
    with racing() as http:
        assert access_token(manager, "google-tasks", http)[0] == "winner"
