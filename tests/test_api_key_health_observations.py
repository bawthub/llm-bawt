from __future__ import annotations

import io
import json
import urllib.error
from contextlib import nullcontext
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import create_engine, event, insert, select, update

from llm_bawt.ops import github_workflow, release
from llm_bawt.runtime_settings import RuntimeSetting
from llm_bawt.service.providers import api_key_health as health, base, github_deploy


@pytest.fixture
def stored(monkeypatch):
    engine = create_engine("sqlite://")
    table = RuntimeSetting.__table__
    table.create(engine)
    monkeypatch.setattr(base.crypto, "encrypt", lambda text: text)
    monkeypatch.setattr(base.crypto, "decrypt", lambda text: text)
    events = []
    monkeypatch.setattr("llm_bawt.service.health_changes.publish", events.append)

    def settings(*_args):
        with engine.connect() as conn:
            return {key: json.loads(value) for key, value in conn.execute(select(table.c.key, table.c.value_json))}

    store = object.__new__(base.CredentialStore)
    store._store = SimpleNamespace(engine=engine, get_scope_settings=settings)
    record = base.ConnectionRecord(provider="example-api", status="connected", auth_method="api_key",
                                   secret={"api_key": "test-key"}, meta={"unrelated": "kept"})
    payload = store._payload(record)
    with engine.begin() as conn:
        conn.execute(insert(table).values(scope_type="global", scope_id="*", key=store._key(record.provider),
                                          value_json=json.dumps(payload), updated_at=datetime.now(timezone.utc)))
    events.clear()
    yield store, engine, events
    engine.dispose()


@pytest.mark.parametrize("text", ["2026-11-06 00:00:00 UTC", "2026-11-06T00:00:00Z",
                                "Fri, 06 Nov 2026 00:00:00 GMT"])
def test_github_expiry_formats_are_epoch_milliseconds(text):
    assert health.parse_expiry(text) == 1793923200000


@pytest.mark.parametrize("text", [None, "", "never", "not-a-date", "9999999999", "2026-99-99"])
def test_unknown_expiry_is_not_invented(text):
    assert health.parse_expiry(text) is None


@pytest.mark.parametrize("days,state", [(8, "ok"), (7, "warning"), (1, "warning"), (-1, "broken")])
def test_any_api_key_provider_uses_expiry_health(days, state):
    expiry = int(datetime.now(timezone.utc).timestamp() * 1000) + days * 86_400_000
    record = base.ConnectionRecord(provider="example-api", status="connected", auth_method="api_key",
                                   meta={"expires_at": expiry})
    adapter = object.__new__(base.ProviderAdapter)
    adapter.store = SimpleNamespace(load=lambda _: record)
    block = adapter.health()
    assert block["state"] == state and block["expires_at"] == expiry
    record.status = "disconnected"
    assert adapter.health()["state"] == "unconfigured"
    record.status = "error"
    assert adapter.health()["state"] == "broken"
    record.status, record.auth_method = "connected", "device_oauth"
    assert adapter.health()["expires_at"] is None


@pytest.mark.parametrize("headers,body", [
    ({"X-RateLimit-Remaining": "0"}, b""),
    ({"Retry-After": "60"}, b""),
    ({}, b'{"message":"You have exceeded a secondary rate limit"}'),
])
def test_rate_limited_403_is_not_credential_failure(headers, body):
    assert health.github_observation(403, headers, body) == {}


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failure_updates_same_key_only_and_preserves_secret(stored, status):
    store, engine, events = stored
    table = RuntimeSetting.__table__
    with engine.connect() as conn:
        original_secret = json.loads(conn.execute(select(table.c.value_json)).scalar_one())["secret_enc"]
    meta = health.github_observation(status, {"github-authentication-token-expiration": "2026-11-06 00:00:00 UTC"})
    assert not health.record_observation(store, "example-api", "stale-key", meta)
    assert health.record_observation(store, "example-api", "test-key", meta)
    record = store.load("example-api")
    assert health.api_key_health(record)["state"] == "broken"
    assert f"HTTP {status}" in health.api_key_health(record)["detail"]
    assert record.meta["unrelated"] == "kept"
    with engine.connect() as conn:
        assert json.loads(conn.execute(select(table.c.value_json)).scalar_one())["secret_enc"] == original_secret
    assert len(events) == 1
    assert not health.record_observation(store, "example-api", "test-key", meta)
    assert len(events) == 1
    # Missing/malformed expiry preserves the last known date.
    success = health.github_observation(200, {"github-authentication-token-expiration": "invalid"})
    health.record_observation(store, "example-api", "test-key", success)
    record = store.load("example-api")
    assert record.meta["expires_at"] == meta["expires_at"]
    assert record.meta["auth_error_status"] == (403 if status == 403 else None)


def test_concurrent_rotation_wins_compare_and_swap(stored):
    store, engine, events = stored
    rotated = False

    def rotate(conn, _cursor, statement, _parameters, _context, _many):
        nonlocal rotated
        if statement.startswith("UPDATE") and not rotated:
            rotated = True
            record = base.ConnectionRecord(provider="example-api", status="disconnected", auth_method="api_key")
            conn.execute(update(RuntimeSetting.__table__).values(value_json=json.dumps(store._payload(record))))

    event.listen(engine, "before_cursor_execute", rotate)
    assert not health.record_observation(store, "example-api", "test-key", {"auth_error_status": 401})
    assert store.load("example-api").status == "disconnected"
    assert not any("observation" in item for item in events)


def test_observation_errors_do_not_leak_or_break_request(caplog):
    store = SimpleNamespace(available=True, _store=SimpleNamespace(engine=None), _key=lambda _: "unused")
    assert not health.record_observation(store, "example-api", "secret-test-key", {"auth_error_status": 401})
    assert "secret-test-key" not in caplog.text


def test_connect_captures_header_before_new_record_is_saved(monkeypatch):
    monkeypatch.setattr(github_deploy.httpx, "get", lambda *_a, **_kw: httpx.Response(
        200, json={"login": "account"}, headers={"github-authentication-token-expiration": "2026-11-06 00:00:00 UTC"},
    ))
    records = []
    adapter = object.__new__(github_deploy.GitHubReleaseAdapter)
    adapter.store = SimpleNamespace(save=records.append)
    assert adapter.set_api_key("test-key").ok
    assert records[-1].meta["expires_at"] == 1793923200000


@pytest.mark.parametrize("status", [200, 401, 403])
def test_dispatch_captures_auth_observation_before_raising(monkeypatch, status):
    seen = []
    monkeypatch.setattr(health, "observe_github_response", lambda *args: seen.append(args))
    monkeypatch.setattr(github_workflow, "_stored_dispatch_token", lambda: "test-key")
    client = httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(status, json={})))
    gateway = github_workflow.HttpGitHubWorkflowGateway(client=client)
    if status == 200:
        gateway._request("GET", "/user")
    else:
        with pytest.raises(github_workflow.GitHubWorkflowError):
            gateway._request("GET", "/user")
    assert seen[0][:3] == ("github-release-dispatch", "test-key", status)
    seen.clear()
    gateway._request("GET", "https://blob.example/receipt", auth=False) if status == 200 else None
    assert not seen
    fixed = github_workflow.HttpGitHubWorkflowGateway("injected-key", client=client)
    with pytest.raises(github_workflow.GitHubWorkflowError) if status != 200 else nullcontext():
        fixed._request("GET", "/user")
    assert not seen
    client.close()


def test_verifier_captures_error_headers_but_not_blob_response(monkeypatch):
    seen = []
    monkeypatch.setattr(health, "observe_github_response", lambda *args: seen.append(args))

    class Opener:
        def open(self, request, timeout):
            raise urllib.error.HTTPError(request.full_url, 403, "forbidden",
                                         {"Retry-After": "60"}, io.BytesIO(b"secondary rate limit"))

    verifier = release.GitHubReleaseVerifier(opener=Opener())
    verifier._token = "test-key"
    verifier._fetch("https://api.github.com/user")
    assert seen[0][:3] == ("github-release", "test-key", 403)
    assert seen[0][-1] == b"secondary rate limit"
    seen.clear()
    verifier._fetch("https://blob.example/receipt", auth=False)
    assert not seen
