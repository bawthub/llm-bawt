"""Offline xAI Management API contract, read-only isolation and balance switching."""

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from llm_bawt.service import usage
from llm_bawt.service.providers import base, crypto
from llm_bawt.service.providers.api_key import ApiKeyAdapter, resolve_api_key
from llm_bawt.service.providers.base import ConnectionRecord, CredentialStore
from llm_bawt.service.providers.registry import access_path_adapter, get_adapter
from llm_bawt.service.providers.xai_management import (
    CONNECTION_ID, VALIDATION_URL, XaiManagementAdapter, balance_url,
    prepaid_usd, team_id_from_validation,
)
from llm_bawt.service.routes import providers as provider_routes
from llm_bawt.service.routes import usage as usage_routes
from llm_bawt.service.usage.adapters.xai import XaiUsageAdapter
from llm_bawt.service.usage.balance import BalanceRow, balance_update_for_turn

KEY = "fixture-management-key-NOT-A-REAL-CREDENTIAL"
TEAM = "fixture-team"
VALIDATION = {"scope": "SCOPE_TEAM", "scopeId": TEAM}


def record(**changes):
    values = dict(provider=CONNECTION_ID, status="connected", meta={"team_id": TEAM},
                  secret={"management_key": KEY}, updated_at="2026-11-02T00:00:00Z")
    values.update(changes)
    return ConnectionRecord(**values)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    class Store:
        available = True

        def __init__(self):
            self.records = {}
            self.saved = []

        def load(self, provider):
            return self.records.get(provider)

        def save(self, rec):
            self.records[rec.provider] = rec
            self.saved.append(rec)

        def delete(self, provider):
            return self.records.pop(provider, None) is not None

    store = Store()
    monkeypatch.setattr(base, "CredentialStore", lambda _config: store)
    monkeypatch.setattr("llm_bawt.service.dependencies.get_service", lambda: SimpleNamespace(config=None))
    monkeypatch.setattr(usage, "_cache", {})
    monkeypatch.setattr(usage, "_cache_identities", {})
    monkeypatch.setattr(usage, "_locks", {})
    monkeypatch.setattr(crypto, "_fernet", Fernet(Fernet.generate_key()))

    def no_network(*_args, **_kwargs):
        raise AssertionError("All upstream requests must be explicitly mocked")

    monkeypatch.setattr(httpx, "get", no_network)
    monkeypatch.setattr(httpx.AsyncClient, "get", no_network)
    for env in XaiUsageAdapter.key_env_vars:
        monkeypatch.delenv(env, raising=False)
    return store


@pytest.mark.parametrize("payload,team", [
    (VALIDATION, TEAM),
    ({**VALIDATION, "teamId": "deprecated-other-team"}, TEAM),
    ({"teamId": "legacy-team"}, "legacy-team"),
    ({"scope": "SCOPE_UNSPECIFIED", "teamId": "legacy-team"}, "legacy-team"),
])
def test_validation_resolves_team_scope_and_legacy_shape(payload, team):
    assert team_id_from_validation(payload) == team


@pytest.mark.parametrize("payload", [
    None, [], {}, {"scope": "SCOPE_TEAM"},
    {"scope": "SCOPE_ORGANIZATION", "scopeId": "organization", "teamId": TEAM},
    {"scope": "NEW_SCOPE", "scopeId": TEAM},
    {"scope": "SCOPE_TEAM", "scopeId": "../other-team"},
    {"scope": "SCOPE_TEAM", "scopeId": "team?query=1"},
    {"scope": "SCOPE_TEAM", "scopeId": "https://other.example"},
])
def test_validation_never_uses_organization_or_unsafe_id_as_team(payload):
    with pytest.raises(ValueError):
        team_id_from_validation(payload)


@pytest.mark.parametrize("cents,dollars", [
    ("-1000", "10"), ("-12345", "123.45"), ("0", "0"), ("-0", "0"),
    ("1", "-0.01"), ("1000", "-10"),
    ("-9223372036854775808", "92233720368547758.08"),
    ("9223372036854775807", "-92233720368547758.07"),
])
def test_signed_int64_usd_cents_follow_documented_prepaid_ledger(cents, dollars):
    assert prepaid_usd({"total": {"val": cents}, "changes": [{"amount": {"val": "unneeded"}}]}) == Decimal(dollars)


@pytest.mark.parametrize("payload", [
    None, [], {}, {"total": None}, {"total": {}}, {"total": {"val": None}},
    *({"total": {"val": value}} for value in [True, 0, 1.5, "", " 0", "1.5", "NaN", "Infinity", "1e3", "9223372036854775808", "-9223372036854775809"]),
])
def test_missing_or_invalid_total_is_not_a_fabricated_zero(payload):
    with pytest.raises(ValueError):
        prepaid_usd(payload)


def mock_connect(monkeypatch, *, validation=VALIDATION, validation_status=200, balance=None, balance_status=200):
    seen = []

    def get(url, *, headers, timeout, follow_redirects):
        assert headers["Authorization"] == f"Bearer {KEY}"
        assert timeout == 15.0 and follow_redirects is False
        seen.append(url)
        if url == VALIDATION_URL:
            return httpx.Response(validation_status, json=validation, request=httpx.Request("GET", url))
        assert url == balance_url(TEAM)
        return httpx.Response(balance_status, json=balance if balance is not None else {"total": {"val": "-1000"}},
                              request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    return seen


def test_connect_probes_only_two_read_endpoints_then_saves_separate_encrypted_slot(offline, monkeypatch):
    offline.records["xai"] = ConnectionRecord(provider="xai", status="connected", secret={"api_key": "fixture-inference"})
    seen = mock_connect(monkeypatch)
    adapter = XaiManagementAdapter(None)
    assert adapter.set_api_key(f"  {KEY}  ").ok
    assert seen == [VALIDATION_URL, balance_url(TEAM)]
    [saved] = offline.saved
    assert saved.provider == CONNECTION_ID and saved.meta == {"team_id": TEAM}
    assert saved.secret == {"management_key": KEY}
    assert offline.records["xai"].secret == {"api_key": "fixture-inference"}
    payload = CredentialStore._payload(saved)
    assert payload["secret_enc"].startswith("enc:v1:")
    assert KEY not in json.dumps(payload) and KEY not in json.dumps(saved.public())
    assert json.loads(crypto.decrypt(payload["secret_enc"])) == {"management_key": KEY}
    descriptor = adapter.descriptor()
    assert descriptor["category"] == "usage"
    assert "team-scoped" in descriptor["credential_help"]
    assert KEY not in json.dumps(descriptor)


@pytest.mark.parametrize("validation_status,balance_status,balance", [
    (401, 200, None), (403, 200, None), (429, 200, None), (302, 200, None),
    (200, 401, None), (200, 403, None), (200, 429, None), (200, 503, None),
    (200, 200, {}),
])
def test_bad_validation_or_missing_read_permission_never_overwrites_saved_connection(
        offline, monkeypatch, validation_status, balance_status, balance):
    original = record(secret={"management_key": "previous-fixture"})
    offline.records[CONNECTION_ID] = original
    mock_connect(monkeypatch, validation_status=validation_status, balance_status=balance_status, balance=balance)
    result = XaiManagementAdapter(None).set_api_key(KEY)
    assert not result.ok and KEY not in result.detail
    assert offline.saved == [] and offline.records[CONNECTION_ID] is original


def test_org_scope_rejected_without_a_balance_read_or_persist(offline, monkeypatch):
    seen = mock_connect(monkeypatch, validation={"scope": "SCOPE_ORGANIZATION", "scopeId": "organization", "teamId": TEAM})
    assert not XaiManagementAdapter(None).set_api_key(KEY).ok
    assert seen == [VALIDATION_URL] and not offline.saved


def test_connection_http_route_and_inference_broker_separation(offline, monkeypatch):
    mock_connect(monkeypatch)
    monkeypatch.setattr(provider_routes, "get_service", lambda: SimpleNamespace(config=None))
    monkeypatch.delenv("BRIDGE_CLAUDE_TOKEN_SECRET", raising=False)
    app = FastAPI()
    app.include_router(provider_routes.router)
    client = TestClient(app)
    connected = client.post(f"/v1/providers/{CONNECTION_ID}/connect/api-key", json={"api_key": KEY})
    assert connected.status_code == 200 and KEY not in connected.text
    assert client.get(f"/v1/providers/{CONNECTION_ID}/token").status_code == 404
    assert KEY not in client.get(f"/v1/providers/{CONNECTION_ID}").text
    assert not isinstance(get_adapter(None, CONNECTION_ID), ApiKeyAdapter)
    assert get_adapter(None, CONNECTION_ID).access_path_keys == ()
    assert access_path_adapter(None, "xai-chat").id == "xai"
    assert access_path_adapter(None, "xai-responses").id == "xai"
    assert resolve_api_key(None, CONNECTION_ID) is None
    assert resolve_api_key(None, "xai") is None  # reporting key never supplies inference
    offline.records["xai"] = ConnectionRecord(provider="xai", status="connected", secret={"api_key": "fixture-inference"})
    assert resolve_api_key(None, "xai") == "fixture-inference"


def mock_report(monkeypatch, *, payload=None, status=200, error=None):
    seen = []

    async def get(_self, url, *, headers):
        seen.append(url)
        assert url == balance_url(TEAM) and headers["Authorization"] == f"Bearer {KEY}"
        if error:
            raise error
        return httpx.Response(status, json=payload if payload is not None else {"total": {"val": "-12345"}},
                              request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    return seen


@pytest.mark.asyncio
@pytest.mark.parametrize("cents,remaining", [("-12345", 123.45), ("0", 0), ("150", -1.5)])
async def test_native_balance_including_zero_is_authoritative_without_inference_key(offline, monkeypatch, cents, remaining):
    offline.records[CONNECTION_ID] = record()
    mock_report(monkeypatch, payload={"total": {"val": cents}, "changes": [{"private": KEY}]})
    snap = await XaiUsageAdapter().fetch()
    assert snap.available and snap.status == "ok" and snap.balance_supported and not snap.manual_balance_supported
    assert snap.balance.remaining_usd == remaining
    assert snap.balance.source == "provider_reported"
    assert snap.balance.amount_usd is None and snap.balance.used_usd is None and snap.balance.as_of is None
    assert snap.balance.updated_at is not None and snap.limits == [] and snap.raw is None
    assert KEY not in snap.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize("status,expected", [(401, "unauthorized"), (403, "unauthorized"), (429, "rate_limited"), (503, "error"), (302, "error")])
async def test_native_errors_never_become_zero_or_manual_fallback(offline, monkeypatch, status, expected):
    offline.records[CONNECTION_ID] = record()
    mock_report(monkeypatch, status=status, payload={"secret_echo": KEY})
    snap = await XaiUsageAdapter().fetch()
    assert not snap.available and snap.status == expected and snap.balance is None
    assert snap.balance_supported and not snap.manual_balance_supported
    assert KEY not in snap.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{}, {"total": {"val": "NaN"}}, [], {"total": {"val": 0}}])
async def test_native_parse_failure_is_unknown_not_zero(offline, monkeypatch, payload):
    offline.records[CONNECTION_ID] = record()
    mock_report(monkeypatch, payload=payload)
    snap = await XaiUsageAdapter().fetch()
    assert snap.status == "error" and snap.balance is None and not snap.manual_balance_supported


@pytest.mark.asyncio
async def test_network_errors_do_not_echo_key_or_provider_body(offline, monkeypatch, caplog):
    offline.records[CONNECTION_ID] = record()
    mock_report(monkeypatch, error=httpx.ConnectError(f"private failure {KEY}"))
    snap = await XaiUsageAdapter().fetch()
    assert snap.status == "error" and snap.balance is None
    assert KEY not in snap.model_dump_json() and KEY not in caplog.text
    monkeypatch.setattr(httpx, "get", lambda *_a, **_k: (_ for _ in ()).throw(httpx.ConnectError(KEY)))
    result = XaiManagementAdapter(None).set_api_key(KEY)
    assert not result.ok and KEY not in result.detail and not offline.saved


@pytest.mark.asyncio
@pytest.mark.parametrize("changes", [
    {"status": "error"}, {"secret": {}}, {"meta": {"team_id": "../bad"}},
])
async def test_configured_but_unusable_connection_does_not_enable_manual_mode(offline, changes):
    offline.records[CONNECTION_ID] = record(**changes)
    adapter = XaiUsageAdapter()
    assert adapter.balance_mode == "provider" and not adapter.capabilities()["manual_balance_supported"]
    snap = await adapter.fetch()
    assert not snap.available and snap.balance is None and not snap.manual_balance_supported


@pytest.mark.asyncio
async def test_store_outage_does_not_enable_manual_mode_or_return_cached_credit(offline, monkeypatch):
    offline.records[CONNECTION_ID] = record()
    mock_report(monkeypatch)
    assert (await usage.get_usage("xai")).balance.remaining_usd == 123.45
    offline.available = False
    snap = await usage.get_usage("xai")
    assert snap.status == "error" and snap.balance is None and not snap.manual_balance_supported
    assert usage.get_cached("xai") is None
    with pytest.raises(HTTPException) as error:
        usage_routes._anchored_provider("xai")
    assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_manual_mode_survives_without_management_and_anchor_is_live(offline, monkeypatch):
    now = datetime.now(timezone.utc)
    offline.records["xai"] = ConnectionRecord(provider="xai", status="connected", connected_at=(now - timedelta(days=1)).isoformat())
    rows = [BalanceRow("xai", Decimal(100), now, Decimal(3), 0, now)]

    class Counter:
        def read(self, _provider):
            return rows[0]

        def daily_spend(self, *_args, **_kwargs):
            return []

        def spend(self, *_args, **_kwargs):
            return (Decimal(0),) * 3

    monkeypatch.setattr(usage, "_balance_store", Counter)
    first = await usage.get_usage("xai")
    assert first.manual_balance_supported and first.balance.remaining_usd == 97
    rows[0] = BalanceRow("xai", Decimal(100), now, Decimal(10), 0, now)
    second = await usage.get_usage("xai")
    assert second.cached and second.balance.remaining_usd == 90
    assert second.balance.source == "bawt_recorded_estimate"


@pytest.mark.asyncio
async def test_manual_to_native_cache_switch_never_double_subtracts_or_reuses_manual_report(offline, monkeypatch):
    offline.records["xai"] = ConnectionRecord(provider="xai", status="connected")
    monkeypatch.setattr(usage, "_balance_store", lambda: (_ for _ in ()).throw(RuntimeError("ledger unavailable")))
    initial = await usage.get_usage("xai")
    assert initial.manual_balance_supported
    offline.records[CONNECTION_ID] = record()
    seen = mock_report(monkeypatch)
    snap = await usage.get_usage("xai")
    assert not snap.cached and not snap.manual_balance_supported
    assert snap.balance.remaining_usd == 123.45 and snap.balance.used_usd is None
    assert len(seen) == 1
    cached = await usage.get_usage("xai")
    assert cached.cached and cached.balance.remaining_usd == 123.45 and len(seen) == 1
    assert balance_update_for_turn(None, {"access_path": {"key": "xai-chat"}, "total_cost_usd": 5})["refresh"]
    with pytest.raises(HTTPException) as error:
        usage_routes._anchored_provider("xai")
    assert error.value.status_code == 409
    del offline.records[CONNECTION_ID]
    manual = await usage.get_usage("xai")
    assert manual.manual_balance_supported and manual.balance is None and not manual.cached


@pytest.mark.asyncio
async def test_key_rotation_cannot_reuse_old_account_credit_even_when_new_read_is_rate_limited(offline, monkeypatch):
    offline.records[CONNECTION_ID] = record()
    mock_report(monkeypatch)
    assert (await usage.get_usage("xai")).balance.remaining_usd == 123.45
    offline.records[CONNECTION_ID] = record(secret={"management_key": "rotated-fixture"})

    async def limited(_self, url, *, headers):
        assert headers["Authorization"] == "Bearer rotated-fixture"
        return httpx.Response(429, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", limited)
    snap = await usage.get_usage("xai")
    assert snap.status == "rate_limited" and snap.balance is None and not snap.cached


@pytest.mark.asyncio
async def test_failed_refresh_cannot_resurrect_old_good_credit_on_the_next_read(offline, monkeypatch):
    offline.records[CONNECTION_ID] = record()
    mock_report(monkeypatch)
    await usage.get_usage("xai")
    seen = mock_report(monkeypatch, status=403)
    failed = await usage.get_usage("xai", force=True)
    again = await usage.get_usage("xai")
    assert failed.status == again.status == "unauthorized"
    assert failed.balance is again.balance is None
    assert not again.cached and len(seen) == 2


@pytest.mark.asyncio
async def test_same_source_rate_limit_shows_explicitly_cached_last_good_not_live(offline, monkeypatch):
    offline.records[CONNECTION_ID] = record()
    mock_report(monkeypatch)
    await usage.get_usage("xai")
    mock_report(monkeypatch, status=429)
    snap = await usage.get_usage("xai", force=True)
    assert snap.cached and snap.balance.remaining_usd == 123.45
    assert "rate-limited" in snap.error and "cached" in snap.error


def test_usage_http_reports_native_zero_and_refuses_manual_writes(offline, monkeypatch):
    offline.records[CONNECTION_ID] = record()
    mock_report(monkeypatch, payload={"total": {"val": "0"}})
    monkeypatch.setattr(usage_routes, "set_balance", lambda *_: pytest.fail("Native credit must not accept an anchor"))
    monkeypatch.setattr(usage_routes, "clear_balance", lambda *_: pytest.fail("Native credit must not clear an anchor"))
    app = FastAPI()
    app.include_router(usage_routes.router)
    client = TestClient(app)
    response = client.get("/v1/usage?provider=xai")
    assert response.status_code == 200
    assert response.json()["balance"]["remaining_usd"] == 0
    assert response.json()["balance"]["source"] == "provider_reported"
    assert response.json()["manual_balance_supported"] is False
    assert client.put("/v1/usage/xai/balance", json={"amount_usd": 10}).status_code == 409
    assert client.delete("/v1/usage/xai/balance").status_code == 409


@pytest.mark.asyncio
async def test_switch_during_inflight_read_does_not_publish_old_credit(offline, monkeypatch):
    offline.records[CONNECTION_ID] = record()

    async def switching(_self, url, *, headers):
        offline.records[CONNECTION_ID] = record(meta={"team_id": "new-team"})
        return httpx.Response(200, json={"total": {"val": "-12345"}}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", switching)
    snap = await usage.get_usage("xai")
    assert snap.status == "error" and snap.balance is None and not snap.cached
    assert usage.get_cached("xai") is None
