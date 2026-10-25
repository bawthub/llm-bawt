"""Anthropic pay-per-token usage, credential isolation, and cost normalization."""

import json
from datetime import datetime, timezone
from decimal import Decimal

import httpx
import pytest
from fastapi import HTTPException

from llm_bawt.service.routes import settings as setting_routes
from llm_bawt.service import usage as usage_registry
from llm_bawt.service.usage.canonical import UsageBalance, UsageSpend

from llm_bawt.service.providers.anthropic_key import AnthropicKeyAdapter
from llm_bawt.service.usage.adapters.anthropic_api import (
    AnthropicApiUsageAdapter,
    cost_totals,
    read_local_spend,
)


def test_balance_anchor_requires_global_positive_amount_and_stamps_server_time(monkeypatch):
    class NoopPolicy:
        def __init__(self, _engine):
            pass

        def validate_setting_write(self, *_args):
            pass

    monkeypatch.setattr(setting_routes, "WorkingMemoryPolicy", NoopPolicy)
    key = "anthropic_api_balance_anchor"
    before = datetime.now(timezone.utc)
    value = {"amount_usd": 157}
    setting_routes._validate_setting_writes(None, [("global", "*", key, value)])
    stored = setting_routes._setting_value_for_storage(key, value)
    assert stored["amount_usd"] == "157"
    assert before <= datetime.fromisoformat(stored["as_of"]) <= datetime.now(timezone.utc)
    assert value == {"amount_usd": 157}  # caller cannot choose the timestamp
    for scope, bad in [("bot", value), ("global", {"amount_usd": 0}),
                       ("global", {"amount_usd": -1}), ("global", {"amount_usd": "NaN"}),
                       ("global", {"amount_usd": 157, "as_of": "yesterday"})]:
        with pytest.raises(HTTPException):
            setting_routes._validate_setting_writes(None, [(scope, "*", key, bad)])
    with pytest.raises(HTTPException, match="replaced"):
        setting_routes._validate_setting_writes(None, [("global", "*", "anthropic_api_monthly_budget_usd", 25)])


def test_settings_put_persists_one_server_stamped_anchor_and_invalidates_usage(monkeypatch):
    from types import SimpleNamespace
    from llm_bawt.service.schemas import RuntimeSettingUpsertRequest

    class NoopPolicy:
        def __init__(self, _engine):
            pass

        def validate_setting_write(self, *_args):
            pass

    writes = []
    invalidated = []
    store = SimpleNamespace(engine=object(), set_value=lambda *args: writes.append(args))
    service = SimpleNamespace(config=object(), _default_bot="example",
                              invalidate_all_instances=lambda: 0)
    monkeypatch.setattr(setting_routes, "WorkingMemoryPolicy", NoopPolicy)
    monkeypatch.setattr(setting_routes, "get_service", lambda: service)
    monkeypatch.setattr(setting_routes, "get_runtime_settings_store", lambda _config: store)
    monkeypatch.setattr(setting_routes, "_invalidate_balance_report", invalidated.append)
    response = setting_routes.upsert_runtime_setting(RuntimeSettingUpsertRequest(
        scope_type="global", key="anthropic_api_balance_anchor", value={"amount_usd": 157}))
    assert writes == [("global", "*", "anthropic_api_balance_anchor", response["value"])]
    assert invalidated == ["anthropic_api_balance_anchor"]
    assert datetime.fromisoformat(response["value"]["as_of"]).tzinfo is not None


@pytest.fixture
def report():
    return {
        "data": [
            {
                "starting_at": "2026-10-24T00:00:00Z",
                "results": [
                    {"amount": "123.78912", "currency": "USD", "cost_type": "tokens", "model": "claude-a", "token_type": "output_tokens"},
                    {"amount": "7.21088", "currency": "USD", "cost_type": "tokens", "model": "claude-a", "token_type": "input_tokens"},
                ],
            },
            {
                "starting_at": "2026-10-25T00:00:00Z",
                "results": [
                    {"amount": "29.5", "currency": "USD", "cost_type": "tokens", "model": "claude-b", "token_type": "output_tokens"},
                    {"amount": "0.5", "currency": "USD", "cost_type": "code_execution", "model": None},
                ],
            },
        ],
        "has_more": False,
        "next_page": None,
    }


def test_report_decimal_cents_convert_without_rounding_error(report):
    now = datetime(2026, 10, 25, 16, tzinfo=timezone.utc)
    today, seven, month = cost_totals(report["data"], now=now)
    assert today == Decimal("0.30")
    assert seven == Decimal("1.61")
    assert month == Decimal("1.61")


def test_inference_probe_rejects_admin_key_without_network(monkeypatch):
    def no_http(*_args, **_kwargs):
        raise AssertionError("admin key must not reach the inference endpoint")

    monkeypatch.setattr(httpx, "get", no_http)
    account, detail = AnthropicKeyAdapter._probe("sk-ant-admin01-example")
    assert account is None
    assert "Admin key" in detail
    assert "Usage reporting key" in detail


@pytest.mark.asyncio
async def test_admin_cost_report_paginates_without_unused_requests_or_inference_key(monkeypatch, report):
    seen = []
    pages = [
        {**report, "has_more": True, "next_page": "page1"},
        {"data": [], "has_more": False, "next_page": None},
    ]

    async def fake_get(self, url, *, params, headers):
        seen.append((url, dict(params), dict(headers)))
        assert "cost_report" in url, "unused usage endpoint must not delay or break spend reporting"
        return httpx.Response(200, json=pages.pop(0), request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    adapter = AnthropicApiUsageAdapter()
    monkeypatch.setattr(adapter, "_admin_key", lambda: "sk-ant-admin01-report")
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(adapter, "_anchor_and_connection", lambda: (Decimal("20"), now, now))
    monkeypatch.setattr(adapter, "_local_spend", lambda now, since: (Decimal(0),) * 4)
    snap = await adapter.fetch()
    assert snap.status == "ok"
    assert snap.spend is not None
    assert snap.spend.source == "anthropic_reported"
    assert snap.balance is not None and snap.balance.remaining_usd == 20.0
    assert len(seen) == 2
    assert seen[1][1]["page"] == "page1"
    assert all(call[2]["x-api-key"] == "sk-ant-admin01-report" for call in seen)
    assert all("cost_report" in call[0] for call in seen)


@pytest.mark.asyncio
async def test_rejected_admin_report_never_falls_back_to_mislabelled_estimate(monkeypatch):
    async def forbidden(self, url, *, params, headers):
        return httpx.Response(403, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", forbidden)
    adapter = AnthropicApiUsageAdapter()
    monkeypatch.setattr(adapter, "_admin_key", lambda: "sk-ant-admin01-report")
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(adapter, "_anchor_and_connection", lambda: (None, None, now))
    monkeypatch.setattr(adapter, "_local_spend", lambda now, since: (Decimal(0),) * 4)
    snap = await adapter.fetch()
    assert snap.kind == "spend" and snap.status == "unauthorized"
    assert snap.spend is None and not snap.available


@pytest.mark.asyncio
async def test_bad_admin_key_cannot_hide_a_valid_recorded_balance(monkeypatch):
    async def forbidden(self, url, *, params, headers):
        return httpx.Response(403, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", forbidden)
    adapter = AnthropicApiUsageAdapter()
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(adapter, "_admin_key", lambda: "sk-ant-admin01-bad")
    monkeypatch.setattr(adapter, "_anchor_and_connection", lambda: (Decimal("157"), now, now))
    monkeypatch.setattr(adapter, "_local_spend", lambda _now, since: (Decimal(0), Decimal(0), Decimal(0), Decimal("2.50")))
    snap = await adapter.fetch()
    assert snap.status == "unauthorized" and snap.spend is None
    assert snap.balance.remaining_usd == 154.50
    assert "local balance" in snap.error


@pytest.mark.asyncio
async def test_rate_limited_admin_report_keeps_fresh_local_balance_not_cached_old_one(monkeypatch):
    import time

    adapter = usage_registry.usage_adapter_for_access_path("anthropic-api")
    old_balance = UsageBalance(amount_usd=157, remaining_usd=150, used_usd=7, as_of=1)
    fresh_balance = UsageBalance(amount_usd=157, remaining_usd=145, used_usd=12, as_of=1)
    old = adapter._base(status="ok", available=True, balance=old_balance,
                        spend=UsageSpend(today_usd=7, seven_day_usd=7, month_usd=7,
                                         source="anthropic_reported"))
    monkeypatch.setitem(usage_registry._cache, "anthropic_api", (time.monotonic() - 300, old))

    async def report_rate_limited():
        return adapter._base(status="rate_limited", balance=fresh_balance,
                             error="Admin report returned HTTP 429; local balance remains available")

    monkeypatch.setattr(adapter, "fetch", report_rate_limited)
    snap = await usage_registry.get_usage("anthropic_api", force=True)
    assert snap.status == "rate_limited" and snap.balance.remaining_usd == 145
    assert snap.spend is None
    assert usage_registry.get_cached("anthropic_api").balance.remaining_usd == 150
    usage_registry.invalidate_cache("anthropic_api")
    assert usage_registry.get_cached("anthropic_api") is None


@pytest.mark.asyncio
async def test_no_admin_key_uses_only_recorded_anthropic_api_cost(monkeypatch):
    adapter = AnthropicApiUsageAdapter()
    monkeypatch.setattr(adapter, "_admin_key", lambda: None)
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(adapter, "_anchor_and_connection", lambda: (Decimal("157"), now, now))
    monkeypatch.setattr(adapter, "_local_spend", lambda now, since: (Decimal("3.25"), Decimal("7.5"), Decimal("12"), Decimal("44.70")))
    snap = await adapter.fetch()
    assert snap.status == "ok"
    assert snap.spend.source == "bawt_recorded_estimate"
    assert snap.spend.today_usd == 3.25
    assert snap.balance.remaining_usd == 112.30
    assert snap.balance.used_usd == 44.70
    assert snap.limits == []


def test_local_spend_excludes_pre_connection_pre_anchor_and_legacy_paths():
    from datetime import timedelta

    now = datetime(2026, 10, 25, 16, tzinfo=timezone.utc)
    anchor = now - timedelta(hours=1)

    class FakeConnection:
        def execute(self, _statement, values):
            assert values["start"] == anchor and values["end"] == now
            # Include out-of-range rows deliberately: the parser also checks the
            # time boundary, rather than relying solely on the SQL mock.
            return [
                (anchor - timedelta(days=23), json.dumps({"total_cost_usd": 2925, "access_path": {"key": "anthropic-api"}})),
                (anchor, json.dumps({"total_cost_usd": 100, "access_path": {"key": "anthropic-api"}})),
                (now, json.dumps({"total_cost_usd": 3.25, "access_path": {"key": "anthropic-api"}})),
                (now, json.dumps({"total_cost_usd": 20, "access_path": {"key": "anthropic-oauth"}})),
                (now, json.dumps({"total_cost_usd": 12, "access_path": {"key": "<none>"}})),
                (now, json.dumps({"total_cost_usd": 90})),
                (now, "not json"),
            ]

    class FakeEngine:
        def connect(self):
            from contextlib import nullcontext
            return nullcontext(FakeConnection())

    totals = read_local_spend(FakeEngine(), now, since=anchor)
    assert totals == (Decimal("3.25"),) * 4


def test_no_anchor_does_not_expose_a_fabricated_zero_balance(monkeypatch):
    import asyncio

    adapter = AnthropicApiUsageAdapter()
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(adapter, "_admin_key", lambda: None)
    monkeypatch.setattr(adapter, "_anchor_and_connection", lambda: (None, None, now))
    monkeypatch.setattr(adapter, "_local_spend", lambda _now, since: (Decimal(0),) * 4)
    snapshot = asyncio.run(adapter.fetch())
    assert snapshot.status == "ok" and snapshot.balance is None
    assert snapshot.spend.month_usd == 0


def test_missing_connection_timestamp_fails_closed(monkeypatch):
    import asyncio

    adapter = AnthropicApiUsageAdapter()
    monkeypatch.setattr(adapter, "_admin_key", lambda: None)
    monkeypatch.setattr(adapter, "_anchor_and_connection", lambda: (Decimal("157"), datetime.now(timezone.utc), None))
    snapshot = asyncio.run(adapter.fetch())
    assert snapshot.status == "error" and snapshot.balance is None and snapshot.spend is None


def test_connection_timestamp_is_floor_even_if_anchor_is_older(monkeypatch):
    import asyncio
    from datetime import timedelta

    adapter = AnthropicApiUsageAdapter()
    now = datetime.now(timezone.utc)
    connection = now - timedelta(hours=1)
    monkeypatch.setattr(adapter, "_admin_key", lambda: None)
    monkeypatch.setattr(adapter, "_anchor_and_connection", lambda: (Decimal("157"), now - timedelta(days=2), connection))
    observed = []
    def local_spend(_now, since):
        observed.append(since)
        return (Decimal("0.3"), Decimal("0.3"), Decimal("0.3"), Decimal("0.3"))
    monkeypatch.setattr(adapter, "_local_spend", local_spend)
    snapshot = asyncio.run(adapter.fetch())
    assert observed == [connection] and snapshot.balance.remaining_usd == 156.70
