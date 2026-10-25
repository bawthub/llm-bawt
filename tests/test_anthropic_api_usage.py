"""Anthropic pay-per-token usage, credential isolation, and cost normalization."""

import json
from datetime import datetime, timezone
from decimal import Decimal

import httpx
import pytest

from llm_bawt.service.providers.anthropic_key import AnthropicKeyAdapter
from llm_bawt.service.usage.adapters.anthropic_api import (
    AnthropicApiUsageAdapter,
    cost_totals,
    read_local_spend,
)


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
async def test_admin_report_paginates_and_never_uses_inference_key(monkeypatch, report):
    seen = []
    pages = [
        {**report, "has_more": True, "next_page": "page1"},
        {"data": [], "has_more": False, "next_page": None},
    ]

    async def fake_get(self, url, *, params, headers):
        seen.append((url, dict(params), dict(headers)))
        if "cost_report" in url:
            return httpx.Response(200, json=pages.pop(0), request=httpx.Request("GET", url))
        return httpx.Response(200, json={"data": [], "has_more": False}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    adapter = AnthropicApiUsageAdapter()
    monkeypatch.setattr(adapter, "_admin_key", lambda: "sk-ant-admin01-report")
    monkeypatch.setattr(adapter, "_budget", lambda: Decimal("20"))
    snap = await adapter.fetch()
    assert snap.status == "ok"
    assert snap.spend is not None
    assert snap.spend.source == "anthropic_reported"
    assert snap.spend.today_usd is not None
    assert snap.spend.monthly_budget_usd == 20.0
    assert len(seen) == 3
    assert seen[1][1]["page"] == "page1"
    assert all(call[2]["x-api-key"] == "sk-ant-admin01-report" for call in seen)
    assert any("usage_report/messages" in call[0] for call in seen)


@pytest.mark.asyncio
async def test_no_admin_key_uses_only_recorded_anthropic_api_cost(monkeypatch):
    adapter = AnthropicApiUsageAdapter()
    monkeypatch.setattr(adapter, "_admin_key", lambda: None)
    monkeypatch.setattr(adapter, "_budget", lambda: None)
    monkeypatch.setattr(adapter, "_local_spend", lambda now: (Decimal("3.25"), Decimal("7.5"), Decimal("12")))
    snap = await adapter.fetch()
    assert snap.status == "ok"
    assert snap.spend.source == "bawt_recorded_estimate"
    assert snap.spend.today_usd == 3.25
    assert snap.limits == []


def test_local_spend_filters_by_frozen_access_path_not_model_prefix():
    now = datetime(2026, 10, 25, 16, tzinfo=timezone.utc)

    class FakeConnection:
        def execute(self, _statement, values):
            assert values["model_suffix"] == "%@anthropic-api"
            return [
                (now, "claude-a@anthropic-api", json.dumps({"total_cost_usd": 3.25, "access_path": {"key": "anthropic-api"}})),
                (now, "claude-b@anthropic-api", json.dumps({"total_cost_usd": 20, "access_path": {"key": "anthropic-oauth"}})),
                (now, "claude-c@anthropic-oauth", json.dumps({"total_cost_usd": 0.5, "access_path": {"key": "anthropic-api"}})),
                (now, "claude-d", json.dumps({"total_cost_usd": 90})),
                (now, "claude-e@anthropic-api", "not json"),
            ]

    class FakeEngine:
        def connect(self):
            from contextlib import nullcontext
            return nullcontext(FakeConnection())

    totals = read_local_spend(FakeEngine(), now)
    assert totals == (Decimal("3.75"),) * 3
