"""Pay-per-token usage: Anthropic API spend, balance contract, credential isolation."""

import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import httpx
import pytest
from fastapi import HTTPException

from llm_bawt.service import usage as usage_registry
from llm_bawt.service.providers.anthropic_key import AnthropicKeyAdapter
from llm_bawt.service.routes import settings as setting_routes
from llm_bawt.service.routes import usage as usage_routes
from llm_bawt.service.usage.adapters.anthropic_api import AnthropicApiUsageAdapter, cost_totals
from llm_bawt.service.usage.adapters.moonshot import MoonshotUsageAdapter
from llm_bawt.service.usage.balance import BalanceRow, anchored_balance, balance_update_for_turn
from llm_bawt.service.usage.canonical import UsageSpend

NOW = datetime(2026, 10, 25, 16, tzinfo=timezone.utc)


def _row(*, anchor="100", spent="2.5", unpriced=0, anchor_at=NOW):
    return BalanceRow("anthropic_api", Decimal(anchor), anchor_at, Decimal(spent), unpriced,
                      anchor_at + timedelta(minutes=5))


class _Store:
    def __init__(self, row=None, spend=(Decimal(0),) * 3):
        self.row, self._spend, self.spend_calls = row, spend, []

    def read(self, _provider):
        return self.row

    def spend(self, provider, *, since, starts):
        self.spend_calls.append((provider, since, starts))
        return self._spend


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


# -- canonical balance ------------------------------------------------------


def test_anchored_balance_is_one_row_read_with_every_field_set():
    balance, reason = anchored_balance(_row(), NOW - timedelta(days=1))
    assert reason is None
    assert balance.model_dump() == {
        "amount_usd": 100.0, "remaining_usd": 97.5, "used_usd": 2.5,
        "as_of": int(NOW.timestamp()), "source": "bawt_recorded_estimate",
        "updated_at": (NOW + timedelta(minutes=5)).timestamp(),
    }


def test_no_anchor_is_no_balance_not_a_fabricated_zero():
    assert anchored_balance(None, NOW) == (None, None)


def test_unpriced_turn_fails_closed_instead_of_charging_a_guess():
    balance, reason = anchored_balance(_row(unpriced=2), None)
    assert balance is None and "2 API turn(s)" in reason


def test_key_reconnected_after_anchor_voids_the_balance():
    balance, reason = anchored_balance(_row(), NOW + timedelta(seconds=1))
    assert balance is None and "key changed" in reason


def test_provider_reported_balance_invents_no_deposit_debit_or_anchor(monkeypatch):
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"code": 0, "data": {"available_balance": 49.53}}

    async def fake_get(self, url, *, headers):
        return Response()

    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test")
    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    import asyncio

    snap = asyncio.run(MoonshotUsageAdapter().fetch())
    assert snap.kind == "balance" and snap.balance_supported and snap.status == "ok"
    balance = snap.balance
    assert (balance.remaining_usd, balance.source) == (49.53, "provider_reported")
    assert balance.amount_usd is None and balance.used_usd is None and balance.as_of is None
    assert balance.updated_at is not None


# -- registry overlay: balances are never cached ----------------------------


@pytest.mark.asyncio
async def test_cached_report_always_gets_a_fresh_counter_read(monkeypatch):
    adapter = usage_registry.get_adapter("anthropic_api")
    report = adapter._base(status="ok", available=True,
                           spend=UsageSpend(today_usd=1, seven_day_usd=1, month_usd=1,
                                            source="bawt_recorded_estimate"))
    monkeypatch.setitem(usage_registry._cache, "anthropic_api", (time.monotonic(), report))
    store = _Store(_row(spent="2.5"))
    monkeypatch.setattr(usage_registry, "_balance_store", lambda: store)
    monkeypatch.setattr(adapter, "credential_connected_at", lambda: None)
    first = await usage_registry.get_usage("anthropic_api")
    store.row = _row(spent="7")
    second = await usage_registry.get_usage("anthropic_api")
    assert first.cached and second.cached  # the spend report came from cache...
    assert (first.balance.remaining_usd, second.balance.remaining_usd) == (97.5, 93.0)  # ...the balance did not
    assert usage_registry.get_cached("anthropic_api").balance is None


@pytest.mark.asyncio
async def test_unreadable_counter_reports_unknown_credit_not_zero(monkeypatch):
    adapter = usage_registry.get_adapter("anthropic_api")
    monkeypatch.setitem(usage_registry._cache, "anthropic_api",
                        (time.monotonic(), adapter._base(status="ok", available=True)))

    def broken():
        raise RuntimeError("db down")

    monkeypatch.setattr(usage_registry, "_balance_store", broken)
    monkeypatch.setattr(adapter, "credential_connected_at", lambda: None)
    snap = await usage_registry.get_usage("anthropic_api")
    assert snap.balance is None and snap.balance_supported
    assert snap.error == "API balance unavailable"


# -- Anthropic spend detail (Admin report preserved, ledger otherwise) ------


def test_report_decimal_cents_convert_without_rounding_error(report):
    today, seven, month = cost_totals(report["data"], now=NOW)
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
    monkeypatch.setattr(adapter, "_ledger_spend",
                        lambda *_: (_ for _ in ()).throw(AssertionError("Admin report replaces the ledger")))
    snap = await adapter.fetch()
    assert snap.status == "ok" and snap.kind == "balance" and snap.balance_supported
    assert snap.spend.source == "anthropic_reported"
    assert snap.balance is None  # the counter overlay owns the balance
    assert len(seen) == 2
    assert seen[1][1]["page"] == "page1"
    assert all(call[2]["x-api-key"] == "sk-ant-admin01-report" for call in seen)


@pytest.mark.asyncio
async def test_rejected_admin_report_never_falls_back_to_mislabelled_estimate(monkeypatch):
    async def forbidden(self, url, *, params, headers):
        return httpx.Response(403, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", forbidden)
    adapter = AnthropicApiUsageAdapter()
    monkeypatch.setattr(adapter, "_admin_key", lambda: "sk-ant-admin01-report")
    snap = await adapter.fetch()
    assert snap.status == "unauthorized"
    assert snap.spend is None and not snap.available


@pytest.mark.asyncio
async def test_bad_admin_key_cannot_hide_a_valid_recorded_balance(monkeypatch):
    async def forbidden(self, url, *, params, headers):
        return httpx.Response(403, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", forbidden)
    adapter = usage_registry.get_adapter("anthropic_api")
    monkeypatch.setattr(adapter, "_admin_key", lambda: "sk-ant-admin01-bad")
    monkeypatch.setattr(adapter, "credential_connected_at", lambda: NOW - timedelta(days=1))
    monkeypatch.setattr(usage_registry, "_balance_store", lambda: _Store(_row(spent="2.5")))
    usage_registry.invalidate_cache("anthropic_api")
    snap = await usage_registry.get_usage("anthropic_api", force=True)
    assert snap.status == "unauthorized" and snap.spend is None
    assert snap.balance.remaining_usd == 97.5


@pytest.mark.asyncio
async def test_no_admin_key_reads_ledger_spend_since_key_connection(monkeypatch):
    adapter = AnthropicApiUsageAdapter()
    connected = datetime.now(timezone.utc) - timedelta(days=2)
    store = _Store(spend=(Decimal("3.25"), Decimal("7.5"), Decimal("12")))
    monkeypatch.setattr(adapter, "_admin_key", lambda: None)
    monkeypatch.setattr(adapter, "credential_connected_at", lambda: connected)
    monkeypatch.setattr(usage_registry, "_balance_store", lambda: store)
    snap = await adapter.fetch()
    assert snap.status == "ok" and snap.limits == []
    assert snap.spend.source == "bawt_recorded_estimate"
    assert (snap.spend.today_usd, snap.spend.seven_day_usd, snap.spend.month_usd) == (3.25, 7.5, 12.0)
    [(provider, since, starts)] = store.spend_calls
    assert provider == "anthropic_api" and since == connected and len(starts) == 3


@pytest.mark.asyncio
async def test_missing_connection_timestamp_fails_closed(monkeypatch):
    adapter = AnthropicApiUsageAdapter()
    monkeypatch.setattr(adapter, "_admin_key", lambda: None)
    monkeypatch.setattr(adapter, "credential_connected_at", lambda: None)
    snap = await adapter.fetch()
    assert snap.status == "error" and snap.spend is None and snap.balance is None


@pytest.mark.asyncio
async def test_admin_spend_stays_available_without_inference_connection(monkeypatch, report):
    async def fake_get(self, url, *, params, headers):
        return httpx.Response(200, json=report, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    adapter = AnthropicApiUsageAdapter()
    monkeypatch.setattr(adapter, "_admin_key", lambda: "sk-ant-admin01-report")
    monkeypatch.setattr(adapter, "credential_connected_at", lambda: None)
    snap = await adapter.fetch()
    assert snap.status == "ok" and snap.spend.source == "anthropic_reported"


# -- HTTP contract ------------------------------------------------------------


@pytest.fixture
def anchors(monkeypatch):
    calls = []
    monkeypatch.setattr(usage_routes, "set_balance", lambda provider, amount: calls.append(("set", provider, amount)))
    monkeypatch.setattr(usage_routes, "clear_balance", lambda provider: calls.append(("clear", provider)))

    async def report(provider, *, force=False):
        return usage_registry.get_adapter(provider)._base(status="ok", available=True)

    monkeypatch.setattr(usage_routes, "get_usage", report)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("amount", ["157.25", 157, 0.5])
async def test_put_balance_anchors_the_amount_and_returns_the_report(anchors, amount):
    snap = await usage_routes.set_balance_endpoint(
        "anthropic_api", usage_routes.BalanceAnchorRequest(amount_usd=amount))
    assert anchors == [("set", "anthropic_api", Decimal(str(amount)))]
    assert snap.provider == "anthropic_api" and snap.kind == "balance"


@pytest.mark.asyncio
@pytest.mark.parametrize("amount", [0, -1, "NaN", "Infinity", True, None, "abc", [5], {"v": 1}, 10**9])
async def test_put_balance_rejects_non_positive_non_finite_or_absurd_amounts(anchors, amount):
    with pytest.raises(HTTPException) as err:
        await usage_routes.set_balance_endpoint(
            "anthropic_api", usage_routes.BalanceAnchorRequest(amount_usd=amount))
    assert err.value.status_code == 400 and anchors == []


@pytest.mark.asyncio
async def test_delete_balance_clears_the_anchor(anchors):
    await usage_routes.clear_balance_endpoint("xai")
    assert anchors == [("clear", "xai")]


@pytest.mark.asyncio
async def test_balance_writes_refuse_provider_reported_unknown_and_subscription_providers(anchors, monkeypatch):
    monkeypatch.setitem(usage_registry._REGISTRY, "moonshot", MoonshotUsageAdapter())
    for provider, code in [("moonshot", 409), ("claude", 409), ("nope", 404)]:
        for call in (lambda: usage_routes.set_balance_endpoint(
                         provider, usage_routes.BalanceAnchorRequest(amount_usd=5)),
                     lambda: usage_routes.clear_balance_endpoint(provider)):
            with pytest.raises(HTTPException) as err:
                await call()
            assert err.value.status_code == code
    assert anchors == []


@pytest.mark.asyncio
async def test_path_query_echoes_the_resolved_access_path(anchors):
    snap = await usage_routes.get_usage_endpoint(provider=None, access_path=" Anthropic-API ",
                                                 bot_id=None, force=False)
    assert snap.provider == "anthropic_api" and snap.access_path_key == "anthropic-api"


def test_retired_balance_settings_point_at_the_balance_endpoint(monkeypatch):
    class NoopPolicy:
        def __init__(self, _engine):
            pass

        def validate_setting_write(self, *_args):
            pass

    monkeypatch.setattr(setting_routes, "WorkingMemoryPolicy", NoopPolicy)
    for key in ("anthropic_api_balance_anchor", "anthropic_api_monthly_budget_usd"):
        with pytest.raises(HTTPException, match="/v1/usage/"):
            setting_routes._validate_setting_writes(None, [("global", "*", key, {"amount_usd": 5})])


# -- turn_complete.balance_update ---------------------------------------------


def test_balance_update_for_a_provider_reported_turn_asks_for_a_refresh(monkeypatch):
    monkeypatch.setitem(usage_registry._ACCESS_PATH_ADAPTERS, "moonshot-anthropic", MoonshotUsageAdapter())
    update = balance_update_for_turn(None, {"access_path": {"key": "moonshot-anthropic"}, "total_cost_usd": 0.1})
    assert update["provider"] == "moonshot" and update["access_path_key"] == "moonshot-anthropic"
    assert update["balance"] is None and update["refresh"] is True


def test_balance_update_for_an_anchored_turn_carries_the_committed_counter(monkeypatch):
    from llm_bawt.service.usage import balance as balance_module

    class Store:
        def __init__(self, _engine):
            pass

        def read(self, _provider):
            return _row(spent="4")

    monkeypatch.setattr(balance_module, "ApiBalanceStore", Store)
    monkeypatch.setattr(usage_registry.get_adapter("anthropic_api"), "credential_connected_at", lambda: None)
    update = balance_update_for_turn(object(), '{"access_path": {"key": "anthropic-api"}}')
    assert update["provider"] == "anthropic_api" and "refresh" not in update
    assert update["balance"]["remaining_usd"] == 96.0
    assert update["updated_at"] == update["balance"]["updated_at"]


@pytest.mark.parametrize("usage", [None, "not json", {"total_cost_usd": 1},
                                   {"access_path": {"key": "anthropic-oauth"}}])
def test_turns_off_balance_paths_get_no_balance_update(usage):
    assert balance_update_for_turn(None, usage) is None


def test_a_failed_balance_read_degrades_to_a_refresh_not_a_dropped_event(monkeypatch):
    from llm_bawt.service.usage import balance as balance_module

    class Broken:
        def __init__(self, _engine):
            raise RuntimeError("db down")

    monkeypatch.setattr(balance_module, "ApiBalanceStore", Broken)
    update = balance_update_for_turn(object(), {"access_path": {"key": "anthropic-api"}})
    assert update["provider"] == "anthropic_api" and update["refresh"] is True and update["balance"] is None


def test_balance_update_for_turn_id_reads_nothing_without_a_database():
    from llm_bawt.service.usage.balance import balance_update_for_turn_id

    assert balance_update_for_turn_id(None, "turn") is None


# -- every turn_complete producer carries balance_update ------------------------


@pytest.mark.asyncio
async def test_nonstream_turn_complete_carries_the_committed_balance(monkeypatch):
    from types import SimpleNamespace

    from llm_bawt.service.chat_nonstream import ChatNonStreamMixin
    from llm_bawt.service.usage import balance as balance_module

    published = []

    class Subscriber:
        async def publish_tool_event(self, _bot, _user, event):
            published.append(event)

    monkeypatch.setattr(balance_module, "balance_update_for_turn_id",
                        lambda engine, turn_id: {"engine": engine, "turn": turn_id})
    service = SimpleNamespace(_redis_subscriber=Subscriber(), _turn_log_store=SimpleNamespace(engine="db"))
    for kind in ("text_delta", "turn_complete"):
        await ChatNonStreamMixin._publish_nonstream_turn_event(
            service, bot_id="b", user_id="u", event={"_type": kind, "turn_id": "T"})
    assert "balance_update" not in published[0]
    assert published[1]["balance_update"] == {"engine": "db", "turn": "T"}


@pytest.mark.asyncio
async def test_abort_snapshot_turn_complete_carries_the_committed_balance(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from llm_bawt.service import turn_abort
    from llm_bawt.service.usage import balance as balance_module

    row = SimpleNamespace(id="T", bot_id="b", user_id="u", trigger_message_id=None, assistant_message_id=None,
                          status="aborted", end_reason="aborted", response_text="", token_usage_json="{usage}")
    store = SimpleNamespace(engine="db", get_turn=lambda _t: row, changed_files_summary=lambda _t: None)
    subscriber = SimpleNamespace(publish_tool_event=AsyncMock(return_value="1-0"))
    monkeypatch.setattr(balance_module, "balance_update_for_turn",
                        lambda engine, usage: {"engine": engine, "usage": usage})
    await turn_abort._publish_abort_snapshot(
        SimpleNamespace(_turn_log_store=store, _redis_subscriber=subscriber), "T")
    event = subscriber.publish_tool_event.await_args.args[2]
    assert event["_type"] == "turn_complete"
    assert event["balance_update"] == {"engine": "db", "usage": "{usage}"}


# -- generic pay-per-token API keys and the manual-balance capability -----------


def test_reports_say_whether_a_manual_balance_is_accepted():
    flags = {name: usage_registry.get_adapter(name).capabilities()
             for name in ("anthropic_api", "xai", "openai_api", "openrouter", "claude", "zai")}
    flags["moonshot"] = MoonshotUsageAdapter().capabilities()
    manual = {"balance_supported": True, "manual_balance_supported": True}
    assert {name: flags[name] for name in ("anthropic_api", "xai", "openai_api", "openrouter")} == dict.fromkeys(
        ("anthropic_api", "xai", "openai_api", "openrouter"), manual)
    assert flags["moonshot"] == {"balance_supported": True, "manual_balance_supported": False}
    assert flags["claude"] == flags["zai"] == {"balance_supported": False, "manual_balance_supported": False}


@pytest.mark.asyncio
async def test_error_and_no_balance_reports_keep_the_capability_flags(monkeypatch):
    from llm_bawt.service.usage.adapters import OpenRouterUsageAdapter

    for env in ("MOONSHOT_API_KEY", "LLM_BAWT_MOONSHOT_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    native = await MoonshotUsageAdapter().fetch()
    assert native.balance is None and native.balance_supported and not native.manual_balance_supported

    adapter = OpenRouterUsageAdapter()

    async def boom():
        raise RuntimeError("upstream exploded")

    monkeypatch.setattr(adapter, "fetch", boom)
    monkeypatch.delitem(usage_registry._cache, adapter.provider, raising=False)
    failed = await usage_registry._report(adapter, force=True)
    assert failed.status == "error" and failed.provider == "openrouter"
    assert failed.balance_supported and failed.manual_balance_supported


@pytest.mark.parametrize("provider,paths", [
    ("openai_api", ("openai-api", "openai-responses")),
    ("openrouter", ("openrouter-anthropic",)),
    ("xai", ("xai-chat", "xai-responses")),
    ("anthropic_api", ("anthropic-api",)),
])
def test_every_pay_per_token_api_key_path_reports_an_anchored_balance(provider, paths):
    from llm_bawt.service.usage.adapters import ApiKeyBalanceAdapter
    from llm_bawt.service.usage.balance import BALANCE_ANCHOR, balance_adapter_for_path

    adapter = usage_registry.get_adapter(provider)
    assert isinstance(adapter, ApiKeyBalanceAdapter) and adapter.kind == "balance"
    assert adapter.balance_mode == BALANCE_ANCHOR
    for path in paths:
        assert balance_adapter_for_path(path) is adapter


@pytest.mark.asyncio
async def test_api_key_report_follows_the_connected_credential(monkeypatch):
    from types import SimpleNamespace

    from llm_bawt.service.providers.base import STATUS_CONNECTED
    from llm_bawt.service.usage.adapters import OpenAIApiUsageAdapter, OpenRouterUsageAdapter

    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")  # never another provider's key
    openrouter = OpenRouterUsageAdapter()
    monkeypatch.setattr(openrouter, "_connection", lambda: None)
    missing = await openrouter.fetch()
    assert missing.status == "unauthorized" and not missing.available and missing.manual_balance_supported

    connected = SimpleNamespace(status=STATUS_CONNECTED, connected_at="2026-10-25T12:00:00Z")
    monkeypatch.setattr(openrouter, "_connection", lambda: connected)
    snap = await openrouter.fetch()
    assert snap.status == "ok" and snap.available and snap.limits == []
    assert openrouter.credential_connected_at() == datetime(2026, 10, 25, 12, tzinfo=timezone.utc)

    openai = OpenAIApiUsageAdapter()
    monkeypatch.setattr(openai, "_connection", lambda: None)
    assert (await openai.fetch()).status == "ok"  # the clients' legacy env key still pays
    assert openai.credential_connected_at() is None
