"""TASK-1733: background call usage normalization, pricing and the factory hook."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llm_bawt.clients.anthropic_messages_client import AnthropicMessagesClient
from llm_bawt.models.message import Message
from llm_bawt.service import background_clients as bc
from llm_bawt.service.usage import background_spend as bs


def test_chat_completions_usage_with_xai_ticks():
    usage = bs.normalize_usage({"id": "r1", "usage": {
        "prompt_tokens": 1000, "completion_tokens": 200,
        "prompt_tokens_details": {"cached_tokens": 600},
        "completion_tokens_details": {"reasoning_tokens": 150},
        "cost_in_usd_ticks": 12_345_000_000,
    }})
    assert usage == {"input_tokens": 400, "cache_read_tokens": 600, "cache_write_tokens": 0,
                     "output_tokens": 200, "reasoning_tokens": 150, "reported_cost_usd": "1.23450000"}


def test_responses_usage():
    usage = bs.normalize_usage(SimpleNamespace(id="r", usage=SimpleNamespace(model_dump=lambda: {
        "input_tokens": 50, "output_tokens": 7,
        "input_tokens_details": {"cached_tokens": 20}, "output_tokens_details": {"reasoning_tokens": 3},
    })))
    assert usage == {"input_tokens": 30, "cache_read_tokens": 20, "cache_write_tokens": 0,
                     "output_tokens": 7, "reasoning_tokens": 3}


def test_anthropic_messages_usage_with_openrouter_cost():
    usage = bs.normalize_usage({"usage": {
        "input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 100,
        "cache_creation_input_tokens": 40, "cost": 0.0021,
    }})
    assert usage == {"input_tokens": 10, "cache_read_tokens": 100, "cache_write_tokens": 40,
                     "output_tokens": 5, "reasoning_tokens": 0, "reported_cost_usd": "0.00210000"}


def test_response_without_usage_is_none():
    assert bs.normalize_usage({"id": "x"}) is None
    assert bs.normalize_usage(SimpleNamespace(usage=None)) is None


def test_reported_cost_beats_catalog_pricing():
    usage = {"input_tokens": 1_000_000, "output_tokens": 0, "reported_cost_usd": "0.5"}
    assert bs.call_cost(usage, {"input": 9}) == (Decimal("0.5"), "provider_reported")


def test_catalog_pricing_prices_every_token_class():
    usage = {"input_tokens": 1_000_000, "cache_read_tokens": 2_000_000, "cache_write_tokens": 0,
             "output_tokens": 500_000}
    cost, source = bs.call_cost(usage, {"input": 1.25, "output": 2.5, "cache_read": 0.2})
    assert (cost, source) == (Decimal("2.90000000"), "catalog_pricing")


@pytest.mark.parametrize("pricing", [None, {}, {"input": 1}, {"input": 1, "output": "bad"}])
def test_partial_or_missing_rates_leave_the_call_unpriced(pricing):
    usage = {"input_tokens": 10, "output_tokens": 10}
    assert bs.call_cost(usage, pricing) == (None, None)


def test_entry_id_is_stable_per_response_and_unique_without_one():
    a = bs._entry_id("xai-chat", {"id": "resp-1"})
    assert a == bs._entry_id("xai-chat", {"id": "resp-1"}) and a.startswith("bg:") and len(a) <= 128
    assert a != bs._entry_id("xai-responses", {"id": "resp-1"})
    assert bs._entry_id("xai-chat", {}) != bs._entry_id("xai-chat", {})


def test_recorder_skips_paths_without_a_balance():
    engine_getter = MagicMock()
    bs.BackgroundSpendRecorder({"access_path": "local-vllm"}, engine_getter=engine_getter)(
        {"id": "r", "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    engine_getter.assert_not_called()


def test_factory_installs_the_recorder_once_for_every_client():
    model_def = {"type": "openai-compatible", "protocol": "chat-completions", "base_url": "https://x/v1",
                 "access_path": "xai-chat", "api_key": "k", "model_id": "grok-4.3"}
    client, _ = bc.build_background_client(MagicMock(), model_def, "grok-4.3")
    assert isinstance(client.usage_observer, bs.BackgroundSpendRecorder)
    assert client.usage_observer.path == "xai-chat"


def test_client_reports_each_response_and_survives_a_recorder_failure():
    client = AnthropicMessagesClient("m", MagicMock(TEMPERATURE=0.2), base_url="https://a", api_key="k")
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")], usage=None, id="r")
    client.client = MagicMock()
    client.client.messages.create.return_value = response
    seen = []
    client.usage_observer = seen.append
    assert client.query([Message("user", "hi")], max_tokens=5) == "ok"
    assert seen == [response]

    def boom(_response):
        raise RuntimeError("db down")

    client.usage_observer = boom
    assert client.query([Message("user", "hi")], max_tokens=5) == "ok"


def test_report_usage_ignores_clients_without_an_observer():
    from llm_bawt.clients.usage_hook import report_usage

    report_usage(object(), {"id": "r"})  # no attribute, no error


# -- OpenRouter: provider-reported credit -------------------------------------


def _openrouter_fetch(monkeypatch, *, status=200, payload=None, key="sk-or-test"):
    import asyncio

    import httpx

    from llm_bawt.service.usage.adapters.openrouter import CREDITS_URL, OpenRouterUsageAdapter
    from tests.stored_provider_keys import stub_stored_keys

    seen = {}

    class Response:
        status_code = status

        @staticmethod
        def json():
            return payload

    async def fake_get(self, url, *, headers):
        seen.update(url=url, headers=headers)
        return Response()

    stub_stored_keys(monkeypatch, {"openrouter": key})
    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    return asyncio.run(OpenRouterUsageAdapter().fetch()), seen, CREDITS_URL


def test_openrouter_reports_credit_left_from_the_credits_endpoint(monkeypatch):
    snap, seen, url = _openrouter_fetch(
        monkeypatch, payload={"data": {"total_credits": 10, "total_usage": 0.966084842}})
    assert seen["url"] == url and seen["headers"]["Authorization"] == "Bearer sk-or-test"
    assert snap.status == "ok" and snap.available and snap.balance_supported
    assert not snap.manual_balance_supported
    balance = snap.balance
    assert balance.source == "provider_reported"
    assert (balance.amount_usd, balance.used_usd) == (10.0, 0.966084842)
    assert balance.remaining_usd == pytest.approx(9.033915158)


@pytest.mark.parametrize("status,payload,expected", [
    (401, {}, "unauthorized"),
    (429, {}, "rate_limited"),
    (500, {}, "error"),
    (200, {"data": {"total_credits": 10}}, "error"),
])
def test_openrouter_failures_show_no_balance(monkeypatch, status, payload, expected):
    snap, _seen, _url = _openrouter_fetch(monkeypatch, status=status, payload=payload)
    assert snap.status == expected and snap.balance is None and not snap.available


def test_openrouter_without_a_key_never_calls_upstream(monkeypatch):
    snap, seen, _url = _openrouter_fetch(monkeypatch, key=None)
    assert snap.status == "unauthorized" and seen == {}


def test_unrecognized_usage_shape_is_unpriced_not_free():
    usage = bs.normalize_usage({"id": "r", "usage": {"total_tokens": 100}})
    assert usage == {}
    assert bs.call_cost(usage, {"input": 1, "output": 2}) == (None, None)
    assert bs.catalog_cost({}, {"input": 1, "output": 2}) is None


def test_unrecognized_usage_shape_keeps_a_reported_cost():
    usage = bs.normalize_usage({"usage": {"total_tokens": 100, "cost": 0.25}})
    assert bs.call_cost(usage, None)[1] == "provider_reported"
