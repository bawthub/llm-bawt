"""Dollar amounts retain billing meaning rather than masquerading as spend."""
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from llm_bawt.service.turn_usage import classify_turn_cost, encode_token_usage
from llm_bawt.service.usage.balance import _cost


@pytest.mark.parametrize(("access", "amount", "kind", "expected"), [
    ({"key": "openai-oauth", "billing_mode": "subscription"}, 15.5, "notional", 15.5),
    ({"billing_mode": "subscription"}, None, "none", None),
    ({"vendor": "local"}, 15.5, "none", 0),
    ({"engine_kind": "ollama", "billing_mode": "metered"}, 15.5, "none", 0),
    ({"billing_mode": "metered"}, 0.125, "billed", 0.125),
    ({"billing_mode": "metered"}, 0, "billed", 0),
    ({"billing_mode": "metered"}, None, "unknown", None),
    ({"auth_mechanism": "api-key"}, 15.5, "unknown", None),
    ({}, 15.5, "unknown", None),
    ({"billing_mode": "metered"}, float("nan"), "unknown", None),
    ({"billing_mode": "metered"}, float("inf"), "unknown", None),
    ({"billing_mode": "metered"}, -1, "unknown", None),
    ({"billing_mode": "metered"}, True, "unknown", None),
])
def test_frozen_billing_classifies_without_inferring_from_auth(access, amount, kind, expected):
    original = {"access_path": access, "total_cost_usd": amount, "output_tokens": 3}
    result = classify_turn_cost(original)
    assert result["cost_kind"] == kind
    assert result["total_cost_usd"] == expected
    assert result["output_tokens"] == 3
    assert "cost_kind" not in original


@pytest.mark.parametrize("kind", ["notional", "none", "unknown"])
def test_explicit_nonbilled_cost_cannot_be_upgraded_or_enter_spend(kind):
    usage = {"total_cost_usd": 1.5, "cost_kind": kind, "access_path": {"billing_mode": "metered"}}
    assert classify_turn_cost(usage)["cost_kind"] == kind
    assert _cost(usage) is None


def test_billed_and_legacy_costs_remain_available_to_metered_ledger():
    assert _cost({"total_cost_usd": 1.5, "cost_kind": "billed"}) == Decimal("1.50000000")
    assert _cost({"total_cost_usd": 1.5}) == Decimal("1.50000000")


def test_common_encoder_classifies_bridge_usage_and_preserves_frozen_facts():
    catalog = SimpleNamespace(endpoint_by_ref=lambda _: SimpleNamespace(access_path=SimpleNamespace(
        key="openai-oauth", vendor="openai", protocol="responses",
        auth_mechanism="oauth", engine_kind=None,
    )))
    raw = {"total_cost_usd": 99, "input_tokens": 12}
    encoded = json.loads(encode_token_usage(raw, catalog=catalog, model_ref="example@openai-oauth"))
    assert encoded["cost_kind"] == "notional"
    assert encoded["access_path"]["billing_mode"] == "subscription"
    assert raw == {"total_cost_usd": 99, "input_tokens": 12}
    frozen = {**raw, "access_path": {"billing_mode": "metered"}}
    result = json.loads(encode_token_usage(frozen, catalog=catalog, model_ref="example@openai-oauth"))
    assert result["cost_kind"] == "billed"


@pytest.mark.parametrize("path_key", ["anthropic-api", "xai-chat", "xai-responses"])
@pytest.mark.parametrize("amount", [1.25, "3.10000001", Decimal("0.00000123"), "0.125"])
def test_legacy_frozen_route_and_decimal_cost_survive_encoder_to_ledger(path_key, amount):
    usage = {"access_path": {"key": path_key}, "total_cost_usd": amount}
    # A current catalog pointing elsewhere must not override the frozen key.
    catalog = SimpleNamespace(endpoint_by_ref=lambda _: pytest.fail("must use frozen path"))
    encoded = json.loads(encode_token_usage(usage, catalog=catalog, model_ref="other@openai-oauth"))
    assert encoded["cost_kind"] == "billed"
    assert isinstance(encoded["total_cost_usd"], (int, float))
    assert _cost(encoded) == Decimal(str(amount)).quantize(Decimal("0.00000001"))
    assert usage == {"access_path": {"key": path_key}, "total_cost_usd": amount}


@pytest.mark.parametrize(("access", "kind", "cost"), [
    ({"key": "openai-oauth"}, "notional", 1.25),
    ({"key": "anthropic-oauth"}, "notional", 1.25),
    ({"key": "no-such-route", "auth_mechanism": "api-key"}, "unknown", None),
    ({"auth_mechanism": "api-key"}, "unknown", None),
    ({"key": "anthropic-api", "billing_mode": "unknown"}, "unknown", None),
    ({"key": "anthropic-api", "billing_mode": None}, "unknown", None),
    ({"key": "anthropic-api", "billing_mode": "subscription"}, "notional", 1.25),
    ({"key": "openai-oauth", "billing_mode": "metered"}, "billed", 1.25),
    ({"key": "anthropic-api", "vendor": "local"}, "none", 0),
    ({"key": "anthropic-api", "engine_kind": "ollama"}, "none", 0),
])
def test_legacy_key_fallback_preserves_frozen_billing_and_suppression(access, kind, cost):
    encoded = json.loads(encode_token_usage(
        {"access_path": access, "total_cost_usd": "1.25"}, catalog=None, model_ref=None,
    ))
    assert encoded["cost_kind"] == kind
    assert encoded["total_cost_usd"] == cost
    if kind != "billed":
        assert _cost(encoded) is None


@pytest.mark.parametrize("amount", [
    None, True, False, -1, "-0.1", "NaN", "Infinity", "-Infinity", "abc",
    Decimal("NaN"), Decimal("Infinity"), Decimal("-1"), Decimal("1e9999"), Decimal("1e-9999"),
])
def test_legacy_metered_invalid_cost_remains_unpriced(amount):
    encoded = json.loads(encode_token_usage(
        {"access_path": {"key": "anthropic-api"}, "total_cost_usd": amount},
        catalog=None, model_ref=None,
    ))
    assert encoded["cost_kind"] == "unknown"
    assert encoded["total_cost_usd"] is None
    assert _cost(encoded) is None


@pytest.mark.parametrize("kind", ["notional", "none", "unknown"])
def test_legacy_metered_route_does_not_upgrade_explicit_nonbilled_cost(kind):
    encoded = json.loads(encode_token_usage(
        {"access_path": {"key": "anthropic-api"}, "total_cost_usd": "1.25", "cost_kind": kind},
        catalog=None, model_ref=None,
    ))
    assert encoded["cost_kind"] == kind
    assert _cost(encoded) is None
