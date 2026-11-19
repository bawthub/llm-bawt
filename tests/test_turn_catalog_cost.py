"""Unpriced bridge turns get an API-equivalent cost from catalog rates (TASK-2503)."""

import json

import pytest

from claude_code_bridge.send_usage import LiveUsagePublisher
from llm_bawt.model_catalog import AccessPath, ModelCatalog, ModelEndpoint, ModelIdentity
from llm_bawt.service.turn_usage import encode_token_usage

RATES = {"input": 2, "output": 10, "cache_read": 0.1}
REF = "gpt-6-sol@openai-oauth"
THRESHOLD = 272_000


def _catalog(pricing, threshold=THRESHOLD) -> ModelCatalog:
    return ModelCatalog([
        ModelEndpoint(
            id=20,
            model=ModelIdentity(20, "gpt-6-sol", "openai", "GPT-6 Sol", pricing_threshold=threshold),
            access_path=AccessPath(11, "openai-oauth", "openai", "responses", "https://example.invalid", "oauth"),
            upstream_model_id="gpt-6-sol",
            pricing=pricing,
        ),
    ])


def _usage(peak=150_000, **extra):
    # Top-level counts are one request; only the per-request sums are priced.
    return {
        "input_tokens": 631, "cache_read_tokens": 28416, "cache_creation_tokens": 0,
        "output_tokens": 999_999, "total_cost_usd": None,
        "turn_request_totals": {"requests": 7, "input_tokens": 1_100_000, "cache_read_tokens": 1_000_000,
                                "cache_creation_tokens": 0, "output_tokens": 1000, "unmeasured_requests": 0,
                                "max_request_input_tokens": peak},
        **extra,
    }


def _encode(usage, pricing=RATES, threshold=THRESHOLD):
    return json.loads(encode_token_usage(usage, model_ref=REF, catalog=_catalog(pricing, threshold)))


def test_subscription_turn_gets_notional_cost_from_request_sums() -> None:
    out = _encode(_usage())
    # 100k fresh * $2 + 1M cached * $0.10 + 1k output * $10, per 1M tokens.
    assert out["total_cost_usd"] == pytest.approx(0.2 + 0.1 + 0.01)
    assert out["cost_source"] == "catalog_pricing"
    assert out["cost_kind"] == "notional"


@pytest.mark.parametrize("peak,priced", [(THRESHOLD - 1, True), (THRESHOLD, True), (THRESHOLD + 1, False)])
def test_long_context_request_stays_unpriced(peak, priced) -> None:
    out = _encode(_usage(peak=peak))
    assert (out["total_cost_usd"] is not None) is priced


def test_unset_threshold_uses_conservative_default() -> None:
    # Unset model threshold means the catalog default (200k), not "no tier".
    assert _encode(_usage(peak=250_000), threshold=None)["total_cost_usd"] is None


def test_missing_rates_stay_unpriced_not_zero() -> None:
    for pricing in (None, {}, {"input": 2, "output": 10}):  # cache_read used but unpriced
        out = _encode(_usage(), pricing)
        assert out["total_cost_usd"] is None
        assert out["cost_kind"] == "none"
        assert "cost_source" not in out


@pytest.mark.parametrize("missing", [None, "output_tokens", "max_request_input_tokens", "unmeasured_requests"])
def test_incomplete_totals_stay_unpriced(missing) -> None:
    usage = _usage()
    if missing is None:
        del usage["turn_request_totals"]
    else:
        del usage["turn_request_totals"][missing]
    assert _encode(usage)["total_cost_usd"] is None


def test_existing_cost_is_kept() -> None:
    out = _encode(_usage(total_cost_usd=0.5))
    assert out["total_cost_usd"] == 0.5
    assert "cost_source" not in out


def _stream(pub, inp, cached, outputs):
    pub.observe_stream_event({"type": "message_start", "message": {
        "id": "m", "usage": {"input_tokens": inp, "cache_read_input_tokens": cached}}})
    for out in outputs:  # message_delta output is cumulative within a request
        pub.observe_stream_event({"type": "message_delta", "usage": {"output_tokens": out}})


def test_interrupted_turn_counts_every_requests_output() -> None:
    # Codex's case: 2,000 input, first request 100 output, interrupted second at 10.
    pub = LiveUsagePublisher(None, "r", "s", "gpt-6-sol", "gpt-6-sol", 400_000)
    _stream(pub, 1000, 0, [40, 100])
    _stream(pub, 1000, 0, [10])
    interrupted = pub.with_totals({"input_tokens": 1000, "output_tokens": 10, "total_cost_usd": None})
    totals = interrupted["turn_request_totals"]
    assert totals["output_tokens"] == 110
    assert totals["max_request_input_tokens"] == 1000
    out = _encode(interrupted, {"input": 2, "output": 10})
    assert out["total_cost_usd"] == pytest.approx(0.0051)


def test_cached_input_counts_toward_request_peak() -> None:
    pub = LiveUsagePublisher(None, "r", "s", "gpt-6-sol", "gpt-6-sol", 400_000)
    _stream(pub, 1000, THRESHOLD, [5])
    assert pub.request_totals()["max_request_input_tokens"] == THRESHOLD + 1000
    assert _encode(pub.with_totals({"total_cost_usd": None}))["total_cost_usd"] is None


def test_cancelled_gpt_request_with_zero_placeholder_start_stays_unpriced() -> None:
    # GPT proxy: synthetic all-zero message_start, content, then cancel before
    # the terminal usage delta. That request's usage is unknown.
    pub = LiveUsagePublisher(None, "r", "s", "gpt-6-sol", "gpt-6-sol", 400_000)
    _stream(pub, 1000, 0, [100])
    pub.observe_stream_event({"type": "message_start", "message": {"id": "m2", "usage": {
        "input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}}})
    pub.observe_stream_event({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "partial"}})
    usage = pub.with_totals({"input_tokens": 1000, "output_tokens": 100, "total_cost_usd": None})
    totals = usage["turn_request_totals"]
    assert totals["requests"] == 2 and totals["unmeasured_requests"] == 1
    out = _encode(usage, {"input": 2, "output": 10})
    assert out["total_cost_usd"] is None
    assert out["turn_request_totals"]["input_tokens"] == 1000  # counts preserved


def test_cancelled_native_request_with_measured_start_stays_unpriced() -> None:
    # Native Anthropic: message_start carries real input and output=1; content
    # streams; cancel before the terminal message_delta. Output is unknown.
    pub = LiveUsagePublisher(None, "r", "s", "claude", "claude", 400_000)
    pub.observe_stream_event({"type": "message_start", "message": {"id": "m", "usage": {
        "input_tokens": 1000, "output_tokens": 1}}})
    pub.observe_stream_event({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "partial"}})
    usage = pub.with_totals({"input_tokens": 1000, "output_tokens": 1, "total_cost_usd": None})
    assert usage["turn_request_totals"]["unmeasured_requests"] == 1
    assert _encode(usage, {"input": 2, "output": 10})["total_cost_usd"] is None
