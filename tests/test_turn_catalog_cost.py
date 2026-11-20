"""Unpriced bridge turns get an API-equivalent cost from catalog rates (TASK-2503)."""

import json
from types import SimpleNamespace

import pytest

from claude_code_bridge.proxy.adapters.openai_chatgpt import OpenAIChatGPTAdapter
from claude_code_bridge.proxy.stream import responses_to_anthropic_sse
from claude_code_bridge.send_usage import LiveUsagePublisher
from llm_bawt.model_catalog import AccessPath, ModelCatalog, ModelEndpoint, ModelIdentity
from llm_bawt.service.turn_usage import catalog_turn_cost, encode_token_usage

BASE_RATES = {"input": 2, "output": 10, "cache_read": 0.1}
RATES = {**BASE_RATES, "cache_write": 2.5}
REF = "gpt-6.2-sol@openai-oauth"
THRESHOLD = 272_000


def _catalog(pricing, threshold=THRESHOLD, *, access="openai-oauth") -> ModelCatalog:
    return ModelCatalog([
        ModelEndpoint(
            id=20,
            model=ModelIdentity(20, "gpt-6.2-sol", "openai", "GPT-6.2 Sol", pricing_threshold=threshold),
            access_path=AccessPath(11, access, "openai", "responses", "https://example.invalid", "oauth"),
            upstream_model_id="gpt-6.2-sol",
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
                                "unpriced_cache_write_requests": 0, "max_request_input_tokens": peak},
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
        pub.observe_stream_event({"type": "message_delta", "usage": {
            "output_tokens": out, "cache_write_pricing_complete": True,
        }})


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


async def _responses_request(pub, details, *, kind="completed", oauth=True, output=0):
    async def upstream():
        yield SimpleNamespace(type=f"response.{kind}", response=SimpleNamespace(
            status=kind, usage={"input_tokens": 12_000, "output_tokens": output,
                                "input_tokens_details": {"cached_tokens": 4_000, **details}},
        ))

    delta = None
    async for frame in responses_to_anthropic_sse(
        upstream(), "fixture", reports_cache_write_usage=oauth and OpenAIChatGPTAdapter.reports_cache_write_usage,
    ):
        event = json.loads(frame.split(b"\ndata: ", 1)[1])
        pub.observe_stream_event(event)
        if event["type"] == "message_delta":
            delta = event
    assert delta is not None
    return delta


def _writes(total, short, long):
    return {"cache_write_tokens": total, "cache_write_tokens_details": {
        "ttl_30m_tokens": short, "ttl_12h_tokens": long,
    }}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,stop", [("completed", "end_turn"), ("incomplete", "max_tokens")])
async def test_oauth_cache_writes_are_not_fresh_tokens(kind, stop) -> None:
    pub = LiveUsagePublisher(None, "r", "s", "gpt-6.2-sol", "gpt-6.2-sol", 400_000)
    delta = await _responses_request(pub, _writes(7000, 7000, 0), kind=kind)
    assert delta["delta"]["stop_reason"] == stop
    assert delta["usage"]["input_tokens"] == 1000
    usage = pub.with_totals(None)
    totals = usage["turn_request_totals"]
    assert totals["max_request_input_tokens"] == totals["input_tokens"] == 12_000
    assert totals["cache_read_tokens"] == 4000 and totals["cache_creation_tokens"] == 7000
    assert totals["unmeasured_requests"] == totals["unpriced_cache_write_requests"] == 0
    out = _encode(usage)
    assert out["total_cost_usd"] == pytest.approx(0.0199)
    assert out["cost_kind"] == "notional"
    base = _encode(usage, BASE_RATES)
    assert base["total_cost_usd"] == pytest.approx(0.0164)
    assert base["turn_request_totals"] == totals
    assert base["cost_kind"] == "notional"
    assert catalog_turn_cost(usage, _catalog(BASE_RATES, access="openai-api"), "gpt-6.2-sol@openai-api") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("details", [{"cache_write_tokens": 0}, _writes(0, 0, 0)])
async def test_explicit_zero_writes_are_priceable(details) -> None:
    pub = LiveUsagePublisher(None, "r", "s", "gpt-6.2-sol", "gpt-6.2-sol", 400_000)
    await _responses_request(pub, details)
    assert _encode(pub.with_totals(None))["total_cost_usd"] == pytest.approx(0.0164)


@pytest.mark.asyncio
@pytest.mark.parametrize("details,known_writes", [
    ({}, 0), ({"cache_write_tokens": None}, 0), ({"cache_write_tokens": False}, 0),
    ({"cache_write_tokens": "7000"}, 0), ({"cache_write_tokens": -1}, 0),
    (_writes(9000, 9000, 0), 0), (_writes(7000, 6900, 0), 7000),
    (_writes(7000, 6000, 1000), 7000), (_writes(7000, 7000, None), 7000),
    ({"cache_write_tokens": 7000, "cache_write_tokens_details": None}, 7000),
])
async def test_bad_write_telemetry_keeps_tokens_but_invalidates_whole_turn(details, known_writes) -> None:
    pub = LiveUsagePublisher(None, "r", "s", "gpt-6.2-sol", "gpt-6.2-sol", 400_000)
    await _responses_request(pub, _writes(0, 0, 0), output=11)
    await _responses_request(pub, details, output=7)
    usage = pub.with_totals(None)
    totals = usage["turn_request_totals"]
    assert totals["requests"] == 2 and totals["output_tokens"] == 18
    assert totals["max_request_input_tokens"] == 12_000
    assert totals["input_tokens"] == 24_000 and totals["cache_read_tokens"] == 8000
    assert totals["cache_creation_tokens"] == known_writes
    assert totals["unmeasured_requests"] == 0 and totals["unpriced_cache_write_requests"] == 1
    assert _encode(usage)["total_cost_usd"] is None
    assert _encode(usage, BASE_RATES)["total_cost_usd"] == pytest.approx(0.03298)


@pytest.mark.asyncio
async def test_providers_without_write_telemetry_keep_existing_pricing() -> None:
    pub = LiveUsagePublisher(None, "r", "s", "fixture", "fixture", 400_000)
    delta = await _responses_request(pub, {}, oauth=False)
    assert "cache_write_pricing_complete" not in delta["usage"]
    usage = pub.with_totals(None)
    assert "unpriced_cache_write_requests" not in usage["turn_request_totals"]
    assert catalog_turn_cost(usage, _catalog(RATES, access="openai-api"), "gpt-6.2-sol@openai-api") == pytest.approx(0.0164)
    # A stale OAuth bridge can price only the explicitly configured base rates.
    assert _encode(usage)["total_cost_usd"] is None
    assert _encode(usage, BASE_RATES)["total_cost_usd"] == pytest.approx(0.0164)


@pytest.mark.asyncio
async def test_one_terminal_frame_missing_coverage_cannot_hide_behind_measured_request() -> None:
    pub = LiveUsagePublisher(None, "r", "s", "gpt-6.2-sol", "gpt-6.2-sol", 400_000)
    await _responses_request(pub, {}, oauth=False)
    await _responses_request(pub, _writes(0, 0, 0))
    usage = pub.with_totals(None)
    assert usage["turn_request_totals"]["unpriced_cache_write_requests"] == 1
    assert _encode(usage)["total_cost_usd"] is None
