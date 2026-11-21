from types import SimpleNamespace

import pytest

from claude_code_bridge.context_env import prompt_cache_ttl_env
from claude_code_bridge.send_usage import LiveUsagePublisher
from claude_code_bridge.turn_prompt_cache import TurnPromptCache
from llm_bawt.service.turn_usage import encode_token_usage


def attempt(ttl, *, proxy=False, resumed=False):
    return TurnPromptCache.from_sdk_env(
        prompt_cache_ttl_env(ttl), use_proxy=proxy,
        access_path="openai-oauth" if proxy else "anthropic-oauth",
        model="gpt-fixture" if proxy else "claude-fixture", resumed=resumed,
    )


def activity(*, five=0, hour=0, read=0):
    return {"type": "message_start", "message": {
        "id": "msg_fixture", "model": "claude-fixture",
        "usage": {"cache_read_input_tokens": read, "cache_creation_input_tokens": five + hour,
                  "cache_creation": {"ephemeral_5m_input_tokens": five, "ephemeral_1h_input_tokens": hour}},
    }}


@pytest.mark.parametrize(("ttl", "proxy", "expected"), [
    ("5m", False, "5m"), (" 1H ", False, "1h"), (None, False, "unset"),
    ("unknown", False, "unset"), ("1h", True, "not_applicable"),
])
def test_records_sdk_option_without_a_cache_claim(ttl, proxy, expected):
    cache = attempt(ttl, proxy=proxy)
    cache.bind_session("opaque-session")
    snapshot = cache.snapshot()
    assert snapshot["sdk_ttl"] == expected
    assert snapshot["route"] == ("proxy" if proxy else "native")
    assert snapshot["started_at_ms"] > 0
    assert snapshot["access_path"] == ("openai-oauth" if proxy else "anthropic-oauth")
    assert "opaque-session" not in str(snapshot)
    assert "estimated_reuse_until_ms" not in snapshot


def test_flip_and_rapid_resume_do_not_shorten_previous_long_write(monkeypatch):
    times = iter([1_000, 1_010, 1_050, 1_060])
    monkeypatch.setattr("claude_code_bridge.turn_prompt_cache._clock_ms", lambda: next(times))
    long = attempt("1h")
    long.bind_session("session-a")
    long.observe_start(activity(hour=5000), session_id="session-a", parent=None)
    previous = long.snapshot()
    short = attempt("5m", resumed=True)
    short.bind_session("session-a")
    short.observe_start(activity(five=5000, read=5000), session_id="session-a", parent=None)
    now = short.snapshot()
    assert previous["started_at_ms"] == 1_000
    assert previous["estimated_reuse_until_ms"] == 3_601_010
    assert now["started_at_ms"] == 1_050 and now["observed_at_ms"] == 1_060
    assert now["sdk_ttl"] == "5m" and now["read_tokens"] == 5000
    assert now["sdk_session_ref"] == previous["sdk_session_ref"]
    assert "estimated_reuse_until_ms" not in now
    assert long.snapshot() == previous


def test_only_new_foreground_unmixed_write_produces_short_estimate():
    cache = attempt("5m")
    cache.bind_session("session-a")
    cache.observe_start(activity(five=5000), session_id="session-a", parent="helper")
    cache.observe_start(activity(five=5000), session_id="session-b", parent=None)
    assert "observed_at_ms" not in cache.snapshot()
    cache.observe_start(activity(five=5000), session_id="session-a", parent=None)
    first = cache.snapshot()
    assert first["estimated_reuse_until_ms"] - first["observed_at_ms"] == 300_000
    assert first["actual_model"] == "claude-fixture"
    cache.observe_start(activity(read=5000), session_id="session-a", parent=None)
    assert "estimated_reuse_until_ms" not in cache.snapshot()
    cache.bind_session("session-b")
    assert cache.snapshot()["sdk_session_ref"] != first["sdk_session_ref"]
    assert "observed_at_ms" not in cache.snapshot()


@pytest.mark.parametrize(("ttl", "proxy", "five", "hour", "read"), [
    ("1h", False, 10, 10, 20), ("1h", False, 10, 0, 0),
    ("1h", False, 0, 0, 20), (None, False, 10, 0, 0), ("1h", True, 0, 10, 0),
    ("5m", False, 10, 0, 20),
])
def test_unsupported_or_ambiguous_activity_has_no_estimate(ttl, proxy, five, hour, read):
    cache = attempt(ttl, proxy=proxy)
    cache.bind_session("fixture")
    cache.observe_start(activity(five=five, hour=hour, read=read), session_id="fixture", parent=None)
    assert "estimated_reuse_until_ms" not in cache.snapshot()


def test_unbound_stream_and_compaction_cannot_claim_reuse():
    cache = attempt("1h")
    event = activity(hour=5000)
    cache.observe_start(event, session_id="unconfirmed", parent=None)
    assert "sdk_session_ref" not in cache.snapshot()
    assert "observed_at_ms" not in cache.snapshot()

    cache.bind_session("confirmed")
    cache.observe_start(event, session_id="confirmed", parent=None)
    assert "estimated_reuse_until_ms" in cache.snapshot()
    cache.suppress_estimate()
    cache.observe_start(event, session_id="confirmed", parent=None)
    assert cache.snapshot()["write_tokens"] == 5000
    assert "estimated_reuse_until_ms" not in cache.snapshot()


def test_only_supported_native_relay_write_has_clock_bound_evidence(monkeypatch):
    clock_samples = iter([("fixture-domain", 1_000), ("fixture-domain", 2_000), None])
    monkeypatch.setattr("claude_code_bridge.turn_prompt_cache.prompt_cache_clock", lambda: next(clock_samples))
    relay = "http://127.0.0.1:1/claude-oauth"
    env = {**prompt_cache_ttl_env("1h"), "ANTHROPIC_BASE_URL": relay}

    def open_attempt(url):
        cache = TurnPromptCache.from_sdk_env(env, use_proxy=False, access_path="anthropic-oauth",
                                            model="claude-fixture", resumed=True, oauth_gateway_url=url)
        cache.bind_session("sdk-fixture")
        return cache

    direct = open_attempt(None)
    direct.observe_start(activity(hour=3000), session_id="sdk-fixture", parent=None)
    assert "protection_write" not in direct.snapshot()

    cache = open_attempt(relay)
    cache.observe_start(activity(five=1000, hour=1000), session_id="sdk-fixture", parent=None)
    assert "protection_write" not in cache.snapshot()
    cache.observe_start(activity(hour=3500), session_id="sdk-fixture", parent=None)
    assert cache.snapshot()["protection_write"] == {
        "source": "native_oauth_relay_1h_write", "clock_domain": "fixture-domain",
        "observed_ms": 1_000, "model": "claude-fixture", "write_tokens": 3500,
    }
    cache.observe_start(activity(read=3500), session_id="sdk-fixture", parent=None)
    assert cache.snapshot()["protection_write"]["observed_ms"] == 1_000
    cache.observe_start(activity(hour=4000), session_id="sdk-fixture", parent=None)
    assert cache.snapshot()["protection_write"]["observed_ms"] == 2_000
    cache.observe_start(activity(hour=4000), session_id="sdk-fixture", parent=None)
    assert cache.snapshot()["protection_write"]["observed_ms"] == 2_000
    cache.bind_session("replacement-fixture")
    assert "protection_write" not in cache.snapshot()


def test_usage_transport_preserves_metadata_without_inventing_counts():
    import json

    cache = attempt(None)
    publisher = LiveUsagePublisher(SimpleNamespace(), "request", "sender", "model", "model", None,
                                   prompt_cache=cache)
    usage = publisher.with_totals(None)
    assert usage == {"prompt_cache": cache.snapshot()}
    encoded = encode_token_usage(usage, catalog=None, model_ref="claude-fixture")
    assert json.loads(encoded)["prompt_cache"] == usage["prompt_cache"]
    assert "input_tokens" not in json.loads(encoded)
