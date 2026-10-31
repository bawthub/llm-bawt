"""TASK-1895: recovered broker credentials can unblock one safe SDK resend."""

import asyncio
import time
from types import SimpleNamespace as NS

import httpx
import openai
import pytest

from claude_code_bridge.proxy import routes
from claude_code_bridge.proxy.adapters.openai_chatgpt import OpenAIChatGPTAdapter
from claude_code_bridge.proxy.retry import BrokerAuthorizationError, RetryBudgetRegistry


class BrokerAdapter(OpenAIChatGPTAdapter):
    """Real broker-cache/Responses retry code, with all external I/O replaced."""

    def __init__(self):
        super().__init__()
        self.broker_token = "fixture-stale"
        self.broker_error = False
        self.stale_now_valid = False
        self.broker_account = "fixture-account"
        self.broker_calls = 0
        self.samples = []
        self.sample_headers = []
        self.cancelled = False
        self.statuses = []
        self._cached_token = self.broker_token
        self._cached_account_id = "fixture-account"
        self._cached_expires_at = time.time() + 3600
        self._responses_client = NS(with_options=lambda **_: NS())

    async def start(self):
        pass

    def _fetch_broker_token(self, *, force=False):
        assert not force  # The app, never this proxy, owns OAuth refresh.
        self.broker_calls += 1
        if self.broker_error:
            raise BrokerAuthorizationError("fixture token broker unavailable")
        return self.broker_token, self.broker_account, time.time() + 3600

    async def open_stream(self, *, bearer, headers, **_):
        self.samples.append(bearer)
        self.sample_headers.append(dict(headers))
        if bearer == "fixture-stale" and not self.stale_now_valid:
            response = httpx.Response(401, request=httpx.Request("POST", "https://fixture.invalid"))
            raise openai.AuthenticationError("fixture expired credential", response=response, body=None)

        async def events():
            yield NS(type="response.output_item.added", item=NS(type="message"))
            yield NS(type="response.output_text.delta", delta="recovered")
            yield NS(type="response.completed", response=NS(usage=None, status="completed"))

        return events()


@pytest.fixture
def harness(monkeypatch):
    async def instant(_):
        pass

    monkeypatch.setattr(asyncio, "sleep", instant)
    monkeypatch.setattr(routes, "_PING_INTERVAL", 0)
    monkeypatch.setattr("claude_code_bridge.proxy.usage_capture.schedule_capture", lambda _: None)
    monkeypatch.setattr("claude_code_bridge.proxy.cache_diag.record", lambda _: None)
    adapter = BrokerAdapter()
    monkeypatch.setattr(routes, "lookup", lambda _: adapter)
    registry = RetryBudgetRegistry()
    app = NS(state=NS(
        proxy_retry_budgets=registry,
        proxy_cancellations=NS(is_cancelled=lambda _: adapter.cancelled),
        proxy_status_callback=lambda _request, status: adapter.statuses.append(status),
    ))

    class Request:
        headers = {"X-LLM-Bawt-Request-ID": "fixture-auth-turn"}

        def __init__(self, stream):
            self.app = app
            self.stream = stream

        async def json(self):
            return {"model": "openai_chatgpt/fixture", "stream": self.stream,
                    "messages": [{"role": "user", "content": "fixture"}]}

    async def send(stream=True):
        response = await routes.messages(Request(stream))
        if stream:
            return b"".join([chunk async for chunk in response.body_iterator])
        return response

    return adapter, registry, send


@pytest.mark.parametrize("stream", [True, False])
def test_terminal_auth_at_two_recovers_once_without_resetting_budget(harness, stream):
    async def run():
        adapter, _, send = harness
        assert b"authentication_error" in await send()
        second = await send(False)  # Same stale bearer grants no extra sample.
        assert second.status_code == 500
        assert len(adapter.samples) == 2
        adapter.broker_token = "fixture-recovered"
        recovered = await send(stream)
        if stream:
            assert b"event: message_stop" in recovered
            assert b'"type":"error"' not in recovered
        else:
            assert recovered.status_code == 200
            assert b"recovered" in recovered.body
        assert adapter.samples == ["fixture-stale"] * 2 + ["fixture-recovered"]
        # Successful recovery retains its spent-allowance tombstone.
        adapter.broker_token = "fixture-another-generation"
        assert b"error" in await send()
        assert len(adapter.samples) == 3
        assert adapter.statuses[-1]["state"] == "failed"

    asyncio.run(run())


def test_initial_broker_failures_do_not_strand_later_recovered_resend(harness):
    async def run():
        adapter, _, send = harness
        adapter._cached_expires_at = 0
        adapter.broker_error = True
        for _ in range(3):
            assert b"error" in await send()
        assert adapter.samples == []
        adapter.broker_error = False
        adapter.broker_token = "fixture-recovered"
        assert b"event: message_stop" in await send()
        assert adapter.samples == ["fixture-recovered"]

    asyncio.run(run())


def test_broker_only_failure_can_recover_same_unsampled_bearer(harness):
    async def run():
        adapter, _, send = harness
        adapter._cached_expires_at = 0
        adapter.broker_error = True
        for _ in range(3):
            await send()
        assert adapter.samples == []
        adapter.broker_error = False
        adapter.stale_now_valid = True
        assert b"message_stop" in await send()
        assert adapter.samples == ["fixture-stale"]

    asyncio.run(run())


def test_unknown_broker_account_cannot_grant_recovery(harness):
    async def run():
        adapter, _, send = harness
        adapter._cached_expires_at = 0
        adapter._cached_account_id = None
        adapter._cached_token = None
        adapter.broker_error = True
        for _ in range(3):
            await send()
        adapter.broker_error = False
        adapter.broker_token = "fixture-recovered"
        assert b"budget exhausted" in await send()
        assert adapter.samples == []

    asyncio.run(run())


def test_failed_probe_has_no_sample_and_preserves_later_rotation(harness):
    async def run():
        adapter, _, send = harness
        await send()
        adapter.broker_error = True
        assert b"authentication_error" in await send()
        assert len(adapter.samples) == 2
        adapter.broker_error = False
        adapter.broker_token = "fixture-recovered"
        assert b"message_stop" in await send()
        assert len(adapter.samples) == 3

    asyncio.run(run())


@pytest.mark.parametrize("drift", ["account", "base_url"])
def test_probe_rejects_identity_change_without_sampling(harness, monkeypatch, drift):
    async def run():
        adapter, _, send = harness
        await send()
        adapter.broker_token = "fixture-recovered"
        if drift == "account":
            adapter.broker_account = "fixture-other-account"
        else:
            monkeypatch.setenv("OPENAI_BASE_URL", "https://other.fixture.invalid")
        assert b"same-account" in await send()
        assert len(adapter.samples) == 2
        # A subsequent broker failure cannot rewrite the original account
        # baseline to the rejected account, then admit it on the next probe.
        adapter.broker_error = True
        assert b"authentication_error" in await send()
        adapter.broker_error = False
        adapter.broker_token = "fixture-another-generation"
        assert b"same-account" in await send()
        assert len(adapter.samples) == 2

    asyncio.run(run())


def test_registered_cancellation_during_probe_prevents_grant_and_sample(harness, monkeypatch):
    async def run():
        adapter, _, send = harness
        await send()
        adapter.broker_token = "fixture-recovered"
        prepare = adapter.prepare_auth_recovery

        async def cancel_after_prepare(*args):
            bundle = await prepare(*args)
            adapter.cancelled = True
            return bundle

        monkeypatch.setattr(adapter, "prepare_auth_recovery", cancel_after_prepare)
        with pytest.raises(asyncio.CancelledError):
            await send()
        assert len(adapter.samples) == 2

    asyncio.run(run())


def test_probe_sample_uses_frozen_bearer_and_account_headers(harness, monkeypatch):
    async def run():
        adapter, _, send = harness
        await send()
        adapter.broker_token = "fixture-recovered"
        prepare = adapter.prepare_auth_recovery

        async def rotate_after_prepare(*args):
            bundle = await prepare(*args)
            adapter._cached_token = "fixture-unadmitted-token"
            adapter._cached_account_id = "fixture-unadmitted-account"
            return bundle

        monkeypatch.setattr(adapter, "prepare_auth_recovery", rotate_after_prepare)
        assert b"message_stop" in await send()
        assert adapter.samples[-1] == "fixture-recovered"
        assert adapter.sample_headers[-1]["chatgpt-account-id"] == "fixture-account"

    asyncio.run(run())


def test_recovery_bonus_never_enters_internal_retries(harness, monkeypatch):
    from claude_code_bridge.proxy.retry import RetryPolicy

    async def run():
        adapter, _, send = harness
        monkeypatch.setattr(adapter, "retry_policy", lambda *_: RetryPolicy(max_attempts=5))
        await send()  # F is terminal at2, leaving3 unused original slots.
        adapter.broker_token = "fixture-recovered"

        async def fail_stream(**kwargs):
            adapter.samples.append(kwargs["bearer"])
            raise ConnectionError("fixture transient after auth recovered")

        monkeypatch.setattr(adapter, "open_stream", fail_stream)
        assert b"api_error" in await send()
        assert len(adapter.samples) == 3
        assert b"already spent" in await send()
        assert len(adapter.samples) == 3

    asyncio.run(run())


@pytest.mark.parametrize("terminal", ["cancel", "clear", "concurrent"])
def test_probe_lease_fences_cancellation_cleanup_and_concurrent_resend(harness, monkeypatch, terminal):
    async def run():
        adapter, registry, send = harness
        await send()
        adapter.broker_token = "fixture-recovered"
        entered, release = asyncio.Event(), asyncio.Event()
        prepare = adapter.prepare_auth_recovery

        async def suspended(*args):
            result = await prepare(*args)
            entered.set()
            await release.wait()
            return result

        monkeypatch.setattr(adapter, "prepare_auth_recovery", suspended)
        pending = asyncio.create_task(send())
        await entered.wait()
        if terminal == "cancel":
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert len(adapter.samples) == 2
        elif terminal == "clear":
            registry.clear_request("fixture-auth-turn")
            release.set()
            assert b"error" in await pending
            assert len(adapter.samples) == 2
        else:
            assert b"already running" in await send()
            assert len(adapter.samples) == 2
            release.set()
            assert b"message_stop" in await pending
            assert len(adapter.samples) == 3

    asyncio.run(run())


@pytest.mark.parametrize("message", ["broker unavailable", "broker returned 403"])
def test_generic_broker_text_is_not_auth_recovery_evidence(harness, monkeypatch, message):
    async def run():
        adapter, _, send = harness

        async def fail():
            raise RuntimeError(message)

        authorize = adapter.authorize
        monkeypatch.setattr(adapter, "authorize", fail)
        for _ in range(3):
            await send()
        monkeypatch.setattr(adapter, "authorize", authorize)
        adapter.broker_token = "fixture-recovered"
        assert b"budget exhausted" in await send()
        assert adapter.samples == []

    asyncio.run(run())


@pytest.mark.parametrize("payload", [
    {"type": "content_block_start", "content_block": {"type": "text", "text": "visible"}},
    {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "visible"}},
    {"type": "content_block_start", "content_block": {"type": "thinking", "signature": "native"}},
    {"type": "content_block_delta", "delta": {"type": "signature_delta", "signature": "native"}},
    {"type": "content_block_start", "content_block": {"type": "unknown"}},
    {"type": "content_block_delta", "delta": []},
    {"type": "content_block_start", "content_block": {"type": "tool_use"}},
])
def test_forwarded_content_blocks_auth_probe_even_with_split_frames(harness, monkeypatch, payload):
    import json

    async def run():
        adapter, _, send = harness
        calls = []

        async def call(_body, _model, context):
            calls.append(True)
            lease = context.retry_lease
            lease.record_authorization(("fixture",), "old-scope")
            lease.mark_auth_failure()
            raw = (f'event: {payload["type"]}\ndata: {json.dumps(payload)}\n\n').encode()
            yield raw[:17]
            yield raw[17:]
            yield b'event: error\ndata: {"type":"error","error":{"type":"api_error"}}\n\n'

        monkeypatch.setattr(adapter, "call", call)
        await send()
        assert b"cannot be replayed" in await send()
        assert len(calls) == 1
        assert adapter.broker_calls == 0

    asyncio.run(run())


def test_exhausted_auth_policy_grants_only_one_sample_and_fences_stale_owner():
    from claude_code_bridge.proxy.retry import RetryAdmissionError, RetryPolicy

    registry = RetryBudgetRegistry()
    def factory():
        return RetryPolicy(max_attempts=2)
    old = registry.acquire("turn", "body", factory)
    assert old.policy.claim_attempt() == 1
    assert old.policy.claim_attempt() == 2
    old.record_authorization(("provider", "endpoint", "account"), "failed-bearer")
    old.mark_auth_failure()
    old.release()
    probe = registry.acquire("turn", "body", factory, allow_auth_probe=True)
    with pytest.raises(RetryAdmissionError, match="same-account"):
        old.grant_auth_recovery(("provider", "endpoint", "account"), "new-bearer")
    old.clear_auth_failure()
    old.release()
    probe.grant_auth_recovery(("provider", "endpoint", "account"), "new-bearer")
    assert probe.recovery_sample
    assert probe.policy.max_attempts == 2  # no reset or limit increase
    assert probe.policy.attempt == 2
    probe.release()  # failure/cancellation after grant never refunds bonus
    with pytest.raises(RetryAdmissionError, match="already spent"):
        registry.acquire("turn", "body", factory, allow_auth_probe=True)
    registry.clear_request("turn")
    assert registry.acquire("turn", "body", factory).policy.claim_attempt() == 1


def test_terminal_stall_cannot_be_reopened_by_auth_marker():
    from claude_code_bridge.proxy.retry import (
        FailureBucket, RetryAdmissionError, RetryDecision, RetryPolicy,
    )

    registry = RetryBudgetRegistry()
    lease = registry.acquire("turn", "body", RetryPolicy)
    lease.policy.record_decision(FailureBucket.G_PROGRESS_STALL, RetryDecision(False))
    lease.record_authorization(("provider", "endpoint", "account"), "old-bearer")
    lease.mark_auth_failure()
    lease.release()
    with pytest.raises(RetryAdmissionError, match="unavailable"):
        registry.acquire("turn", "body", RetryPolicy, allow_auth_probe=True)
