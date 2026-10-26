"""TASK-1079: SDK resends share one bounded, side-effect-aware proxy budget."""

import asyncio
import json
from types import SimpleNamespace as NS

import pytest

from claude_code_bridge.proxy import routes
from claude_code_bridge.proxy.retry import (
    FailureBucket,
    RetryAdmissionError,
    RetryBudgetRegistry,
    RetryPhase,
    RetryPolicy,
    decide,
)


def _policy():
    return RetryPolicy(max_attempts=2, jitter_s=0)


def _lease(registry, *, request="turn", body="same"):
    return registry.acquire(request, body, _policy)


def test_serial_sdk_resends_share_two_attempts_and_keep_terminal_failure():
    registry = RetryBudgetRegistry()
    first = _lease(registry)
    assert first.policy.claim_attempt() == 1
    decision = decide(bucket=FailureBucket.G_PROGRESS_STALL,
                      phase=RetryPhase.PRE_OUTPUT, policy=first.policy)
    assert decision.retry
    assert first.policy.claim_attempt() == 2
    final = decide(bucket=FailureBucket.G_PROGRESS_STALL,
                   phase=RetryPhase.PRE_OUTPUT, policy=first.policy)
    assert not final.retry and final.final_error_type == "api_error"
    first.policy.record_decision(FailureBucket.G_PROGRESS_STALL, final)
    first.release()

    with pytest.raises(RetryAdmissionError, match="exhausted"):
        _lease(registry)
    registry.clear_request("turn")
    fresh_turn = _lease(registry)
    assert fresh_turn.policy.claim_attempt() == 1
    fresh_turn.release()


def test_concurrent_identical_request_rejected_before_sampling():
    registry = RetryBudgetRegistry()
    owner = _lease(registry)
    assert owner.policy.claim_attempt() == 1
    with pytest.raises(RetryAdmissionError, match="already running"):
        _lease(registry)
    assert owner.policy.attempt == 1
    owner.release()
    next_sdk_request = _lease(registry)
    assert next_sdk_request.policy.claim_attempt() == 2
    next_sdk_request.release()


def test_completed_tool_and_open_tool_tombstones_deny_identical_resends():
    for successful in (False, True):
        registry = RetryBudgetRegistry()
        owner = _lease(registry)
        assert owner.policy.claim_attempt() == 1
        owner.mark_tool_committed()  # before forwarding content_block_start
        if successful:
            owner.mark_success()
        owner.release()
        with pytest.raises(RetryAdmissionError, match="tool"):
            _lease(registry)
        assert _lease(registry, body="tool-result-added").policy.claim_attempt() == 1
        registry.clear_request("turn")


def test_stale_finalizer_cannot_clear_new_owner_or_tool_tombstone():
    registry = RetryBudgetRegistry()
    old = _lease(registry)
    old.release()
    current = _lease(registry)
    old.mark_tool_committed()
    old.mark_success()
    old.release()
    with pytest.raises(RetryAdmissionError, match="already running"):
        _lease(registry)
    current.mark_tool_committed()
    current.release()
    with pytest.raises(RetryAdmissionError, match="tool"):
        _lease(registry)
    registry.clear_request("turn")
    new_turn = _lease(registry)
    current.release()  # stale release after request cleanup
    with pytest.raises(RetryAdmissionError, match="already running"):
        _lease(registry)
    new_turn.release()


def test_text_success_releases_lease_but_fail_closed_at_capacity():
    registry = RetryBudgetRegistry(max_entries=1)
    first = _lease(registry)
    with pytest.raises(RetryAdmissionError, match="capacity"):
        _lease(registry, request="another")
    first.mark_success()
    first.release()
    another = _lease(registry, request="another")
    assert another.policy.claim_attempt() == 1
    another.release()


def test_route_rejects_in_flight_duplicate_without_second_upstream_call(monkeypatch):
    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = []

        class Adapter:
            def account_hash(self):
                return "fixture"

            def retry_policy(self, *_):
                return _policy()

            async def call(self, _body, _model, context):
                calls.append(context.request_id)
                entered.set()
                await release.wait()
                # The route, not the adapter, must tombstone the tool before
                # forwarding even when passthrough transport splits the SSE.
                yield b'event: content_block_start\ndata: {"type":"content_block_start",'
                yield (b'"index":0,"content_block":'
                       b'{"type":"tool_use","id":"tool-1","name":"work","input":{}}}\n\n')

        registry = RetryBudgetRegistry()
        adapter = Adapter()
        monkeypatch.setattr(routes, "lookup", lambda _: adapter)
        app = NS(state=NS(proxy_retry_budgets=registry, proxy_cancellations=None,
                           proxy_status_callback=None))

        class Request:
            headers = {"X-LLM-Bawt-Request-ID": "turn-123"}

            def __init__(self, body):
                self.body = body
                self.app = app

            async def json(self):
                return self.body

        body = {"model": "fixture/model", "stream": True,
                "messages": [{"role": "user", "content": "fixture"}]}
        first = await routes.messages(Request(body))
        first_task = asyncio.create_task(_collect(first))
        await entered.wait()  # A has entered upstream sampling
        second = await routes.messages(Request(body))
        duplicate = await _collect(second)
        assert len(calls) == 1
        assert _frames(duplicate)[0]["error"]["type"] == "api_error"
        release.set()
        first_output = await first_task
        assert b'"tool_use"' in first_output
        third = await routes.messages(Request({**body, "stream": False}))
        # The same body with another SDK stream mode cannot execute it again.
        assert third.status_code == 500
        assert len(calls) == 1
        changed = await routes.messages(Request({**body, "messages": [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tool-1",
                                               "name": "work", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool-1",
                                          "content": "done"}]},
        ]}))
        await _collect(changed)
        assert len(calls) == 2  # a changed body advances before turn-final cleanup
        registry.clear_request("turn-123")

    asyncio.run(run())


def test_stream_and_json_sdk_resends_spend_one_shared_budget(monkeypatch):
    async def run():
        calls = []

        class Adapter:
            def account_hash(self):
                return "fixture"

            def retry_policy(self, *_):
                return _policy()

            async def call(self, _body, _model, context):
                calls.append(context.attempt)
                yield b'event: error\ndata: {"type":"error","error":{"type":"api_error","message":"upstream stopped"}}\n\n'

        monkeypatch.setattr(routes, "lookup", lambda _: Adapter())
        registry = RetryBudgetRegistry()
        app = NS(state=NS(proxy_retry_budgets=registry, proxy_cancellations=None,
                           proxy_status_callback=None))

        class Request:
            headers = {"X-LLM-Bawt-Request-ID": "same-bridge-turn"}

            def __init__(self, stream):
                self.app = app
                self.stream = stream

            async def json(self):
                return {"model": "fixture/model", "stream": self.stream,
                        "messages": [{"role": "user", "content": "fixture"}]}

        first = await routes.messages(Request(True))
        assert _frames(await _collect(first))[0]["type"] == "error"
        second = await routes.messages(Request(False))
        assert second.status_code == 500
        third = await routes.messages(Request(True))
        assert "exhausted" in _frames(await _collect(third))[0]["error"]["message"]
        assert calls == [1, 2]  # no third upstream sampling
        registry.clear_request("same-bridge-turn")
        fresh = await routes.messages(Request(True))
        await _collect(fresh)
        assert calls == [1, 2, 1]

    asyncio.run(run())


async def _collect(response):
    return b"".join([chunk async for chunk in response.body_iterator])


def _frames(raw):
    return [json.loads(line[6:]) for line in raw.splitlines() if line.startswith(b"data: ")]
