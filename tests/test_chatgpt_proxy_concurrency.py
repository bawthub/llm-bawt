"""TASK-685 regression and deterministic concurrency coverage."""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from claude_code_bridge.proxy.adapters.openai_chatgpt import (
    _CACHE_BUFFER_S,
    OpenAIChatGPTAdapter,
)
from claude_code_bridge.proxy.request_context import (
    BOT_HEADER,
    CONVERSATION_HEADER,
    REQUEST_HEADER,
    RESPONSES_TRANSPORT_HEADER,
    ProxyRequestContext,
    custom_header_env,
    durable_conversation_identity,
)
from claude_code_bridge.proxy.routes import _proxy_iter, messages
from claude_code_bridge.send_stream import ClaudeStreamMixin


def _run(coro):
    return asyncio.run(coro)


def test_broker_epoch_seconds_cache_and_refresh_buffer(monkeypatch) -> None:
    """Healthy repeated auth fetches once; near-expiry auth fetches again."""
    adapter = OpenAIChatGPTAdapter()
    now = 1_800_000_000.0
    fetches: list[int] = []

    monkeypatch.setattr(
        "claude_code_bridge.proxy.adapters.openai_chatgpt.time.time", lambda: now
    )

    def fetch(*, force=False):
        fetches.append(1)
        return f"token-{len(fetches)}", "account-1", now + 3600

    monkeypatch.setattr(adapter, "_fetch_broker_token", fetch)

    async def healthy_then_near_expiry():
        first = await adapter.authorize()
        second = await adapter.authorize()
        assert first == second
        assert adapter._cached_expires_at == now + 3600
        assert len(fetches) == 1

        adapter._cached_expires_at = now + _CACHE_BUFFER_S
        third = await adapter.authorize()
        assert third[0] == "token-2"

    _run(healthy_then_near_expiry())
    assert len(fetches) == 2


def test_broker_route_preserves_epoch_seconds_contract(monkeypatch) -> None:
    from llm_bawt.service.routes.providers import chatgpt_access_token
    from llm_bawt.service import bridge_gate
    from llm_bawt.service.usage import codex_oauth

    expires_at_seconds = 1_800_003_600.0
    monkeypatch.delenv("BRIDGE_CLAUDE_TOKEN_SECRET", raising=False)
    # This test covers the payload contract; gate policy is covered in test_bridge_gate.
    monkeypatch.setattr(bridge_gate, "check", lambda request, purpose: None)
    monkeypatch.setattr(
        codex_oauth,
        "get_access_token",
        lambda **_: SimpleNamespace(
            token="token",
            account_id="account-1",
            expires_at=expires_at_seconds,
            state="ok",
        ),
    )
    request = SimpleNamespace(headers={})

    payload = _run(chatgpt_access_token(request))
    assert payload["expires_at"] == expires_at_seconds
    assert payload["expires_at"] < 100_000_000_000  # seconds, not milliseconds


def test_concurrent_cold_authorize_is_single_flight(monkeypatch) -> None:
    adapter = OpenAIChatGPTAdapter()
    now = time.time()
    fetches = 0

    def fetch(*, force=False):
        nonlocal fetches
        fetches += 1
        # Runs in asyncio.to_thread; a short sleep makes every coroutine reach
        # the lock while the cold fetch is still in flight.
        time.sleep(0.03)
        return "shared-token", "account-1", now + 3600

    monkeypatch.setattr(adapter, "_fetch_broker_token", fetch)

    async def burst():
        return await asyncio.gather(*(adapter.authorize() for _ in range(12)))

    results = _run(burst())
    assert fetches == 1
    assert {result[0] for result in results} == {"shared-token"}


def test_broker_millisecond_expiry_is_tolerated_at_boundary(monkeypatch) -> None:
    adapter = OpenAIChatGPTAdapter()
    now = 1_800_000_000.0
    monkeypatch.setattr(
        "claude_code_bridge.proxy.adapters.openai_chatgpt.time.time", lambda: now
    )
    monkeypatch.setattr(
        adapter,
        "_fetch_broker_token",
        lambda **_: ("token", "account", (now + 3600) * 1000),
    )

    _run(adapter.authorize())
    assert adapter._cached_expires_at == now + 3600
    assert adapter._cache_valid()


def test_proxy_lifespan_starts_and_closes_adapter_pools_once(monkeypatch) -> None:
    from claude_code_bridge.proxy import app as proxy_app

    calls: list[str] = []

    async def start_all():
        calls.append("start")

    async def close_all():
        calls.append("close")

    monkeypatch.setattr(proxy_app, "start_all", start_all)
    monkeypatch.setattr(proxy_app, "close_all", close_all)
    app = proxy_app.create_app()

    async def enter_lifespan():
        async with app.router.lifespan_context(app):
            assert calls == ["start"]

    _run(enter_lifespan())
    assert calls == ["start", "close"]


def _conversation(bot: str, user: str, thread: str) -> str:
    return durable_conversation_identity(
        bot_id=bot, session_key=f"{bot}:{user}", thread_session_id=thread
    )


def test_durable_identity_is_stable_isolated_and_rotates_on_new() -> None:
    snark_thread = _conversation("snark", "operator", "thread-1")
    assert snark_thread == _conversation("snark", "operator", "thread-1")
    assert snark_thread != _conversation("al", "operator", "thread-1")
    assert snark_thread != _conversation("snark", "other-user", "thread-1")
    # /new rotates the durable DB thread before bridge dispatch.
    assert snark_thread != _conversation("snark", "operator", "thread-2")


def test_custom_header_channel_carries_only_opaque_metadata() -> None:
    conversation_id = _conversation("snark", "operator", "thread-1")
    context = ProxyRequestContext(
        request_id="request-1",
        provider="openai_chatgpt",
        bot_id="snark",
        conversation_id=conversation_id,
        responses_transport="sse",
    )

    headers = dict(
        line.split(": ", 1) for line in custom_header_env(context).splitlines()
    )
    assert headers == {
        CONVERSATION_HEADER: conversation_id,
        BOT_HEADER: "snark",
        REQUEST_HEADER: "request-1",
        RESPONSES_TRANSPORT_HEADER: "sse",
    }
    serialized = json.dumps(headers)
    assert "operator" not in serialized
    assert "thread-1" not in serialized


def test_sdk_env_threads_durable_identity_into_proxy_headers() -> None:
    class Harness(ClaudeStreamMixin):
        _proxy_base_url = "http://127.0.0.1:12345"

    env = Harness()._build_sdk_env(
        use_proxy=True,
        model="openai_chatgpt/gpt-5.4",
        subagent_model=None,
        force_refresh=False,
        bot_id="snark",
        session_key="snark:operator",
        thread_session_id="thread-1",
        request_id="request-1",
        responses_transport="sse",
    )
    headers = dict(
        line.split(": ", 1) for line in env["ANTHROPIC_CUSTOM_HEADERS"].splitlines()
    )
    assert headers[CONVERSATION_HEADER] == _conversation("snark", "operator", "thread-1")
    assert headers[BOT_HEADER] == "snark"
    assert headers[REQUEST_HEADER] == "request-1"
    assert headers[RESPONSES_TRANSPORT_HEADER] == "sse"


def test_proxy_route_extracts_request_local_metadata(monkeypatch) -> None:
    captured: list[ProxyRequestContext] = []
    conversation_id = _conversation("snark", "operator", "thread-1")

    class Adapter:
        def account_hash(self):
            return "mock-account"

        async def call(self, body, upstream_model, context):
            captured.append(context)
            yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'

    class Request:
        headers = {
            CONVERSATION_HEADER: conversation_id,
            BOT_HEADER: "snark",
            REQUEST_HEADER: "request-1",
            RESPONSES_TRANSPORT_HEADER: "lite_ws",
        }

        async def json(self):
            return {
                "model": "openai_chatgpt/gpt-5.4",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            }

    monkeypatch.setattr(
        "claude_code_bridge.proxy.routes.lookup", lambda _provider: Adapter()
    )

    async def run_route():
        response = await messages(Request())
        return b"".join([chunk async for chunk in response.body_iterator])

    _run(run_route())
    assert len(captured) == 1
    assert captured[0].request_id == "request-1"
    assert captured[0].bot_id == "snark"
    assert captured[0].conversation_id == conversation_id
    assert captured[0].responses_transport == "lite_ws"


def test_session_and_prompt_cache_use_same_durable_identity() -> None:
    adapter = OpenAIChatGPTAdapter()
    adapter._cached_account_id = "account-1"
    context = ProxyRequestContext(
        request_id="request-1",
        provider=adapter.name,
        bot_id="snark",
        conversation_id=_conversation("snark", "operator", "thread-1"),
    )
    body = {
        "instructions": "Stable persona",
        "input": [{"role": "user", "content": "opening"}],
    }

    prepared = adapter.prepare_request(dict(body), context)
    headers = adapter.extra_headers(prepared, context)
    assert prepared["prompt_cache_key"] == context.conversation_id
    assert headers["session_id"] == context.conversation_id

    # Tail growth and resume calls do not alter the explicit durable key.
    resumed = adapter.prepare_request(
        {**body, "input": [*body["input"], {"role": "assistant", "content": "tail"}]},
        context,
    )
    assert resumed["prompt_cache_key"] == prepared["prompt_cache_key"]

    other = ProxyRequestContext(
        request_id="request-2",
        provider=adapter.name,
        bot_id="al",
        conversation_id=_conversation("al", "operator", "thread-1"),
    )
    assert (
        adapter.prepare_request(dict(body), other)["prompt_cache_key"]
        != prepared["prompt_cache_key"]
    )


class _FakeResponsesClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.close_count = 0

    def with_options(self, **options):
        parent = self

        class RequestClient:
            class Responses:
                async def create(self, **body):
                    parent.calls.append({"options": options, "body": body})

                    async def events():
                        yield SimpleNamespace(
                            type="response.output_text.delta",
                            delta=body["input"][0]["content"][0]["text"],
                        )
                        yield SimpleNamespace(
                            type="response.completed",
                            response=SimpleNamespace(
                                status="completed",
                                usage=SimpleNamespace(
                                    input_tokens=100,
                                    output_tokens=1,
                                    input_tokens_details=SimpleNamespace(
                                        cached_tokens=90
                                    ),
                                ),
                            ),
                        )

                    return events()

            responses = Responses()

        return RequestClient()

    async def close(self):
        self.close_count += 1


def test_structured_log_splits_setup_ttfb_stream_and_cache(caplog) -> None:
    adapter = OpenAIChatGPTAdapter()
    adapter._cached_token = "token"
    adapter._cached_account_id = "account-1"
    adapter._cached_expires_at = time.time() + 3600
    adapter._http_client = object()
    adapter._responses_client = _FakeResponsesClient()
    context = ProxyRequestContext(
        request_id="observed-request",
        provider=adapter.name,
        bot_id="snark",
        conversation_id=_conversation("snark", "operator", "thread-observed"),
    )
    body = {
        "model": "openai_chatgpt/gpt-5.4",
        "messages": [{"role": "user", "content": "observed-payload"}],
        "stream": True,
    }

    async def run_stream():
        try:
            return b"".join(
                [
                    chunk
                    async for chunk in _proxy_iter(
                        adapter,
                        body,
                        "gpt-5.4",
                        provider=adapter.name,
                        context=context,
                    )
                ]
            )
        finally:
            await adapter.close()

    with caplog.at_level("INFO", logger="claude_code_bridge.proxy.routes"):
        _run(run_stream())

    completed = next(
        record.getMessage()
        for record in caplog.records
        if "proxy_stream_complete request_id=observed-request" in record.getMessage()
    )
    assert "active_provider=" in completed
    assert "active_account=" in completed
    assert "queue_ms=0.0" in completed
    assert "setup_ms=na" not in completed
    assert "upstream_ttfb_ms=na" not in completed
    assert "stream_ms=" in completed
    assert "input=100 cached=90 output=1 cache_hit=90.0%" in completed
    assert "errors_429_total=" in completed
    assert "errors_5xx_total=" in completed


@pytest.mark.parametrize("parallel", [1, 2, 3, 5])
def test_parallel_responses_calls_reuse_pool_without_header_or_payload_crosstalk(
    parallel: int,
) -> None:
    adapter = OpenAIChatGPTAdapter()
    adapter._cached_token = "token"
    adapter._cached_account_id = "account-1"
    adapter._cached_expires_at = time.time() + 3600
    root = _FakeResponsesClient()
    adapter._http_client = object()  # mark lifecycle pool as started
    adapter._responses_client = root

    contexts: list[ProxyRequestContext] = []

    async def one(index: int) -> bytes:
        context = ProxyRequestContext(
            request_id=f"request-{index}",
            provider=adapter.name,
            bot_id=f"bot-{index}",
            conversation_id=_conversation(f"bot-{index}", "operator", f"thread-{index}"),
        )
        contexts.append(context)
        body = {
            "model": "openai_chatgpt/gpt-5.4",
            "messages": [{"role": "user", "content": f"payload-{index}"}],
            "stream": True,
        }
        return b"".join(
            [chunk async for chunk in adapter.call(body, "gpt-5.4", context)]
        )

    async def burst():
        outputs = await asyncio.gather(*(one(i) for i in range(parallel)))
        await adapter.close()
        await adapter.close()
        return outputs

    outputs = _run(burst())
    assert len(root.calls) == parallel
    assert root.close_count == 1
    assert len({id(call["options"]) for call in root.calls}) == parallel

    seen_cache_keys = set()
    calls_by_payload = {
        call["body"]["input"][0]["content"][0]["text"]: call for call in root.calls
    }
    for index, output in enumerate(outputs):
        call = calls_by_payload[f"payload-{index}"]
        expected = _conversation(f"bot-{index}", "operator", f"thread-{index}")
        assert call["body"]["prompt_cache_key"] == expected
        assert call["options"]["set_default_headers"]["session_id"] == expected
        assert f"payload-{index}".encode() in output
        seen_cache_keys.add(call["body"]["prompt_cache_key"])
    assert len(seen_cache_keys) == parallel
    assert all(context.cache_hit_pct == 90.0 for context in contexts)


@pytest.mark.parametrize("parallel", [1, 2, 3, 5])
def test_parallel_proxy_streams_make_progress_without_serialization(
    parallel: int,
) -> None:
    entered = 0
    all_entered = asyncio.Event()

    class Adapter:
        def account_hash(self):
            return "mock-account"

        async def call(self, body, upstream_model, context):
            nonlocal entered
            entered += 1
            if entered == parallel:
                all_entered.set()
            await asyncio.wait_for(all_entered.wait(), timeout=0.5)
            for step in range(3):
                await asyncio.sleep(0)
                yield f"{context.request_id}:{step}\n\n".encode()

    async def one(index: int):
        context = ProxyRequestContext(
            request_id=f"r{index}",
            provider="mock",
            bot_id=f"bot-{index}",
            conversation_id=_conversation(f"bot-{index}", "operator", f"t{index}"),
        )
        return b"".join(
            [
                chunk
                async for chunk in _proxy_iter(
                    Adapter(), {}, "model", provider="mock", context=context
                )
            ]
        )

    async def burst():
        return await asyncio.wait_for(
            asyncio.gather(*(one(i) for i in range(parallel))), timeout=1.0
        )

    outputs = _run(burst())
    assert len(outputs) == parallel
    for index, output in enumerate(outputs):
        assert output == b"".join(f"r{index}:{step}\n\n".encode() for step in range(3))
