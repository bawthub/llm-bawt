"""TASK-502: provider-bound proxy bodies are captured after every rewrite."""
import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from websockets.exceptions import InvalidStatus

from claude_code_bridge.proxy.chatgpt_transport import ChatGPTResponsesTransport, TURN_HEADER
from claude_code_bridge.proxy.request_context import ProxyRequestContext
from tests.test_chatgpt_transport import DONE, Socket, kwargs


def _context(captures):
    return ProxyRequestContext(
        request_id="a" * 32, provider="openai-chatgpt", bot_id="bot",
        conversation_id="conversation",
        egress_callback=lambda rid, capture: captures.append((rid, capture)),
    )


def test_context_reports_attempt_and_transport_and_never_raises():
    captures = []
    context = _context(captures)
    context.attempt = 2
    context.report_egress({"model": "m"}, transport="responses_http")
    rid, capture = captures[0]
    assert rid == "a" * 32
    assert capture == {"provider": "openai-chatgpt", "bot_id": "bot", "attempt": 2,
                       "transport": "responses_http", "body": {"model": "m"}}

    def boom(*_):
        raise RuntimeError("down")
    context.egress_callback = boom
    context.report_egress({"model": "m"}, transport="responses_http")  # swallowed


def test_lite_ws_capture_is_the_exact_frame_sent():
    async def run():
        captures = []
        socket = Socket()
        client = ChatGPTResponsesTransport(connector=AsyncMock(return_value=socket))
        context = _context(captures)
        stream = await client.open(**kwargs(context=context))
        await socket.events.put(DONE)
        await anext(stream)
        await stream.close()
        await client.close()
        (_, capture), = captures
        assert capture["transport"] == "chatgpt_lite_ws"
        assert capture["body"] == socket.sent[0]
        assert "tools" not in capture["body"] and "instructions" not in capture["body"]
    asyncio.run(run())


def test_lite_http_fallback_capture_is_the_posted_body():
    async def run():
        captures = []
        connector = AsyncMock(side_effect=InvalidStatus(NS(status_code=426, headers={})))
        client = ChatGPTResponsesTransport(connector=connector)
        args = kwargs(context=_context(captures))

        async def events():
            yield NS(type="response.completed")

        class Stream:
            response = NS(headers={TURN_HEADER: "t"})
            close = AsyncMock()

            def __aiter__(self):
                return events()
        args["http_client"].post.return_value = Stream()
        stream = await client.open(**args)
        await anext(stream)
        await stream.close()
        await client.close()
        (_, capture), = captures
        assert capture["transport"] == "chatgpt_lite_sse"
        assert capture["body"] == args["http_client"].post.call_args.kwargs["body"]
    asyncio.run(run())


def test_bridge_serializes_at_capture_time_and_skips_inactive_requests():
    from claude_code_bridge.event_ops import ClaudeEventMixin

    async def run():
        posted = []
        bridge = ClaudeEventMixin.__new__(ClaudeEventMixin)
        bridge._app_api_url = "http://app"
        bridge._proxy_request_sessions = {"live": "session"}

        async def post(payload):
            posted.append(payload)
        bridge._post_proxy_egress = post
        body = {"model": "m", "input": ["one"]}
        bridge.publish_proxy_egress("live", {"attempt": 1, "transport": "x", "body": body})
        body["input"].append("mutated later")
        bridge.publish_proxy_egress("gone", {"attempt": 1, "transport": "x", "body": body})
        await asyncio.sleep(0)
        (payload,) = posted
        assert payload["seq"] == 1 and payload["request_id"] == "live"
        assert json.loads(payload["body_json"]) == {"model": "m", "input": ["one"]}
        assert "body" not in payload
    asyncio.run(run())


@pytest.fixture
def store():
    from sqlalchemy import create_engine
    from llm_bawt.service.egress_capture_store import EgressCaptureStore

    engine = create_engine("sqlite://")
    store = EgressCaptureStore(engine)
    store.ensure_schema()
    return store


def _capture(seq, body="{\"model\": \"m\"}"):
    return {"request_id": "r", "seq": seq, "attempt": 1, "provider": "p",
            "transport": "responses_http", "body_json": body,
            "captured_at": f"2026-11-15T17:00:0{seq}+00:00"}


def test_store_keeps_first_and_latest_bodies_and_is_idempotent(store):
    for seq in (1, 2, 3):
        assert store.record(turn_id="t", capture=_capture(seq))
    assert not store.record(turn_id="t", capture=_capture(3))
    items = store.for_turn("t")
    assert [i["body_state"] for i in items] == ["retained", "superseded", "retained"]
    assert items[0]["body"] == {"model": "m"} and items[1]["body"] is None
    assert items[2]["model"] == "m" and items[1]["body_bytes"] > 0


def test_store_records_oversized_body_by_size_and_hash_only(store, monkeypatch):
    from llm_bawt.service import egress_capture_store

    monkeypatch.setattr(egress_capture_store, "MAX_BODY_BYTES", 4)
    store.record(turn_id="t", capture=_capture(1))
    (item,) = store.for_turn("t")
    assert item["body_state"] == "over_cap" and item["body"] is None
    assert len(item["body_sha256"]) == 64
