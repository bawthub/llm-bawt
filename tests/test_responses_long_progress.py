"""Long productive responses must survive without weakening stall detection."""
import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from claude_code_bridge.proxy import chatgpt_transport, responses_supervisor
from claude_code_bridge.proxy.chatgpt_transport import (
    ChatGPTEventTimeout, ChatGPTResponsesTransport,
)
from claude_code_bridge.proxy.responses_supervisor import ResponsesSSEStream


@pytest.mark.parametrize("transport", ["websocket", "sse"])
@pytest.mark.parametrize("kind", [
    "response.output_text.delta",
    "response.reasoning_summary_text.delta",
    "response.function_call_arguments.delta",
])
@pytest.mark.parametrize("ending", ["completed", "stalled"])
def test_long_productive_response(monkeypatch, transport, kind, ending):
    # Replace only these modules' clock, not asyncio's clock. Ten minutes of
    # upstream generation are exercised deterministically without wall-time waits.
    now = 1000.0
    clock = NS(monotonic=lambda: now)
    monkeypatch.setattr(chatgpt_transport, "time", clock)
    monkeypatch.setattr(responses_supervisor, "time", clock)

    async def run():
        nonlocal now
        events = asyncio.Queue()
        close = AsyncMock()
        owner = None
        if transport == "websocket":
            async def recv():
                return json.dumps(await events.get())
            socket = NS(response=NS(headers={}), send=AsyncMock(),
                        recv=recv, close=close)
            owner = ChatGPTResponsesTransport(connector=AsyncMock(return_value=socket))
            stream = await owner.open(
                body={"model": "test-model", "input": []},
                headers={"session_id": "test"}, bearer="test",
                base_url="https://example.test", context=None,
                http_client=AsyncMock(),
            )
        else:
            async def incoming():
                while True:
                    yield NS(**await events.get())
            iterator = incoming()
            class HTTPStream:
                def __aiter__(self):
                    return iterator
            http = HTTPStream()
            http.close = close
            stream = ResponsesSSEStream(AsyncMock(return_value=http))
            await stream.prepare()

        try:
            for _ in range(20):
                now += 30
                await events.put({"type": kind, "delta": "chunk"})
                assert (await anext(stream)).delta == "chunk"
            assert now - stream.started_at == 600
            if ending == "completed":
                # Bookkeeping frames are valid even after the old absolute cap.
                await events.put({"type": "response.in_progress"})
                assert (await anext(stream)).type == "response.in_progress"
                await events.put({"type": "response.completed"})
                assert (await anext(stream)).type == "response.completed"
                with pytest.raises(StopAsyncIteration):
                    await anext(stream)
            else:
                # Progress does not permanently disable supervision. Lifecycle
                # chatter keeps transport alive, but cannot renew productive idle.
                for _ in range(2):
                    now += 30
                    await events.put({"type": "response.in_progress"})
                    await anext(stream)
                now += 31
                with pytest.raises(ChatGPTEventTimeout) as error:
                    await anext(stream)
                assert error.value.phase == "productive"
                assert error.value.productive_idle_seconds == 91
        finally:
            await stream.close()
            if owner is not None:
                await owner.close()
        close.assert_awaited_once()

    asyncio.run(run())
