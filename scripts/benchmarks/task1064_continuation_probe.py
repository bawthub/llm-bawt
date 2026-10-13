"""Opt-in TASK-1064 live recovery probe; no tools, bot sessions or service edits.

Run with --model <explicit upstream model>. Uses the app-owned token broker.
Injects a disconnect after visible text through the ACTUAL adapter retry path,
parses its spliced wire with the Anthropic SDK, then sends the SDK-shaped history
back upstream. Prints synthetic text/counts only, never credentials/reasoning.
"""
from __future__ import annotations

import argparse
import asyncio
import json

import httpx
from anthropic import AsyncAnthropic

from claude_code_bridge.proxy.adapters.openai_chatgpt import OpenAIChatGPTAdapter


class InterruptedStream:
    def __init__(self, stream):
        self.stream = stream
        self.chars = 0

    async def __aiter__(self):
        async for event in self.stream:
            yield event
            if event.type == "response.output_text.delta":
                self.chars += len(event.delta)
                if self.chars >= 25:
                    raise ConnectionError("TASK-1064 synthetic post-text disconnect")

    async def close(self):
        await self.stream.close()


class ProbeAdapter(OpenAIChatGPTAdapter):
    def __init__(self):
        super().__init__()
        self.opens = 0

    async def open_stream(self, **kwargs):
        self.opens += 1
        stream = await super().open_stream(**kwargs)
        return InterruptedStream(stream) if self.opens == 1 else stream


async def sdk_blocks(frames):
    transport = httpx.MockTransport(lambda _: httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=b"".join(frames),
    ))
    async with AsyncAnthropic(api_key="local-fixture", http_client=httpx.AsyncClient(
        transport=transport,
    )) as client:
        async with client.messages.stream(model="local-fixture", max_tokens=100,
                                          messages=[{"role": "user", "content": "probe"}]) as stream:
            message = await stream.get_final_message()
            return json.loads(message.model_dump_json())["content"]


async def run(model: str) -> None:
    adapter = ProbeAdapter()
    body = {
        "model": "openai_chatgpt/" + model, "max_tokens": 1024,
        "system": "Follow the user's exact output format. No tools.",
        "thinking": {"type": "enabled", "budget_tokens": 32000},
        "messages": [{"role": "user", "content": (
            "Calculate squares of integers 1 through 30. Output ONLY the "
            "comma-separated results in order, no explanations."
        )}],
    }
    try:
        frames = [frame async for frame in adapter.call(body, model)]
        blocks = await sdk_blocks(frames)
        answer = "".join(block.get("text", "") for block in blocks if block["type"] == "text")
        reasoning_count = sum(block["type"] == "thinking" and bool(block.get("signature")) for block in blocks)
        expected = ",".join(str(i * i) for i in range(1, 31))
        receipt = {"phase": "recovered", "opens": adapter.opens,
                   "reasoning_blocks": reasoning_count, "text": answer,
                   "exactly_once": answer.replace(" ", "") == expected}
        print(json.dumps(receipt), flush=True)
        assert adapter.opens == 2 and reasoning_count > 0 and receipt["exactly_once"]
        body["messages"] += [{"role": "assistant", "content": blocks},
                             {"role": "user", "content": "What is the last square? Reply only the number."}]
        followup = await sdk_blocks([frame async for frame in adapter.call(body, model)])
        result = "".join(block.get("text", "") for block in followup if block["type"] == "text")
        print(json.dumps({"phase": "sdk_history_followup", "text": result,
                          "correct": result.strip() == "900", "opens": adapter.opens}), flush=True)
        assert result.strip() == "900" and adapter.opens == 3
    finally:
        await adapter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    asyncio.run(asyncio.wait_for(run(args.model), timeout=180))
