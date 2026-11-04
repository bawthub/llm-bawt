"""TASK-774: images ride a mid-turn steer into the live run.

The Claude bridge puts them on the same user message as image blocks for
``next``, ``now`` and an escalation replay; codex app-server adds local-image
items to ``turn/steer``. The wire field is parsed fail-closed.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agent_bridge.steer_contract import MAX_STEER_IMAGES, parse_steer_attachments
from tests.test_claude_active_run_next_steer import (
    CMD,
    FRAME,
    _INIT,
    _FakeClient,
    _finish,
    _lifecycle,
    _result,
    _run,
    _start,
)

PNG = base64.b64encode(b"\x89PNG fake").decode()
IMAGE = {"mimeType": "image/png", "content": PNG}
BLOCK = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}}


class _ImageClient(_FakeClient):
    """Records multimodal replacement queries written as one user frame."""

    async def query(self, prompt):
        if isinstance(prompt, str):
            return await super().query(prompt)
        frames = [frame async for frame in prompt]
        for frame in frames:
            if "uuid" in frame:
                self.written.append(frame)
                await self._query.frames.put(_lifecycle(frame["uuid"], "queued"))
            else:
                self.replacements.append(frame["message"]["content"])


def test_parse_steer_attachments_is_fail_closed():
    assert parse_steer_attachments(None) == []
    assert parse_steer_attachments(json.dumps([IMAGE])) == [IMAGE]
    for bad in ("{", json.dumps({"a": 1}), json.dumps([{"mimeType": "text/plain", "content": "x"}]),
                json.dumps([{"mimeType": "image/png", "content": ""}]),
                json.dumps([IMAGE] * (MAX_STEER_IMAGES + 1))):
        with pytest.raises(ValueError):
            parse_steer_attachments(bad)


@pytest.mark.anyio
async def test_next_steer_with_images_sends_one_multimodal_command():
    client = _ImageClient()
    run = _run(client)
    _yielded, task = await _start(run)
    await run.steer("look", priority="next", message_id=CMD, origin="user",
                    framed_message="NEXT[look]", interrupt_item="ITEM[look]", images=[IMAGE])
    [frame] = client.written
    assert frame["uuid"] == CMD and frame["priority"] == "next"
    assert frame["message"]["content"] == [{"type": "text", "text": "NEXT[look]"}, BLOCK]
    await client._query.frames.put(_result("done"))
    await _finish(client, task)


@pytest.mark.anyio
async def test_text_only_next_steer_content_stays_a_string():
    client = _ImageClient()
    run = _run(client)
    _yielded, task = await _start(run)
    await run.steer("plain", priority="next", message_id=CMD, origin="user",
                    framed_message="NEXT[plain]", interrupt_item="ITEM[plain]")
    assert client.written[0]["message"]["content"] == "NEXT[plain]"
    await _finish(client, task)


@pytest.mark.anyio
async def test_now_steer_with_images_replaces_with_a_multimodal_query():
    client = _ImageClient()
    run = _run(client)
    _yielded, task = await _start(run)
    await run.steer("stop", origin="user", framed_message="NOW[stop]", images=[IMAGE])
    client.interrupt.assert_awaited_once_with()
    assert client.replacements == [[{"type": "text", "text": "NOW[stop]"}, BLOCK]]
    await _finish(client, task)


@pytest.mark.anyio
async def test_escalation_replays_the_swept_commands_images():
    client = _ImageClient()
    await client._query.frames.put(_INIT)
    run = _run(client, [])
    _yielded, task = await _start(run)
    await run.steer("see this", priority="next", message_id=CMD, origin="user",
                    framed_message="NEXT[see this]", interrupt_item="ITEM[see this]", images=[IMAGE])
    await client._query.frames.put(_result("old turn"))
    await asyncio.sleep(0.01)
    assert await run.escalate(CMD, interrupt_frame=FRAME) == "escalated"
    assert client.replacements == [[{"type": "text", "text": "INTERRUPT[ITEM[see this]]END"}, BLOCK]]
    await client._query.frames.put(_lifecycle(CMD, "cancelled"))
    await client._query.frames.put(_result("replacement"))
    await _finish(client, task)


@pytest.mark.anyio
async def test_codex_steer_adds_local_images_and_tracks_temp_files():
    from codex_bridge.app_server import CodexAppServerRun

    client = AsyncMock()
    client.add_observer = lambda _fn: None
    run = CodexAppServerRun(client, request_id="r", thread_id="th")
    run.set_turn("t1")
    await run.steer("look", message_id=None, origin="user", steer_rpc_id=None, images=[IMAGE])
    method, params = client.request.await_args.args[:2]
    assert method == "turn/steer"
    text, image = params["input"]
    assert text == {"type": "text", "text": "look", "text_elements": []}
    assert image["type"] == "localImage" and image["path"] == run.tmp_paths[0]
    assert Path(run.tmp_paths[0]).read_bytes() == base64.b64decode(PNG)
    Path(run.tmp_paths[0]).unlink()
