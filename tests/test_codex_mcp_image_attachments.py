"""TASK-1324 — Codex bridge persists MCP image results as media attachments."""

from __future__ import annotations

import asyncio

from agent_bridge.events import AgentEventKind
from codex_bridge.bridge import CodexBridge
from codex_bridge.event_ops import TURN_IMAGES_KEY


def _bridge(upload_ids: list[str | None]) -> CodexBridge:
    bridge = CodexBridge.__new__(CodexBridge)
    bridge._backend_name = "codex"
    bridge._app_api_url = "http://app:8642"
    bridge._trigger_message_ids = {}
    bridge.emitted = []
    bridge.uploads = []
    bridge._publish_run_event_with_changed_file = lambda request_id, event: bridge.emitted.append(event)
    ids = iter(upload_ids)

    async def upload(data_url, user_id, tool_use_id):
        bridge.uploads.append((data_url[:22], user_id, tool_use_id))
        asset_id = next(ids)
        return {"asset_id": asset_id} if asset_id else None

    bridge._upload_data_url = upload
    return bridge


def _item(item_id: str, tool: str = "browser_take_screenshot") -> dict:
    return {"type": "mcpToolCall", "id": item_id, "tool": tool, "result": {"content": [
        {"type": "text", "text": "shot"},
        {"type": "image", "data": "aW1hZ2U=", "mimeType": "image/png"},
    ]}}


def _complete(bridge: CodexBridge, item: dict, buffers: dict) -> None:
    buffers.setdefault(item["id"], {"tool_name": item["tool"], "type": "mcp"})
    asyncio.run(bridge._persist_mcp_images(item, "caid:nick", buffers))
    bridge._emit_tool_end(item, "req", "caid:nick", 0, buffers, [])


def _tool_ends(bridge: CodexBridge) -> list:
    return [e for e in bridge.emitted if e.kind == AgentEventKind.TOOL_END]


def test_screenshot_uploads_and_attaches_to_tool_end_and_turn():
    bridge = _bridge(["ma_1", "ma_2"])
    buffers: dict = {}
    _complete(bridge, _item("a"), buffers)
    _complete(bridge, _item("b", "mcp__bawthub__generate_image"), buffers)

    assert bridge.uploads == [
        ("data:image/png;base64,", "nick", "a"),
        ("data:image/png;base64,", "nick", "b"),
    ]
    assert [e.attachments for e in _tool_ends(bridge)] == [
        [{"asset_id": "ma_1", "kind": "image"}],
        [{"asset_id": "ma_2", "kind": "image"}],
    ]
    assert buffers[TURN_IMAGES_KEY] == [
        {"asset_id": "ma_1", "kind": "image"},
        {"asset_id": "ma_2", "kind": "image"},
    ]


def test_non_image_tool_and_failed_upload_attach_nothing():
    bridge = _bridge([None])
    buffers: dict = {}
    _complete(bridge, _item("a", "browser_snapshot"), buffers)
    _complete(bridge, _item("b"), buffers)

    assert len(bridge.uploads) == 1  # only the screenshot tried to upload
    assert [e.attachments for e in _tool_ends(bridge)] == [None, None]
    assert TURN_IMAGES_KEY not in buffers


def test_sdk_model_result_is_dumped_before_extraction():
    class Result:
        def model_dump(self, mode="json"):
            return _item("x")["result"]

    bridge = _bridge(["ma_9"])
    buffers: dict = {}
    item = {"type": "mcpToolCall", "id": "x", "tool": "browser_take_screenshot", "result": Result()}
    _complete(bridge, item, buffers)
    assert _tool_ends(bridge)[0].attachments == [{"asset_id": "ma_9", "kind": "image"}]
