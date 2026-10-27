"""TASK-1585 slice 1: tool-result accounting provenance and the codex media fix.

The codex bridge used to publish structured MCP results as ``str(result)``, a
Python repr, so inline image base64 was tokenized as text (701k "tokens" for a
screenshot). These tests drive the real codex tool-end handler through the
real ``CodexBridge._publish_event`` and check the emitted versioned payload,
then persist it through the app's coordinator and store.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from agent_bridge.events import AgentEventKind
from agent_bridge.tool_results import (
    TOOL_RESULT_ACCOUNTING_VERSION,
    ToolResultPayload,
    payload_from_event,
)
from codex_bridge.bridge import CodexBridge
from llm_bawt.service.tool_call_store import ToolCallRecord, ToolCallResultPayloadRecord
from llm_bawt.service.tool_event_coordinator import ToolEventCoordinator
from llm_bawt.service.turn_logs import TurnLog

# Small enough that tokenizing the old repr stays inside the unit budget.
BASE64_PNG = "iVBORw0KGgo" + "A" * 4_000
OTHER_PNG = "iVBORw0KGgo" + "B" * 4_000
TEXT = "### Result\n- [Screenshot of viewport](/home/bridge/dev/.playwright-mcp/page.png)\n" + "word " * 1000


def _bridge() -> CodexBridge:
    bridge = CodexBridge.__new__(CodexBridge)
    bridge._backend_name = "codex"
    bridge._trigger_message_ids = {}
    bridge.emitted = []
    bridge._publish_run_event_with_changed_file = lambda request_id, event: bridge.emitted.append(event)
    return bridge


def _mcp_item(result, item_id="t1"):
    return {"type": "mcpToolCall", "id": item_id, "tool": "browser_take_screenshot", "result": result}


def _screenshot_result(image=BASE64_PNG):
    return {"content": [
        {"type": "text", "text": TEXT},
        {"type": "image", "data": image, "mimeType": "image/png"},
    ]}


def _tool_end(bridge, item):
    buffers = {item["id"]: {"tool_name": item["tool"]}}
    bridge._emit_tool_end(item, "req", "bot:user", 0, buffers, [])
    (event,) = [e for e in bridge.emitted if e.kind == AgentEventKind.TOOL_END]
    return event


def test_codex_mcp_image_result_publishes_accounted_payload_without_base64():
    event = _tool_end(_bridge(), _mcp_item(_screenshot_result()))

    assert isinstance(event.tool_result, str)
    payload = payload_from_event(event.to_dict()["tool_result_payload"], event.tool_result)
    assert payload.accounting == TOOL_RESULT_ACCOUNTING_VERSION
    assert payload.media_blocks == 1
    assert BASE64_PNG[:64] not in payload.content
    assert payload.total_tokens < 3_000
    # Path shortening still applies inside the structured value.
    assert "/home/bridge/" not in payload.content and "~/dev/.playwright-mcp" in payload.content


def test_codex_sdk_object_result_is_structured_too():
    class Block:
        def __init__(self, **kw):
            self.kw = kw

        def model_dump(self, mode="json"):
            return dict(self.kw)

    class Result:
        def model_dump(self, mode="json"):
            return {"content": [Block(type="text", text="hi").model_dump(),
                                {"type": "image", "data": BASE64_PNG, "mimeType": "image/png"}]}

    event = _tool_end(_bridge(), _mcp_item(Result()))
    payload = payload_from_event(event.to_dict()["tool_result_payload"], event.tool_result)
    assert payload.media_blocks == 1 and BASE64_PNG[:64] not in payload.content


def test_codex_string_result_still_publishes_unchanged_with_paths_shortened():
    event = _tool_end(_bridge(), _mcp_item("saved /home/bridge/dev/out.txt"))

    assert event.tool_result == "saved ~/dev/out.txt"
    payload = payload_from_event(event.to_dict()["tool_result_payload"], event.tool_result)
    assert payload.content == "saved ~/dev/out.txt"
    assert payload.media_blocks == 0 and payload.accounting == TOOL_RESULT_ACCOUNTING_VERSION


def test_different_images_with_identical_text_share_sha_but_carry_media_blocks():
    first = ToolResultPayload.from_producer(_screenshot_result(BASE64_PNG))
    second = ToolResultPayload.from_producer(_screenshot_result(OTHER_PNG))

    # The hash cannot tell the images apart, so the report must exclude media rows.
    assert first.sha256 == second.sha256
    assert first.media_blocks == second.media_blocks == 1


def test_unaccounted_fallbacks_stay_unverified():
    # Old producer: no payload, only a result string.
    assert payload_from_event(None, "repr with base64").accounting is None
    # Legacy app-side from_value callers do not claim producer accounting.
    assert ToolResultPayload.from_value("x").accounting is None
    # A payload dict from an old producer lacks the provenance fields.
    old = ToolResultPayload.from_value("x").to_dict()
    old.pop("accounting"), old.pop("media_blocks")
    restored = ToolResultPayload.from_dict(old)
    assert restored.accounting is None and restored.media_blocks is None


@pytest.mark.parametrize("bad", [True, -1, 1.5, "2", None])
def test_from_dict_rejects_malformed_provenance(bad):
    data = ToolResultPayload.from_producer("x").to_dict()
    data["accounting"] = bad
    data["media_blocks"] = bad
    restored = ToolResultPayload.from_dict(data)
    assert restored.accounting is None and restored.media_blocks is None


def test_unknown_accounting_version_is_not_trusted():
    data = ToolResultPayload.from_producer("x").to_dict()
    data["accounting"] = TOOL_RESULT_ACCOUNTING_VERSION + 1
    assert ToolResultPayload.from_dict(data).accounting is None


@pytest.fixture
def engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(
        engine, tables=[TurnLog.__table__, ToolCallRecord.__table__, ToolCallResultPayloadRecord.__table__],
    )
    with Session(engine) as session:
        session.add(TurnLog(id="turn", bot_id="bot", user_id="user", status="ok",
                            ended_at=datetime.now(timezone.utc)))
        session.commit()
    return engine


def _persist(engine, event_dict):
    coordinator = ToolEventCoordinator(engine)
    coordinator.end({"turn_id": "turn", "call_id": "c1", "tool_name": "browser_take_screenshot",
                     "bot_id": "bot", "user_id": "user", "ts": 1.0, **event_dict})
    with Session(engine) as session:
        return session.exec(select(ToolCallRecord)).one()


def test_store_persists_producer_provenance(engine):
    event = _tool_end(_bridge(), _mcp_item(_screenshot_result()))
    row = _persist(engine, {"result": event.tool_result,
                            "tool_result_payload": event.to_dict()["tool_result_payload"]})
    assert row.result_accounting == TOOL_RESULT_ACCOUNTING_VERSION
    assert row.result_media_blocks == 1
    assert row.result_total_tokens < 3_000


def test_store_leaves_old_producer_rows_unverified(engine):
    row = _persist(engine, {"result": str(_screenshot_result())})
    assert row.result_accounting is None and row.result_media_blocks is None
