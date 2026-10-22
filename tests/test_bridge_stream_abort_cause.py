"""TASK-1231B: the OpenClaw bridge stream stamps the bridge's abort cause.

The cause rides the same ``update_turn`` write that ends the row aborted, so a
host-shutdown kill is distinguishable from a plain abort without moving
``end_reason`` (which clients switch on).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent_bridge.abort_contract import HOST_SHUTDOWN
from agent_bridge.events import AgentEvent, AgentEventKind
from llm_bawt.service.chat_streaming_bridge import ChatStreamingBridgeMixin


def _event(kind, *, raw=None, text=None, run_id="run-1"):
    return AgentEvent(event_id="e", session_key="s", run_id=run_id, kind=kind,
                      origin="user", text=text, raw=raw or {})


def _run(events):
    updates: list[dict] = []

    async def subscribe(_session_key):
        for event in events:
            yield event

    async def send_user_message(_session_key, _prompt):
        return "run-1"

    async def history(_session_key, limit):
        return []

    svc = ChatStreamingBridgeMixin.__new__(ChatStreamingBridgeMixin)
    svc._session_bridge = SimpleNamespace(
        send_user_message=send_user_message, _api_run_ids=set(),
        _fanout=SimpleNamespace(subscribe=subscribe),
        _ws_client=SimpleNamespace(get_chat_history=history),
    )
    svc._redis_subscriber = None
    svc._turn_log_store = SimpleNamespace(engine=None)
    svc._persist_turn_log = lambda **_kw: None
    svc._update_turn_log = lambda **kw: updates.append(kw)
    svc._finalize_turn = lambda **_kw: None

    async def drain():
        async for _chunk in svc._stream_via_bridge(
            user_prompt="hi", session_key="s", response_id="r", created=0, model_alias="m",
            bot_id="b", user_id="u", ctx=SimpleNamespace(request_id="q", path="/v1/chat/completions"),
            turn_log_id="turn-1",
            llm_bawt=SimpleNamespace(history_manager=SimpleNamespace(add_message=lambda *_a: None)),
        ):
            pass

    asyncio.run(drain())
    return [u for u in updates if u.get("status") == "aborted"]


@pytest.mark.parametrize("raw, cause", [
    ({"end_reason": "aborted", "abort_cause": HOST_SHUTDOWN}, HOST_SHUTDOWN),
    ({"end_reason": "aborted"}, None),                      # plain abort: no cause
    ({"end_reason": "aborted", "abort_cause": "forged"}, None),  # unknown cause is not trusted
], ids=["host-shutdown", "plain-abort", "unknown-cause"])
def test_aborted_done_ends_the_row_with_its_cause_in_one_write(raw, cause):
    aborted = _run([_event(AgentEventKind.ASSISTANT_DONE, raw=raw),
                    _event(AgentEventKind.RUN_COMPLETED)])
    assert aborted == [{"turn_id": "turn-1", "status": "aborted", "end_reason": "aborted",
                        "abort_cause": cause}]
