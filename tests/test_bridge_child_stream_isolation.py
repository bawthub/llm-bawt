"""TASK-2272: a worker's snapshot must not replace the parent's answer or usage."""
from __future__ import annotations

from types import SimpleNamespace


def _host():
    from claude_code_bridge import send_stream

    cls = next(v for v in vars(send_stream).values()
               if isinstance(v, type) and hasattr(v, "_on_assistant_message"))
    host = type("Host", (cls,), {})()
    host.events = []
    host._publish_event = lambda rid, sk, seq, **kw: host.events.append((seq, kw))
    host._is_ask_user_question = lambda name: False
    return host


def _msg(text, *, parent=None, usage=None):
    return SimpleNamespace(
        content=[{"type": "text", "text": text}], parent_tool_use_id=parent,
        usage=usage, message_id=None, model="m",
    )


def _call(host, msg, usage, snapshot):
    return host._on_assistant_message(
        msg, request_id="r", session_key="s", seq=0, tool_names_by_id={},
        tool_arguments_by_id={}, latest_assistant_usage=usage, assistant_snapshot_text=snapshot,
    )


def test_child_snapshot_keeps_parent_text_and_usage():
    host = _host()
    parent_usage = {"input_tokens": 10}
    _, usage, text = _call(host, _msg("CHILD PRIVATE DRAFT", parent="toolu_w",
                                      usage={"input_tokens": 99}), parent_usage, "PARENT")
    assert text == "PARENT" and usage == parent_usage


def test_top_level_snapshot_still_updates():
    host = _host()
    _, usage, text = _call(host, _msg("PARENT", usage={"input_tokens": 5}), None, "")
    assert text == "PARENT" and usage == {"input_tokens": 5}
