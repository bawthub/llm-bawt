"""TASK-1689: ~ shortening is limited to shell tool-card display.

Prose and non-shell tool arguments/results must round-trip byte-identical
through both bridges' ``_publish_event``.
"""
from __future__ import annotations

import pytest

from agent_bridge.events import AgentEventKind
from claude_code_bridge.event_ops import ClaudeEventMixin
from codex_bridge.bridge import CodexBridge

PATH = "/home/bridge/dev/llm-bawt/src/x.py"
PROSE = f"Edited {PATH} and /home/bridge."


class _Claude(ClaudeEventMixin):
    pass


def _make(cls):
    bridge = cls.__new__(cls)
    bridge._backend_name = "test"
    bridge._trigger_message_ids = {}
    bridge.emitted = []
    bridge._publish_run_event_with_changed_file = lambda _rid, event: bridge.emitted.append(event)
    return bridge


@pytest.mark.parametrize("cls,shell,other", [
    (_Claude, "Bash", "Edit"),
    (CodexBridge, "shell", "mcp__bawthub__tasks_update"),
])
def test_prose_and_non_shell_tools_round_trip_byte_identical(cls, shell, other):
    bridge = _make(cls)
    bridge._publish_event("r", "b:u", 1, kind=AgentEventKind.ASSISTANT_DONE, text=PROSE)
    args = {"file_path": PATH, "nested": {"p": PATH}}
    bridge._publish_event("r", "b:u", 2, kind=AgentEventKind.TOOL_START,
                          tool_name=other, tool_arguments=args)
    bridge._publish_event("r", "b:u", 3, kind=AgentEventKind.TOOL_END,
                          tool_name=other, tool_result=PROSE)

    prose, start, end = bridge.emitted
    assert prose.text == PROSE
    assert start.tool_arguments == {"file_path": PATH, "nested": {"p": PATH}}
    assert end.tool_result == PROSE


@pytest.mark.parametrize("cls,shell", [(_Claude, "Bash"), (CodexBridge, "shell")])
def test_shell_tool_card_still_shortens(cls, shell):
    bridge = _make(cls)
    bridge._publish_event("r", "b:u", 1, kind=AgentEventKind.TOOL_START,
                          tool_name=shell, tool_arguments={"command": f"cat {PATH}"})
    bridge._publish_event("r", "b:u", 2, kind=AgentEventKind.TOOL_END,
                          tool_name=shell, tool_result=f"{PATH}: ok")

    start, end = bridge.emitted
    assert start.tool_arguments == {"command": "cat ~/dev/llm-bawt/src/x.py"}
    assert end.tool_result == "~/dev/llm-bawt/src/x.py: ok"
