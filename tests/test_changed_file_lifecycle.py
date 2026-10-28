from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from agent_bridge.changed_file_lifecycle import ChangedFileLifecycleMixin
from agent_bridge.events import AgentEvent, AgentEventKind
from claude_code_bridge.bridge import ClaudeCodeBridge
from codex_bridge.bridge import CodexBridge


class _Bridge(ChangedFileLifecycleMixin):
    def __init__(self, cwd: Path):
        self._publisher = MagicMock()
        self._backend_name = "test-bridge"
        self._cwd = str(cwd)


def _event(kind: AgentEventKind, *, tool_use_id: str, path: Path) -> AgentEvent:
    return AgentEvent(
        event_id=f"e-{kind.value}",
        session_key="s",
        run_id="req",
        kind=kind,
        origin="system",
        tool_name="Edit",
        tool_arguments={"file_path": str(path)},
        tool_use_id=tool_use_id,
        provider="test-bridge",
        trigger_message_id="msg",
    )


def test_shared_lifecycle_publishes_incremental_file_event(tmp_path: Path):
    path = tmp_path / "a.py"
    path.write_text("before\n")
    bridge = _Bridge(tmp_path)

    bridge._publish_run_event_with_changed_file(
        "req", _event(AgentEventKind.TOOL_START, tool_use_id="tool-1", path=path)
    )
    path.write_text("after\n")
    bridge._publish_run_event_with_changed_file(
        "req", _event(AgentEventKind.TOOL_END, tool_use_id="tool-1", path=path)
    )

    events = [call.args[1] for call in bridge._publisher.publish_run_event.call_args_list]
    assert [event.kind for event in events] == [
        AgentEventKind.TOOL_START,
        AgentEventKind.TOOL_END,
        AgentEventKind.FILE_CHANGED,
    ]
    file_event = events[-1]
    assert file_event.raw["file"]["path"] == str(path)
    assert file_event.provider == "test-bridge"
    assert file_event.trigger_message_id == "msg"


def test_active_bridge_classes_share_the_lifecycle_contract():
    assert issubclass(ClaudeCodeBridge, ChangedFileLifecycleMixin)
    assert issubclass(CodexBridge, ChangedFileLifecycleMixin)
