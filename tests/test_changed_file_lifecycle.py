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


def _git(cwd: Path, *args: str) -> None:
    import os
    import subprocess

    subprocess.run(
        ("git", "-C", str(cwd), *args), check=True, capture_output=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x",
             "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x"},
    )


def _terminal(kind: AgentEventKind) -> AgentEvent:
    return AgentEvent(
        event_id="done", session_key="s", run_id="req", kind=kind, origin="system",
        provider="test-bridge", trigger_message_id="msg",
    )


def test_command_edits_publish_before_terminal_and_skip_tool_paths(tmp_path: Path):
    """TASK-1218: a Bash edit lands as FILE_CHANGED before ASSISTANT_DONE."""
    import asyncio

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "edited.py").write_text("a\n")
    (repo / "sed.py").write_text("a\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "init")
    bridge = _Bridge(repo)

    asyncio.run(bridge._observe_tool_for_turn_snapshot(
        "req", "s", {"tool_name": "Bash", "tool_input": {"command": "sed -i s/a/b/ sed.py"}, "cwd": str(repo)},
    ))
    bridge._publish_run_event_with_changed_file(
        "req", _event(AgentEventKind.TOOL_START, tool_use_id="tool-1", path=repo / "edited.py"))
    (repo / "edited.py").write_text("edit\n")
    (repo / "sed.py").write_text("b\n")
    bridge._publish_run_event_with_changed_file(
        "req", _event(AgentEventKind.TOOL_END, tool_use_id="tool-1", path=repo / "edited.py"))
    bridge._publish_run_event_with_changed_file("req", _terminal(AgentEventKind.ASSISTANT_DONE))

    events = [call.args[1] for call in bridge._publisher.publish_run_event.call_args_list]
    assert [e.kind for e in events][-2:] == [AgentEventKind.FILE_CHANGED, AgentEventKind.ASSISTANT_DONE]
    command = [e for e in events if e.kind == AgentEventKind.FILE_CHANGED and e.tool_name == "Bash"]
    assert [Path(e.raw["file"]["path"]).name for e in command] == ["sed.py"]
    assert command[0].raw["file"]["provenance"] == "command_observation"
    assert command[0].trigger_message_id == "msg"


def test_mcp_tools_and_disabled_flag_do_not_observe(tmp_path: Path, monkeypatch):
    import asyncio

    bridge = _Bridge(tmp_path)
    asyncio.run(bridge._observe_tool_for_turn_snapshot(
        "req", "s", {"tool_name": "mcp__bawthub__x", "tool_input": {"path": str(tmp_path)}, "cwd": str(tmp_path)},
    ))
    monkeypatch.setenv("AGENT_TURN_SNAPSHOTS", "0")
    asyncio.run(bridge._observe_tool_for_turn_snapshot(
        "req", "s", {"tool_name": "Bash", "tool_input": {"command": "true"}, "cwd": str(tmp_path)},
    ))
    assert getattr(bridge, "_turn_snapshots", None) is None
