"""TASK-2023: Codex shell edits reach the changes panel via turn snapshots."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_bridge.events import AgentEventKind
from codex_bridge.bridge import CodexBridge

SESSION = "vex:nick"


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ("git", "-C", str(cwd), *args), check=True, capture_output=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x",
             "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x"},
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "kept.txt").write_text("kept\n")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "init")
    return root


def _bridge(tmp_path: Path) -> CodexBridge:
    bridge = object.__new__(CodexBridge)
    bridge._publisher = MagicMock()
    bridge._backend_name = "codex"
    bridge._cwd = str(tmp_path)
    bridge._trigger_message_ids = {}
    bridge._turn_token_usage = AsyncMock(return_value=None)
    return bridge


async def _event(bridge: CodexBridge, request_id: str, event: SimpleNamespace, seq: int = 0) -> int:
    seq, _, _ = await bridge._handle_event(
        event, bot_slug="vex", session_key=SESSION, request_id=request_id, seq=seq,
        text_parts=[], item_text={}, model="gpt", actual_model_ref=["gpt"],
        resume_id="thread-1", session_persisted=True, item_buffers={},
        thread=SimpleNamespace(id="thread-1"),
    )
    return seq


def _command(command: str, cwd: Path) -> SimpleNamespace:
    return SimpleNamespace(
        type="item.started",
        item=SimpleNamespace(type="command_execution", id="cmd-1", command=command, cwd=str(cwd)),
    )


def _published(bridge: CodexBridge, request_id: str) -> list:
    return [c.args[1] for c in bridge._publisher.publish_run_event.call_args_list if c.args[0] == request_id]


def _command_rows(events: list) -> list[dict]:
    return [e.raw["file"] for e in events if e.kind == AgentEventKind.FILE_CHANGED and e.tool_name == "Bash"]


@pytest.mark.asyncio
@pytest.mark.parametrize("oversized_untracked", [False, True])
async def test_codex_shell_add_then_delete_are_observed(repo: Path, tmp_path: Path, monkeypatch, oversized_untracked) -> None:
    if oversized_untracked:
        monkeypatch.setattr("agent_bridge.turn_snapshots.MAX_UNTRACKED_BYTES", 32)
        (repo / "excluded.txt").write_text("x" * 33)
    bridge = _bridge(tmp_path)

    # Turn 1: the command names the tree, then creates a file.
    await _event(bridge, "r1", SimpleNamespace(type="thread.started", thread_id="thread-1"))
    await _event(bridge, "r1", _command(f"cd {repo} && echo hi > new.txt", tmp_path))
    (repo / "new.txt").write_text("hi\n")
    await _event(bridge, "r1", SimpleNamespace(type="turn.completed", usage=None))

    events = _published(bridge, "r1")
    rows = _command_rows(events)
    assert [(Path(r["path"]).name, r["change_kind"], r["provenance"]) for r in rows] == [
        ("new.txt", "added", "command_observation"),
    ]
    assert events[-1].kind == AgentEventKind.ASSISTANT_DONE

    # Turn 2: the delete lands before Codex's item/started notification is
    # handled (the race). The tree carried from turn 1 was snapshotted at
    # thread.started, so the delete is still observed.
    await _event(bridge, "r2", SimpleNamespace(type="thread.started", thread_id="thread-1"))
    (repo / "new.txt").unlink()
    await _event(bridge, "r2", _command("rm new.txt", repo))
    await _event(bridge, "r2", SimpleNamespace(type="turn.completed", usage=None))

    rows = _command_rows(_published(bridge, "r2"))
    assert [(Path(r["path"]).name, r["change_kind"]) for r in rows] == [("new.txt", "deleted")]


@pytest.mark.asyncio
async def test_read_only_tree_is_not_carried(repo: Path, tmp_path: Path) -> None:
    """A tree the session only read must not be re-registered next turn."""
    bridge = _bridge(tmp_path)
    await _event(bridge, "r1", _command(f"cat {repo}/kept.txt", tmp_path))
    await _event(bridge, "r1", SimpleNamespace(type="turn.completed", usage=None))
    assert _command_rows(_published(bridge, "r1")) == []
    assert getattr(bridge, "_turn_snapshot_carried", {}).get(SESSION) is None

    observe = AsyncMock()
    bridge._observe_tool_for_turn_snapshot = observe  # type: ignore[method-assign]
    await _event(bridge, "r2", SimpleNamespace(type="thread.started", thread_id="thread-1"))
    observe.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_commands_are_joined_for_path_discovery(repo: Path, tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    await _event(bridge, "r1", _command(["bash", "-lc", f"touch {repo}/x.txt"], tmp_path))  # type: ignore[arg-type]
    (repo / "x.txt").write_text("")
    await _event(bridge, "r1", SimpleNamespace(type="turn.completed", usage=None))
    assert [Path(r["path"]).name for r in _command_rows(_published(bridge, "r1"))] == ["x.txt"]


@pytest.mark.asyncio
async def test_observe_failure_never_blocks_the_command(repo: Path, tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    bridge._observe_tool_for_turn_snapshot = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
    bridge._begin_turn_snapshot = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]

    await _event(bridge, "r1", SimpleNamespace(type="thread.started", thread_id="thread-1"))
    await _event(bridge, "r1", _command(f"touch {repo}/y.txt", tmp_path))

    kinds = [e.kind for e in _published(bridge, "r1")]
    assert AgentEventKind.TOOL_START in kinds
    bridge._observe_tool_for_turn_snapshot.assert_awaited_once()


@pytest.mark.asyncio
async def test_disabled_flag_skips_codex_observation(repo: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_TURN_SNAPSHOTS", "0")
    bridge = _bridge(tmp_path)
    await _event(bridge, "r1", _command(f"touch {repo}/z.txt", tmp_path))
    (repo / "z.txt").write_text("")
    await _event(bridge, "r1", SimpleNamespace(type="turn.completed", usage=None))
    assert _command_rows(_published(bridge, "r1")) == []
