from __future__ import annotations

import base64
from pathlib import Path

from agent_bridge.changed_files import ToolChangedFileCapture, is_file_modifying_tool


def _decode(value: str | None) -> bytes | None:
    return base64.b64decode(value) if value is not None else None


def test_recognizes_phase_one_file_tools_and_excludes_bash():
    assert is_file_modifying_tool("Edit")
    assert is_file_modifying_tool("mcp__claude__Write")
    assert is_file_modifying_tool("NotebookEdit")
    assert is_file_modifying_tool("file_change")
    assert not is_file_modifying_tool("Bash")
    assert not is_file_modifying_tool("Read")


def test_capture_reports_only_the_declared_file(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    target = repo / "target.py"
    other = repo / "other.py"
    target.write_text("one\n")
    other.write_text("old\n")

    capture = ToolChangedFileCapture(cwd=repo)
    capture.start(
        request_id="turn-a",
        tool_use_id="tool-1",
        tool_name="Edit",
        arguments={"file_path": str(target)},
    )
    target.write_text("one\ntwo\n")
    other.write_text("changed by another turn\n")

    finished = capture.finish(request_id="turn-a", tool_use_id="tool-1", tool_name="Edit")
    assert finished is not None
    tool_name, arguments, changed = finished
    assert tool_name == "Edit"
    assert arguments["file_path"] == str(target)
    assert changed["repo_key"] == "workspace"
    assert changed["path"] == str(target)
    assert changed["change_kind"] == "modified"
    assert changed["additions"] == 1
    assert changed["deletions"] == 0
    assert _decode(changed["before_b64"]) == b"one\n"
    assert _decode(changed["after_b64"]) == b"one\ntwo\n"


def test_repeated_edits_keep_first_before_and_latest_after(tmp_path: Path):
    target = tmp_path / "a.py"
    target.write_text("v1\n")
    capture = ToolChangedFileCapture(cwd=tmp_path)

    capture.start(request_id="turn-r", tool_use_id="tool-1", tool_name="Edit", arguments={"file_path": str(target)})
    target.write_text("v2\n")
    assert capture.finish(request_id="turn-r", tool_use_id="tool-1", tool_name="Edit") is not None

    capture.start(request_id="turn-r", tool_use_id="tool-2", tool_name="Edit", arguments={"file_path": str(target)})
    target.write_text("v3\n")
    second = capture.finish(request_id="turn-r", tool_use_id="tool-2", tool_name="Edit")
    assert second is not None
    changed = second[2]
    assert _decode(changed["before_b64"]) == b"v1\n"
    assert _decode(changed["after_b64"]) == b"v3\n"


def test_idless_provider_calls_pair_sequentially(tmp_path: Path):
    target = tmp_path / "a.py"
    target.write_text("before\n")
    capture = ToolChangedFileCapture(cwd=tmp_path)
    capture.start(request_id="turn-o", tool_use_id=None, tool_name="Write", arguments={"file_path": str(target)})
    target.write_text("after\n")
    finished = capture.finish(request_id="turn-o", tool_use_id=None, tool_name="Write")
    assert finished is not None
    assert finished[2]["path"] == str(target)


def test_noop_write_emits_nothing(tmp_path: Path):
    target = tmp_path / "same.py"
    target.write_text("same\n")
    capture = ToolChangedFileCapture(cwd=tmp_path)
    capture.start(request_id="turn-n", tool_use_id="tool-n", tool_name="Edit", arguments={"file_path": str(target)})
    assert capture.finish(request_id="turn-n", tool_use_id="tool-n", tool_name="Edit") is None


def test_binary_file_keeps_metadata_without_content(tmp_path: Path):
    target = tmp_path / "image.bin"
    target.write_bytes(b"\x00old")
    capture = ToolChangedFileCapture(cwd=tmp_path)
    capture.start(request_id="turn-b", tool_use_id="tool-b", tool_name="Write", arguments={"file_path": str(target)})
    target.write_bytes(b"\x00new")
    finished = capture.finish(request_id="turn-b", tool_use_id="tool-b", tool_name="Write")
    assert finished is not None
    changed = finished[2]
    assert changed["binary"] is True
    assert changed["before_b64"] is None
    assert changed["after_b64"] is None


def test_capture_records_git_head_and_branch(tmp_path: Path):
    import subprocess

    repo = tmp_path / "gitrepo"
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)
    run("init", "-q", "-b", "trunk")
    run("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x")
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    target = repo / "f.py"
    target.write_text("1\n")
    capture = ToolChangedFileCapture(cwd=repo)
    capture.start(request_id="r", tool_use_id="t", tool_name="Edit", arguments={"file_path": str(target)})
    target.write_text("2\n")
    finished = capture.finish(request_id="r", tool_use_id="t", tool_name="Edit")
    payload = finished[2]
    assert payload["git_head_sha"] == head
    assert payload["git_branch"] == "trunk"
    assert payload["provenance"] == "tool"
    assert payload["absolute_path"] == str(target)
