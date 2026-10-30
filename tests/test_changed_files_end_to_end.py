from __future__ import annotations

from pathlib import Path

import pytest
from sqlmodel import SQLModel, create_engine

from agent_bridge.changed_files import ToolChangedFileCapture
from llm_bawt.service.changed_files_store import (
    TurnChangedFile,
    TurnDiffPayloadRecord,
    reset_diff_blob_backend,
)
from llm_bawt.service.changed_file_identity import WorkspaceIdentityResolver
from llm_bawt.service.tool_changed_files import ToolChangedFilesCoordinator


class StubResolver:
    """Deterministic identity resolver; never touches the network."""

    def __init__(self) -> None:
        self.payload: dict | None = None
        self.calls: list[tuple] = []

    def resolve(self, absolute_path, bot_id):
        self.calls.append((absolute_path, bot_id))
        if self.payload is None or absolute_path is None:
            return WorkspaceIdentityResolver.unresolved()
        return WorkspaceIdentityResolver.from_payload(self.payload)


@pytest.fixture()
def coordinator(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LLM_BAWT_DIFFBLOBS_FS_ROOT", str(tmp_path / "diffs"))
    monkeypatch.setenv("LLM_BAWT_STORAGE_BACKEND", "fs")
    reset_diff_blob_backend()
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(
        engine, tables=[TurnChangedFile.__table__, TurnDiffPayloadRecord.__table__]
    )
    yield ToolChangedFilesCoordinator(engine, identity_resolver=StubResolver())
    reset_diff_blob_backend()


def test_tool_capture_to_durable_summary_and_diff(tmp_path: Path, coordinator):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    path = repo / "app.py"
    path.write_text("print('one')\n")

    capture = ToolChangedFileCapture(cwd=repo)
    capture.start(request_id="turn-1", tool_use_id="tool-1", tool_name="Edit", arguments={"file_path": str(path)})
    path.write_text("print('one')\nprint('two')\n")
    finished = capture.finish(request_id="turn-1", tool_use_id="tool-1", tool_name="Edit")
    assert finished is not None

    persisted = coordinator.persist({
        "turn_id": "turn-1",
        "bot_id": "snark",
        "user_id": "nick",
        "trigger_message_id": "msg-1",
        "tool_use_id": "tool-1",
        "tool_name": finished[0],
        "arguments": finished[1],
        "file": finished[2],
    })
    assert persisted is not None
    assert persisted["summary"]["total_files"] == 1
    assert persisted["summary"]["total_additions"] == 1
    assert persisted["file"]["has_content"] is True

    repo_key = persisted["file"]["repo_key"]
    before, after = coordinator.store.get_file_content(
        turn_id="turn-1", repo_key=repo_key, path=str(path)
    )
    assert before.text == "print('one')\n"
    assert after.text == "print('one')\nprint('two')\n"


def test_overlapping_turns_keep_only_their_declared_files(tmp_path: Path, coordinator):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    a = repo / "a.py"
    b = repo / "b.py"
    a.write_text("a0\n")
    b.write_text("b0\n")

    first = ToolChangedFileCapture(cwd=repo)
    second = ToolChangedFileCapture(cwd=repo)
    first.start(request_id="turn-a", tool_use_id="tool-a", tool_name="Edit", arguments={"file_path": str(a)})
    second.start(request_id="turn-b", tool_use_id="tool-b", tool_name="Edit", arguments={"file_path": str(b)})
    a.write_text("a1\n")
    b.write_text("b1\n")

    for turn_id, tool_id, capture, tool_name in (
        ("turn-a", "tool-a", first, "Edit"),
        ("turn-b", "tool-b", second, "Edit"),
    ):
        finished = capture.finish(request_id=turn_id, tool_use_id=tool_id, tool_name=tool_name)
        assert finished is not None
        assert coordinator.persist({
            "turn_id": turn_id,
            "bot_id": "snark",
            "user_id": "nick",
            "trigger_message_id": f"msg-{turn_id}",
            "tool_use_id": tool_id,
            "tool_name": finished[0],
            "arguments": finished[1],
            "file": finished[2],
        }) is not None

    assert [f["path"] for f in coordinator.store.summary_for_turn("turn-a")["files"]] == [str(a)]
    assert [f["path"] for f in coordinator.store.summary_for_turn("turn-b")["files"]] == [str(b)]


def test_metadata_fallback_uses_completed_tool_arguments(coordinator):
    persisted = coordinator.persist({
        "turn_id": "turn-meta",
        "bot_id": "snark",
        "user_id": "nick",
        "trigger_message_id": "msg-meta",
        "tool_use_id": None,
        "tool_name": "Write",
        "arguments": {"file_path": "/workspace/generated.py"},
    })
    assert persisted is not None
    assert persisted["file"]["path"] == "/workspace/generated.py"
    assert persisted["file"]["has_content"] is False


def test_bridge_content_enriches_metadata_first_row(tmp_path: Path, coordinator):
    path = tmp_path / "enriched.py"
    path.write_text("before\n")
    capture = ToolChangedFileCapture(cwd=tmp_path)
    capture.start(
        request_id="turn-enrich",
        tool_use_id="tool-enrich",
        tool_name="Edit",
        arguments={"file_path": str(path)},
    )
    path.write_text("after\n")

    first = coordinator.persist({
        "turn_id": "turn-enrich",
        "bot_id": "snark",
        "user_id": "nick",
        "trigger_message_id": "msg-enrich",
        "tool_use_id": "tool-enrich",
        "tool_name": "Edit",
        "arguments": {"file_path": str(path)},
    })
    assert first is not None
    assert first["file"]["has_content"] is False

    finished = capture.finish(
        request_id="turn-enrich", tool_use_id="tool-enrich", tool_name="Edit"
    )
    assert finished is not None
    second = coordinator.persist({
        "turn_id": "turn-enrich",
        "bot_id": "snark",
        "user_id": "nick",
        "trigger_message_id": "msg-enrich",
        "tool_use_id": "tool-enrich",
        "tool_name": finished[0],
        "arguments": finished[1],
        "file": finished[2],
    })
    assert second is not None
    assert second["summary"]["total_files"] == 1
    assert second["file"]["has_content"] is True
    before, after = coordinator.store.get_file_content(
        turn_id="turn-enrich", repo_key="workspace", path=str(path)
    )
    assert before.text == "before\n"
    assert after.text == "after\n"


def _canonical_payload(path: Path) -> dict:
    return {
        "schemaVersion": 1,
        "repositories": [{"id": "repo-1", "name": "llm-bawt", "path": str(path.parent)}],
        "workspaces": [],
        "captureRoots": [],
        "resolved": {
            "repoId": "repo-1", "workspaceId": "canonical:repo-1", "kind": "canonical",
            "relativePath": path.name, "state": "ACTIVE", "ownerBotId": None,
        },
    }


def _event(turn: str, finished=None, path: Path | None = None) -> dict:
    event = {"turn_id": turn, "bot_id": "vex", "user_id": "u", "tool_use_id": "t1"}
    if finished is not None:
        event.update(tool_name=finished[0], arguments=finished[1], file=finished[2])
    else:
        event.update(tool_name="Edit", arguments={"file_path": str(path)})
    return event


def _rows(coordinator, turn: str) -> list[TurnChangedFile]:
    from sqlmodel import Session, select

    with Session(coordinator.store.engine) as session:
        return list(session.exec(select(TurnChangedFile).where(TurnChangedFile.turn_id == turn)))


def _capture_edit(tmp_path: Path, turn: str):
    path = tmp_path / "x.py"
    path.write_text("a\n")
    capture = ToolChangedFileCapture(cwd=tmp_path)
    capture.start(request_id=turn, tool_use_id="t1", tool_name="Edit", arguments={"file_path": str(path)})
    path.write_text("b\n")
    return path, capture.finish(request_id=turn, tool_use_id="t1", tool_name="Edit")


def test_resolved_companion_rekeys_unresolved_fallback_row(tmp_path: Path, coordinator):
    path, finished = _capture_edit(tmp_path, "turn-rk")
    coordinator.persist(_event("turn-rk", path=path))  # fallback, unresolved
    assert [r.repo_key for r in _rows(coordinator, "turn-rk")] == ["workspace"]

    coordinator.identity_resolver.payload = _canonical_payload(path)
    persisted = coordinator.persist(_event("turn-rk", finished))
    rows = _rows(coordinator, "turn-rk")
    assert len(rows) == 1
    row = rows[0]
    assert row.repo_key == "canonical:repo-1" and row.path == str(path)
    assert row.identity_status == "resolved"
    assert row.workspace_repo_id == "repo-1" and row.workspace_repo_name == "llm-bawt"
    assert row.workspace_kind == "canonical" and row.repo_relative_path == "x.py"
    assert row.provenance == "tool" and row.git_base_sha is None
    assert persisted["file"]["workspace_id"] == "canonical:repo-1"
    assert coordinator.identity_resolver.calls[-1] == (str(path), "vex")


def test_unresolved_fallback_after_resolved_merges_into_resolved_row(tmp_path: Path, coordinator):
    path, finished = _capture_edit(tmp_path, "turn-mg")
    coordinator.identity_resolver.payload = _canonical_payload(path)
    coordinator.persist(_event("turn-mg", finished))
    coordinator.identity_resolver.payload = None  # BawtHub unreachable now
    persisted = coordinator.persist(_event("turn-mg", path=path))
    rows = _rows(coordinator, "turn-mg")
    assert len(rows) == 1
    assert rows[0].repo_key == "canonical:repo-1"
    assert rows[0].identity_status == "resolved"
    assert persisted is not None and persisted["file"]["repo_key"] == "canonical:repo-1"


def test_resolver_failure_keeps_raw_path_unresolved(tmp_path: Path, coordinator):
    path, finished = _capture_edit(tmp_path, "turn-un")
    coordinator.persist(_event("turn-un", finished))
    (row,) = _rows(coordinator, "turn-un")
    assert row.repo_key == "workspace" and row.path == str(path)
    assert row.identity_status == "unresolved" and row.workspace_id is None


def test_isolated_payload_carries_base_sha_and_name():
    payload = {
        "repositories": [{"id": "r", "name": "bawthub", "path": "/x"}],
        "workspaces": [{"id": "ws-9", "repoId": "r", "kind": "isolated", "baseSha": "abc123"}],
        "resolved": {"repoId": "r", "workspaceId": "ws-9", "kind": "isolated", "relativePath": "a/b.ts"},
    }
    got = WorkspaceIdentityResolver.from_payload(payload)
    assert got["repo_key"] == "ws-9" and got["git_base_sha"] == "abc123"
    assert got["workspace_repo_name"] == "bawthub" and got["repo_relative_path"] == "a/b.ts"


@pytest.mark.parametrize("payload", [
    {"resolved": None},
    {"resolved": {"repoId": "r", "workspaceId": "w", "kind": "weird"}},
    {"resolved": {"workspaceId": "w", "kind": "canonical", "relativePath": "a"}},
    {"resolved": {"repoId": "", "workspaceId": "w", "kind": "canonical", "relativePath": "a"}},
    {"resolved": {"repoId": "r", "workspaceId": " ", "kind": "canonical", "relativePath": "a"}},
    {"resolved": {"repoId": "r", "workspaceId": "w", "kind": "canonical"}},
    {"resolved": {"repoId": "r", "workspaceId": "w", "kind": "canonical", "relativePath": ""}},
    {"resolved": {"repoId": "r", "workspaceId": "w", "kind": "canonical", "relativePath": 7}},
    "not-a-dict",
])
def test_unusable_payloads_are_unresolved(payload):
    assert WorkspaceIdentityResolver.from_payload(payload) == WorkspaceIdentityResolver.unresolved()


def test_resolver_rejects_relative_and_survives_transport_errors(monkeypatch):
    resolver = WorkspaceIdentityResolver(base_url="http://bawthub.invalid")
    assert resolver.resolve("rel/path.py", "vex") == WorkspaceIdentityResolver.unresolved()

    import httpx

    def boom(self, *a, **k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx.Client, "get", boom)
    assert resolver.resolve("/abs/path.py", "vex") == WorkspaceIdentityResolver.unresolved()
