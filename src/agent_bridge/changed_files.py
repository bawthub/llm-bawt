"""Tool-call-scoped changed-file capture.

File attribution comes from completed file-modifying tool calls, never from a
workspace-wide before/after diff.  The bridge snapshots only the declared
``file_path`` at TOOL_START and TOOL_END so the app can persist an exact,
per-tool before/after payload without attributing concurrent workspace changes
to the wrong turn.
"""

from __future__ import annotations

import base64
import difflib
import hashlib
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_BYTES_PER_SIDE = 1_048_576
FILE_MODIFYING_TOOLS = frozenset({"Edit", "Write", "NotebookEdit", "file_change"})


def canonical_tool_name(tool_name: str | None) -> str:
    """Return the leaf tool name (MCP-prefixed names remain compatible)."""
    return str(tool_name or "").rsplit("__", 1)[-1]


def is_file_modifying_tool(tool_name: str | None) -> bool:
    return canonical_tool_name(tool_name) in FILE_MODIFYING_TOOLS


@dataclass(frozen=True)
class _FileSnapshot:
    exists: bool
    data: bytes | None
    binary: bool
    truncated: bool
    digest: str | None


@dataclass(frozen=True)
class _PendingCapture:
    tool_name: str
    arguments: dict[str, Any]
    absolute_path: Path
    repo_key: str
    repo_label: str
    repo_path: str
    in_repo: bool
    repo_root: Path | None
    before: _FileSnapshot


class ToolChangedFileCapture:
    """Capture only paths explicitly named by Edit/Write/NotebookEdit calls."""

    def __init__(self, *, cwd: str | Path, max_bytes_per_side: int = MAX_BYTES_PER_SIDE):
        self._cwd = Path(cwd).expanduser()
        self._max_bytes_per_side = max_bytes_per_side
        self._pending: dict[tuple[str, str], _PendingCapture] = {}
        self._pending_without_id: dict[tuple[str, str], list[_PendingCapture]] = {}
        self._baselines: dict[tuple[str, Path], _FileSnapshot] = {}

    @staticmethod
    def _key(request_id: str, tool_use_id: str | None) -> tuple[str, str] | None:
        if not request_id or not tool_use_id:
            return None
        return request_id, tool_use_id

    def start(
        self,
        *,
        request_id: str,
        tool_use_id: str | None,
        tool_name: str | None,
        arguments: dict[str, Any] | None,
    ) -> None:
        """Remember the declared file and its bytes immediately before execution."""
        key = self._key(request_id, tool_use_id)
        if not request_id or not is_file_modifying_tool(tool_name):
            return
        args = arguments if isinstance(arguments, dict) else {}
        raw_path = args.get("file_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return
        try:
            absolute_path = self._resolve_path(raw_path)
            repo_root = self._find_repo_root(absolute_path)
            repo_label = repo_root.name if repo_root is not None else absolute_path.parent.name
            repo_key = "workspace"
            # Keep the exact normalized tool argument as the persistence key.
            # The app-side TOOL_END path sees this same value before the richer
            # bridge companion arrives, so both upserts converge on one row.
            repo_path = raw_path.replace("\\", "/")
            pending = _PendingCapture(
                tool_name=canonical_tool_name(tool_name),
                arguments=dict(args),
                absolute_path=absolute_path,
                repo_key=repo_key,
                repo_label=repo_label or repo_key,
                repo_path=repo_path,
                in_repo=repo_root is not None,
                repo_root=repo_root,
                before=self._baselines.setdefault(
                    (request_id, absolute_path), self._read_snapshot(absolute_path)
                ),
            )
            if key is not None:
                self._pending[key] = pending
            else:
                self._pending_without_id.setdefault(
                    (request_id, canonical_tool_name(tool_name)), []
                ).append(pending)
        except Exception:
            logger.debug("tool changed-file capture failed to start", exc_info=True)

    def finish(
        self,
        *,
        request_id: str,
        tool_use_id: str | None,
        tool_name: str | None = None,
    ) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
        """Return ``(tool_name, arguments, captured_file)`` for one completed tool."""
        key = self._key(request_id, tool_use_id)
        pending = self._pending.pop(key, None) if key is not None else None
        if pending is None and request_id and is_file_modifying_tool(tool_name):
            pending_key = (request_id, canonical_tool_name(tool_name))
            queue = self._pending_without_id.get(pending_key)
            if queue:
                pending = queue.pop(0)
                if not queue:
                    self._pending_without_id.pop(pending_key, None)
        if pending is None:
            return None
        try:
            after = self._read_snapshot(pending.absolute_path)
            if (
                pending.before.exists == after.exists
                and pending.before.binary == after.binary
                and pending.before.digest == after.digest
                and not pending.before.truncated
                and not after.truncated
            ):
                return None
            additions, deletions = self._line_counts(pending.before, after)
            if not pending.before.exists and after.exists:
                change_kind = "added"
            elif pending.before.exists and not after.exists:
                change_kind = "deleted"
            else:
                change_kind = "modified"
            binary = pending.before.binary or after.binary
            captured = {
                "repo_key": pending.repo_key,
                "repo_label": pending.repo_label,
                "path": pending.repo_path,
                "in_repo": pending.in_repo,
                "change_kind": change_kind,
                "additions": additions,
                "deletions": deletions,
                "binary": binary,
                "truncated": pending.before.truncated or after.truncated,
                "content_type": None if binary else "text/plain; charset=utf-8",
                "before_b64": self._encode(pending.before.data) if pending.before.exists else None,
                "after_b64": self._encode(after.data) if after.exists else None,
                # TASK-1275 S2: raw evidence for app-side workspace identity.
                # The app resolves repo/workspace identity at persist time;
                # the bridge only reports what it observed at TOOL_END.
                "absolute_path": str(pending.absolute_path),
                "provenance": "tool",
                **self._git_facts(pending.repo_root),
            }
            return pending.tool_name, pending.arguments, captured
        except Exception:
            logger.debug("tool changed-file capture failed to finish", exc_info=True)
            return pending.tool_name, pending.arguments, {
                "repo_key": pending.repo_key,
                "repo_label": pending.repo_label,
                "path": pending.repo_path,
                "in_repo": pending.in_repo,
                "change_kind": "modified",
                "binary": False,
                "truncated": False,
            }

    def discard_request(self, request_id: str) -> None:
        stale = [key for key in self._pending if key[0] == request_id]
        for key in stale:
            self._pending.pop(key, None)
        stale_queues = [key for key in self._pending_without_id if key[0] == request_id]
        for key in stale_queues:
            self._pending_without_id.pop(key, None)
        baseline_keys = [key for key in self._baselines if key[0] == request_id]
        for key in baseline_keys:
            self._baselines.pop(key, None)

    def _resolve_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = self._cwd / candidate
        return candidate.resolve(strict=False)

    @staticmethod
    def _git_facts(repo_root: Path | None) -> dict[str, str]:
        """Branch and HEAD of the checkout at TOOL_END; omitted when unknown."""
        if repo_root is None:
            return {}
        facts: dict[str, str] = {}
        for key, args in (
            ("git_head_sha", ("rev-parse", "--verify", "-q", "HEAD")),
            ("git_branch", ("symbolic-ref", "-q", "--short", "HEAD")),
        ):
            try:
                result = subprocess.run(
                    ("git", "-C", str(repo_root), *args),
                    capture_output=True, text=True, timeout=2, check=False,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            value = result.stdout.strip()
            if result.returncode == 0 and value:
                facts[key] = value
        return facts

    @staticmethod
    def _find_repo_root(path: Path) -> Path | None:
        current = path if path.is_dir() else path.parent
        for candidate in (current, *current.parents):
            if (candidate / ".git").exists():
                return candidate
        return None

    def _read_snapshot(self, path: Path) -> _FileSnapshot:
        try:
            if not path.is_file():
                return _FileSnapshot(False, None, False, False, None)
            with path.open("rb") as handle:
                data = handle.read(self._max_bytes_per_side + 1)
            truncated = len(data) > self._max_bytes_per_side
            if truncated:
                data = data[: self._max_bytes_per_side]
            binary = b"\0" in data[:8192]
            return _FileSnapshot(
                True,
                None if binary else data,
                binary,
                truncated,
                hashlib.sha256(data).hexdigest(),
            )
        except OSError:
            return _FileSnapshot(False, None, False, False, None)

    @staticmethod
    def _encode(data: bytes | None) -> str | None:
        if data is None:
            return None
        return base64.b64encode(data).decode("ascii")

    @staticmethod
    def _line_counts(before: _FileSnapshot, after: _FileSnapshot) -> tuple[int | None, int | None]:
        if before.binary or after.binary or before.data is None and before.exists or after.data is None and after.exists:
            return None, None
        before_lines = (before.data or b"").decode("utf-8", errors="replace").splitlines()
        after_lines = (after.data or b"").decode("utf-8", errors="replace").splitlines()
        additions = 0
        deletions = 0
        for line in difflib.ndiff(before_lines, after_lines):
            if line.startswith("+ "):
                additions += 1
            elif line.startswith("- "):
                deletions += 1
        return additions, deletions
