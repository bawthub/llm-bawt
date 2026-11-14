"""Per-turn Git snapshots for files changed through shell commands (TASK-1218).

Edit/Write calls are captured per tool (``changed_files.py``).  Shell commands
(sed, heredocs, formatters, codegen) are invisible to that path, so this module
follows the VS Code agent-host model: snapshot each touched working tree as a
Git tree at the start of the turn and again at the end, and diff the two.

* Snapshots use a temporary ``GIT_INDEX_FILE``; the real index, HEAD and
  branches are never modified.  Both trees are pinned under the private ref
  namespace ``refs/bawt/turns/<turn>/<tree>/{start,end}`` so a diff still has
  its objects after the worktree is removed.
* A tree is a candidate once a tool call's cwd or a path it names resolves
  inside it.  The start snapshot is taken from the PreToolUse hook, which the
  CLI awaits before running the tool, so it precedes the tool's effects.
* A tree where another turn was active at any point after this turn started
  using it is *ineligible*: both turns' snapshots would contain each other's
  edits, so nothing is attributed (tool-reported Edit/Write rows still count).
* HEAD may advance only through commits made during the turn *by this
  turn's bot* (author name or email local part); those stay in the diff. Any
  other HEAD movement — a rebase, checkout, a fast-forward of someone else's
  commits — makes the tree ineligible rather than attributing others' work.
* Tool calls that provably cannot write (``readonly_commands``) register no
  marker, so a reviewer's ``git diff`` or Read does not make the tree busy.
* Only registered bridge turns are visible to the concurrency rule.  An
  *uncommitted* edit by any other writer (a host shell, a process outside the
  bridges) during the turn is still attributed to the turn.
* Snapshot refs are kept only for trees with an observed diff.  They are
  ordinary refs, so ``git log --all`` lists them; use ``--exclude=refs/bawt/*``.
* Per-turn budgets bound the work: past them, rows are metadata-only (no
  content) or the tree is skipped as too large.
"""

from __future__ import annotations

import base64
import difflib
import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
import socket
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .readonly_commands import is_read_only_tool_call

logger = logging.getLogger(__name__)

PROVENANCE = "command_observation"
REF_PREFIX = "refs/bawt/turns"
MAX_BYTES_PER_SIDE = 1_048_576
# Untracked files larger than this are not copied into the object store.
MAX_UNTRACKED_BYTES = 16 * 1_048_576
MAX_FILES_PER_TREE = 500
MAX_FILES_PER_TURN = 300
# Diff bytes (before + after) shipped per turn; later rows are metadata-only.
MAX_CONTENT_BYTES_PER_TURN = 8 * 1_048_576
# A tree with more changed paths than this is not snapshotted at all.
MAX_STATUS_PATHS = 5000
# A registry marker older than this belongs to a dead turn and is ignored.
MARKER_STALE_SECONDS = 6 * 3600
REF_RETENTION_SECONDS = 30 * 86400
PRUNE_INTERVAL_SECONDS = 86400
GIT_TIMEOUT_SECONDS = 20
# Commits older than the turn's first use of a tree (with clock slack) were not
# made during the turn.
COMMIT_CLOCK_SLACK_SECONDS = 5

_SNAPSHOT_IDENTITY = {
    "GIT_AUTHOR_NAME": "bawt-turn-snapshot",
    "GIT_AUTHOR_EMAIL": "bawt-turn-snapshot@localhost",
    "GIT_COMMITTER_NAME": "bawt-turn-snapshot",
    "GIT_COMMITTER_EMAIL": "bawt-turn-snapshot@localhost",
}
_PATH_TOKEN = re.compile(r"(?:(?<=^)|(?<=[\s'\"=:(]))(/[^\s'\"`;|&<>()$*?]+)")
_SAFE_REF_PART = re.compile(r"[^A-Za-z0-9._-]+")
_PATH_ARG_KEYS = ("file_path", "path", "notebook_path", "cwd")


class GitError(RuntimeError):
    pass


class SnapshotTooLarge(RuntimeError):
    pass


# Identifies this bridge process in registry markers, so markers a crashed or
# restarted process left behind are recognised as dead immediately.
_PROCESS_TOKEN = f"{socket.gethostname()}:{uuid.uuid4().hex}"
_PROCESS_STARTED = time.time()
# Turns with a live observation in this process, across observer instances.
_LIVE_TURNS: set[str] = set()
_LIVE_TURNS_LOCK = threading.Lock()


def _git(
    cwd: Path,
    *args: str,
    env: dict[str, str] | None = None,
    input_bytes: bytes | None = None,
    check: bool = True,
) -> bytes:
    full_env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"}
    if env:
        full_env.update(env)
    try:
        result = subprocess.run(
            ("git", "-C", str(cwd), *args),
            input=input_bytes,
            capture_output=True,
            env=full_env,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GitError(f"git {args[0]} failed: {exc}") from exc
    if check and result.returncode != 0:
        raise GitError(
            f"git {args[0]} exited {result.returncode}: "
            f"{result.stderr.decode('utf-8', 'replace').strip()[:300]}"
        )
    return result.stdout


def _git_text(cwd: Path, *args: str, check: bool = True, **kw: Any) -> str:
    return _git(cwd, *args, check=check, **kw).decode("utf-8", "replace").strip()


def find_tree_root(path: Path) -> Path | None:
    """Nearest ancestor holding a ``.git`` entry (repo or linked worktree)."""
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _inode_key(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_dev, st.st_ino


def file_identity(path: Path) -> tuple[Any, str] | None:
    """Identity stable across bind-mount aliases: (parent inode, basename)."""
    parent = _inode_key(path.parent)
    if parent is None:
        return None
    return parent, path.name


def referenced_paths(tool_input: Any, cwd: str | None) -> list[Path]:
    """Absolute paths a tool call names: cwd, path arguments and command text."""
    base = Path(cwd) if isinstance(cwd, str) and cwd.startswith("/") else None
    found: list[Path] = []
    if base is not None:
        found.append(base)
    if not isinstance(tool_input, dict):
        return found
    for key in _PATH_ARG_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            candidate = Path(value).expanduser()
            if not candidate.is_absolute() and base is not None:
                candidate = base / candidate
            if candidate.is_absolute():
                found.append(candidate)
    command = tool_input.get("command")
    if isinstance(command, str) and command:
        for match in _PATH_TOKEN.finditer(command):
            found.append(Path(match.group(1)))
        if base is not None:
            found.extend(_relative_cd_targets(command, base))
    return found


def _relative_cd_targets(command: str, base: Path) -> list[Path]:
    targets: list[Path] = []
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return targets
    for index, token in enumerate(tokens[:-1]):
        if token in ("cd", "pushd") and (index == 0 or tokens[index - 1] in (";", "&&", "||", "|", "(")):
            target = tokens[index + 1]
            if target and not target.startswith(("/", "-", "~", "$")):
                targets.append(base / target)
    return targets


@dataclass
class Snapshot:
    commit: str
    tree: str
    head: str | None
    branch: str | None
    taken_at: float


@dataclass
class TreeState:
    root: Path
    git_dir: Path
    key: str
    start: Snapshot | None = None
    start_error: str | None = None
    marker: Path | None = None
    # Missing snapshot content is not evidence that a file was absent.
    excluded_paths: set[str] = field(default_factory=set)


@dataclass
class TurnObservation:
    """One request's candidate trees and outcome receipts."""

    turn_id: str
    bot_id: str | None
    started_at: float = field(default_factory=time.time)
    trees: dict[tuple[int, int], TreeState] = field(default_factory=dict)
    pending_paths: dict[Path, Path | None] = field(default_factory=dict)
    timings: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    finished: bool = False
    content_bytes: int = 0
    files_emitted: int = 0


class TurnSnapshotObserver:
    """Bridge-side owner of per-turn command-edit observation."""

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._turns: dict[str, TurnObservation] = {}
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def observe_tool(
        self,
        *,
        request_id: str,
        bot_id: str | None,
        tool_input: Any,
        cwd: str | None,
        tool_name: str | None = None,
    ) -> None:
        """Make every tree this tool call references a candidate (PreToolUse).

        A call that provably cannot write (TASK-2024: Read/Grep/Glob, or a shell
        command on the conservative read-only allowlist) registers nothing, so
        a reviewer reading a worktree does not make its owner's turn busy.
        Anything unclassifiable still registers.
        """
        if not request_id:
            return
        turn = self._turn(request_id, bot_id)
        read_only = tool_name is not None and is_read_only_tool_call(tool_name, tool_input)
        with turn.lock:
            if turn.finished:
                return
            if read_only:
                self._refresh_markers(turn)
                return
            for path in referenced_paths(tool_input, cwd):
                try:
                    self._add_candidate(turn, path)
                except Exception:
                    logger.debug("turn snapshot candidate failed path=%s", path, exc_info=True)
            self._refresh_markers(turn)

    def finish(
        self,
        request_id: str,
        *,
        covered: Iterable[tuple[Any, str]] = (),
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """End-of-turn snapshots.  Returns ``(changed_files, tree_receipts)``.

        ``covered`` are :func:`file_identity` keys of paths already reported by
        Edit/Write captures; they are not reported again.
        """
        with self._lock:
            turn = self._turns.pop(request_id, None)
        if turn is None:
            return [], []
        with turn.lock:
            turn.finished = True
            try:
                return self._finish_locked(turn, set(covered))
            finally:
                with _LIVE_TURNS_LOCK:
                    _LIVE_TURNS.discard(request_id)

    def _finish_locked(
        self, turn: TurnObservation, covered_set: set[tuple[Any, str]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        files: list[dict[str, Any]] = []
        receipts: list[dict[str, Any]] = []
        for path in list(turn.pending_paths):
            try:
                self._add_created_candidate(turn, path, finishing=True)
            except Exception:
                logger.debug("created-tree candidate failed path=%s", path, exc_info=True)
        for state in turn.trees.values():
            receipt: dict[str, Any] = {"tree": str(state.root)}
            started = time.monotonic()
            status = "error"
            try:
                status, tree_files = self._finish_tree(turn, state, covered_set)
                receipt["status"] = status
                receipt["files"] = len(tree_files)
                if state.excluded_paths:
                    receipt["excluded_count"] = len(state.excluded_paths)
                    receipt["excluded_paths"] = sorted(state.excluded_paths)[:MAX_FILES_PER_TREE]
                files.extend(tree_files)
            except Exception as exc:
                receipt["status"] = "error"
                receipt["error"] = str(exc)[:300]
                logger.warning("turn snapshot finish failed tree=%s: %s", state.root, exc)
            finally:
                self._unregister(state, turn.turn_id)
                if status not in {"observed", "observed_partial"}:
                    self._delete_refs(state, turn.turn_id)
            receipt["end_ms"] = round((time.monotonic() - started) * 1000, 1)
            receipts.append(receipt)
        receipts.extend(turn.timings)
        if receipts:
            logger.info("turn_snapshot_finish turn=%s receipts=%s", turn.turn_id, json.dumps(receipts))
        return files, receipts

    def discard(self, request_id: str) -> None:
        with self._lock:
            turn = self._turns.pop(request_id, None)
        if turn is None:
            return
        with _LIVE_TURNS_LOCK:
            _LIVE_TURNS.discard(request_id)
        with turn.lock:
            turn.finished = True
            for state in turn.trees.values():
                self._unregister(state, turn.turn_id)
                self._delete_refs(state, turn.turn_id)

    # -- candidates --------------------------------------------------------

    def _turn(self, request_id: str, bot_id: str | None) -> TurnObservation:
        with self._lock:
            turn = self._turns.get(request_id)
            if turn is None:
                turn = TurnObservation(turn_id=request_id, bot_id=bot_id, started_at=self._clock())
                self._turns[request_id] = turn
                with _LIVE_TURNS_LOCK:
                    _LIVE_TURNS.add(request_id)
            return turn

    def _add_candidate(self, turn: TurnObservation, path: Path) -> None:
        if not path.is_absolute():
            return
        previous_pending = tuple(turn.pending_paths)
        root = find_tree_root(path)
        if not path.exists():
            # Remember the original root even if registration later fails.
            if len(turn.pending_paths) < 64:
                turn.pending_paths.setdefault(path, root.resolve() if root else None)
        if root is None:
            return
        key = _inode_key(root)
        if key is None or key in turn.trees:
            return
        # A later tool can discover a tree whose creating command already
        # edited files. Prefer its creation baseline before registering it with
        # a late working-copy snapshot. Never replace an existing baseline.
        for pending in previous_pending:
            pending_root = find_tree_root(pending)
            if pending_root is not None and _inode_key(pending_root) == key:
                self._add_created_candidate(turn, pending)
                if key in turn.trees:
                    return
        started = time.monotonic()
        state = self._new_state(root)
        if state is None:
            return
        turn.trees[key] = state
        self._register(state, turn)
        try:
            state.start = self._snapshot(state, turn.turn_id, "start")
        except Exception as exc:
            state.start_error = str(exc)[:300]
            logger.debug("turn snapshot start failed tree=%s: %s", root, exc)
        turn.timings.append({
            "tree": str(root), "phase": "start",
            "ms": round((time.monotonic() - started) * 1000, 1),
        })

    def _add_created_candidate(self, turn: TurnObservation, path: Path, *, finishing: bool = False) -> None:
        """A tree that did not exist at first reference starts at its creation commit."""
        root = find_tree_root(path)
        if root is None:
            return
        key = _inode_key(root)
        if key is None or key in turn.trees:
            return
        state = self._new_state(root)
        if state is None:
            return
        original_root = turn.pending_paths.get(path)
        existed = original_root is not None and (
            root.resolve() == original_root or _inode_key(original_root) == key
        )
        created = None if existed else self._creation_commit(root, turn.started_at)
        if created is None:
            if finishing:
                # At finish a working-copy baseline would falsely say unchanged.
                turn.trees[key] = state
            return
        turn.trees[key] = state
        self._register(state, turn)
        tree = _git_text(root, "rev-parse", f"{created}^{{tree}}")
        state.start = Snapshot(commit=created, tree=tree, head=created, branch=None, taken_at=turn.started_at)
        self._update_ref(state, turn.turn_id, "start", created)

    @staticmethod
    def _creation_commit(root: Path, since: float) -> str | None:
        """Use the reflog event time, not the possibly old base commit time."""
        out = _git_text(root, "reflog", "show", "--format=%H %gD", "--date=unix", "HEAD", check=False)
        lines = [line for line in out.splitlines() if line.strip()]
        if not lines:
            return None
        match = re.fullmatch(r"([0-9a-f]+) HEAD@\{([0-9]+)\}", lines[-1])
        if match is None or int(match[2]) < since - COMMIT_CLOCK_SLACK_SECONDS:
            return None
        return match[1]

    @staticmethod
    def _new_state(root: Path) -> TreeState | None:
        try:
            git_dir = Path(_git_text(root, "rev-parse", "--absolute-git-dir"))
        except GitError:
            return None
        key = hashlib.sha256(str(git_dir).encode()).hexdigest()[:16]
        return TreeState(root=root, git_dir=git_dir, key=key)

    # -- snapshots ---------------------------------------------------------

    def _snapshot(self, state: TreeState, turn_id: str, phase: str) -> Snapshot:
        root = state.root
        head = _git_text(root, "rev-parse", "--verify", "-q", "HEAD", check=False) or None
        branch = _git_text(root, "symbolic-ref", "-q", "--short", "HEAD", check=False) or None
        status = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--no-renames")
        excluded: set[str] = set()
        paths = self._changed_paths(root, status, excluded=excluded)
        path_count = len(paths) + len(excluded)
        if path_count > MAX_STATUS_PATHS:
            raise SnapshotTooLarge(f"{path_count} changed paths exceeds {MAX_STATUS_PATHS}")
        state.excluded_paths.update(excluded)
        if not paths and head is not None:
            tree = _git_text(root, "rev-parse", f"{head}^{{tree}}")
        else:
            tree = self._write_tree(state, paths)
        parents = ("-p", head) if head else ()
        commit = _git_text(
            root, "commit-tree", tree, *parents, "-m", f"bawt turn snapshot {phase} {turn_id}",
            env=_SNAPSHOT_IDENTITY,
        )
        self._update_ref(state, turn_id, phase, commit)
        return Snapshot(commit=commit, tree=tree, head=head, branch=branch, taken_at=self._clock())

    @staticmethod
    def _changed_paths(root: Path, status: bytes, *, excluded: set[str]) -> list[str]:
        paths: list[str] = []
        for entry in status.split(b"\0"):
            if len(entry) < 4:
                continue
            code, raw = entry[:2], entry[3:]
            rel = raw.decode("utf-8", "surrogateescape")
            if code == b"??":
                try:
                    if (root / rel).stat().st_size > MAX_UNTRACKED_BYTES:
                        excluded.add(rel)
                        continue
                except OSError:
                    excluded.add(rel)
                    continue
            paths.append(rel)
        return paths

    def _write_tree(self, state: TreeState, paths: list[str]) -> str:
        root = state.root
        real_index = state.git_dir / "index"
        with tempfile.TemporaryDirectory(prefix="bawt-turn-index-") as tmp:
            temp_index = Path(tmp) / "index"
            if real_index.exists():
                temp_index.write_bytes(real_index.read_bytes())
            env = {"GIT_INDEX_FILE": str(temp_index), "GIT_LITERAL_PATHSPECS": "1"}
            if paths:
                spec = b"\0".join(p.encode("utf-8", "surrogateescape") for p in paths)
                _git(
                    root, "add", "-A", "--pathspec-from-file=-", "--pathspec-file-nul",
                    env=env, input_bytes=spec,
                )
            return _git_text(root, "write-tree", env=env)

    @staticmethod
    def _ref(state: TreeState, turn_id: str, phase: str) -> str:
        safe_turn = _SAFE_REF_PART.sub("_", turn_id)[:120] or "turn"
        return f"{REF_PREFIX}/{safe_turn}/{state.key}/{phase}"

    def _update_ref(self, state: TreeState, turn_id: str, phase: str, commit: str) -> None:
        _git(state.root, "update-ref", self._ref(state, turn_id, phase), commit)

    def _delete_refs(self, state: TreeState, turn_id: str) -> None:
        """Drop pins for a tree without an observed diff (nothing to render)."""
        try:
            commands = "".join(
                f"delete {self._ref(state, turn_id, phase)}\n" for phase in ("start", "end")
            ).encode()
            _git(state.root, "update-ref", "--stdin", input_bytes=commands, check=False)
        except GitError:
            logger.debug("turn snapshot ref cleanup failed tree=%s", state.root, exc_info=True)

    # -- end of turn -------------------------------------------------------

    def _finish_tree(
        self,
        turn: TurnObservation,
        state: TreeState,
        covered: set[tuple[Any, str]],
    ) -> tuple[str, list[dict[str, Any]]]:
        if state.start is None:
            return "no_start_snapshot", []
        if not state.root.exists():
            return "tree_removed", []
        busy = self._overlapped(state, turn.turn_id)
        try:
            end = self._snapshot(state, turn.turn_id, "end")
        except SnapshotTooLarge:
            return "skipped_too_many_paths", []
        self._prune_refs(state)
        if busy:
            return "ineligible_shared_tree_busy", []
        if not self._head_movement_is_own(state, state.start, end, turn.bot_id):
            return "ineligible_head_moved", []
        if end.tree == state.start.tree:
            return ("excluded_only" if state.excluded_paths else "unchanged"), []
        raw = _git(state.root, "diff-tree", "-r", "-z", "--no-renames", "--raw", state.start.tree, end.tree)
        entries = self._parse_raw_diff(raw)
        files: list[dict[str, Any]] = []
        for old_mode, new_mode, old_blob, new_blob, status, rel in entries:
            if rel in state.excluded_paths:
                continue
            if len(files) >= MAX_FILES_PER_TREE or turn.files_emitted >= MAX_FILES_PER_TURN:
                break
            absolute = state.root / rel
            identity = file_identity(absolute)
            if identity is not None and identity in covered:
                continue
            if old_mode.startswith("160") or new_mode.startswith("160"):
                continue  # submodule pointer, not file content
            files.append(self._file_payload(turn, state, end, rel, status, old_blob, new_blob))
            turn.files_emitted += 1
        if state.excluded_paths:
            return ("observed_partial" if files else "excluded_only"), files
        return "observed", files

    @staticmethod
    def _parse_raw_diff(raw: bytes) -> list[tuple[str, str, str, str, str, str]]:
        parts = raw.split(b"\0")
        entries = []
        index = 0
        while index + 1 < len(parts):
            meta = parts[index].decode("ascii", "replace")
            if not meta.startswith(":"):
                index += 1
                continue
            fields = meta[1:].split()
            path = parts[index + 1].decode("utf-8", "surrogateescape")
            index += 2
            if len(fields) < 5:
                continue
            old_mode, new_mode, old_blob, new_blob, status = fields[:5]
            entries.append((old_mode, new_mode, old_blob, new_blob, status[0], path))
        return entries

    def _head_movement_is_own(
        self, state: TreeState, start: Snapshot, end: Snapshot, bot_id: str | None,
    ) -> bool:
        if start.head == end.head:
            return True
        if not bot_id:
            return False
        if start.head is None or end.head is None:
            return start.head is None and end.head is not None and self._commits_are_own(
                state, end.head, None, start.taken_at, bot_id)
        ancestor = subprocess.run(
            ("git", "-C", str(state.root), "merge-base", "--is-ancestor", start.head, end.head),
            capture_output=True, timeout=GIT_TIMEOUT_SECONDS, check=False,
        )
        if ancestor.returncode != 0:
            return False
        return self._commits_are_own(state, end.head, start.head, start.taken_at, bot_id)

    @staticmethod
    def _commits_are_own(
        state: TreeState, tip: str, base: str | None, since: float, bot_id: str,
    ) -> bool:
        """Every new commit is recent and authored by this turn's bot.

        A rebase or fast-forward by an integrator gives fresh committer dates,
        so recency alone would attribute their publish to this turn.
        """
        spec = f"{base}..{tip}" if base else tip
        out = _git_text(
            state.root, "rev-list", "--format=%ct%x00%an%x00%ae", "--no-commit-header", spec,
        )
        wanted = bot_id.strip().lower()
        for line in out.splitlines():
            fields = line.strip().split("\0")
            if len(fields) != 3 or not fields[0].isdigit():
                return False
            committed, name, email = int(fields[0]), fields[1].strip().lower(), fields[2].strip().lower()
            if committed < since - COMMIT_CLOCK_SLACK_SECONDS:
                return False
            if wanted not in (name, email.split("@", 1)[0]):
                return False
        return True

    def _file_payload(
        self, turn: TurnObservation, state: TreeState, end: Snapshot,
        rel: str, status: str, old_blob: str, new_blob: str,
    ) -> dict[str, Any]:
        null = "0" * len(old_blob)
        sizes = [
            self._blob_size(state.root, blob) for blob in (old_blob, new_blob) if blob != null
        ]
        cost = sum(min(size, MAX_BYTES_PER_SIDE) for size in sizes)
        with_content = turn.content_bytes + cost <= MAX_CONTENT_BYTES_PER_TURN
        if with_content:
            turn.content_bytes += cost
            before = self._read_blob(state.root, old_blob) if old_blob != null else None
            after = self._read_blob(state.root, new_blob) if new_blob != null else None
        else:
            before = after = None
        change_kind = {"A": "added", "D": "deleted"}.get(status, "modified")
        if not with_content:
            # Over the turn's content budget: the row still lists the file.
            absolute = str(state.root / rel)
            return {
                "repo_key": "workspace", "repo_label": state.root.name, "path": absolute,
                "in_repo": True, "change_kind": change_kind, "additions": None, "deletions": None,
                "binary": False, "truncated": True, "content_type": None,
                "before_b64": None, "after_b64": None, "absolute_path": absolute,
                "provenance": PROVENANCE,
                "git_before_blob": old_blob if old_blob != null else None,
                "git_after_blob": new_blob if new_blob != null else None,
                **({"git_head_sha": end.head} if end.head else {}),
                **({"git_branch": end.branch} if end.branch else {}),
            }
        before_data, before_trunc, before_bin = before if before else (None, False, False)
        after_data, after_trunc, after_bin = after if after else (None, False, False)
        binary = before_bin or after_bin
        additions, deletions = (None, None)
        if not binary and not before_trunc and not after_trunc:
            additions, deletions = _line_counts(before_data, after_data)
        absolute = str(state.root / rel)
        payload: dict[str, Any] = {
            "repo_key": "workspace",
            "repo_label": state.root.name,
            "path": absolute,
            "in_repo": True,
            "change_kind": change_kind,
            "additions": additions,
            "deletions": deletions,
            "binary": binary,
            "truncated": before_trunc or after_trunc,
            "content_type": None if binary else "text/plain; charset=utf-8",
            "before_b64": _b64(None if before_bin else before_data) if before is not None else None,
            "after_b64": _b64(None if after_bin else after_data) if after is not None else None,
            "absolute_path": absolute,
            "provenance": PROVENANCE,
            "git_before_blob": old_blob if old_blob != null else None,
            "git_after_blob": new_blob if new_blob != null else None,
        }
        if end.head:
            payload["git_head_sha"] = end.head
        if end.branch:
            payload["git_branch"] = end.branch
        return payload

    @staticmethod
    def _blob_size(root: Path, blob: str) -> int:
        size_text = _git_text(root, "cat-file", "-s", blob)
        return int(size_text) if size_text.isdigit() else 0

    @staticmethod
    def _read_blob(root: Path, blob: str) -> tuple[bytes, bool, bool]:
        size = TurnSnapshotObserver._blob_size(root, blob)
        data = _git(root, "cat-file", "blob", blob)
        truncated = size > MAX_BYTES_PER_SIDE
        if truncated:
            data = data[:MAX_BYTES_PER_SIDE]
        binary = b"\0" in data[:8192]
        return data, truncated, binary

    # -- concurrency registry ----------------------------------------------

    @staticmethod
    def _registry_dir(state: TreeState) -> Path:
        return state.git_dir / "bawt-turns"

    def _register(self, state: TreeState, turn: TurnObservation) -> None:
        """Write this turn's marker, then flag every other live turn (and us)."""
        registry = self._registry_dir(state)
        try:
            registry.mkdir(parents=True, exist_ok=True)
            marker = registry / f"{_SAFE_REF_PART.sub('_', turn.turn_id)[:120]}.json"
            marker.write_text(json.dumps({
                "turn_id": turn.turn_id, "bot_id": turn.bot_id, "registered_at": self._clock(),
                "process": _PROCESS_TOKEN, "host": socket.gethostname(),
            }))
            state.marker = marker
            for other in self._live_markers(registry, exclude=marker):
                (other.with_suffix(".overlap")).touch()
                marker.with_suffix(".overlap").touch()
        except OSError:
            logger.debug("turn registry write failed tree=%s", state.root, exc_info=True)

    def _live_markers(self, registry: Path, *, exclude: Path | None) -> list[Path]:
        live = []
        now = self._clock()
        for marker in registry.glob("*.json"):
            if exclude is not None and marker == exclude:
                continue
            try:
                if now - marker.stat().st_mtime > MARKER_STALE_SECONDS or self._dead_process_marker(marker):
                    marker.unlink(missing_ok=True)
                    marker.with_suffix(".overlap").unlink(missing_ok=True)
                    continue
            except OSError:
                continue
            live.append(marker)
        return live

    def _dead_process_marker(self, marker: Path) -> bool:
        """A marker from this host is dead if its process is gone or its turn ended.

        Same host + different process token = a previous bridge process (crash
        or restart).  Same token = this process, live only while the turn is.
        """
        try:
            data = json.loads(marker.read_text() or "{}")
        except (OSError, ValueError):
            return False
        if not isinstance(data, dict) or data.get("host") != socket.gethostname():
            return False
        if data.get("process") != _PROCESS_TOKEN:
            registered = data.get("registered_at")
            return isinstance(registered, (int, float)) and registered < _PROCESS_STARTED
        with _LIVE_TURNS_LOCK:
            return data.get("turn_id") not in _LIVE_TURNS

    def _refresh_markers(self, turn: TurnObservation) -> None:
        """Keep a long turn's markers younger than the stale cutoff."""
        for state in turn.trees.values():
            if state.marker is not None:
                try:
                    os.utime(state.marker)
                except OSError:
                    pass

    def _overlapped(self, state: TreeState, turn_id: str) -> bool:
        if state.marker is None:
            # Without a marker this turn could not announce itself: be safe.
            return True
        registry = self._registry_dir(state)
        if state.marker.with_suffix(".overlap").exists():
            return True
        return bool(self._live_markers(registry, exclude=state.marker))

    @staticmethod
    def _unregister(state: TreeState, turn_id: str) -> None:
        if state.marker is None:
            return
        state.marker.unlink(missing_ok=True)
        state.marker.with_suffix(".overlap").unlink(missing_ok=True)

    # -- retention ---------------------------------------------------------

    def _prune_refs(self, state: TreeState) -> None:
        try:
            common = Path(_git_text(state.root, "rev-parse", "--path-format=absolute", "--git-common-dir"))
            stamp = common / "bawt-turns-pruned"
            now = self._clock()
            if stamp.exists() and now - stamp.stat().st_mtime < PRUNE_INTERVAL_SECONDS:
                return
            stamp.touch()
            out = _git_text(
                state.root, "for-each-ref", "--format=%(refname) %(committerdate:unix)", REF_PREFIX,
            )
            stale = [
                line.split()[0] for line in out.splitlines()
                if len(line.split()) == 2 and line.split()[1].isdigit()
                and now - int(line.split()[1]) > REF_RETENTION_SECONDS
            ]
            if stale:
                commands = "".join(f"delete {ref}\n" for ref in stale).encode()
                _git(state.root, "update-ref", "--stdin", input_bytes=commands)
                logger.info("turn snapshot refs pruned repo=%s count=%d", common, len(stale))
        except (OSError, GitError):
            logger.debug("turn snapshot prune failed tree=%s", state.root, exc_info=True)


def _b64(data: bytes | None) -> str | None:
    return None if data is None else base64.b64encode(data).decode("ascii")


def _line_counts(before: bytes | None, after: bytes | None) -> tuple[int, int]:
    before_lines = (before or b"").decode("utf-8", "replace").splitlines()
    after_lines = (after or b"").decode("utf-8", "replace").splitlines()
    additions = deletions = 0
    for line in difflib.ndiff(before_lines, after_lines):
        if line.startswith("+ "):
            additions += 1
        elif line.startswith("- "):
            deletions += 1
    return additions, deletions
