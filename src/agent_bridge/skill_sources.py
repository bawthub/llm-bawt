"""Read-only skill content sources (TASK-1309): the one reader of skill files.

Skill content is authored in git and never written here. The catalog API, the
injected-skills prompt section and the bridge's Skill-call token estimate all
read through a :class:`SkillSource`, so frontmatter parsing, size caps and path
confinement exist once.

A skill's ``ref`` is the selector the harness already uses: the bare directory
name for the ambient tree. Registry packages (``package:skill``) get their own
source once TASK-871 activates the registry.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .tool_results import count_result_tokens

logger = logging.getLogger(__name__)

SKILL_FILE = "SKILL.md"
MAX_SKILL_BYTES = 1_000_000
MAX_VIEW_BYTES = 512_000
MAX_TREE_ENTRIES = 2_000
_FRONTMATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)
_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_GIT_TIMEOUT = 5


class SkillSourceError(Exception):
    """A read the caller asked for cannot be served; ``status`` maps to HTTP."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class SkillEntry:
    ref: str
    name: str
    description: str
    source: str
    root: Path  # resolved skill directory (the path the harness sees)


@dataclass(frozen=True)
class SkillBody:
    ref: str
    text: str  # SKILL.md without frontmatter
    tokens: int
    content_sha: str  # sha256 of ``text``


@dataclass(frozen=True)
class SkillFileNode:
    path: str  # relative to the skill root, POSIX separators
    size: int


@dataclass(frozen=True)
class SkillFileContent:
    path: str
    size: int
    text: str


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Return (frontmatter mapping, body). Malformed frontmatter reads as empty."""
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    body = text[match.end():]
    try:
        import yaml

        data = yaml.safe_load(match.group(1))
    except Exception:
        return {}, body
    return (data if isinstance(data, dict) else {}), body


def _confined(root: Path, relative: str) -> Path:
    """Resolve ``relative`` under ``root``; refuse traversal and symlink escape."""
    if not relative or relative.startswith(("/", "\\")) or "\x00" in relative:
        raise SkillSourceError(400, "path must be relative to the skill root")
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise SkillSourceError(400, "path escapes the skill root")
    return target


def _read_text(path: Path, limit: int) -> str:
    size = path.stat().st_size
    if size > limit:
        raise SkillSourceError(413, f"file is {size} bytes; the limit is {limit}")
    data = path.read_bytes()
    if b"\x00" in data:
        raise SkillSourceError(415, "binary file")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SkillSourceError(415, "not UTF-8 text") from exc


class SkillSource(ABC):
    """One place skills come from. Implementations never write."""

    kind: str

    @abstractmethod
    def list(self) -> tuple[SkillEntry, ...]:
        """Every skill this source offers, sorted by ref."""

    def get(self, ref: str) -> SkillEntry | None:
        return next((entry for entry in self.list() if entry.ref == ref), None)

    def read_body(self, ref: str) -> SkillBody | None:
        """SKILL.md minus frontmatter; None when missing, oversized or unreadable."""
        entry = self.get(ref)
        if entry is None:
            return None
        try:
            path = _confined(entry.root, SKILL_FILE)
            stat = path.stat()
            key = (ref, str(path), stat.st_mtime_ns, stat.st_size)
            cached = _BODY_CACHE.get(key)
            if cached is not None:
                return cached
            _, body = split_frontmatter(_read_text(path, MAX_SKILL_BYTES))
        except (OSError, SkillSourceError):
            return None
        result = SkillBody(ref, body, count_result_tokens(body),
                           hashlib.sha256(body.encode()).hexdigest())
        _BODY_CACHE.put(key, result)
        return result

    def tree(self, ref: str) -> tuple[SkillFileNode, ...]:
        """Regular files under the skill root, confined to it, hidden entries skipped."""
        entry = self._require(ref)
        nodes: list[SkillFileNode] = []
        for directory, dirs, files in os.walk(entry.root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            for name in sorted(files):
                if name.startswith("."):
                    continue
                path = Path(directory) / name
                try:
                    resolved = path.resolve()
                    if not resolved.is_relative_to(entry.root) or not resolved.is_file():
                        continue
                    nodes.append(SkillFileNode(path.relative_to(entry.root).as_posix(),
                                               resolved.stat().st_size))
                except OSError:
                    continue
                if len(nodes) >= MAX_TREE_ENTRIES:
                    return tuple(nodes)
        return tuple(nodes)

    def read_file(self, ref: str, relative: str) -> SkillFileContent:
        entry = self._require(ref)
        target = _confined(entry.root, relative)
        if not target.is_file():
            raise SkillSourceError(404, "no such file")
        text = _read_text(target, MAX_VIEW_BYTES)
        return SkillFileContent(target.relative_to(entry.root).as_posix(), len(text.encode()), text)

    def provenance(self, ref: str) -> dict:
        """Where the bytes come from; git facts fail soft to None."""
        entry = self._require(ref)
        return {"source": self.kind, "path": str(entry.root), **_git_provenance(entry.root)}

    def _require(self, ref: str) -> SkillEntry:
        entry = self.get(ref)
        if entry is None:
            raise SkillSourceError(404, f"unknown skill {ref!r}")
        return entry


class AmbientSource(SkillSource):
    """The operator's skill checkout(s): ``<root>/<name>/SKILL.md``; first root wins."""

    kind = "ambient"

    def __init__(self, roots: tuple[Path, ...] | None = None):
        self._roots = roots

    @property
    def roots(self) -> tuple[Path, ...]:
        if self._roots is not None:
            return self._roots
        from .skill_selection import legacy_claude_skill_roots

        return legacy_claude_skill_roots()

    def list(self) -> tuple[SkillEntry, ...]:
        found: dict[str, SkillEntry] = {}
        for configured in self.roots:
            try:
                root = configured.resolve()
                children = sorted(root.iterdir())
            except OSError:
                continue
            for child in children:
                if child.name in found or not _REF.fullmatch(child.name):
                    continue
                try:
                    resolved = child.resolve()
                    # A link out of the checkout, or a SKILL.md linked out of
                    # its own skill directory, is not a skill of this source.
                    if not resolved.is_relative_to(root) or not resolved.is_dir():
                        continue
                    skill_file = _confined(resolved, SKILL_FILE)
                    if not skill_file.is_file():
                        continue
                    meta = _frontmatter_of(skill_file)
                except (OSError, SkillSourceError):
                    continue
                description = meta.get("description")
                found[child.name] = SkillEntry(
                    ref=child.name, name=str(meta.get("name") or child.name),
                    description=description.strip() if isinstance(description, str) else "",
                    source=self.kind, root=resolved)
        return tuple(found[ref] for ref in sorted(found))


class _LruCache:
    def __init__(self, size: int):
        self._size = size
        self._items: dict = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            value = self._items.pop(key, None)
            if value is not None:
                self._items[key] = value
            return value

    def put(self, key, value) -> None:
        with self._lock:
            self._items.pop(key, None)
            self._items[key] = value
            while len(self._items) > self._size:
                self._items.pop(next(iter(self._items)))


_BODY_CACHE = _LruCache(256)
_META_CACHE = _LruCache(512)


def _frontmatter_of(skill_file: Path) -> dict:
    stat = skill_file.stat()
    key = (str(skill_file), stat.st_mtime_ns, stat.st_size)
    cached = _META_CACHE.get(key)
    if cached is not None:
        return cached
    if stat.st_size > MAX_SKILL_BYTES:
        meta: dict = {}
    else:
        try:
            meta, _ = split_frontmatter(skill_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError):
            meta = {}
    _META_CACHE.put(key, meta)
    return meta


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", str(root), *args],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _git_provenance(root: Path) -> dict:
    toplevel = _git(root, "rev-parse", "--show-toplevel")
    if not toplevel:
        return {"repo": None, "repo_path": None, "commit": None, "committed_at": None}
    last = _git(root, "log", "-1", "--format=%H%x1f%cI", "--", ".") or ""
    commit, _, committed_at = last.partition("\x1f")
    try:
        repo_path = root.relative_to(Path(toplevel).resolve()).as_posix()
    except ValueError:
        repo_path = None
    return {"repo": Path(toplevel).name, "repo_path": repo_path,
            "commit": commit or None, "committed_at": committed_at or None}
