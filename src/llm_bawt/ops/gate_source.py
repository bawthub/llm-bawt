"""Unused exact-commit source staging for the future socketless gate runner.

Only trusted configuration supplies repository/staging paths. Reads Git objects,
never the source index/worktree, and runs no repository code, checkout filters,
archive substitutions, network fetch, hooks or templates. Each context owns a
new directory and an independent shallow Git repository; no canonical metadata
is copied. Staging must remain owned/frozen until handed to the sandbox. This
is initial source identity, not authorization, runner attestation or SHIP proof.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import zlib
from contextlib import contextmanager
from dataclasses import dataclass

from .gate_plan import GatePlan

_SHA = re.compile(r"[0-9a-f]{40}\Z")
_MAX_FILES = 100_000
_MAX_ENTRIES = 100_000
_MAX_BYTES = 512 * 1024 * 1024
_MAX_OBJECT = 64 * 1024 * 1024
_MAX_TREE_BYTES = 64 * 1024 * 1024
_MAX_DEPTH = 64
_CONFIG = b"[core]\nrepositoryformatversion=0\nbare=false\nfilemode=true\nhooksPath=/dev/null\nlogallrefupdates=false\n"


class GateSourceError(ValueError):
    """Fixed refusal text: never expose Git stderr, config or source contents."""


@dataclass(frozen=True)
class GateSource:
    directory: Path
    source_sha: str
    tree_sha: str
    file_count: int
    byte_count: int


def _oid(kind: str, body: bytes) -> str:
    return hashlib.sha1(f"{kind} {len(body)}\0".encode() + body).hexdigest()


class _Git:
    def __init__(self, repository: Path):
        self.repository = repository
        self.binary = shutil.which("git", path=os.defpath)
        if self.binary is None:
            raise GateSourceError("trusted system Git is unavailable")

    def run(self, *args: str) -> bytes:
        # Do not inherit GIT_DIR, config injection, alternates, PATH, loader
        # settings, credentials, signing agents or the operator's HOME.
        env = {"PATH": os.defpath, "HOME": "/nonexistent", "XDG_CONFIG_HOME": "/nonexistent",
            "LC_ALL": "C", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_NO_REPLACE_OBJECTS": "1", "GIT_NO_LAZY_FETCH": "1",
            "GIT_ALLOW_PROTOCOL": "", "GIT_PROTOCOL_FROM_USER": "0",
            "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0", "GIT_ATTR_NOSYSTEM": "1"}
        try:
            result = subprocess.run([self.binary, "--no-pager", "--no-replace-objects",
                "-c", f"safe.directory={self.repository}", "-c", "core.hooksPath=/dev/null",
                "-c", "core.fsmonitor=false", "-C", str(self.repository), *args],
                env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired):
            raise GateSourceError("source Git is unavailable or timed out") from None
        if result.returncode:
            raise GateSourceError("source Git refused fixed object plumbing")
        return result.stdout

    def object(self, kind: str, oid: str) -> bytes:
        if not _SHA.fullmatch(oid):
            raise GateSourceError("source object identity is invalid")
        size = self.run("cat-file", "-s", oid).strip()
        if not size.isdigit() or int(size) > _MAX_OBJECT:
            raise GateSourceError("source object exceeds the staging bound")
        body = self.run("cat-file", kind, oid)
        if len(body) != int(size) or _oid(kind, body) != oid:
            raise GateSourceError("source object bytes do not match their identity")
        return body


def _directory(raw: Path, *, private: bool = False) -> Path:
    path = Path(raw)
    try:
        if not path.is_absolute() or path != path.resolve(strict=True) or not path.is_dir():
            raise GateSourceError("source locations must be real absolute directories")
        info = path.stat()
        if private and (info.st_uid != os.geteuid() or info.st_mode & 0o022):
            raise GateSourceError("staging parent must be owned and not group/other writable")
    except OSError:
        raise GateSourceError("source location is unavailable") from None
    return path


def _name(raw: bytes) -> str:
    try:
        name = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise GateSourceError("source path is not UTF-8") from None
    if (not name or len(raw) > 255 or name in (".", "..") or name.casefold() == ".git"
            or any(c in name for c in "/\\") or any(ord(c) < 32 or ord(c) == 127 for c in name)):
        raise GateSourceError("source path is unsafe")
    return name


def _inventory(git: _Git, tree: str):
    files, objects, directories = {}, {}, set()
    entries = tree_bytes = 0

    def walk(oid: str, prefix: str, depth: int):
        nonlocal entries, tree_bytes
        if depth > _MAX_DEPTH:
            raise GateSourceError("source tree exceeds the depth bound")
        body = git.object("tree", oid)
        tree_bytes += len(body)
        if tree_bytes > _MAX_TREE_BYTES:
            raise GateSourceError("source tree metadata exceeds the staging bound")
        objects[oid] = ("tree", body)
        offset, seen = 0, set()
        while offset < len(body):
            space, nul = body.find(b" ", offset), body.find(b"\0", offset)
            if space < offset or nul <= space or nul + 21 > len(body):
                raise GateSourceError("source tree encoding is invalid")
            mode, name = body[offset:space], _name(body[space + 1:nul])
            child = body[nul + 1:nul + 21].hex()
            offset = nul + 21
            if name in seen:
                raise GateSourceError("source tree contains duplicate paths")
            seen.add(name)
            entries += 1
            if entries > _MAX_ENTRIES:
                raise GateSourceError("source tree exceeds the entry bound")
            path = prefix + name
            if len(path.encode()) > 4095:
                raise GateSourceError("source path exceeds the staging bound")
            if mode == b"40000":
                directories.add(path)
                walk(child, path + "/", depth + 1)
            elif mode in (b"100644", b"100755"):
                files[path] = (0o755 if mode == b"100755" else 0o644, child)
                if len(files) > _MAX_FILES:
                    raise GateSourceError("source tree exceeds the file bound")
            else:
                raise GateSourceError("source tree requires unsupported links or special files")

    walk(tree, "", 0)
    return files, objects, directories


def _put_object(root: Path, kind: str, oid: str, body: bytes):
    target = root / ".git" / "objects" / oid[:2] / oid[2:]
    target.parent.mkdir(mode=0o700, exist_ok=True)
    # Duplicate blobs/trees are harmless, but never overwrite an existing path.
    if target.exists():
        return
    with target.open("xb") as handle:
        handle.write(zlib.compress(f"{kind} {len(body)}\0".encode() + body))
    target.chmod(0o600)


def _verify(root: Path, files: dict, directories: set, sha: str, tree: str):
    actual, actual_dirs = set(), set()
    if not (root / ".git").is_dir() or (root / ".git").is_symlink():
        raise GateSourceError("private Git metadata is unavailable")
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            if (Path(directory) / name).is_symlink():
                raise GateSourceError("private source contains an unexpected link")
        if Path(directory) == root:
            dirs.remove(".git")
        actual_dirs.update((Path(directory) / name).relative_to(root).as_posix() for name in dirs)
        for name in names:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            expected = files.get(relative)
            if expected is None or not stat.S_ISREG(path.stat().st_mode):
                raise GateSourceError("private source contains an unexpected file")
            mode, oid = expected
            if stat.S_IMODE(path.stat().st_mode) != mode or _oid("blob", path.read_bytes()) != oid:
                raise GateSourceError("private source bytes or modes changed during staging")
            actual.add(relative)
    if actual != files.keys() or actual_dirs != directories:
        raise GateSourceError("private source file or directory inventory differs")
    git = _Git(root)
    if (git.run("rev-parse", "--verify", "HEAD^{commit}") != f"{sha}\n".encode()
            or git.run("rev-parse", "--verify", "HEAD^{tree}") != f"{tree}\n".encode()):
        raise GateSourceError("private source commit or tree identity differs")


@contextmanager
def staged_gate_source(plan: GatePlan, *, repository: Path, staging_parent: Path):
    """Yield one independent source repository, removing it on every exit.

    Paths are trusted configuration, never body/tool inputs. All leaves must be
    regular committed files. Bounds are explicit: 100k files/tree entries,
    512MiB file bytes, 64MiB per object/total tree metadata, 64 tree levels.
    Unsupported source refuses, never truncates.
    The commit retains its original identity; only that commit/tree/blob closure
    is exported. Parent history is intentionally unavailable (shallow boundary).
    Committed secrets cannot be scrubbed while preserving exact source identity.
    """
    plan = GatePlan.from_dict(plan.to_dict())
    repository, parent = _directory(repository), _directory(staging_parent, private=True)
    if parent.is_relative_to(repository):
        raise GateSourceError("staging must not modify the source repository")
    git = _Git(repository)
    if git.run("rev-parse", "--show-toplevel") != f"{repository}\n".encode():
        raise GateSourceError("source location is not the repository root")
    commit = git.object("commit", plan.source_sha)
    header = commit.split(b"\n", 1)[0]
    if not header.startswith(b"tree ") or not _SHA.fullmatch(header[5:].decode("ascii", "replace")):
        raise GateSourceError("source commit tree identity is invalid")
    tree = header[5:].decode("ascii")
    files, objects, directories = _inventory(git, tree)
    with tempfile.TemporaryDirectory(prefix="ops-gate-source-", dir=parent) as temporary:
        root = Path(temporary) / "source"
        root.mkdir(mode=0o700)
        for directory in (".git", ".git/objects", ".git/refs"):
            (root / directory).mkdir(mode=0o700)
        (root / ".git/config").write_bytes(_CONFIG)
        (root / ".git/HEAD").write_text(plan.source_sha + "\n")
        (root / ".git/shallow").write_text(plan.source_sha + "\n")
        _put_object(root, "commit", plan.source_sha, commit)
        for oid, (kind, body) in objects.items():
            _put_object(root, kind, oid, body)
        for directory in sorted(directories):
            (root / directory).mkdir(mode=0o700, exist_ok=True)
        total = 0
        for name, (mode, oid) in files.items():
            body = git.object("blob", oid)
            total += len(body)
            if total > _MAX_BYTES:
                raise GateSourceError("source files exceed the staging byte bound")
            _put_object(root, "blob", oid, body)
            target = root / name
            target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(body)
            target.chmod(mode)
        private = _Git(root)
        private.run("read-tree", plan.source_sha)  # Populate only a PRIVATE index, no checkout/filter.
        _verify(root, files, directories, plan.source_sha, tree)
        yield GateSource(root, plan.source_sha, tree, len(files), total)
