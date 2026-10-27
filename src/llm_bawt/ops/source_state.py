"""Is the app's bind-mounted source exactly a commit? STANDARD LIBRARY ONLY (TASK-1474).

The llm-bawt app runs its source from bind mounts of the operator's checkout,
so recreating the container runs whatever bytes the checkout holds. The
same-image reconcile therefore refuses unless every byte the container can
import or execute from a source mount is the checkout's ``HEAD`` commit, and
the app (not this module) proves that commit is the published branch head.

The catalog spec names the container's mount destinations in two disjoint
lists, ``source_mounts`` and ``data_mounts``. :func:`source_plan` classifies
every mount the recreated container will have, checks how each source mount
maps into the Compose project directory, and refuses a runtime environment
that could load code from elsewhere. :class:`SourceChecker` then runs fixed
``git`` argv in one :class:`sealed_container.SealedContainer` session (no
network, read-only bind of the project directory, ``noexec`` scratch tmpfs):

1. the project directory is the repository's top level; ``HEAD`` is a commit;
2. the real index carries no assume-unchanged/skip-worktree flag on source;
3. a FRESH index from ``HEAD`` (the operator's index is never trusted) shows
   no tracked change anywhere in the repository;
4. no untracked or ignored file lives under a source path, except
   ``__pycache__`` while Python is told to keep bytecode elsewhere
   (``PYTHONPYCACHEPREFIX`` outside every mount);
5. ``HEAD`` holds only regular files under each source path (no symlink,
   no gitlink) and at least one;
6. every one of those files, and every Compose file, hashed RAW
   (``--no-filters``: a clean filter or ``text``/``eol`` attribute cannot
   launder it) equals its ``HEAD`` blob.

Steps 3 and 6 overlap on purpose: status catches type changes and deletions;
the raw hash catches content a filter would hide. Git's stderr is never read
back and any path shown is sanitised. Any unexpected exit or output is
``source_unverifiable``: this check only ever fails closed.
"""
from __future__ import annotations

import posixpath
import re

if __package__:
    from .compose_render import ComposeError, SAFE_PATH
    from .sealed_container import SealedContainer, SealedRunError
else:  # standalone immutable worker
    from compose_render import ComposeError, SAFE_PATH
    from sealed_container import SealedContainer, SealedRunError

SOURCE_FIELDS = ("source_repository", "source_branch", "source_mounts", "data_mounts")
SOURCE_CHECK_LABEL = "llm-bawt.ops.source-check"
SOURCE_ROLE = "source-check"
REPOSITORY = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/[A-Za-z0-9._-]{1,100}")
_BRANCH = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}")
SHA = re.compile(r"[0-9a-f]{40}")
MAX_MOUNTS = 32
# Runtime variables that make Python (or the dynamic loader) take code from a
# path the source check does not cover. Names are reported, never values.
UNSAFE_ENV = ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PYTHONPLATLIBDIR", "PYTHONSTARTUP",
              "LD_PRELOAD", "LD_AUDIT")
PYCACHE_PREFIX = "PYTHONPYCACHEPREFIX"
PYCACHE_DIR = "__pycache__"
MAX_SHOWN = 5
_SHOWN = re.compile(r"[A-Za-z0-9._/-]{1,160}")
HASH_CHUNK = 256
_REGULAR = ("100644", "100755")
_GIT_ENV = ["GIT_CONFIG_NOSYSTEM=1", "HOME=/nonexistent", "XDG_CONFIG_HOME=/nonexistent",
            "GIT_LITERAL_PATHSPECS=1", "GIT_TERMINAL_PROMPT=0", "GIT_OPTIONAL_LOCKS=0",
            "GIT_ATTR_NOSYSTEM=1", "GIT_PAGER=cat", "LC_ALL=C"]
_FRESH_INDEX = "GIT_INDEX_FILE=/tmp/index"


class SourceStateError(ComposeError):
    """Fixed code plus sanitised path NAMES (at most :data:`MAX_SHOWN`)."""

    def __init__(self, code: str, names=(), *, detail: str = ""):
        names = sorted(set(names))
        shown = [n if _SHOWN.fullmatch(n) else "<unprintable>" for n in names[:MAX_SHOWN]]
        if len(names) > MAX_SHOWN:
            detail = f"{len(names)} paths" + (f"; {detail}" if detail else "")
        super().__init__(code, shown, detail=detail)


def _clean_path(value) -> bool:
    return (isinstance(value, str) and bool(SAFE_PATH.fullmatch(value)) and value != "/"
            and posixpath.normpath(value) == value and ".." not in value.split("/"))


def _within(path: str, root: str) -> bool:
    """``path`` is ``root`` or below it."""
    return path == root or path.startswith(root.rstrip("/") + "/")


def valid_branch(branch) -> bool:
    return (isinstance(branch, str) and bool(_BRANCH.fullmatch(branch)) and ".." not in branch
            and "//" not in branch and not branch.endswith(("/", ".", ".lock"))
            and not any(part.startswith(".") for part in branch.split("/")))


def validate_source_spec(spec: dict) -> None:
    """The four source fields of an app spec; raises ValueError."""
    if not REPOSITORY.fullmatch(str(spec["source_repository"])):
        raise ValueError("source_repository must be owner/repo")
    if not valid_branch(spec["source_branch"]):
        raise ValueError("source_branch is not a plain branch name")
    lists = {}
    for key in ("source_mounts", "data_mounts"):
        value = spec[key]
        if (not isinstance(value, list) or len(value) > MAX_MOUNTS or len(set(value)) != len(value)
                or not all(_clean_path(p) for p in value)):
            raise ValueError(f"{key} must be distinct absolute normalised container paths")
        lists[key] = value
    sources, data = lists["source_mounts"], lists["data_mounts"]
    if not sources:
        raise ValueError("source_mounts must name at least one destination")
    for path in sources:
        if any(other != path and _within(path, other) for other in sources):
            raise ValueError("a source mount may not sit inside another source mount")
        if any(_within(path, d) or _within(d, path) for d in data):
            raise ValueError("source and data mounts must be disjoint, neither inside the other")


def _mount_destinations(container: dict, image: dict, desired: dict) -> set:
    """Every destination the RECREATED container has, plus the current one's.

    The recreated container takes the desired Compose mounts, the image's own
    ``VOLUME`` declarations and the cloned body's tmpfs mounts. The current
    container's mounts are included so a stray mount fails before the swap."""
    host = container.get("HostConfig") or {}
    destinations = set(desired["mounts"])
    destinations |= {m.get("Target") for m in host.get("Mounts") or []}
    destinations |= {raw.split(":")[1] for raw in host.get("Binds") or [] if raw.count(":") >= 1}
    destinations |= {m.get("Destination") for m in container.get("Mounts") or []}
    destinations |= set(host.get("Tmpfs") or {})
    destinations |= set((image.get("Config") or {}).get("Volumes") or {})
    return {d if isinstance(d, str) else "" for d in destinations}


def source_plan(spec: dict, source: dict, container: dict, image: dict, desired: dict) -> dict:
    """``{"paths": [repo-relative source paths], "compose_files": [names], "bytecode_ok": bool}``.

    Raises :class:`SourceStateError` for a mount or environment the source
    check cannot vouch for. Values are never reported, only names."""
    sources, data = spec["source_mounts"], spec["data_mounts"]
    destinations = _mount_destinations(container, image, desired)
    shadowed = [d for d in destinations if d not in sources
                and any(_within(posixpath.normpath(d or "/"), s) for s in sources)]
    if shadowed:
        raise SourceStateError("source_mount_shadowed", shadowed)
    unclassified = [d for d in destinations if d not in sources and d not in data]
    if unclassified:
        raise SourceStateError("source_mount_unclassified", unclassified)
    missing = [s for s in sources if s not in desired["mounts"]]
    if missing:
        raise SourceStateError("source_mount_missing", missing)
    project = source["project_dir"]
    paths = []
    for destination in sources:
        mount = desired["mounts"][destination]
        host = mount.get("source")
        if (mount.get("type") != "bind" or not _clean_path(host) or host == project
                or not _within(host, project)):
            raise SourceStateError("source_mount_invalid", [destination], detail="not a bind inside the project")
        rel = posixpath.relpath(host, project)
        if ".git" in rel.split("/"):
            raise SourceStateError("source_mount_invalid", [destination], detail="inside the git directory")
        paths.append(rel)
    env = {**dict(e.partition("=")[::2] for e in (image.get("Config") or {}).get("Env") or []),
           **desired["env"]}
    unsafe = [name for name in UNSAFE_ENV if name in env]
    if unsafe:
        raise SourceStateError("source_env_unsafe", unsafe)
    prefix = env.get(PYCACHE_PREFIX)
    bytecode_ok = (_clean_path(prefix)
                   and not any(_within(prefix, d) or _within(d, prefix) for d in destinations if d))
    compose_files = [posixpath.relpath(f, project) for f in source["compose_files"]]
    return {"paths": sorted(paths), "compose_files": compose_files, "bytecode_ok": bool(bytecode_ok)}


def _records(out: bytes) -> list[str]:
    try:
        text = out.decode("utf-8")
    except UnicodeDecodeError:
        raise SourceStateError("source_unverifiable", detail="git output is not UTF-8") from None
    return [r for r in text.split("\0") if r]


def _status_names(records: list[str]) -> list[str]:
    """Paths of ``status --porcelain=v2 -z`` records (types 1, 2, u)."""
    names, skip = [], False
    for record in records:
        if skip:  # a rename's original path
            skip = False
            continue
        kind, fields = record[:1], {"1": 8, "2": 9, "u": 10}.get(record[:1])
        parts = record.split(" ", fields) if fields else []
        names.append(parts[-1] if fields and len(parts) == fields + 1 else "<unparsed>")
        skip = kind == "2"
    return names


class SourceChecker:
    """Runs the git steps in one sealed session. ``api`` is the worker's
    stdlib DockerAPI shape; ``image`` the digest-pinned renderer image (it
    ships git and ``sleep``)."""

    def __init__(self, api, image: str, *, tag: str):
        self.sealed = SealedContainer(api, image, role=SOURCE_ROLE, tag=tag, labels={SOURCE_CHECK_LABEL: "1"})

    def check(self, project_dir: str, plan: dict) -> str:
        """The ``HEAD`` commit the source mounts are exactly; raises SourceStateError."""
        try:
            with self.sealed.session(env=list(_GIT_ENV), directory=project_dir) as run:
                return _Steps(run, project_dir, plan).verify()
        except SealedRunError as exc:
            raise SourceStateError("source_unverifiable", detail=f"sealed git {exc.kind}") from None


class _Steps:
    def __init__(self, run, project_dir: str, plan: dict):
        self.run, self.project, self.plan = run, project_dir, plan
        self.base = ["git", "-c", f"safe.directory={project_dir}", "-c", "core.fsmonitor=false",
                     "-c", "core.ignoreStat=false", "-c", "core.untrackedCache=false",
                     "-c", "core.checkStat=default", "-c", "core.trustctime=true",
                     "-c", "core.fileMode=true", "-c", "core.symlinks=true", "--no-optional-locks"]

    def git(self, *args, fresh=False) -> bytes:
        code, out, _err = self.run([*self.base, *args], [_FRESH_INDEX] if fresh else [])
        if code != 0:
            raise SourceStateError("source_unverifiable", detail=f"git {args[0]} exited {code}")
        return out

    def verify(self) -> str:
        paths, compose = self.plan["paths"], self.plan["compose_files"]
        lines = self.git("rev-parse", "--show-toplevel", "HEAD^{commit}").decode("utf-8", "replace").split("\n")
        if len(lines) != 3 or lines[2] != "" or not SHA.fullmatch(lines[1]):
            raise SourceStateError("source_unverifiable", detail="rev-parse output")
        if lines[0] != self.project:
            raise SourceStateError("source_tree_unsupported", detail="project directory is not the repository root")
        head = lines[1]
        flagged = [r[2:] for r in _records(self.git("ls-files", "-z", "-v", "--", *paths)) if not r.startswith("H ")]
        if flagged:
            raise SourceStateError("source_index_flags", flagged)
        self.git("read-tree", head, fresh=True)
        dirty = _records(self.git("status", "--porcelain=v2", "-z", "--untracked-files=no", fresh=True))
        if dirty:
            raise SourceStateError("source_dirty", _status_names(dirty))
        others = _records(self.git("ls-files", "-z", "--others", "--directory", "--no-empty-directory",
                                   "--", *paths, fresh=True))
        cache = [p for p in others if PYCACHE_DIR in p.split("/")]
        stray = [p for p in others if p not in cache]
        if stray:
            raise SourceStateError("source_untracked", stray)
        if cache and not self.plan["bytecode_ok"]:
            # Python would load a forged .pyc whose header matches its source.
            raise SourceStateError("source_bytecode_unsafe", cache, detail=f"{PYCACHE_PREFIX} not outside every mount")
        blobs = self._tree(head, paths)
        tracked = self._tree(head, compose, recursive=False)
        bad_compose = [c for c in compose if c not in tracked or tracked[c][0] != "100644"]
        if bad_compose:
            raise SourceStateError("source_compose_untracked", bad_compose)
        empty = [p for p in paths if not any(_within(f, p) for f in blobs)]
        if empty:
            raise SourceStateError("source_mount_invalid", empty, detail="no tracked files")
        odd = [f for f, (mode, _sha) in blobs.items() if mode not in _REGULAR]
        if odd:
            raise SourceStateError("source_unsupported_entry", odd)
        expected = {**blobs, **tracked}
        changed = []
        names = sorted(expected)
        for i in range(0, len(names), HASH_CHUNK):
            chunk = names[i:i + HASH_CHUNK]
            hashes = self.git("hash-object", "--no-filters", "--", *chunk).decode("ascii", "replace").split("\n")
            if len(hashes) != len(chunk) + 1 or hashes[-1] != "" or not all(SHA.fullmatch(h) for h in hashes[:-1]):
                raise SourceStateError("source_unverifiable", detail="hash-object output")
            changed += [name for name, digest in zip(chunk, hashes) if digest != expected[name][1]]
        if changed:
            raise SourceStateError("source_dirty", changed, detail="raw bytes differ from HEAD")
        return head

    def _tree(self, head: str, paths: list, *, recursive: bool = True) -> dict:
        """``{path: (mode, blob sha)}`` of HEAD under ``paths``."""
        args = ["ls-tree", "-z", "--full-tree", *(["-r"] if recursive else []), head, "--", *paths]
        entries = {}
        for record in _records(self.git(*args)):
            meta, tab, path = record.partition("\t")
            parts = meta.split(" ")
            if not tab or len(parts) != 3 or not SHA.fullmatch(parts[2]):
                raise SourceStateError("source_unverifiable", detail="ls-tree output")
            entries[path] = (parts[0], parts[2])
        return entries


def source_commit(spec: dict, head: str) -> dict:
    return {"repository": spec["source_repository"], "branch": spec["source_branch"], "head": head}


def valid_source_commit(spec: dict, value) -> bool:
    return (isinstance(value, dict) and set(value) == {"repository", "branch", "head"}
            and value["repository"] == spec["source_repository"] and value["branch"] == spec["source_branch"]
            and isinstance(value["head"], str) and bool(SHA.fullmatch(value["head"])))


__all__ = ["PYCACHE_PREFIX", "SOURCE_CHECK_LABEL", "SOURCE_FIELDS", "SourceChecker", "SourceStateError",
           "UNSAFE_ENV", "source_commit", "source_plan", "valid_source_commit", "validate_source_spec"]
