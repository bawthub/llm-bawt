"""Paths whose contents must never enter turn snapshots or diff payloads."""

from __future__ import annotations

from pathlib import PurePosixPath

_SOURCE_SUFFIXES = {".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".sh"}


def is_sensitive_path(path: str, *, tracked: bool = False) -> bool:
    """Protect data paths without suppressing ordinary tracked source diffs."""
    normalized = PurePosixPath(path.replace("\\", "/"))
    parts = normalized.parts
    if parts and parts[-1].lower().endswith((".example", ".template")):
        return False
    # The tracked Docker settings template is nonsecret configuration. Unlike
    # example/template suffixes, this exception needs tracked provenance.
    if tracked and normalized.name.lower() == ".env.docker":
        parts = parts[:-1]
    tracked_source = tracked and normalized.suffix.lower() in _SOURCE_SUFFIXES
    return any(
        part.lower() == ".env"
        or part.lower().startswith(".env.")
        or part.lower().endswith(".key")
        or ("secret" in part.lower() and not tracked_source)
        for part in parts
    )


def path_stat_signature(path) -> tuple[int, ...] | None:
    """Detect changes without opening, hashing or retaining sensitive bytes."""
    try:
        stat = path.lstat()
    except OSError:
        return None
    return (stat.st_mode, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
