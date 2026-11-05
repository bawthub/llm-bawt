"""Small, deployment-independent noise policy for command-observed snapshots.

Do not exclude these paths from capture: history and the Ignored view retain
them, and an explicit Restore can make an exact snapshot actionable again.
Tool-attributed edits and ordinary source paths are never classified here.
Tracking status deliberately does not exempt generated paths: cached bytecode
and vendored node_modules remain restorable in Ignored even when committed.
"""
from pathlib import PurePosixPath


GENERATED_CACHE_DIRECTORIES = frozenset({
    "__pycache__", ".pytest_cache", "node_modules", ".mypy_cache", ".ruff_cache",
})


def default_ignored(*, provenance: str | None, path: str) -> bool:
    if provenance != "command_observation":
        return False
    parts = PurePosixPath(path).parts
    return any(part in GENERATED_CACHE_DIRECTORIES for part in parts[:-1]) or path.endswith(".pyc")
