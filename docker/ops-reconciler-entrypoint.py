"""Run the baked reconciler only if its source and Alembic graph match its pin.

The repository mount is a read-only Git reference (and batch-gate input),
not a runtime source overlay. Check the pinned commit, not the moving HEAD.
"""

import io
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

ROOTS = ("alembic.ini", "migrations", "src")


def verify(app: Path, repo: Path, expected: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise ValueError("LLM_BAWT_OPS_RECONCILER_SOURCE_SHA must be a full commit SHA")
    revision = (app / ".image-source-sha").read_text().strip()
    if revision != expected:
        raise ValueError(f"baked image revision {revision!r} does not match pin {expected}")

    # Compare with the committed blobs even if the build context was incomplete,
    # dirty, or incorrectly labeled. This never reads the checkout's worktree.
    archive = subprocess.check_output([
        "git", "-c", f"safe.directory={repo}", "-C", str(repo),
        "archive", "--format=tar", expected, "--", *ROOTS,
    ])
    remaining = set()
    for root in ROOTS:
        path = app / root
        if path.is_symlink():
            raise ValueError(f"unexpected symlink in baked image: {root}")
        for entry in path.rglob("*") if path.is_dir() else (path,):
            rel = entry.relative_to(app)
            if entry.is_symlink():
                raise ValueError(f"unexpected symlink in baked image: {rel}")
            if entry.is_dir() or ("__pycache__" in rel.parts and rel.suffix == ".pyc"):
                continue
            remaining.add(rel.as_posix())
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as files:
        for item in files:
            if item.isdir():
                continue
            if not item.isfile() or item.name not in remaining:
                raise ValueError(f"missing or unsupported file in baked image: {item.name}")
            if (app / item.name).read_bytes() != files.extractfile(item).read():
                raise ValueError(f"baked image content differs at {item.name}")
            remaining.remove(item.name)
    if remaining:
        raise ValueError(f"unexpected file in baked image: {min(remaining)}")


def main() -> None:
    expected = os.environ.get("LLM_BAWT_OPS_RECONCILER_SOURCE_SHA", "")
    try:
        verify(Path("/app"), Path(os.environ["LLM_BAWT_OPS_GATE_REPOSITORY"]), expected)
    except (OSError, KeyError, ValueError, tarfile.TarError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"ops reconciler image/source mismatch: {exc}") from None
    print(f"ops reconciler verified baked source and migrations at {expected}", flush=True)
    os.execv(sys.executable, [sys.executable, "-I", "-m", "llm_bawt.ops.reconciler"])


if __name__ == "__main__":
    main()
