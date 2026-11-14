"""Tests never inherit a caller's repo-locating git variables (TASK-2302).

On 2026-11-14 a host run exported ``GIT_DIR`` (a linked worktree's admin
dir) and ``GIT_WORK_TREE`` so file-listing tests could find a bridge-created
worktree. A test's ``git -C <tmp> init`` then re-initialised that repository
instead of the temp one and wrote ``core.worktree`` into the shared checkout's
``.git/config``; the next fast-forward of the live tree landed elsewhere.

``install()`` runs at root-conftest import, before any test module, and drops
those variables from this process so every child git call targets its own
``cwd``. Tests that list tracked files then need the worktree's ``.git`` link
to resolve where pytest runs; when it does not, a warning names the fix.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRUBBED_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR",
                "GIT_OBJECT_DIRECTORY", "GIT_NAMESPACE", "GIT_PREFIX")


def scrub(environ: dict[str, str] | os._Environ = os.environ) -> list[str]:
    """Remove repo-locating git variables; return the names removed."""
    removed = [name for name in SCRUBBED_ENV if name in environ]
    for name in removed:
        environ.pop(name)
    return removed


def link_hint(root: Path = ROOT) -> str | None:
    """A fix when ``root``'s ``.git`` link does not resolve here, else None."""
    probe = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root,
                           capture_output=True, text=True, env=os.environ.copy())
    if probe.returncode == 0 and Path(probe.stdout.strip()).resolve() == root.resolve():
        return None
    link = root / ".git"
    target = ""
    if link.is_file():
        text = link.read_text(errors="replace").strip()
        target = text.removeprefix("gitdir:").strip() if text.startswith("gitdir:") else ""
    marker = "/.git/worktrees/"
    generic = f"{root} is not a git checkout here; tests that list tracked files will fail."
    if marker not in target or root.parent.name != ".worktrees":
        return generic
    repo_rel, admin = target.split(marker, 1)
    # Same layout on both mounts; the tree sits in <dev>/.worktrees (sibling
    # repo) or <dev>/<repo>/.worktrees (nested). Suggest only a link that
    # resolves here to a real worktree admin dir.
    candidates = (Path(root.parent.parent, Path(repo_rel).name, ".git", "worktrees", admin),
                  Path(root.parent.parent, ".git", "worktrees", admin))
    found = next((c for c in candidates if (c / "commondir").is_file()), None)
    if found is None:
        return generic
    relative = os.path.relpath(found, root)
    return (f"{link} points at {target}, which does not exist here (a worktree created "
            f"under another mount). Make the link relative instead of exporting GIT_DIR: "
            f"printf 'gitdir: {relative}\\n' > {link}")


def install() -> None:
    removed = scrub()
    if removed:
        print(f"tests: ignoring inherited {', '.join(removed)} (TASK-2302)", file=sys.stderr)
    hint = link_hint()
    if hint:
        print(f"tests: {hint}", file=sys.stderr)
