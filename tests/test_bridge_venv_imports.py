"""Bridge entrypoints must import with only the lean agent-bridge venv.

Dockerfile.agent-bridge installs the project with ``--no-deps`` into a venv
pinned by ``docker/agent-bridge-constraints.txt``. Its build-time import check
only runs on an image rebuild, while source-mounted changes reach the running
bridge without one. A top-level import of a non-bridge dependency (TASK-2380:
``rich`` via ``llm_bawt/__init__``) then crashes the bridge at startup.

This test imports each entrypoint in a fresh interpreter whose meta-path
refuses any third-party top-level module not provided by a distribution in the
constraints snapshot.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONSTRAINTS = ROOT / "docker" / "agent-bridge-constraints.txt"

ENTRYPOINTS = [
    "claude_code_bridge.__main__",
    "codex_bridge.__main__",
    "agent_bridge.house_preview_supervisor",
]

_PROBE = textwrap.dedent(
    """
    import importlib, importlib.metadata as md, re, sys

    allowed = set()
    for line in open(sys.argv[1], encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if line:
            allowed.add(re.split(r"[=<>!~;\\[ ]", line, 1)[0].lower().replace("_", "-"))
    owners = md.packages_distributions()
    project = {"llm_bawt", "agent_bridge", "claude_code_bridge", "codex_bridge",
               "local_model_bridge"}

    import importlib.machinery, os, sysconfig
    stdlib_dirs = tuple({os.path.realpath(sysconfig.get_paths()[k]) + os.sep
                         for k in ("stdlib", "platstdlib")})

    def in_stdlib(top):
        spec = importlib.machinery.PathFinder.find_spec(top)
        origin = spec and (spec.origin or next(iter(spec.submodule_search_locations or ()), None))
        return bool(origin) and os.path.realpath(origin).startswith(stdlib_dirs) \\
            and "site-packages" not in origin

    class LeanVenv:
        def find_spec(self, name, path=None, target=None):
            top = name.partition(".")[0]
            if top in sys.stdlib_module_names or top in project or top in sys.builtin_module_names:
                return None
            dists = {d.lower().replace("_", "-") for d in owners.get(top, ())}
            if dists & allowed or in_stdlib(top):
                return None
            raise ModuleNotFoundError(
                f"{top!r} is not in the agent-bridge venv", name=top)

    sys.meta_path.insert(0, LeanVenv())
    for module in sys.argv[2:]:
        importlib.import_module(module)
    """
)


PROJECT = ("llm_bawt", "agent_bridge", "claude_code_bridge", "codex_bridge")


def _project_imports(module: str) -> list[str]:
    """The module plus every project module it imports, including deferred
    imports inside functions (bridges load api_auth.outbound from main())."""
    source = ROOT / "src" / (module.replace(".", "/") + ".py")
    found = {module}
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        else:
            continue
        found.update(n for n in names if n.partition(".")[0] in PROJECT)
    return sorted(found)


def test_constraints_snapshot_present() -> None:
    assert CONSTRAINTS.is_file()


@pytest.mark.parametrize("module", ENTRYPOINTS)
def test_bridge_entrypoint_imports_in_lean_venv(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, str(CONSTRAINTS), *_project_imports(module)],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"{module} needs a module outside the agent-bridge venv:\n{result.stderr[-3000:]}"
    )
