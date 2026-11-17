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
import re
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


# Static pass for imports that only run once startup executes (inside main()
# or a helper it calls), which the import probe above never reaches. Scope:
# the entrypoints, their deferred project imports (one hop), and everything
# reachable from those through module-level project imports. Imports guarded
# by TYPE_CHECKING or an ImportError/Exception handler are treated as optional.


def _allowed_dists() -> set[str]:
    allowed = set()
    for line in CONSTRAINTS.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            allowed.add(re.split(r"[=<>!~;\[ ]", line, maxsplit=1)[0].lower().replace("_", "-"))
    return allowed


def _module_file(src: Path, module: str) -> Path | None:
    base = src / module.replace(".", "/")
    for candidate in (base / "__init__.py", base.with_suffix(".py")):
        if candidate.is_file():
            return candidate
    return None


def _is_optional(parents: list[ast.AST]) -> bool:
    for node in parents:
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            return True
        if isinstance(node, ast.Try) and any(
            h.type is None or re.search(r"ImportError|ModuleNotFoundError|Exception", ast.unparse(h.type))
            for h in node.handlers
        ):
            return True
    return False


def _imports(src: Path, module: str) -> list[tuple[str, int, bool, bool]]:
    """(name, line, deferred, optional) for each import in ``module``."""
    path = _module_file(src, module)
    assert path is not None
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    found: list[tuple[str, int, bool, bool]] = []

    def visit(node: ast.AST, parents: list[ast.AST]) -> None:
        for child in ast.iter_child_nodes(node):
            chain = [*parents, node]
            names: list[str] = []
            if isinstance(child, ast.ImportFrom):
                base = child.module or ""
                if child.level:
                    anchor = package
                    for _ in range(child.level - 1):
                        anchor = anchor.rpartition(".")[0]
                    base = f"{anchor}.{base}" if base else anchor
                names = [base, *(f"{base}.{a.name}" for a in child.names)]
            elif isinstance(child, ast.Import):
                names = [a.name for a in child.names]
            deferred = any(isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)) for p in chain)
            found.extend((n, child.lineno, deferred, _is_optional(chain)) for n in names)
            visit(child, chain)

    visit(ast.parse(path.read_text(encoding="utf-8")), [])
    return found


def _startup_violations(src: Path, roots: list[str], allowed: set[str], owners) -> list[str]:
    project = {p.name for p in src.iterdir() if p.is_dir()}
    seen: set[str] = set()
    todo = list(roots)
    bad: list[str] = []
    while todo:
        module = todo.pop()
        if module in seen or _module_file(src, module) is None:
            continue
        seen.add(module)
        for name, line, deferred, optional in _imports(src, module):
            top = name.partition(".")[0]
            if top in project:
                if not deferred or module in roots:
                    todo.append(name)
                continue
            if optional or top in sys.stdlib_module_names or top in sys.builtin_module_names:
                continue
            dists = {d.lower().replace("_", "-") for d in owners.get(top, ())}
            dists.add(top.lower().replace("_", "-"))
            if not dists & allowed:
                bad.append(f"{module}:{line} imports {top!r}")
    return sorted(set(bad))


def test_bridge_startup_imports_stay_in_lean_venv() -> None:
    import importlib.metadata as md

    bad = _startup_violations(ROOT / "src", ENTRYPOINTS, _allowed_dists(), md.packages_distributions())
    assert not bad, "bridge startup needs modules outside the agent-bridge venv:\n" + "\n".join(bad)


@pytest.mark.parametrize(
    "files",
    [
        {"brg/__main__.py": "def main():\n    import rich\n"},
        {
            "brg/__main__.py": "from brg import helper\ndef main():\n    helper.start()\n",
            "brg/helper.py": "def start():\n    import rich\n",
        },
        {
            "brg/__main__.py": "def main():\n    from brg.helper import start\n    start()\n",
            "brg/helper.py": "def start():\n    from rich import console\n",
        },
    ],
    ids=["main", "imported-helper", "deferred-helper"],
)
def test_startup_scan_catches_deferred_imports(tmp_path: Path, files: dict[str, str]) -> None:
    for rel, body in {"brg/__init__.py": "", **files}.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(body, encoding="utf-8")
    bad = _startup_violations(tmp_path, ["brg.__main__"], {"httpx"}, {"rich": ["rich"]})
    assert bad and all("'rich'" in b for b in bad)


def test_startup_scan_allows_optional_and_allowed_imports(tmp_path: Path) -> None:
    (tmp_path / "brg").mkdir()
    (tmp_path / "brg/__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "brg/__main__.py").write_text(
        "import json\nimport httpx\nfrom typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    import rich\n"
        "def main():\n    try:\n        import rich\n    except ImportError:\n        pass\n",
        encoding="utf-8",
    )
    assert _startup_violations(tmp_path, ["brg.__main__"], {"httpx"}, {"httpx": ["httpx"]}) == []
