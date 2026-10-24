"""TASK-1329 §11: code never branches on a bot slug; behaviour comes from bot tags/data.

Every string constant under ``src/`` is checked against the deployed bot roster.
Every file is parsed: the AST folds implicit concatenation (``"mi" "ra"``) and
escapes (``"\\x6dira"``), so no text prefilter can safely skip a file. To stay
inside the unit time budget, the files are split into ``SHARDS`` test cases.
Docstrings and comments don't count. A slug that is also harness vocabulary
(``codex`` is a bot and a bridge) is skipped, because the harness name is code
vocabulary. Strings built at runtime (``"mi" + "ra"``) are not literals and are
out of scope.

``KNOWN_DEBT`` is a count ratchet over literals that predate the guardrail: a
file's count per slug may only shrink. More literals than the pinned count
fail, and so does a pinned count larger than the code. Swapping one pinned
literal for a new one of the same slug in the same file keeps the count and is
not caught.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from functools import cache
from pathlib import Path

import pytest

from llm_bawt.bot_types import DURABLE_AGENT_HARNESSES

SRC = Path(__file__).resolve().parents[1] / "src"
# A full parse of src/ takes about 1.1 s on echo; eight shards keep each case
# near 0.15 s, well inside the 1 s per-case unit budget even under xdist load.
SHARDS = 8

# The DB is the source of truth for bots and the bots.yaml seed is gone, so the
# roster is listed here. Add a new bot's slug when it is created.
ROSTER = frozenset(
    {"al", "byte", "caid", "codex", "gavel", "loopy", "mira", "nova", "qlocal", "rook", "snark", "vex"}
)
CHECKED_SLUGS = ROSTER - DURABLE_AGENT_HARNESSES

# (path under src/, slug) -> count. Retiring these is TASK-1373: the "nova"
# defaults should resolve the configured default bot instead of naming one.
KNOWN_DEBT: dict[tuple[str, str], int] = {
    ("llm_bawt/bots.py", "nova"): 3,
    ("llm_bawt/cli/display_cmd.py", "nova"): 1,
    ("llm_bawt/cli/openclaw_handler.py", "vex"): 1,
    ("llm_bawt/core/client.py", "nova"): 1,
    ("llm_bawt/core/initialization.py", "nova"): 1,
    ("llm_bawt/mcp_server/extraction.py", "nova"): 1,
    ("llm_bawt/memory/base.py", "nova"): 1,
    ("llm_bawt/memory/migrations.py", "nova"): 1,
    ("llm_bawt/memory/postgresql.py", "nova"): 2,
    ("llm_bawt/memory/postgresql_short_term.py", "nova"): 1,
    ("llm_bawt/memory/summarization.py", "nova"): 1,
    ("llm_bawt/service/background_service.py", "nova"): 1,
    ("llm_bawt/service/client.py", "nova"): 1,
    ("llm_bawt/service/core.py", "nova"): 1,
    ("llm_bawt/service/tasks.py", "nova"): 7,
    ("llm_bawt/tools/executor.py", "nova"): 1,
    ("llm_bawt/tools/loop.py", "nova"): 2,
    ("llm_bawt/tools/streaming.py", "nova"): 1,
    ("llm_bawt/utils/config.py", "nova"): 1,
    ("llm_bawt/utils/history.py", "nova"): 1,
}


def _docstring_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                ids.add(id(first.value))
    return ids


def slug_literals(source: str, slugs: frozenset[str]) -> list[tuple[int, str]]:
    """(line, slug) for each non-docstring string constant equal to a slug."""
    tree = ast.parse(source)
    docstrings = _docstring_ids(tree)
    return sorted(
        (node.lineno, node.value.strip().lower())
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        and node.value.strip().lower() in slugs
    )


def test_scanner_flags_literals_and_ignores_docstrings_and_comments() -> None:
    source = '''
"""Module docstring naming nova."""
def pick(bot_id: str = "nova") -> bool:
    """Docstring naming 'mira'."""
    # comment naming "mira"
    return bot_id in {"Mira", "e.g. nova"} or bot_id == "vexed"
'''
    assert slug_literals(source, frozenset({"nova", "mira", "vex"})) == [(3, "nova"), (6, "mira")]


def test_harness_vocabulary_is_not_treated_as_a_bot_slug() -> None:
    assert "codex" in ROSTER and "codex" not in CHECKED_SLUGS
    assert {"mira", "nova", "qlocal"} <= CHECKED_SLUGS


def partition(root: Path, shards: int) -> list[list[Path]]:
    """Every ``*.py`` under root in exactly one shard, largest file first onto the lightest shard."""
    sized = sorted(((p.stat().st_size, p) for p in root.rglob("*.py")), key=lambda sp: (-sp[0], sp[1]))
    loads = [0] * shards
    parts: list[list[Path]] = [[] for _ in range(shards)]
    for size, path in sized:
        lightest = loads.index(min(loads))
        parts[lightest].append(path)
        loads[lightest] += size
    return parts


def scan(paths: Iterable[Path], root: Path) -> dict[tuple[str, str], list[str]]:
    """(path under root, slug) -> locations, parsing every file given."""
    found: dict[tuple[str, str], list[str]] = {}
    for path in paths:
        rel = path.relative_to(root).as_posix()
        for line, slug in slug_literals(path.read_text(encoding="utf-8"), CHECKED_SLUGS):
            found.setdefault((rel, slug), []).append(f"{root.name}/{rel}:{line}")
    return found


@cache
def _src_shards() -> list[list[Path]]:
    return partition(SRC, SHARDS)


def test_walker_parses_every_file_and_sees_folded_literals(tmp_path: Path) -> None:
    files = {
        "implicit.py": 'def f(bot_id):\n    return bot_id == "mi" "ra"\n',
        "escaped.py": 'BOT = "\\x6dira"\n',
        "fstring.py": 'LABEL = f"no" "va"\n',
        "spaced.py": "SLUGS = {' Mira '}\n",
        "pkg/clean.py": 'NAME = "mirage"\n',
    }
    for name, body in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(body, encoding="utf-8")

    parts = partition(tmp_path, SHARDS)
    assert sorted(p for part in parts for p in part) == sorted(tmp_path.rglob("*.py"))
    found = {key for part in parts for key in scan(part, tmp_path)}
    assert found == {
        ("implicit.py", "mira"),
        ("escaped.py", "mira"),
        ("fstring.py", "nova"),
        ("spaced.py", "mira"),
    }


def test_known_debt_paths_exist() -> None:
    # A pinned file that was deleted or moved is in no shard; catch it here.
    missing = sorted({rel for rel, _ in KNOWN_DEBT if not (SRC / rel).is_file()})
    assert not missing, f"KNOWN_DEBT names files that are gone; drop these entries: {missing}"


@pytest.mark.parametrize("shard", range(SHARDS))
def test_src_has_no_bot_slug_literals_beyond_known_debt(shard: int) -> None:
    paths = _src_shards()[shard]
    found = scan(paths, SRC)

    new = {key: locs for key, locs in found.items() if len(locs) > KNOWN_DEBT.get(key, 0)}
    assert not new, (
        "Bot-slug literal in code; read bot tags/data instead (TASK-1329): "
        f"{sorted(loc for locs in new.values() for loc in locs)}"
    )
    rels = {path.relative_to(SRC).as_posix() for path in paths}
    stale = {
        key: count
        for key, count in KNOWN_DEBT.items()
        if key[0] in rels and len(found.get(key, ())) < count
    }
    assert not stale, f"KNOWN_DEBT is larger than src; shrink these entries: {stale}"
