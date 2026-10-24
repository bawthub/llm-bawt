"""TASK-1329 §11: code never branches on a bot slug; behaviour comes from bot tags/data.

Every string constant under ``src/`` is checked against the deployed bot roster.
Docstrings and comments don't count. A slug that is also harness vocabulary
(``codex`` is a bot and a bridge) is skipped, because the harness name is code
vocabulary. ``KNOWN_DEBT`` pins the literals that predate the guardrail; it may
only shrink, so a new literal or a stale entry both fail.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

from llm_bawt.bot_types import DURABLE_AGENT_HARNESSES

SRC = Path(__file__).resolve().parents[1] / "src"

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


def test_src_has_no_bot_slug_literals_beyond_known_debt() -> None:
    # Cheap prefilter (unit time budget): a constant equal to a slug appears in
    # the source as a quoted slug, so only those files are parsed; the AST decides.
    quoted = re.compile(
        r"""(['"])\s*(?:%s)\s*\1""" % "|".join(sorted(CHECKED_SLUGS)), re.IGNORECASE
    )
    found: Counter[tuple[str, str]] = Counter()
    where: dict[tuple[str, str], list[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        source = path.read_text(encoding="utf-8")
        if not quoted.search(source):
            continue
        for line, slug in slug_literals(source, CHECKED_SLUGS):
            found[(rel, slug)] += 1
            where.setdefault((rel, slug), []).append(f"src/{rel}:{line}")

    new = {key: where[key] for key, count in found.items() if count > KNOWN_DEBT.get(key, 0)}
    assert not new, (
        "Bot-slug literal in code; read bot tags/data instead (TASK-1329): "
        f"{sorted(loc for locs in new.values() for loc in locs)}"
    )
    stale = {key: count for key, count in KNOWN_DEBT.items() if found.get(key, 0) < count}
    assert not stale, f"KNOWN_DEBT is larger than src; shrink these entries: {stale}"
