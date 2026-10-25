"""TASK-1249: product source never names an operator role tag.

Role tags are operator data in ``bot_tag_catalog``; code only knows the
behaviour tags in ``bot_tags/vocabulary.py``. A role slug baked into ``src/``
(a seed, a default, a ``== "reviewer"`` check) would tie the product to one
install's roster. This guard checks every non-docstring string constant under
``src/`` against the role names an install is most likely to create. It is a
partial ratchet: extend the list, never allowlist around it.

Every file is parsed, with the walker of the bot-slug guard (TASK-1329): the
AST folds implicit concatenation (``"co" "der"``) and escapes (``"\\x63oder"``),
so no text prefilter can safely skip a file. The files are split into shards to
stay inside the unit time budget. Strings built at runtime are out of scope.
"""
from __future__ import annotations

from functools import cache
from pathlib import Path

import pytest

from tests.test_no_bot_slug_literals import SHARDS, partition, slug_literals

SRC = Path(__file__).resolve().parents[1] / "src"

# Representative operator role slugs. Matching is on the whole (stripped,
# case-folded) string constant, so prose that merely mentions a reviewer is fine.
ROLE_SLUGS = frozenset({"coder", "reviewer", "coordinator", "arbiter", "auditor"})


def role_literals(paths, root: Path) -> list[str]:
    """``path:line: slug`` for each role-slug constant, parsing every file given."""
    return [
        f"{path.relative_to(root).as_posix()}:{line}: {slug}"
        for path in paths
        for line, slug in slug_literals(path.read_text(encoding="utf-8"), ROLE_SLUGS)
    ]


@cache
def _src_shards() -> list[list[Path]]:
    return partition(SRC, SHARDS)


def test_guard_parses_every_file_and_sees_folded_literals(tmp_path: Path) -> None:
    files = {
        "seeded.py": 'ROLES = (Tag("coder", "Coder"), Tag("Reviewer ", "x"))\n',
        "implicit.py": 'ROLE = "co" "der"\n',
        "escaped.py": 'ROLE = "\\x63oder"\n',
        "fstring.py": 'ROLE = f"audi" "tor"\n',
        "pkg/clean.py": '"""A coder."""\nok = "a reviewer said"\nname = "coders"\n',
    }
    for name, body in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(body, encoding="utf-8")

    parts = partition(tmp_path, SHARDS)
    assert sorted(p for part in parts for p in part) == sorted(tmp_path.rglob("*.py"))
    assert sorted(hit for part in parts for hit in role_literals(part, tmp_path)) == [
        "escaped.py:1: coder",
        "fstring.py:1: auditor",
        "implicit.py:1: coder",
        "seeded.py:1: coder",
        "seeded.py:1: coder",
        "seeded.py:1: reviewer",
    ]


@pytest.mark.parametrize("shard", range(SHARDS))
def test_src_names_no_role_tag(shard: int) -> None:
    offenders = role_literals(_src_shards()[shard], SRC)
    assert offenders == [], "role tag slugs belong in the DB, not in source:\n" + "\n".join(offenders)
