"""TASK-1249: product source never names an operator role tag.

Role tags are operator data in ``bot_tag_catalog``; code only knows the
behaviour tags in ``bot_tags/vocabulary.py``. A role slug baked into ``src/``
(a seed, a default, a ``== "reviewer"`` check) would tie the product to one
install's roster. This guard scans every string constant under ``src/`` for
the role names an install is most likely to create. It is a partial ratchet:
extend the list, never allowlist around it.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

# Representative operator role slugs. Matching is on the whole (stripped,
# case-folded) string constant, so prose that merely mentions a reviewer is fine.
ROLE_SLUGS = frozenset({"coder", "reviewer", "coordinator", "arbiter", "auditor"})
# Cheap prefilter (the unit time budget): only files with a quoted candidate are
# parsed. A whole-string constant always sits between its own quote characters.
CANDIDATE = re.compile(r"""["']\s*(?:%s)\s*["']""" % "|".join(sorted(ROLE_SLUGS)), re.IGNORECASE)


def role_literals(tree: ast.AST) -> list[tuple[int, str]]:
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and node.value.strip().casefold() in ROLE_SLUGS
    ]


def test_guard_detects_a_seeded_role():
    tree = ast.parse('ROLES = (Tag("coder", "Coder"), Tag("Reviewer ", "x"))\nok = "a reviewer said"')
    assert [value for _, value in role_literals(tree)] == ["coder", "Coder", "Reviewer "]
    assert CANDIDATE.search('x = ( "Reviewer " )') and not CANDIDATE.search('"a reviewer said"')


def test_src_names_no_role_tag():
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        source = path.read_text()
        if not CANDIDATE.search(source):
            continue
        for lineno, value in role_literals(ast.parse(source, filename=str(path))):
            offenders.append(f"{path.relative_to(SRC.parent)}:{lineno}: {value!r}")
    assert offenders == [], "role tag slugs belong in the DB, not in source:\n" + "\n".join(offenders)
