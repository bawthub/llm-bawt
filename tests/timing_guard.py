"""Find the two timing patterns behind most CI-only test failures (TASK-1457).

1. Hand-rolled ``pg_stat_activity`` polling. PostgreSQL snapshots that view
   once per transaction, so a loop that polls inside one transaction keeps its
   first view forever; seven tests copied such a loop before TASK-1451. Tests
   wait for a lock with ``tests/pg_lock_wait.wait_blocked_by`` instead, and
   don't re-derive it from ``pg_locks`` or ``pg_blocking_pids`` either.
2. Asserting that wall-clock time stayed under a fixed limit
   (``assert time.monotonic() - started < 0.5``). The same code takes longer on
   a slower, shared runner, so the assert fails there with no regression. Prove
   the behaviour instead: an event, a barrier, a fake clock, or the call's own
   timeout result. A lower bound (``elapsed >= wait``) is fine: a slow runner
   only makes it larger.

A line may opt out with ``# timing-ok: <reason>`` on the flagged statement or
the line directly above it. The reason is required and is what a reviewer
judges.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re

EXEMPTION = re.compile(r"#\s*timing-ok:\s*\S")
CLOCK_CALLS = frozenset({"monotonic", "monotonic_ns", "perf_counter", "perf_counter_ns", "time", "time_ns"})
# Values measured by the code under test, e.g. ``outcome.elapsed_ms``.
TIMING_NAME = re.compile(r"(?:^|_)(?:elapsed|duration|took)(?:_|$)")
# One sanctioned way to observe a lock wait; a loop over pg_locks re-derives it.
LOCK_VIEWS = ("pg_stat_activity", "pg_locks", "pg_blocking_pids")
UPPER_BOUND_OPS = (ast.Lt, ast.LtE)
# Fields holding a compound statement's nested statements (or handlers/cases).
_BODIES = ("body", "orelse", "finalbody", "handlers", "cases")
LOWER_BOUND_OPS = (ast.Gt, ast.GtE)


@dataclass(frozen=True, order=True)
class Finding:
    path: str
    line: int
    kind: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.kind}"


def _is_clock_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
    return name in CLOCK_CALLS


def _is_timing_name(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return bool(TIMING_NAME.search(node.id))
    if isinstance(node, ast.Attribute):
        return bool(TIMING_NAME.search(node.attr))
    return False


@dataclass
class _Clocks:
    """Names in one scope holding a clock reading, a deadline, or a duration."""

    points: set[str]
    deadlines: set[str]
    durations: set[str]

    def point(self, node: ast.AST) -> bool:
        """A clock reading, or a reading shifted by an offset (a deadline)."""
        if _is_clock_call(node):
            return True
        if isinstance(node, ast.Name):
            return node.id in self.points or node.id in self.deadlines
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            return self.point(node.left) != self.point(node.right)
        return False

    def deadline(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self.deadlines
        return (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)
                and self.point(node.left) != self.point(node.right))

    def duration(self, node: ast.AST) -> bool:
        if _is_timing_name(node) or (isinstance(node, ast.Name) and node.id in self.durations):
            return True
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Sub) and self.point(node.left) and self.point(node.right):
                return True
            if isinstance(node.op, (ast.Mult, ast.Div, ast.FloorDiv)):
                return self.duration(node.left) or self.duration(node.right)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"int", "round", "float"}:
            return any(self.duration(arg) for arg in node.args)
        return False


def _single_names(node: ast.AST) -> list[str]:
    targets: list[ast.AST] = []
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
        targets = [node.target]
    return [target.id for target in targets if isinstance(target, ast.Name)]


def _clocks(scope: ast.AST) -> _Clocks:
    """Follow plain assignments until no name changes kind. Only arithmetic on a
    clock reading counts: a value computed from one by a call (a timestamp
    passed to a builder) is data, not time."""
    assignments = [node for node in ast.walk(scope)
                   if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
                   and getattr(node, "value", None) is not None]
    clocks = _Clocks(set(), set(), set())
    while True:
        before = (len(clocks.points), len(clocks.deadlines), len(clocks.durations))
        for node in assignments:
            for name in _single_names(node):
                if clocks.duration(node.value):
                    clocks.durations.add(name)
                elif clocks.deadline(node.value):
                    clocks.deadlines.add(name)
                elif clocks.point(node.value):
                    clocks.points.add(name)
        if (len(clocks.points), len(clocks.deadlines), len(clocks.durations)) == before:
            return clocks


def _upper_bounds_wall_clock(test: ast.AST, clocks: _Clocks) -> bool:
    """``duration < limit`` or ``value < deadline``, in either operand order.

    A bracket between two readings (``before <= stamp <= time.time()``) and a
    comparison with the current time (``expires_at > time.time()``) hold on any
    runner, so neither counts.
    """
    for compare in (node for node in ast.walk(test) if isinstance(node, ast.Compare)):
        operands = [compare.left, *compare.comparators]
        for left, op, right in zip(operands, compare.ops, operands[1:]):
            if isinstance(op, LOWER_BOUND_OPS):
                left, right = right, left
            elif not isinstance(op, UPPER_BOUND_OPS):
                continue
            if clocks.duration(left) or clocks.deadline(right):
                return True
    return False


def _docstring(node: ast.AST) -> ast.AST | None:
    body = getattr(node, "body", None)
    if (isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)):
        return body[0].value
    return None


def _orders(test: ast.AST) -> bool:
    """Cheap pre-check: only an ordering comparison can bound a time."""
    return any(isinstance(node, ast.Compare)
               and any(isinstance(op, UPPER_BOUND_OPS + LOWER_BOUND_OPS) for op in node.ops)
               for node in ast.walk(test))


def _statement_lines(stmt: ast.stmt) -> range:
    """A simple statement's own lines; a compound statement's header only."""
    body = getattr(stmt, "body", None)
    end = body[0].lineno - 1 if isinstance(body, list) and body else stmt.end_lineno or stmt.lineno
    return range(stmt.lineno, max(stmt.lineno, end) + 1)


# A file without any of these can't hold a finding, so it isn't parsed: the guard
# scans the whole test tree inside the unit budget.
_MAY_MATCH = re.compile("|".join([*LOCK_VIEWS, "elapsed", "duration", "took",
                                  rf"\b(?:{'|'.join(sorted(CLOCK_CALLS))})\s*\("]))


def scan_source(source: str, path: str) -> list[Finding]:
    if not _MAY_MATCH.search(source):
        return []
    lines = source.splitlines()

    def exempt(stmt: ast.stmt) -> bool:
        numbers = [*_statement_lines(stmt), stmt.lineno - 1]
        return any(0 < number <= len(lines) and EXEMPTION.search(lines[number - 1]) for number in numbers)

    findings = []
    clocks_by_scope: dict[int, _Clocks] = {}

    def strings(stmt: ast.stmt):
        """String constants in a simple statement, or in a compound one's header."""
        for name, value in ast.iter_fields(stmt):
            if name in _BODIES:
                continue
            for item in value if isinstance(value, list) else [value]:
                if isinstance(item, ast.AST):
                    yield from (node.value for node in ast.walk(item)
                                if isinstance(node, ast.Constant) and isinstance(node.value, str))

    # Statements only: an assert is one, and a statement's lines show whether
    # it can name a lock view at all, so most expressions are never walked.
    def visit(body: list[ast.stmt], scope: ast.AST, docstring: ast.AST | None) -> None:
        for stmt in body:
            text = "\n".join(lines[number - 1] for number in _statement_lines(stmt))
            if not (isinstance(stmt, ast.Expr) and stmt.value is docstring):
                view = next((view for view in LOCK_VIEWS if view in text
                             and any(view in string for string in strings(stmt))), None)
                if view and not exempt(stmt):
                    findings.append(Finding(path, stmt.lineno, f"hand-rolled {view} query; "
                                            "use tests/pg_lock_wait.wait_blocked_by"))
            if isinstance(stmt, ast.Assert) and _orders(stmt.test):
                if id(scope) not in clocks_by_scope:
                    clocks_by_scope[id(scope)] = _clocks(scope)
                if _upper_bounds_wall_clock(stmt.test, clocks_by_scope[id(scope)]) and not exempt(stmt):
                    findings.append(Finding(path, stmt.lineno, "asserts wall-clock time under a fixed limit"))
            inner = stmt if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) else scope
            for name in _BODIES:
                for child in getattr(stmt, name, None) or []:
                    # handlers and match cases wrap their statements in a body
                    visit(child.body if not isinstance(child, ast.stmt) else [child],
                          inner, _docstring(stmt))

    tree = ast.parse(source)
    visit(tree.body, tree, _docstring(tree))
    return sorted(set(findings))


def scan_tree(root: Path, skip: frozenset[str] = frozenset()) -> list[Finding]:
    findings = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root.parent).as_posix()
        if relative in skip:
            continue
        findings.extend(scan_source(path.read_text(), relative))
    return findings
