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


# ---------------------------------------------------------------------------
# Static scanners (TASK-1457): ``make test-timing-report`` lists what these find.
#   D4: a thread-side wait that can time out silently, or with a short budget
#       (report-only).
#   D7: src/ binding a clock function or sleep by name, which follow_time (or a
#       test patching ``time``) can't redirect. Enforced by test_timing_guard.
#       An injected ``clock=time.monotonic`` default is the sanctioned seam
#       and is not flagged.
# Awaited calls and asyncio.* are loop-side: the virtual clock (D1) covers them.
# ---------------------------------------------------------------------------

SHORT_WAIT_S = 5.0
_WAIT_METHODS = frozenset({"wait", "join", "get", "result"})


def _timeout_literal(call: ast.Call) -> float | None:
    """The call's timeout when it is a numeric literal (``get`` by keyword, or
    positionally after a literal ``block`` bool: ``mapping.get(1)``,
    ``kwargs.get("k", 1)`` and ``session.get(Row, 1)`` are lookups)."""
    name = call.func.attr  # type: ignore[union-attr]
    candidates = [kw.value for kw in call.keywords if kw.arg == "timeout"]
    if (name == "get" and len(call.args) >= 2  # queue.get(True, timeout)
            and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, bool)):
        candidates.append(call.args[1])
    elif name != "get" and call.args:
        candidates.append(call.args[0])
    for node in candidates:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return float(node.value)
    return None


def _has_timeout(call: ast.Call) -> bool:
    return bool(call.args) or any(kw.arg == "timeout" for kw in call.keywords)


def _thread_wait(node: ast.AST, awaited: set[int]) -> ast.Call | None:
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in _WAIT_METHODS and id(node) not in awaited):
        return None
    receiver = node.func.value
    if isinstance(receiver, ast.Name) and receiver.id == "asyncio":
        return None
    if isinstance(receiver, ast.Constant):  # ", ".join(parts)
        return None
    return node


def _barrier_names(tree: ast.AST) -> set[str]:
    """Names bound to ``Barrier(...)``: its wait raises BrokenBarrierError on
    timeout, so its result carries nothing to check.

    Fails closed: the scan is module-wide, not per scope, so a name counts
    only when every binding of it in the module (assignment, parameter,
    loop/with target) is a Barrier. A ``ready`` that is a Barrier in one
    function and an Event in another stays checked.
    """
    barrier_targets = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            if (func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)) == "Barrier":
                barrier_targets.update(id(target) for target in node.targets
                                       if isinstance(target, (ast.Name, ast.Attribute)))
    barriers, others = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            others.add(node.arg)
        elif isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Store):
            (barriers if id(node) in barrier_targets else others).add(ast.unparse(node))
    return barriers - others


def _checks_alive(statements: list[ast.stmt], receiver: str) -> bool:
    return any(isinstance(node, ast.Attribute) and node.attr == "is_alive"
               and ast.unparse(node.value) == receiver
               for stmt in statements for node in ast.walk(stmt))


def scan_thread_waits(source: str, path: str) -> list[Finding]:
    """D4: ``.wait/.join/.get/.result`` with a literal timeout under 5s; a bare
    ``x.wait(t)`` whose bool is dropped; ``x.join(t)`` never followed by an
    ``x.is_alive()`` check."""
    if not re.search(r"\.(?:wait|join|get|result)\(", source):
        return []
    lines = source.splitlines()
    tree = ast.parse(source)
    awaited = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Await)}
    barriers = _barrier_names(tree)
    findings: set[Finding] = set()

    def exempt(line: int) -> bool:
        return any(0 < number <= len(lines) and EXEMPTION.search(lines[number - 1])
                   for number in (line, line - 1))

    def add(line: int, kind: str) -> None:
        if not exempt(line):
            findings.add(Finding(path, line, kind))

    for node in ast.walk(tree):
        call = _thread_wait(node, awaited)
        if call is not None:
            timeout = _timeout_literal(call)
            if timeout is not None and 0 < timeout < SHORT_WAIT_S:
                add(call.lineno, f".{call.func.attr}({timeout:g}) thread wait under {SHORT_WAIT_S:g}s")  # type: ignore[union-attr]
        for field in _BODIES:
            body = getattr(node, field, None)
            if not isinstance(body, list):
                continue
            for index, stmt in enumerate(body):
                if not isinstance(stmt, ast.Expr):
                    continue
                call = _thread_wait(stmt.value, awaited)
                if call is None or not _has_timeout(call):
                    continue
                # Barrier.wait raises on timeout: known by its constructor or its name.
                receiver = ast.unparse(call.func.value)  # type: ignore[union-attr]
                if (call.func.attr == "wait"  # type: ignore[union-attr]
                        and receiver not in barriers and "barrier" not in receiver.lower()):
                    add(stmt.lineno, ".wait(timeout) result ignored; a timeout passes silently")
                elif call.func.attr == "join" and not _checks_alive(  # type: ignore[union-attr]
                        body[index + 1:], ast.unparse(call.func.value)):  # type: ignore[union-attr]
                    add(stmt.lineno, ".join(timeout) without an is_alive() check after it")
    return sorted(findings)


BOUND_TIME_NAMES = CLOCK_CALLS | {"sleep"}


def scan_clock_imports(source: str, path: str) -> list[Finding]:
    """D7: ``from time import monotonic`` (or ``sleep``) binds the function at
    import, so the report's virtual clock (and any test patching
    ``time.monotonic``/``time.sleep``) can't reach that module. Use
    ``import time`` and ``time.monotonic()``."""
    if "from time import" not in source:
        return []
    return sorted({Finding(path, node.lineno, f"from time import {alias.name}; use time.{alias.name}()")
                   for node in ast.walk(ast.parse(source))
                   if isinstance(node, ast.ImportFrom) and node.module == "time"
                   for alias in node.names if alias.name in BOUND_TIME_NAMES})


def report_findings(repo: Path, skip: frozenset[str] = frozenset()) -> dict[str, list[Finding]]:
    """The static half of the timing report: D4 over tests/, D7 over src/."""
    scans = {"D4": (repo / "tests", scan_thread_waits), "D7": (repo / "src", scan_clock_imports)}
    report: dict[str, list[Finding]] = {}
    for key, (root, scan) in scans.items():
        report[key] = [finding for path in sorted(root.rglob("*.py"))
                       if (relative := path.relative_to(repo).as_posix()) not in skip
                       for finding in scan(path.read_text(), relative)]
    return report
