"""Hermetic guards for durable delivery receipts (TASK-1745).

The coverage guard keeps every status-changing ``UPDATE inter_bot_deliveries``
(and the one enqueue INSERT) staging its winning row, after the write, so a new
transition site cannot silently skip the event ledger.
"""

from __future__ import annotations

import ast
import functools
import re
from pathlib import Path

import pytest

from llm_bawt.inter_bot_delivery_record import DeliveryRecord
from llm_bawt.inter_bot_delivery_ledger import delivery_audience
from llm_bawt.service.inter_bot_delivery_events import delivery_event

SRC = Path(__file__).resolve().parents[1] / "src" / "llm_bawt"
_UPDATE = re.compile(r"UPDATE\s+inter_bot_deliveries(?:\s+\w+)?\s+SET\s+(.*?)(?:\bWHERE\b|\bFROM\b|$)",
                     re.DOTALL | re.IGNORECASE)
_STATUS = re.compile(r"(?<![\w.])(?:\w+\.)?status\s*=", re.IGNORECASE)
_DYNAMIC = "\x00"  # stands in for an f-string hole


def _sql_text(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant) else _DYNAMIC for v in node.values)
    return None


def _is_stage(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "stage"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "DeliveryTransitionEvents")


def _innermost_functions(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


@functools.lru_cache(maxsize=None)
def transition_sites(root: Path = SRC):
    """(path, function, sql lineno, staged_after, is_insert) per status-changing write.

    ``staged_after``: a ``DeliveryTransitionEvents.stage`` call follows the
    write in the same function, before that function's next status write.
    """
    sites = []
    for path in sorted(root.rglob("*.py")):
        source = path.read_text()
        if "inter_bot_deliveries" not in source:
            continue  # no SQL on the table: skip the (slow) parse
        tree = ast.parse(source, filename=str(path))
        for func in _innermost_functions(tree):
            nested = {id(n) for inner in ast.walk(func)
                      if inner is not func and isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef))
                      for n in ast.walk(inner)}
            own = [n for n in ast.walk(func) if id(n) not in nested]
            stages = [n.lineno for n in own if _is_stage(n)]
            writes: dict[int, bool] = {}
            for node in own:
                sql = _sql_text(node)
                if not sql:
                    continue
                sets = [m.group(1) for m in _UPDATE.finditer(sql)]
                if any(_STATUS.search(s) or s.lstrip().startswith(_DYNAMIC) for s in sets):
                    writes.setdefault(node.lineno, False)
                if "INSERT INTO inter_bot_deliveries" in sql:
                    writes[node.lineno] = True
            lines = sorted(writes)
            for line, nxt in zip(lines, [*lines[1:], float("inf")]):
                staged = any(line <= stage < nxt for stage in stages)
                sites.append((path.relative_to(root).as_posix(), func.name, line, staged, writes[line]))
    return tuple(sites)


def test_every_status_write_stages_its_row_after_the_write():
    sites = transition_sites()
    updates = [s for s in sites if not s[4]]
    # Not vacuous: the scan must keep finding the known transition sites.
    assert len(updates) >= 25, updates
    missing = [f"{p}:{line} {fn}" for p, fn, line, staged, _ in updates if not staged]
    assert missing == [], "status writes with no delivery event staged after the write"


def test_the_one_insert_is_staged_by_its_only_caller():
    inserts = [s for s in transition_sites() if s[4]]
    assert [(p, fn) for p, fn, *_ in inserts] == [("request_identity.py", "admit")]
    # ``enqueue`` writes through ``admit`` and must stage after it.
    tree = ast.parse((SRC / "inter_bot_delivery.py").read_text())
    func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "enqueue")
    admit = next(n.lineno for n in ast.walk(func) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "admit")
    assert any(_is_stage(n) and n.lineno > admit for n in ast.walk(func))


def test_the_guard_flags_an_unstaged_status_write(tmp_path):
    module = tmp_path / "rogue.py"
    module.write_text(
        "def rogue(conn):\n"
        "    conn.execute('''UPDATE inter_bot_deliveries SET status='FAILED' WHERE id=:id''')\n"
        "def staged_too_early(conn, row):\n"
        "    DeliveryTransitionEvents.stage(conn, row)\n"
        "    conn.execute(f'''UPDATE inter_bot_deliveries SET {x} WHERE id=:id''')\n"
        "def two_writes_one_stage(conn, row):\n"
        "    conn.execute('''UPDATE inter_bot_deliveries SET status='QUEUED' WHERE id=:a''')\n"
        "    conn.execute('''UPDATE inter_bot_deliveries d SET d.status='FAILED' WHERE id=:b''')\n"
        "    DeliveryTransitionEvents.stage(conn, row)\n"
        "def metadata_only(conn):\n"
        "    conn.execute('''UPDATE inter_bot_deliveries SET metadata_json='{}', reset_status='x'\n"
        "        WHERE status='QUEUED' ''')\n"
    )
    flagged = sorted((fn, line, staged) for _, fn, line, staged, _ in transition_sites(tmp_path))
    assert flagged == [("rogue", 2, False), ("staged_too_early", 5, False),
                       ("two_writes_one_stage", 7, False), ("two_writes_one_stage", 8, True)]


def _record(**overrides):
    base = dict(
        id="d1", sender_bot_id="alpha", target_bot_id="beta", author_entity_type="bot",
        author_entity_id="alpha", message="m", status="QUEUED", metadata_json="{}", revision=0,
        user_message_id="u1", turn_id="t1", attempt_count=0, max_attempts=5,
        created_at=None, updated_at=None, available_at=None,
    )
    base.update(overrides)
    return DeliveryRecord.from_mapping(base)


@pytest.mark.parametrize("metadata", [
    {},
    {"ordinary_chat": {"session_id": "s"}},
    {"prompt_schedule": {"id": "p"}},
    {"maintenance_user_queue": True},
    {"maintenance_user_queue": True, "ordinary_chat": {"session_id": "s"}},
])
@pytest.mark.parametrize("author", [("bot", "alpha"), ("user", "reader")])
def test_receipt_and_live_event_share_one_audience(metadata, author):
    import json

    record = _record(metadata_json=json.dumps(metadata), author_entity_type=author[0],
                     author_entity_id=author[1])
    live = delivery_event(record, default_user="owner")
    audience = delivery_audience(record, default_user="owner")
    assert (live["user_id"], live["bot_id"]) == (audience.user_id, audience.bot_id)
    assert bool(live.get("ordinary_chat")) == bool(audience.ordinary_chat_user)
