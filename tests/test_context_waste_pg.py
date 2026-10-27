"""TASK-1585: the context-waste SQL runs unchanged on PostgreSQL."""
from __future__ import annotations

import json
from datetime import timedelta

import pytest
from sqlmodel import SQLModel

from llm_bawt.service.context_waste import build_report, resolve_thresholds
from llm_bawt.service.tool_call_store import ToolCallRecord
from llm_bawt.service.turn_log_schema import TurnLog
from tests.test_context_waste import SECRET_VALUE, Seeder, _usage, scope

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture(scope="module")
def engine(private_app_pg):
    from llm_bawt.utils.db import get_shared_engine
    engine = get_shared_engine(private_app_pg)
    SQLModel.metadata.create_all(engine, tables=[TurnLog.__table__, ToolCallRecord.__table__])
    return engine


def test_report_on_postgres(engine):
    seed = Seeder(engine)
    s1 = seed.turn("alpha", session_key="s1", age=timedelta(hours=10), usage=_usage())
    seed.call(s1, "alpha", tokens=12_000, age=timedelta(hours=10))
    seed.call(seed.turn("alpha", session_key="s1", age=timedelta(hours=9)), "alpha",
              tokens=12_000, age=timedelta(hours=9))
    seed.call(seed.turn("alpha", session_key="s2", age=timedelta(hours=8)), "alpha",
              tokens=12_000, age=timedelta(hours=8))
    seed.call(s1, "alpha", tool="Grep", tokens=12_000, args={"pattern": "x"})
    seed.call(s1, "alpha", tool="mcp__playwright__browser_take_screenshot", sha="f" * 64,
              accounting=None, tokens=700_000)
    seed.call(seed.turn("secret"), "secret", sha="9" * 64, tokens=90_000)
    seed.call(seed.turn("alpha", user="stranger"), "alpha", sha="8" * 64, tokens=80_000)

    result = build_report(engine, scope(), resolve_thresholds(None))
    assert result["accounting_cutover"] is not None
    assert [(t["tool"], t["calls"]) for t in result["totals"]["trusted"]] == [("Read", 3), ("Grep", 1)]
    assert [t["tokens"] for t in result["totals"]["unverified"]] == [700_000]
    assert [(r["repeats"], r["repeat_tokens"]) for r in result["repeats"]["in_session"]] == [(1, 12_000)]
    assert [r["repeats"] for r in result["repeats"]["cross_session"]] == [1]
    assert result["repeats"]["cross_tool"][0]["tools"] == ["Grep", "Read"]
    assert {e["tool"]: e["count"] for e in result["oversized"]} == {"Read": 3, "Grep": 1}
    assert [d["eligible_turns"] for d in result["composition"]] == [1]
    text = json.dumps(result)
    assert SECRET_VALUE not in text and "secret" not in result["bots"]
    assert "90000" not in text and "80000" not in text
