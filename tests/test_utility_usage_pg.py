"""TASK-1538 PostgreSQL contract: utility usage rows persist and roll up per day x caller."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from llm_bawt.service.turn_logs import TurnLogStore
from llm_bawt.service.usage import utility_usage
from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def engine():
    store = TurnLogStore(require_test_db_config())  # bootstraps the schema
    with store.engine.begin() as conn:
        conn.execute(text("DELETE FROM utility_usage"))
    yield store.engine
    with store.engine.begin() as conn:
        conn.execute(text("DELETE FROM utility_usage"))


def _row(caller, *, at, inp=100, out=10, estimated=False, outcome="ok", bot="caid"):
    box = {} if estimated else {"usage": {"input_tokens": inp, "cache_read_tokens": 5,
                                          "cache_write_tokens": 0, "output_tokens": out}}
    row = utility_usage.build_row(caller=caller, bot_id=bot, model="m", box=box,
                                  estimate={"input_tokens": inp, "output_tokens": out},
                                  elapsed_ms=12.7, outcome=outcome)
    row["created_at"] = at
    return row


def test_rows_persist_and_roll_up_per_day_and_caller(engine):
    now = datetime(2026, 11, 8, 15, tzinfo=timezone.utc)
    yesterday = now - timedelta(days=1)
    for row in (_row("recall", at=now), _row("recall", at=now, estimated=True, outcome="timeout"),
                _row("title", at=now, inp=7, out=3, bot=None), _row("recall", at=yesterday),
                _row("recall", at=now - timedelta(days=5))):
        utility_usage.record(row, engine=engine)
    with engine.connect() as conn:
        stored = conn.execute(text("SELECT * FROM utility_usage ORDER BY id")).mappings().all()
        today = utility_usage.rollup(conn, days=1, now=now)
        two = utility_usage.rollup(conn, days=2, now=now)
    assert len(stored) == 5 and stored[0]["elapsed_ms"] == 12 and stored[2]["bot_id"] is None
    assert today == [
        {"day": "2026-11-08", "caller": "recall", "calls": 2, "input_tokens": 200,
         "cache_read_tokens": 5, "cache_write_tokens": 0, "output_tokens": 20,
         "estimated_calls": 1, "failed_calls": 1},
        {"day": "2026-11-08", "caller": "title", "calls": 1, "input_tokens": 7,
         "cache_read_tokens": 5, "cache_write_tokens": 0, "output_tokens": 3,
         "estimated_calls": 0, "failed_calls": 0},
    ]
    assert [(r["day"], r["caller"]) for r in two] == [
        ("2026-11-08", "recall"), ("2026-11-08", "title"), ("2026-11-07", "recall")]
