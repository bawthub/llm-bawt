"""TASK-1308: ``next_due_after`` and the scan watermark on a private PostgreSQL.

The query reads the whole queue, so fixture rows sit far in the future where
no other test's rows land, and the watermark is placed among them.
"""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest
from sqlalchemy import text as sa_text

from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def rig():
    from types import SimpleNamespace

    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
    from llm_bawt.runtime_settings import BotProfileStore

    config = require_test_db_config()
    store = InterBotDeliveryStore(config)
    profiles = BotProfileStore(config)
    tag = uuid.uuid4().hex[:12]
    bots = [f"test-nd-{tag}-{i}" for i in range(2)]
    for slug in bots:
        profiles.upsert({"slug": slug, "name": slug, "bot_type": "agent",
                         "agent_backend": "claude-code"})
    yield SimpleNamespace(store=store, engine=store.engine, sender=bots[0], target=bots[1])
    with store.engine.begin() as conn:
        for slug in bots:
            conn.execute(sa_text("DELETE FROM inter_bot_deliveries WHERE target_bot_id=:b"), {"b": slug})
            conn.execute(sa_text("DELETE FROM bot_profiles WHERE slug=:b"), {"b": slug})


def _row(rig, *, days: float, status: str = "QUEUED") -> None:
    record, _ = rig.store.enqueue(
        sender_bot_id=rig.sender, target_bot_id=rig.target, message="fixture",
        payload={"messages": [{"role": "user", "content": "fixture"}], "bot_id": rig.target,
                 "user": "fixture-user", "stream": True},
    )
    with rig.engine.begin() as conn:
        conn.execute(sa_text("""
            UPDATE inter_bot_deliveries
            SET available_at = now() + make_interval(secs => :secs), status = :status
            WHERE id = :id
        """), {"id": record.id, "secs": days * 86400, "status": status})


def _db_now(rig):
    with rig.engine.connect() as conn:
        return conn.execute(sa_text("SELECT now()")).scalar_one()


def test_returns_the_earliest_queued_row_after_the_watermark(rig):
    _row(rig, days=9000)
    _row(rig, days=9002)
    _row(rig, days=9001, status="CANCELLED")
    _row(rig, days=8999)  # at or before the watermark: excluded
    watermark = _db_now(rig) + timedelta(days=8999, hours=1)

    due, db_now = rig.store.next_due_after(watermark)

    assert due is not None
    assert timedelta(days=8999, hours=23) < due - db_now < timedelta(days=9000, hours=1)


def test_watermark_is_exclusive_and_nothing_after_it_returns_none(rig):
    _row(rig, days=9000)
    with rig.engine.connect() as conn:
        exact = conn.execute(sa_text(
            "SELECT available_at FROM inter_bot_deliveries WHERE target_bot_id=:b"
        ), {"b": rig.target}).scalar_one()

    due, _ = rig.store.next_due_after(exact)

    assert due is None or due > exact


def test_eligible_scan_moves_the_watermark_even_when_empty(rig):
    before = _db_now(rig)
    rig.store.last_scan_at = None

    rig.store.eligible_targets()

    assert rig.store.last_scan_at is not None
    assert rig.store.last_scan_at >= before
