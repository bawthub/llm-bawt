"""PostgreSQL tests for TASK-1252 S1: idle inbox batches settle as one unit.

An idle opted-in bot's head claim takes the plain rows behind it into the
same turn. Followers never head the queue and carry no lease; a trigger
settles them with their head on every head status change.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text as sa_text

from tests.test_delivery_claim_gate_pg import _enqueue
from tests.test_turn_gates_pg import gated  # noqa: F401, F811  (fixture)

pytestmark = pytest.mark.usefixtures("private_app_pg")


def _opt_in(gated, bot, on=True):
    from llm_bawt.runtime_settings import RuntimeSettingsStore

    RuntimeSettingsStore(gated.store.config).set_value("bot", bot, "inbox_batch", on)


def _send(gated, bot, message, **payload_extra):
    deliveries, record = _enqueue(gated, bot, message)
    if payload_extra:
        import json
        with deliveries.engine.begin() as conn:
            conn.execute(sa_text(
                "UPDATE inter_bot_deliveries SET payload_json=payload_json || CAST(:p AS jsonb) WHERE id=:id"
            ), {"id": record.id, "p": json.dumps(payload_extra)})
    return deliveries, record


def _rows(deliveries, ids):
    with deliveries.engine.connect() as conn:
        return {r["id"]: r for r in conn.execute(sa_text(
            "SELECT id, status, delivery_mode, batch_head_id, lease_expires_at, metadata_json"
            " FROM inter_bot_deliveries WHERE id = ANY(:ids)"), {"ids": list(ids)}).mappings()}


def _claim(deliveries, bot):
    return deliveries.claim_next(bot, claim_owner="worker", steer_capable=False)


def _set_head(deliveries, head_id, assignments):
    with deliveries.engine.begin() as conn:
        conn.execute(sa_text(f"UPDATE inter_bot_deliveries SET {assignments} WHERE id=:id"),
                     {"id": head_id})


@pytest.mark.integration
def test_idle_backlog_becomes_one_turn(gated):
    bot = gated.bot()
    _opt_in(gated, bot)
    deliveries, a = _send(gated, bot, "A")
    _, b = _send(gated, bot, "B")
    _, c = _send(gated, bot, "C")

    head = _claim(deliveries, bot)

    assert head is not None and head.id == a.id and head.status == "DISPATCHING"
    prompt = head.metadata["batch_prompt"]
    assert "1/3" in prompt and prompt.index("A") < prompt.index("B") < prompt.index("C")
    assert head.metadata["batch_member_ids"] == [b.id, c.id]
    rows = _rows(deliveries, [b.id, c.id])
    for row in rows.values():
        assert row["status"] == "DISPATCHING" and row["delivery_mode"] == "batched"
        assert row["batch_head_id"] == a.id and row["lease_expires_at"] is None
    # Followers never head the queue: nothing else is claimable.
    assert _claim(deliveries, bot) is None
    with deliveries.engine.connect() as conn:
        prompt_row = conn.execute(sa_text("SELECT user_prompt FROM turn_logs WHERE id=:id"),
                                  {"id": a.turn_id}).scalar_one()
    assert prompt_row == prompt


@pytest.mark.integration
def test_not_opted_in_keeps_one_turn_per_row(gated):
    bot = gated.bot()
    deliveries, a = _send(gated, bot, "A")
    _, b = _send(gated, bot, "B")

    head = _claim(deliveries, bot)

    assert head.id == a.id and "batch_prompt" not in head.metadata
    assert _rows(deliveries, [b.id])[b.id]["status"] == "QUEUED"


@pytest.mark.integration
def test_a_waited_send_ends_the_batch(gated):
    bot = gated.bot()
    _opt_in(gated, bot)
    deliveries, a = _send(gated, bot, "A")
    _, b = _send(gated, bot, "B")
    _, waited = _send(gated, bot, "W", reply_mode="waited")
    _, d = _send(gated, bot, "D")

    head = _claim(deliveries, bot)

    assert head.metadata["batch_member_ids"] == [b.id]
    rows = _rows(deliveries, [waited.id, d.id])
    assert rows[waited.id]["status"] == "QUEUED" and rows[d.id]["status"] == "QUEUED"


@pytest.mark.integration
def test_a_waited_head_runs_alone(gated):
    bot = gated.bot()
    _opt_in(gated, bot)
    deliveries, waited = _send(gated, bot, "W", reply_mode="waited")
    _, b = _send(gated, bot, "B")

    head = _claim(deliveries, bot)

    assert head.id == waited.id and "batch_prompt" not in head.metadata
    assert _rows(deliveries, [b.id])[b.id]["status"] == "QUEUED"


@pytest.mark.integration
def test_batch_settles_with_its_head(gated):
    bot = gated.bot()
    _opt_in(gated, bot)
    deliveries, a = _send(gated, bot, "A")
    _, b = _send(gated, bot, "B")
    _claim(deliveries, bot)

    _set_head(deliveries, a.id, "status='DELIVERED', delivered_at=now()")

    rows = _rows(deliveries, [a.id, b.id])
    assert rows[b.id]["status"] == "DELIVERED"
    assert rows[a.id]["metadata_json"]["batch_member_ids"] == [b.id]  # kept as history


@pytest.mark.integration
def test_requeued_head_releases_its_followers_once(gated):
    bot = gated.bot()
    _opt_in(gated, bot)
    deliveries, a = _send(gated, bot, "A")
    _, b = _send(gated, bot, "B")
    _claim(deliveries, bot)

    _set_head(deliveries, a.id, "status='QUEUED', claim_token=NULL, lease_expires_at=NULL")

    rows = _rows(deliveries, [a.id, b.id])
    assert rows[b.id]["status"] == "QUEUED" and rows[b.id]["delivery_mode"] is None
    assert "batch_member_ids" not in rows[a.id]["metadata_json"]
    assert "batch_prompt" not in rows[a.id]["metadata_json"]
    # The next claim rebatches the same rows on the head's turn.
    again = _claim(deliveries, bot)
    assert again.id == a.id and again.metadata["batch_member_ids"] == [b.id]


@pytest.mark.integration
def test_restart_pending_failure_releases_and_plain_failure_fails_followers(gated):
    bot = gated.bot()
    _opt_in(gated, bot)
    deliveries, a = _send(gated, bot, "A")
    _, b = _send(gated, bot, "B")
    _claim(deliveries, bot)
    _set_head(deliveries, a.id, "status='FAILED', metadata_json=metadata_json || "
              "'{\"restart_retry_pending\": true}'::jsonb")
    assert _rows(deliveries, [b.id])[b.id]["status"] == "QUEUED"

    bot2 = gated.bot()
    _opt_in(gated, bot2)
    deliveries, x = _send(gated, bot2, "X")
    _, y = _send(gated, bot2, "Y")
    _claim(deliveries, bot2)
    _set_head(deliveries, x.id, "status='FAILED', last_error='boom'")
    assert _rows(deliveries, [y.id])[y.id]["status"] == "FAILED"


@pytest.mark.integration
def test_lease_recovery_requeues_the_batch_as_a_unit(gated):
    bot = gated.bot()
    _opt_in(gated, bot)
    deliveries, a = _send(gated, bot, "A")
    _, b = _send(gated, bot, "B")
    _, c = _send(gated, bot, "C")
    _claim(deliveries, bot)
    _set_head(deliveries, a.id, "lease_expires_at=now() - interval '1 second'")

    deliveries.recover_expired()

    rows = _rows(deliveries, [a.id, b.id, c.id])
    assert all(r["status"] == "QUEUED" for r in rows.values())
    with deliveries.engine.connect() as conn:
        link = conn.execute(sa_text(
            "SELECT CAST(request_json AS jsonb)->>'inter_bot_delivery_id' FROM turn_logs WHERE id=:id"
        ), {"id": a.turn_id}).scalar_one()
    assert link == a.id  # one owner: the head, never a follower


@pytest.mark.integration
def test_row_cap_and_head_always_admitted(gated, monkeypatch):
    from llm_bawt import inter_bot_batch

    bot = gated.bot()
    _opt_in(gated, bot)
    deliveries, a = _send(gated, bot, "A" * 50)
    _, b = _send(gated, bot, "B")
    monkeypatch.setattr(inter_bot_batch, "MAX_CHARS", 10)  # head alone is over

    head = _claim(deliveries, bot)

    assert head.id == a.id and "batch_prompt" not in head.metadata
    assert _rows(deliveries, [b.id])[b.id]["status"] == "QUEUED"
