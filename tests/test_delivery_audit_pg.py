"""TASK-2280: claim exemptions, accepted batches, and terminal ownership."""
# ruff: noqa: F811 -- imported pytest fixture is injected by parameter name.
import pytest
from sqlalchemy import text

from llm_bawt.inter_bot_batch import enabled
from llm_bawt.runtime_settings import RuntimeSettingsStore
from tests.test_delivery_claim_gate_pg import _enqueue
from tests.test_delivery_inbox_batch_pg import _opt_in, _send, _claim
from tests.test_delivery_steer_past_held_head_pg import _wait_for_turn_end
from tests.test_turn_gates_pg import _open, gated  # noqa: F401, F811
from tests.test_user_stop_hold_pg import _hold, _user_chat

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def test_steer_jumper_does_not_inherit_user_stop_exemption(gated):
    bot = gated.bot()
    deliveries, _ = _enqueue(gated, bot, "backlog")
    user = _user_chat(gated, deliveries, bot)
    _wait_for_turn_end(deliveries, user)
    _, jumper = _enqueue(gated, bot, "bot steer must wait")
    _open(gated.store, bot)
    _hold(gated, bot)
    assert deliveries.claim_next(bot, claim_owner="worker", steer_capable=True) is None
    assert deliveries.get(jumper.id).status == "QUEUED"
    assert deliveries.get(jumper.id).attempt_count == 0


def test_accepted_batch_followers_do_not_block_steer_jumper(gated):
    bot = gated.bot()
    _opt_in(gated, bot)
    deliveries, _ = _send(gated, bot, "batch head")
    _, follower = _send(gated, bot, "batch follower")
    head = _claim(deliveries, bot)
    assert head.metadata["batch_member_ids"] == [follower.id]
    assert deliveries.mark_transport_accepted(head.id, head.claim_token)
    with deliveries.engine.begin() as conn:
        conn.execute(text("UPDATE turn_logs SET status='streaming' WHERE id=:id"), {"id": head.turn_id})
    _, held = _send(gated, bot, "wait for idle", prefer_steer=False)
    _, jumper = _send(gated, bot, "steer now")
    claimed = deliveries.claim_next(bot, claim_owner="worker", steer_capable=True)
    assert claimed is not None and claimed.id == jumper.id and claimed.status == "STEERING"
    assert deliveries.get(held.id).status == "QUEUED"


@pytest.mark.parametrize("status", ["QUEUED", "DISPATCHING", "STEERING", "CANCELLED", "DELIVERED"])
def test_preclaim_failure_cannot_overwrite_claimed_or_terminal_row(gated, status):
    bot = gated.bot()
    deliveries, row = _enqueue(gated, bot, "fixture")
    with deliveries.engine.begin() as conn:
        conn.execute(text("UPDATE inter_bot_deliveries SET status=:status WHERE id=:id"),
                     {"status": status, "id": row.id})
    changed = deliveries.mark_failed(row.id, "unknown target")
    assert (changed is not None) is (status == "QUEUED")
    assert deliveries.get(row.id).status == ("FAILED" if status == "QUEUED" else status)


@pytest.mark.parametrize("bot_value,expected", [(None, True), (False, False), (True, True)])
def test_inbox_batch_inherits_global_unless_bot_has_nonnull_value(gated, bot_value, expected):
    bot = gated.bot()
    settings = RuntimeSettingsStore(gated.store.config)
    # Roll back both settings so no other fixture observes the global override.
    settings.set_value("bot", bot, "inbox_batch", bot_value)
    with settings.engine.connect() as conn:
        transaction = conn.begin()
        try:
            conn.execute(text("DELETE FROM runtime_settings WHERE scope_type='global' AND key='inbox_batch'"))
            conn.execute(text("INSERT INTO runtime_settings (scope_type, scope_id, key, value_json, updated_at) "
                              "VALUES ('global', '*', 'inbox_batch', 'true', now())"))
            assert enabled(conn, bot) is expected
        finally:
            transaction.rollback()
