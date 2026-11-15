"""Independent TASK-1252 review probes; copy into tests/ to run."""
import json

import pytest
from sqlalchemy import text

from llm_bawt.inter_bot_delivery_retry import fail_claim_with_restart_retry
from llm_bawt.service.gate_steer_escalation import stamp_send_now
from tests.test_delivery_claim_gate_pg import _backstop_gate
from tests.test_delivery_inbox_batch_pg import _claim, _opt_in, _send
from tests.test_inter_bot_delivery_events_pg import _events, ledger  # noqa: F401
from tests.test_inter_bot_delivery_events_pg import _send as ledger_send
from tests.test_inter_bot_delivery_retry import SHUTDOWN
from tests.test_turn_gates_pg import gated  # noqa: F401
from tests.test_user_stop_hold_pg import _user_chat

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def _shutdown(deliveries, head):
    with deliveries.engine.begin() as conn:
        conn.execute(text("""
            UPDATE turn_logs SET status='aborted', ended_at=now(),
                request_json=CAST(:audit AS text) WHERE id=:id
        """), {"audit": json.dumps(SHUTDOWN), "id": head.turn_id})


def test_live_shutdown_retries_the_followers_with_the_head(gated, monkeypatch):
    monkeypatch.setattr("llm_bawt.inter_bot_delivery_retry.resolve_restart_retry_text", lambda _: "Resume")
    bot = gated.bot()
    _opt_in(gated, bot)
    deliveries, a = _send(gated, bot, "head-A")
    _, b = _send(gated, bot, "follower-B")
    head = _claim(deliveries, bot)
    _shutdown(deliveries, head)
    failed, child = fail_claim_with_restart_retry(deliveries, head, head.claim_token, "shutdown")
    assert failed is not None and child is not None
    queued = deliveries.get(b.id)
    assert queued.status != "FAILED" or "follower-B" in child.message, (
        queued.status, child.message, deliveries.payload(child.id)["messages"],
    )


def test_whole_stack_recovery_keeps_original_batch_order(gated, monkeypatch):
    monkeypatch.setattr("llm_bawt.inter_bot_delivery_retry.resolve_restart_retry_text", lambda _: "Resume")
    bot = gated.bot()
    _opt_in(gated, bot)
    deliveries, a = _send(gated, bot, "head-A")
    _, b = _send(gated, bot, "follower-B")
    head = _claim(deliveries, bot)
    _shutdown(deliveries, head)
    with deliveries.engine.begin() as conn:
        conn.execute(text("UPDATE inter_bot_deliveries SET lease_expires_at=now()-interval '1 second' WHERE id=:id"),
                     {"id": head.id})
    recovered = deliveries.recover_expired()
    child = next(row for row in recovered if row.metadata.get("restart_retry_of") == a.id)
    claimed = _claim(deliveries, bot)
    assert claimed is not None and claimed.id == child.id, (claimed.id if claimed else None, b.id, child.id)


@pytest.mark.parametrize("batch", [False, True])
def test_accepted_batch_does_not_fence_user_bypass(gated, batch):
    bot = gated.bot()
    _opt_in(gated, bot, batch)
    deliveries, a = _send(gated, bot, "head-A")
    if batch:
        _send(gated, bot, "follower-B")
    head = _claim(deliveries, bot)
    assert deliveries.mark_transport_accepted(head.id, head.claim_token)
    with deliveries.engine.begin() as conn:
        conn.execute(text("UPDATE turn_logs SET status='streaming' WHERE id=:id"), {"id": a.turn_id})
    _send(gated, bot, "held backlog")
    user = _user_chat(gated, deliveries, bot)
    _backstop_gate(gated, bot)
    stamp_send_now(deliveries.engine, user.id, "nick")
    candidate = deliveries.claim_next(bot, claim_owner="worker", steer_capable=True)
    assert candidate is not None and candidate.id == user.id


@pytest.mark.parametrize("compact_first", [False, True])
def test_compact_is_a_batch_barrier(gated, compact_first):
    bot = gated.bot()
    _opt_in(gated, bot)
    deliveries, a = _send(gated, bot, "/compact" if compact_first else "head-A")
    _, b = _send(gated, bot, "follower-B" if compact_first else "/compact")
    compact = a if compact_first else b
    with deliveries.engine.begin() as conn:
        conn.execute(text("""
            UPDATE inter_bot_deliveries SET message_kind='CONTEXT_MAINTENANCE',
                metadata_json=metadata_json || '{"context_action":"compact"}'::jsonb
            WHERE id=:id
        """), {"id": compact.id})
    head = _claim(deliveries, bot)
    assert "batch_member_ids" not in head.metadata, head.metadata.get("batch_prompt")


@pytest.mark.parametrize("clear_mode", [False, True])
def test_refunded_requeue_emits_follower_receipt(ledger, clear_mode):
    bot = ledger.bot()
    _opt_in(ledger, bot)
    a, _ = ledger_send(ledger, ledger.bot(), bot)
    b, _ = ledger_send(ledger, ledger.bot(), bot)
    head = _claim(ledger.deliveries, bot)
    queued = ledger.deliveries.requeue(
        head.id, head.claim_token, "admission refused", delay_seconds=0,
        clear_delivery_mode=clear_mode, refund_attempt=True,
    )
    assert queued is not None and ledger.deliveries.get(b.id).status == "QUEUED"
    phases = [event["payload"]["phase"] for event in _events(ledger, b.id)]
    assert phases == ["QUEUED", "DISPATCHING", "QUEUED"], phases
